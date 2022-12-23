#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp
import time
from mmcv import ProgressBar
import subprocess
from torch_mir_eval import bss_eval_sources
import random
import numpy as np
import math
from collections import defaultdict
import warnings
import platform
import pickle
import time

from options.train_options import TrainOptions
from utils import utils
from utils.text_decoder import ctc_greedy_decode, compute_cer

from data.data_loader import CreateDataLoader
from data.audioVisual_dataset import charToIx32, charToIx40, IxTochar32
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from models import criterion

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def detect_anomalous_parameters(loss, model):
    parameters_in_graph = set()
    visited = set()

    def traverse(grad_fn):
        if grad_fn is None:
            return
        if grad_fn not in visited:
            visited.add(grad_fn)
            if hasattr(grad_fn, 'variable'):
                parameters_in_graph.add(grad_fn.variable)
            parents = grad_fn.next_functions
            if parents is not None:
                for parent in parents:
                    grad_fn = parent[0]
                    traverse(grad_fn)

    traverse(loss.grad_fn)
    for n, p in model.named_parameters():
        if p not in parameters_in_graph and p.requires_grad:
            print(f"{n} with shape {p.size()} is not in the computational graph \n", flush=True)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_input_valid(opt):
    # check invalidation of recognition parameters
    if opt.audio_augmentation:
        assert opt.noise_file is not None and opt.noise_root is not None


############## DDP relative
def _init_slurm(opt):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if opt.port is not None:
        os.environ['MASTER_PORT'] = str(opt.port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend='nccl')
    opt.rank = proc_id
    opt.local_rank = proc_id % num_gpus
    opt.device = torch.device("cuda", proc_id % num_gpus)


def _init_pytorch(opt):
    rank = int(os.environ['RANK'])
    local_rank = rank % torch.cuda.device_count()
    opt.device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    opt.local_rank = local_rank
    opt.rank = rank
    dist.init_process_group(backend="nccl")


def _reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


################ init
def _init():
    opt = TrainOptions().parse()

    warnings.filterwarnings('ignore')

    for x in vars(opt):
        value = getattr(opt, x)
        if value == "true":
            setattr(opt, x, True)
        elif value == "false":
            setattr(opt, x, False)

    check_input_valid(opt)

    # set random seed
    set_random_seed(opt.seed)

    # init
    _init_slurm(opt)

    # prevent stucking
    if platform.system() != 'Windows':
        # https://github.com/pytorch/pytorch/issues/973
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        hard_limit = rlimit[1]
        soft_limit = min(4096, hard_limit)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

    writer = None
    if opt.rank == 0:
        args = vars(opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = osp.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = osp.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        if opt.tensorboard:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(os.path.join(osp.dirname(opt.checkpoints_dir), 'runs', opt.name))
    opt.writer = writer

    return opt


################ construct model, optimizer, loss
def create_model(opt):
    if opt.visual_feature_type == 'both':
        visual_feature_dim = 512 + opt.identity_feature_dim
    elif opt.visual_feature_type == 'lipmotion':
        visual_feature_dim = 512
    else:  # identity
        visual_feature_dim = opt.identity_feature_dim

    # Network Builders
    builder = ModelBuilder()
    nets = []

    net_lipreading = builder.build_lipreadingnet(
        opt=opt,
        config_path=opt.lipreading_config_path,
        weights=opt.weights_lipreadingnet,
        extract_feats=opt.lipreading_extract_feature)
    if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'lipreading_latest.pth')):
        ckpt_path = osp.join(opt.checkpoints_dir, opt.name, 'lipreading_latest.pth')
        net_lipreading.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        if opt.rank == 0:
            print(f'resume net_lipreading from {ckpt_path}')
    nets.append(net_lipreading)

    # if identity feature dim is not 512, for resnet reduce dimension to this feature dim
    if opt.identity_feature_dim != 512:
        opt.with_fc = True
    else:
        opt.with_fc = False
    net_facial_attribtes = builder.build_facial(
        opt=opt,
        pool_type=opt.visual_pool,
        fc_out=opt.identity_feature_dim,
        with_fc=opt.with_fc,
        weights=opt.weights_facial)
    if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'facial_latest.pth')):
        ckpt_path = osp.join(opt.checkpoints_dir, opt.name, 'facial_latest.pth')
        net_facial_attribtes.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        if opt.rank == 0:
            print(f'resume net_facial_attribtes from {ckpt_path}')
    nets.append(net_facial_attribtes)

    net_unet = builder.build_unet(
        opt=opt,
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
        audioVisual_feature_dim=opt.unet_ngf * 8 + visual_feature_dim,
        weights=opt.weights_unet)
    if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'unet_latest.pth')):
        ckpt_path = osp.join(opt.checkpoints_dir, opt.name, 'unet_latest.pth')
        net_unet.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        if opt.rank == 0:
            print(f'resume net_unet from {ckpt_path}')
    nets.append(net_unet)

    # refine_net = builder.build_refine_net(
    #     opt=opt,
    #     num_layers=opt.refine_num_layers,
    #     residual_last=opt.residual_last,
    #     kernel_size=opt.refine_kernel_size,
    #     weights=opt.weights_refine
    # )
    refine_net = builder.build_refine_net(
        opt=opt,
        num_layers=opt.refine_num_layers,
        weights=opt.weights_refine
    )
    if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'refine_latest.pth')):
        ckpt_path = osp.join(opt.checkpoints_dir, opt.name, 'refine_latest.pth')
        refine_net.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        if opt.rank == 0:
            print(f'resume refine_net from {ckpt_path}')
    nets.append(refine_net)

    if opt.use_contrast_loss:
        net_vocal = builder.build_vocal(
            opt=opt,
            pool_type=opt.audio_pool,
            input_channel=2,
            with_fc=opt.with_fc,
            fc_out=visual_feature_dim,
            weights=opt.weights_vocal
        )
        if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'vocal_latest.pth')):
            ckpt_path = osp.join(opt.checkpoints_dir, opt.name, 'vocal_latest.pth')
            net_vocal.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            if opt.rank == 0:
                print(f'resume net_vocal from {ckpt_path}')
        nets.append(net_vocal)

    # construct our audio-visual model
    model = AudioVisualModel(nets, opt)
    model.to(opt.device)
    model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
    return model


def create_optimizer(model, opt):
    model2 = model.module
    net_lipreading, net_facial_attribtes, net_unet, net_refine = model2.net_lipreading, model2.net_identity, model2.net_unet, model2.net_refine
    param_groups = [{'params': net_lipreading.parameters(), 'lr': opt.lr_lipreading},
                    {'params': net_facial_attribtes.parameters(), 'lr': opt.lr_facial_attributes},
                    {'params': net_unet.parameters(), 'lr': opt.lr_unet},
                    {'params': net_refine.parameters(), 'lr': opt.lr_refine}]
    if opt.use_contrast_loss:
        net_vocal = model2.net_vocal
        param_groups.append({'params': net_vocal.parameters(), 'lr': opt.lr_vocal})
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
    if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'resume_latest.pth')):
        ckpt = osp.join(opt.checkpoints_dir, opt.name, 'resume_latest.pth')
        optimizer.load_state_dict(torch.load(ckpt, map_location='cpu')['optim_state_dict'])
    return optimizer


def create_loss(opt):
    crit = {}
    if opt.use_mixandseparate_loss:
        if opt.mask_loss_type == 'L1':
            loss_mixandseparate = criterion.L1Loss()
        elif opt.mask_loss_type == 'L2':
            loss_mixandseparate = criterion.L2Loss()
        else:  # BCE
            loss_mixandseparate = criterion.BCELoss()
        loss_mixandseparate.to(opt.device)
        crit['loss_mixandseparate'] = loss_mixandseparate
    if opt.use_sisnr_loss:
        loss_sisnr = criterion.SISNRLoss()
        loss_sisnr.to(opt.device)
        crit['loss_sisnr'] = loss_sisnr
    if opt.use_contrast_loss:
        if opt.contrast_loss_type == 'TripletLossCosine':
            loss_contrast = criterion.TripletLossCosine(opt.contrast_margin)
        elif opt.contrast_loss_type == 'TripletLossCosine2':
            loss_contrast = criterion.TripletLossCosine2(opt.contrast_margin)
        elif opt.contrast_loss_type == "NCELoss":
            loss_contrast = criterion.NCELoss(opt.contrast_temp)
        elif opt.contrast_loss_type == "NCELoss2":
            loss_contrast = criterion.NCELoss2(opt.contrast_temp)
        loss_contrast.to(opt.device)
        crit['loss_contrast'] = loss_contrast
    return crit


##################


def decrease_learning_rate(optimizer, decay_factor=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor


def display_val(model, crit, writer, index, data_loader_val, epoch, opt):
    # print(f"val, rank:{dist.get_rank()}, begin val\n\n\n", flush=True)
    data_loader_val.set_epoch(epoch)
    if opt.rank == 0:
        pb = ProgressBar(len(data_loader_val), start=False)
        pb.start()
    window = opt.window
    mixandseparate_losses = []
    sisnr_losses = []
    contrast_losses = []
    # recognition_losses = []
    # distillation_losses = []
    sdrs_dict, sirs_dict, sars_dict = defaultdict(list), defaultdict(list), defaultdict(list)
    with torch.no_grad():
        # print(f"val, rank:{dist.get_rank()} has {len(data_loader_val)} batches", flush=True)
        # num_batch = len(data_loader_val)

        time.sleep(5)

        for i, val_data in enumerate(data_loader_val):
            # print(f"val, batch:{i + 1} / {num_batch}, rank:{dist.get_rank()}, before forward", flush=True)
            output = model.forward(val_data)
            if opt.use_mixandseparate_loss:
                mixandseparate_loss = get_mixandseparate_loss(opt, output, crit['loss_mixandseparate']) * opt.mixandseparate_loss_weight
                reduced_mix_loss = _reduce_tensor(mixandseparate_loss.data)
                mixandseparate_losses.append(reduced_mix_loss.item())
            if opt.use_sisnr_loss:
                sisnr_loss = get_sisnr_loss(opt, output, crit['loss_sisnr']) * opt.sisnr_loss_weight
                reduced_sisnr_loss = _reduce_tensor(sisnr_loss.data)
                sisnr_losses.append(reduced_sisnr_loss.item())
            if opt.use_contrast_loss:
                contrast_loss = get_contrast_loss(opt, output, crit['loss_contrast']) * opt.contrast_loss_weight
                reduced_con_loss = _reduce_tensor(contrast_loss.data)
                contrast_losses.append(reduced_con_loss.item())
            # print(f"val, batch:{i + 1} / {num_batch}, rank:{dist.get_rank()}, after loss", flush=True)
            try:
                sdr_d, sir_d, sar_d = calculate_sdr(output, window, opt)
                for key in sdr_d.keys():
                    sdrs_dict[key] += sdr_d[key]
                for key in sir_d.keys():
                    sirs_dict[key] += sir_d[key]
                for key in sar_d.keys():
                    sars_dict[key] += sar_d[key]
            except Exception as e:
                pass
            # for mix in range(2, 6):
            #     sdrs_dict[f"sdr{mix}"] += [1.0]
            #     sirs_dict[f"sir{mix}"] += [1.0]
            #     sars_dict[f"sar{mix}"] += [1.0]
            #     sisnrs_dict[f"sisnr{mix}"] += [1.0]
            if opt.rank == 0:
                pb.update()
            # print(f"val, batch: {i + 1} / {num_batch}, rank:{dist.get_rank()}, before barrier", flush=True)
            dist.barrier()
            # print(f"val, batch:{i + 1} / {num_batch}, rank:{dist.get_rank()}, after metric", flush=True)
    for key in sdrs_dict.keys():
        sdrs_dict[key] = sum(sdrs_dict[key]) / len(sdrs_dict[key])
    for key in sirs_dict.keys():
        sirs_dict[key] = sum(sirs_dict[key]) / len(sirs_dict[key])
    for key in sars_dict.keys():
        sars_dict[key] = sum(sars_dict[key]) / len(sars_dict[key])
    # for key in sisnrs_dict.keys():
    #     sisnrs_dict[key] = sum(sisnrs_dict[key]) / len(sisnrs_dict[key])
    if opt.rank == 0:
        if opt.use_mixandseparate_loss:
            avg_mixandseparate_loss = sum(mixandseparate_losses) / len(mixandseparate_losses)
            writer.add_scalar('data/val_mixandseparate_loss', avg_mixandseparate_loss, index)
            print('mix-sep loss: %.5f, ' % avg_mixandseparate_loss, end='')
        if opt.use_sisnr_loss:
            avg_sisnr_loss = sum(sisnr_losses) / len(sisnr_losses)
            writer.add_scalar('data/val_sisnr_loss', avg_sisnr_loss, index)
            print('sisnr loss: %.5f, ' % avg_sisnr_loss, end='')
        if opt.use_contrast_loss:
            avg_contrast_loss = sum(contrast_losses) / len(contrast_losses)
            writer.add_scalar('data/val_contrast_loss', avg_contrast_loss, index)
            print('contrast loss: %.5f, ' % avg_contrast_loss, end='')
        for key in sdrs_dict.keys():
            writer.add_scalar(f'data/val_{key}', sdrs_dict[key], index)
        for key in sirs_dict.keys():
            writer.add_scalar(f'data/val_{key}', sirs_dict[key], index)
        for key in sars_dict.keys():
            writer.add_scalar(f'data/val_{key}', sars_dict[key], index)
        # for key in sisnrs_dict.keys():
        #     writer.add_scalar(f'data/val_{key}', sisnrs_dict[key], index)
    # print(f"val, rank:{dist.get_rank()}, finish val", flush=True)
    return sdrs_dict['sdr2']


################## loss
def get_mixandseparate_loss(opt, output, loss_mixandseparate):
    gt_masks = output['gt_masks']  # (B, 2, 257, 256)
    mask_predictions_pre = output['mask_predictions_pre']  # (B, 2, 256, 256)
    mask_predictions_aft = output['mask_predictions_aft']  # (B, 2, 256, 256)
    weight = output['weight']
    mixandseparate_loss = loss_mixandseparate(mask_predictions_pre, gt_masks[:, :, :-1, :], weight) * (1 - opt.after_refine_ratio) + \
                          loss_mixandseparate(mask_predictions_aft, gt_masks[:, :, :-1, :], weight) * opt.after_refine_ratio
    return mixandseparate_loss


def get_sisnr_loss(opt, output, loss_sisnr):
    gt_specs = output['audio_specs']  # (B, 257, 256, 2)
    pred_specs_pre = output['pred_specs_pre']  # (B, 257, 256, 2)
    pred_specs_aft = output['pred_specs_aft']  # (B, 257, 256, 2)

    pred_audios_pre = torch.istft(pred_specs_pre, n_fft=opt.n_fft, hop_length=opt.hop_size, win_length=opt.window_size, window=opt.window, center=True)  # (B, L)
    pred_audios_aft = torch.istft(pred_specs_aft, n_fft=opt.n_fft, hop_length=opt.hop_size, win_length=opt.window_size, window=opt.window, center=True)  # (B, L)
    gt_audios = torch.istft(gt_specs, n_fft=opt.n_fft, hop_length=opt.hop_size, win_length=opt.window_size, window=opt.window, center=True)  # (B, L)

    if opt.after_refine_ratio < 1:
        sisnr_loss_pre = loss_sisnr(pred_audios_pre, gt_audios) * (1 - opt.after_refine_ratio)
    else:
        sisnr_loss_pre = 0
    if opt.after_refine_ratio > 0:
        sisnr_loss_aft = loss_sisnr(pred_audios_aft, gt_audios) * opt.after_refine_ratio
    else:
        sisnr_loss_aft = 0
    sisnr_loss = sisnr_loss_pre + sisnr_loss_aft
    return sisnr_loss


def get_contrast_loss(opt, output, loss_contrast):
    num_speakers = output['num_speakers']  # (B1,)
    cumsum = torch.cumsum(num_speakers, dim=0) - num_speakers
    indexes = torch.cat([torch.multinomial(torch.ones(n).cuda(), 2) + cumsum[i] for i, n in enumerate(num_speakers)], dim=0)
    indexes1, indexes2 = indexes[::2], indexes[1::2]
    visual_features1 = output['visual_features'][indexes1]  # (B1, 640, 1, 64)
    visual_features2 = output['visual_features'][indexes2]  # (B1, 640, 1, 64)
    audio_embeds_mix1 = output['audio_embeds_mix'][indexes1]  # (B1, 640, 1, 1)
    audio_embeds_mix2 = output['audio_embeds_mix'][indexes2]  # (B1, 640, 1, 1)
    if random.random() <= opt.contrast_gt_percentage:
        audio_embeds1 = output['audio_embeds'][indexes1]  # (B1, 640, 1, 1)
        audio_embeds2 = output['audio_embeds'][indexes2]  # (B1, 640, 1, 1)
    else:
        audio_embeds1 = output['pred_embeds'][indexes1]  # (B1, 640, 1, 1)
        audio_embeds2 = output['pred_embeds'][indexes2]  # (B1, 640, 1, 1)

    if isinstance(loss_contrast, criterion.TripletLossCosine) or isinstance(loss_contrast, criterion.TripletLossCosine2):
        if opt.contrast_type == "audio":  # (V1, A1) > (V1, A2) + (V2, A2) > (V2, A1)
            contrast_loss = loss_contrast(visual_features1, audio_embeds1, audio_embeds2) + loss_contrast(
                visual_features2, audio_embeds2, audio_embeds1)
        elif opt.contrast_type == "mixture":  # ((V1, A1) > (V1, M1) + (V1, M1) > (V1, A2) + (V2, A2) > (V2, M2) + (V2, M2) > (V2, A1)) / 2
            contrast_loss = (loss_contrast(visual_features1, audio_embeds1, audio_embeds_mix1) + loss_contrast(
                visual_features1, audio_embeds_mix1, audio_embeds2) +
                             loss_contrast(visual_features2, audio_embeds2, audio_embeds_mix2) + loss_contrast(
                        visual_features2, audio_embeds_mix2, audio_embeds1)) / 2
    # elif isinstance(loss_contrast, criterion.NCELoss) or isinstance(loss_contrast, criterion.NCELoss2):
    #     contrast_loss = loss_contrast(visual_features, audio_embeds, audio_embeds_mix)

    return contrast_loss


################3  metrics
def _getSeparationMetrics(src, dst):  # src: (B, N, L), N为混合数
    # 一定是同一个混合样本的 src 和 dst
    # audio1: (batch, length)
    # src = torch.stack((audio1, audio2), dim=1).cuda()  # (B, N, L)
    # dst = torch.stack((audio1_gt, audio2_gt), dim=1).cuda()
    sdr, sir, sar, perm = bss_eval_sources(dst, src, compute_permutation=False)
    return torch.mean(sdr), torch.mean(sir), torch.mean(sar)


def _cal_SISNR(source, estimate_source, EPS=1e-6):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    assert source.size() == estimate_source.size()

    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis=-1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis=-1, keepdim=True)

    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis=-1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis=-1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis=-1) / (torch.sum(noise ** 2, axis=-1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)

    return sisnr


def calculate_sdr(output, window, opt):
    with torch.no_grad():
        # fetch data and predictions
        gt_specs = output['audio_specs']
        pred_specs_aft = output['pred_specs_aft']
        num_speakers = output['num_speakers']

        pred_audios = torch.istft(pred_specs_aft, n_fft=opt.n_fft, hop_length=opt.hop_size,
                                  win_length=opt.window_size, window=window, center=True)  # (B, L)

        gt_audios = torch.istft(gt_specs, n_fft=opt.n_fft, hop_length=opt.hop_size,
                                win_length=opt.window_size, window=window, center=True)  # (B, L)

        sdr_dict, sir_dict, sar_dict, sisnr_dict = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        pre_num = 0
        # sisnrs = _cal_SISNR(gt_audios, pred_audios)
        for i in range(len(num_speakers)):
            num_spk = num_speakers[i].item()  # 2, 3, 4, 5
            cur_num = pre_num + num_spk
            if num_spk != 5:
                sdr, sir, sar = _getSeparationMetrics(pred_audios[pre_num: cur_num].unsqueeze(0), gt_audios[pre_num: cur_num].unsqueeze(0))
                if not math.isnan(sdr) and not math.isinf(sdr):
                    sdr_dict[f'sdr{num_spk}'].append(sdr.item())
                if not math.isnan(sir) and not math.isinf(sir):
                    sir_dict[f'sir{num_spk}'].append(sir.item())
                if not math.isnan(sar) and not math.isinf(sar):
                    sar_dict[f'sar{num_spk}'].append(sar.item())
            pre_num = cur_num

    return sdr_dict, sir_dict, sar_dict


def main():
    # initialize
    opt = _init()

    # create data loader
    data_loader = CreateDataLoader(opt)
    if opt.validation_on:
        opt.mode = 'val'
        data_loader_val = CreateDataLoader(opt)
        opt.mode = 'train'  # set it back

    # create model
    model = create_model(opt)
    model2 = model.module
    net_lipreading, net_facial_attribtes, net_unet, net_refine = model2.net_lipreading, model2.net_identity, \
                                                                 model2.net_unet, model2.net_refine

    if opt.use_contrast_loss:
        net_vocal = model2.net_vocal
    # create optimizer
    optimizer = create_optimizer(model, opt)
    # create loss
    crit = create_loss(opt)
    opt.window = torch.hann_window(opt.window_size).cuda()

    cudnn.benchmark = True

    # initialization
    data_loading_time = []
    model_forward_time = []
    model_backward_time = []

    batch_mixandseparate_loss = []
    batch_sisnr_loss = []
    batch_contrast_loss = []

    best_sdr = -float("inf")
    start_epoch = 0
    start_batch = 0
    cumsum_batch = 0

    if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'resume_latest.pth')):
        ckpt = torch.load(osp.join(opt.checkpoints_dir, opt.name, 'resume_latest.pth'), map_location='cpu')
        start_epoch = ckpt['epoch']
        start_batch = ckpt['batch']
        cumsum_batch = ckpt['cumsum_batch']
        best_sdr = ckpt['best_sdr']
        if opt.rank == 0:
            print(f'resume from epoch {start_epoch}, batch {start_batch}, cumsum_batch: {cumsum_batch}, best_sdr {best_sdr}')

    batches_per_epoch = len(data_loader)
    total_batches = batches_per_epoch * opt.epochs
    # for epoch in range(opt.epochs):
    #     data_loader.set_epoch(epoch)
    #     total_batches += len(data_loader)

    if opt.rank == 0:
        pb = ProgressBar(total_batches, start=False)
        pb.start()
        for pbt in range(cumsum_batch):
            pb.update()

    for epoch in range(start_epoch, opt.epochs):

        data_loader.set_epoch(epoch)
        iter_start_time = time.time()

        # print(f"train, rank:{dist.get_rank()} has {len(data_loader)} batches", flush=True)
        # num_batch = len(data_loader)
        time.sleep(5)

        # for batch, data in enumerate(data_loader):
        #     # print(f"train, batch:{batch + 1} / {num_batch}, rank:{dist.get_rank()}, after load data", flush=True)
        #     if epoch == start_epoch and (batch + 1) <= start_batch:
        #         continue

        if epoch == start_epoch and start_batch > 0:
            indices = data_loader._get_indices()
            data_loader.dataloader.sampler.indices = indices[start_batch * opt.batchSize:]

        for batch, data in enumerate(data_loader):
            if epoch == start_epoch:
                batch += start_batch

            iter_data_loaded_time = time.time()

            # zero grad
            model.zero_grad()
            optimizer.zero_grad()

            # forward pass
            # print(f"train, batch:{batch + 1} / {num_batch}, rank:{dist.get_rank()}, before model forward", flush=True)
            output = model.forward(data)
            # print(f"train, batch:{batch + 1} / {num_batch}, rank:{dist.get_rank()}, after model forward", flush=True)

            iter_data_forwarded_time = time.time()

            # calculate loss
            loss = 0
            if opt.use_mixandseparate_loss:
                mixandseparate_loss = get_mixandseparate_loss(opt, output, crit['loss_mixandseparate']) * opt.mixandseparate_loss_weight
                loss = loss + mixandseparate_loss
                reduced_mix_loss = _reduce_tensor(mixandseparate_loss.data)
                batch_mixandseparate_loss.append(reduced_mix_loss.item())
            if opt.use_sisnr_loss:
                sisnr_loss = get_sisnr_loss(opt, output, crit['loss_sisnr']) * opt.sisnr_loss_weight
                loss = loss + sisnr_loss
                reduced_sisnr_loss = _reduce_tensor(sisnr_loss.data)
                batch_sisnr_loss.append(reduced_sisnr_loss.item())
            if opt.use_contrast_loss:
                contrast_loss = get_contrast_loss(opt, output, crit['loss_contrast']) * opt.contrast_loss_weight
                loss = loss + contrast_loss
                reduced_con_loss = _reduce_tensor(contrast_loss.data)
                batch_contrast_loss.append(reduced_con_loss.item())

            # print(f"train, batch:{batch + 1} / {num_batch}, rank:{dist.get_rank()}, before loss backward", flush=True)
            loss.backward()
            optimizer.step()
            # print(f"train, batch:{batch + 1} / {num_batch}, rank:{dist.get_rank()}, after loss backward", flush=True)

            iter_model_backward_time = time.time()

            if opt.rank == 0:
                data_loading_time.append(iter_data_loaded_time - iter_start_time)
                model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
                model_backward_time.append(iter_model_backward_time - iter_data_forwarded_time)

            if (batch + 1) % opt.display_freq == 0 and opt.rank == 0:
                # print('Display training progress at (epoch %d, batch %d)' % (epoch, batch))
                print(f'Epoch %d, Batch %d: ' % (epoch, batch), end='')
                if opt.use_mixandseparate_loss:
                    avg_mixandseparate_loss = sum(batch_mixandseparate_loss) / len(batch_mixandseparate_loss)
                    print('mix-sep loss: %.5f, ' % avg_mixandseparate_loss, end='')
                if opt.use_sisnr_loss:
                    avg_sisnr_loss = sum(batch_sisnr_loss) / len(batch_sisnr_loss)
                    print('sisnr loss: %.5f, ' % avg_sisnr_loss, end='')
                if opt.use_contrast_loss:
                    avg_contrast_loss = sum(batch_contrast_loss) / len(batch_contrast_loss)
                    print('contrast loss: %.5f, ' % avg_contrast_loss, end='')

                batch_mixandseparate_loss = []
                batch_sisnr_loss = []
                batch_contrast_loss = []

                if opt.tensorboard:
                    opt.writer.add_scalar('data/lipreading_lr', optimizer.state_dict()['param_groups'][0]['lr'],
                                          cumsum_batch)
                    opt.writer.add_scalar('data/identity_lr', optimizer.state_dict()['param_groups'][1]['lr'],
                                          cumsum_batch)
                    opt.writer.add_scalar('data/unet_lr', optimizer.state_dict()['param_groups'][2]['lr'], cumsum_batch)
                    opt.writer.add_scalar('data/refine_lr', optimizer.state_dict()['param_groups'][3]['lr'],
                                          cumsum_batch)
                    net_idx = 4
                    if opt.use_contrast_loss:
                        opt.writer.add_scalar('data/vocal_lr', optimizer.state_dict()['param_groups'][net_idx]['lr'], cumsum_batch)
                        net_idx += 1
                        opt.writer.add_scalar('data/contrast_loss', avg_contrast_loss, cumsum_batch)
                    if opt.use_mixandseparate_loss:
                        opt.writer.add_scalar('data/mixandseparate_loss', avg_mixandseparate_loss, cumsum_batch)
                    if opt.use_sisnr_loss:
                        opt.writer.add_scalar('data/sisnr_loss', avg_sisnr_loss, cumsum_batch)
                print('data load: %.3f s, ' % (sum(data_loading_time)/len(data_loading_time)), end='')
                print('forward: %.3f s, ' % (sum(model_forward_time)/len(model_forward_time)), end='')
                print('backward: %.3f s' % (sum(model_backward_time)/len(model_backward_time)))
                data_loading_time = []
                model_forward_time = []
                model_backward_time = []

            if (batch + 1) % opt.validation_freq == 0 and opt.validation_on:
                model.eval()
                opt.mode = 'val'
                if opt.rank == 0:
                    print('Begin evaluate at Epoch %d, Batch %d: ' % (epoch, batch), end='')
                val_sdr = display_val(model, crit, opt.writer, cumsum_batch, data_loader_val, epoch, opt)
                model.train()
                opt.mode = 'train'
                # save the model that achieves the smallest validation error
                if val_sdr > best_sdr or math.isnan(best_sdr) or math.isinf(best_sdr):
                    best_sdr = val_sdr
                    if opt.rank == 0:
                        print('saving the best model (epoch %d, batch %d) with validation sdr %.3f\n' % (epoch, batch, val_sdr))
                        torch.save(net_lipreading.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'lipreading_best.pth'))
                        torch.save(net_facial_attribtes.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'facial_best.pth'))
                        torch.save(net_unet.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'unet_best.pth'))
                        torch.save(net_refine.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'refine_best.pth'))
                        if opt.use_contrast_loss:
                            torch.save(net_vocal.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'vocal_best.pth'))

            if (batch + 1) % opt.save_latest_freq == 0 and opt.rank == 0:
                print('saving the latest model (epoch %d, batch %d)' % (epoch, batch))
                torch.save(net_lipreading.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'lipreading_latest.pth'))
                torch.save(net_facial_attribtes.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'facial_latest.pth'))
                torch.save(net_unet.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'unet_latest.pth'))
                torch.save(net_refine.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'refine_latest.pth'))
                if opt.use_contrast_loss:
                    torch.save(net_vocal.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'vocal_latest.pth'))
                ckpt_dict = {'optim_state_dict': optimizer.state_dict(), 'epoch': epoch, 'batch': batch + 1, 'cumsum_batch': cumsum_batch + 1, 'best_sdr': best_sdr}
                torch.save(ckpt_dict, os.path.join('.', opt.checkpoints_dir, opt.name, 'resume_latest.pth'))

            iter_start_time = time.time()

            cumsum_batch += 1

            if opt.rank == 0:
                pb.update()

            # 防止进程互锁
            dist.barrier()

            # print(f"train, batch:{batch + 1} / {num_batch}, rank:{dist.get_rank()}, finish batch", flush=True)

        if opt.rank == 0:  # save latest model for each epoch
            print('saving the latest model (epoch %d)' % epoch)
            torch.save(net_lipreading.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'lipreading_latest.pth'))
            torch.save(net_facial_attribtes.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'facial_latest.pth'))
            torch.save(net_unet.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'unet_latest.pth'))
            torch.save(net_refine.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'refine_latest.pth'))
            if opt.use_contrast_loss:
                torch.save(net_vocal.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'vocal_latest.pth'))
            ckpt_dict = {'optim_state_dict': optimizer.state_dict(), 'epoch': epoch + 1, 'batch': 0, 'cumsum_batch': cumsum_batch, 'best_sdr': best_sdr}
            torch.save(ckpt_dict, os.path.join('.', opt.checkpoints_dir, opt.name, 'resume_latest.pth'))

        if (epoch + 1) == opt.epochs:  # the last epoch, evaluate model
            model.eval()
            opt.mode = 'val'
            if opt.rank == 0:
                print('Begin evaluate at last Epoch %d, Batch %d: ' % (epoch, len(data_loader)), end='')
            val_sdr = display_val(model, crit, opt.writer, cumsum_batch, data_loader_val, epoch, opt)
            model.train()
            opt.mode = 'train'
            # save the model that achieves the smallest validation error
            if val_sdr > best_sdr:
                best_sdr = val_sdr
                if opt.rank == 0:
                    print('saving the best model (epoch %d, batch %d) with validation sdr %.3f\n' % (epoch, len(data_loader), val_sdr))
                    torch.save(net_lipreading.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'lipreading_best.pth'))
                    torch.save(net_facial_attribtes.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'facial_best.pth'))
                    torch.save(net_unet.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'unet_best.pth'))
                    torch.save(net_refine.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'refine_best.pth'))
                    if opt.use_contrast_loss:
                        torch.save(net_vocal.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'vocal_best.pth'))

        # decrease learning rate
        if (epoch + 1) in opt.lr_steps:
            decrease_learning_rate(optimizer, opt.decay_factor)

            if opt.rank == 0:
                print('decreased learning rate by ', opt.decay_factor)


if __name__ == '__main__':
    main()
