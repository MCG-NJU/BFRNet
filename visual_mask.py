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
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from mmcv.fileio import FileClient
import io
from collections import defaultdict

from options.train_options import TrainOptions
from options.visual_options import VisualOptions
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


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_input_valid(opt):
    # check invalidation of recognition parameters
    if opt.with_recognition:
        assert opt.recog_mode is not None, 'please input recog mode: recog or distill'
        params = []
        if opt.recog_mode == "recog":
            for x in vars(opt):
                if x.startswith("recog"):
                    params.append(x)
        elif opt.recog_mode == "distill":
            for x in vars(opt):
                if x.startswith("distill"):
                    params.append(x)
        for par in params:
            assert getattr(opt, par) is not None, f'please input {par}'
        if opt.recog_mode == "recog":
            assert (opt.recog_blank_label == 0 and opt.recog_num_classes == 40 and not opt.recog_origin_fc) \
                   or (opt.recog_blank_label == 1 and opt.recog_num_classes == 32), \
                   "[recog_num_classes] and [recog_blank_label] are not match!"
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
    opt = VisualOptions().parse()

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

    if opt.rank == 0:
        args = vars(opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

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
        identity_feature_dim=opt.identity_feature_dim,
        weights=opt.weights_unet)
    if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'unet_latest.pth')):
        ckpt_path = osp.join(opt.checkpoints_dir, opt.name, 'unet_latest.pth')
        net_unet.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        if opt.rank == 0:
            print(f'resume net_unet from {ckpt_path}')
    nets.append(net_unet)

    mask_net = builder.build_mask_net(
        opt=opt,
        weights=opt.weights_mask
    )
    if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'mask_latest.pth')):
        ckpt_path = osp.join(opt.checkpoints_dir, opt.name, 'mask_latest.pth')
        mask_net.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        if opt.rank == 0:
            print(f'resume mask_net from {ckpt_path}')
    nets.append(mask_net)

    # construct our audio-visual model
    model = AudioVisualModel(nets, opt)
    model.to(opt.device)
    model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
    return model


# def _getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt):
#     # audio1: (batch, length)
#     src = torch.stack((audio1, audio2), dim=1).cuda()  # (B, N, L)
#     dst = torch.stack((audio1_gt, audio2_gt), dim=1).cuda()
#     sdr, sir, sar, perm = bss_eval_sources(dst, src, compute_permutation=False)
#     return torch.mean(sdr), torch.mean(sir), torch.mean(sar)


def _getSeparationMetrics(pred_audios, gt_audios):
    # pred_audios: (batch, num, length)
    sdr, sir, sar, perm = bss_eval_sources(gt_audios, pred_audios, compute_permutation=False)
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

    return torch.mean(sisnr)


def isanumber(data):
    if not math.isnan(data) and not math.isinf(data):
        return True


def calculate_sdr(output, window, opt, save_dir, names):
    # name1: (B,)
    with torch.no_grad():
        gt_specs = output['audio_specs']  # (B, 257, 256)
        pred_specs_pre = output['pred_specs_pre']
        pred_specs_aft = output['pred_specs_aft']
        num_speakers = output['num_speakers']

        pred_audios_pre = torch.istft(pred_specs_pre, n_fft=opt.n_fft, hop_length=opt.hop_size,
                                      win_length=opt.window_size, window=window, center=True)  # (B, L)
        pred_audios_aft = torch.istft(pred_specs_aft, n_fft=opt.n_fft, hop_length=opt.hop_size,
                                      win_length=opt.window_size, window=window, center=True)  # (B, L)

        gt_audios = torch.istft(gt_specs, n_fft=opt.n_fft, hop_length=opt.hop_size,
                                win_length=opt.window_size, window=window, center=True)  # (B, L)

        sdr_pre_dict, sdr_aft_dict, sisnr_pre_dict, sisnr_aft_dict = defaultdict(list), defaultdict(list), \
                                                                     defaultdict(list), defaultdict(list)

        cumsum = 0
        B = len(num_speakers)
        names_copy = names.tolist()  # (B,)
        for b in range(B):
            num = num_speakers[b]
            if num != 5:
                sdr_pre, _, _ = _getSeparationMetrics(pred_audios_pre[cumsum:cumsum+num].unsqueeze(0), gt_audios[cumsum:cumsum+num].unsqueeze(0))
                sdr_aft, _, _ = _getSeparationMetrics(pred_audios_aft[cumsum:cumsum+num].unsqueeze(0), gt_audios[cumsum:cumsum+num].unsqueeze(0))
                # sisnr_pre = _cal_SISNR(gt_audios, pred_audios_pre)
                # sisnr_aft = _cal_SISNR(gt_audios, pred_audios_aft)
                # if isanumber(sdr_pre) and isanumber(sdr_aft) and isanumber(sisnr_pre) and isanumber(sisnr_aft):
                #     sdr_pre_dict[f'sdr{num}'].append(sdr_pre)
                #     sdr_aft_dict[f'sdr{num}'].append(sdr_aft)
                #     sisnr_pre_dict[f'sdr{num}'].append(sisnr_pre)
                #     sisnr_aft_dict[f'sdr{num}'].append(sisnr_aft)
                sdr_pre = _reduce_tensor(sdr_pre.cuda().data)
                sdr_aft = _reduce_tensor(sdr_aft.cuda().data)
                # if not math.isnan(sdr_pre) and not math.isinf(sdr_pre):
                #     sdr_dict[f'sdr{num_spk}'].append(sdr.item())
                # if opt.rank == 0:
                #     print(float(sdr_pre), flush=True)
                if isanumber(sdr_pre) and isanumber(sdr_aft):
                    sdr_pre_dict[f'sdr{num}'].append(float(sdr_pre))
                    sdr_aft_dict[f'sdr{num}'].append(float(sdr_aft))
            for n in range(num):
                names_copy[cumsum+n] = names_copy[cumsum+n].replace('/', '_')
            cumsum += num

        # gt_audios = gt_audios.detach().cpu().numpy()
        # pred_audios_pre = pred_audios_pre.detach().cpu().numpy()
        # pred_audios_aft = pred_audios_aft.detach().cpu().numpy()
        # cumsum = 0
        # for b in range(B):
        #     num = num_speakers[b]
        #     last_dir = '@'.join(names_copy[cumsum:cumsum+num])
        #     os.makedirs(osp.join(save_dir, last_dir), exist_ok=True)
        #     for n in range(num):
        #         sf.write(osp.join(save_dir, last_dir, f'gt{n+1}.wav'), gt_audios[cumsum+n], opt.audio_sampling_rate)
        #         sf.write(osp.join(save_dir, last_dir, f'pred{n+1}_pre.wav'), pred_audios_pre[cumsum+n], opt.audio_sampling_rate)
        #         sf.write(osp.join(save_dir, last_dir, f'pred{n+1}_aft.wav'), pred_audios_aft[cumsum+n], opt.audio_sampling_rate)

    # return sdr_pre, sdr_aft, sisnr_pre, sisnr_aft
    return sdr_pre_dict, sdr_aft_dict


def draw(spec, save_dir, name):
    plt.cla()
    plt.clf()
    plt.close()
    spec = librosa.amplitude_to_db(spec)
    librosa.display.specshow(spec, y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_dir, name))


def draw_spec(gt_specs, pred_specs_pre, pred_specs_aft, save_dir, names):
    num = len(names)
    # gt_specs: (B, 257, 256)
    names_copy = names.tolist()
    for n in range(num):
        names_copy[n] = names[n].replace('/', '_')
    last_dir = '@'.join(names_copy)
    save_dir = os.path.join(save_dir, last_dir)
    os.makedirs(save_dir, exist_ok=True)
    for n in range(num):
        draw(gt_specs[n], save_dir, f'gt{n+1}.jpg')
        draw(pred_specs_pre[n], save_dir, f'pred{n+1}_pre.jpg')
        draw(pred_specs_aft[n], save_dir, f'pred{n+1}_aft.jpg')


def main():
    # initialize
    opt = _init()

    opt.mode = 'val'
    data_loader_val = CreateDataLoader(opt)
    opt.mode = 'train'  # set it back

    # create model
    model = create_model(opt)
    model.eval()
    opt.window = torch.hann_window(opt.window_size).cuda()

    cudnn.benchmark = True

    if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'resume_latest.pth')):
        ckpt = torch.load(osp.join(opt.checkpoints_dir, opt.name, 'resume_latest.pth'), map_location='cpu')
        start_epoch = ckpt['epoch']
        start_batch = ckpt['batch']
        best_sdr = ckpt['best_sdr']
        if opt.rank == 0:
            print(f'resume from epoch {start_epoch}, batch {start_batch}, best_sdr {best_sdr}')

    batches_per_epoch = len(data_loader_val)

    def get_L2_norm(data, axis=3):
        data = np.linalg.norm(data, axis=axis)
        return data

    if opt.rank == 0:
        pb = ProgressBar(len(data_loader_val), start=False)
        pb.start()
    window = opt.window
    sdr_pre_dict, sdr_aft_dict, sisnr_pre_dict, sisnr_aft_dict = defaultdict(list), defaultdict(list), \
                                                                 defaultdict(list), defaultdict(list)
    client = FileClient(backend='petrel')
    with io.BytesIO(client.get(opt.val_file)) as af:
        videos_path = [d.decode('utf-8').strip()[6:] for d in af.readlines()]
    videos_path = np.array(videos_path)
    with torch.no_grad():
        for i, val_data in enumerate(data_loader_val):
            output = model.forward(val_data, batches_per_epoch)
            # save spectrogram into dir opt.visual_dir, 保存gt和再分离前以及再分离后的结果
            # gt_specs = output['audio_specs'].detach().cpu().numpy()  # (B, 257, 256, 2)
            # pred_specs_pre = output['pred_specs_pre'].detach().cpu().numpy()  # (B, 257, 256, 2)
            # pred_specs_aft = output['pred_specs_aft'].detach().cpu().numpy()  # (B, 257, 256, 2)
            # num_speakers = output['num_speakers']
            indexes = output['indexes'].detach().cpu().numpy()  # (N,)

            # gt_specs = get_L2_norm(gt_specs)  # (B, 257, 256)
            # pred_specs_pre = get_L2_norm(pred_specs_pre)  # (B, 257, 256)
            # pred_specs_aft = get_L2_norm(pred_specs_aft)  # (B, 257, 256)

            # cumsum = 0
            # B = len(num_speakers)
            # for b in range(B):
            #     num = num_speakers[b]  # n
            #     names = videos_path[indexes[cumsum:cumsum+num]]
            #     draw_spec(gt_specs[cumsum:cumsum+num], pred_specs_pre[cumsum:cumsum+num], pred_specs_aft[cumsum:cumsum+num], opt.visual_dir, names)
            #     cumsum += num
            # calculate the validation metrics
            try:
                # sdr_pre, sdr_aft, sisnr_pre, sisnr_aft = calculate_sdr(output, window, opt, opt.visual_dir, videos_path[indexes])
                sdr_pre, sdr_aft = calculate_sdr(output, window, opt, opt.visual_dir, videos_path[indexes])
                for key in sdr_pre:
                    sdr_pre_dict[key] += (sdr_pre[key])
                    sdr_aft_dict[key] += (sdr_aft[key])
                    # sisnr_pre_dict[key].append(sisnr_pre[key])
                    # sisnr_aft_dict[key].append(sisnr_aft[key])
                # if opt.rank == 0:
                #     print(sdr_pre_dict)
                #     print(sdr_aft_dict)
                #     print(sum(sdr_pre_dict[key]))
                #     print(len(sdr_pre_dict[key]))
            except Exception as e:
                pass
            if opt.rank == 0:
                pb.update()
    for key in sdr_pre_dict:
        sdr_pre = sum(sdr_pre_dict[key]) / len(sdr_pre_dict[key])
        sdr_aft = sum(sdr_aft_dict[key]) / len(sdr_aft_dict[key])
        # sisnr_pre = sum(sisnr_pre_dict[key]) / len(sisnr_pre_dict[key])
        # sisnr_aft = sum(sisnr_aft_dict[key]) / len(sisnr_aft_dict[key])
        # print(f'sdr{key}_pre: %.5f, sdr{key}_aft: %.5f, sisnr{key}_pre: %.5f, sisnr{key}_aft: %.5f' % (sdr_pre, sdr_aft, sisnr_pre, sisnr_aft))
        print(f'sdr{key}_pre: %.5f, sdr{key}_aft: %.5f' % (sdr_pre, sdr_aft))


if __name__ == '__main__':
    main()
