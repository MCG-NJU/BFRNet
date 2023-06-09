import os
import os.path as osp
from mmengine import ProgressBar
import subprocess
from torch_mir_eval import bss_eval_sources
import random
import numpy as np
import math
from collections import defaultdict
import warnings
import platform
import time

from options.train_options import TrainOptions
from dataset.data_loader import CreateDataLoader
from models.build_models import ModelBuilder
from models.audioVisual_model import BFRNet
from models import criterion

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _init_slurm(opt):
    """the setup of slurm launch."""
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
    """the setup of pytorch distributed launch."""
    rank = int(os.environ['RANK'])
    local_rank = rank % torch.cuda.device_count()
    opt.device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    opt.local_rank = local_rank
    opt.rank = rank
    dist.init_process_group(backend="nccl")


def _init():
    """load options, prepare for distributed training, create a logger."""
    opt = TrainOptions().parse()

    warnings.filterwarnings('ignore')

    for x in vars(opt):
        value = getattr(opt, x)
        if value == "true":
            setattr(opt, x, True)
        elif value == "false":
            setattr(opt, x, False)

    # set random seed
    set_random_seed(opt.seed)

    _init_slurm(opt)

    # log
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


def create_model(opt):
    """construct model."""
    # Network Builders
    builder = ModelBuilder()
    nets = []

    # build lip net
    lip_net = builder.build_lipnet(
        opt=opt,
        config_path=opt.lipnet_config_path,
        weights=opt.weights_lipnet)
    # resume from checkpoint (optional)
    if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'lipnet_latest.pth')):
        ckpt_path = osp.join(opt.checkpoints_dir, opt.name, 'lipnet_latest.pth')
        lip_net.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        if opt.rank == 0:
            print(f'resume lip_net from {ckpt_path}')
    nets.append(lip_net)

    # if face feature dim is not 512, for resnet reduce dimension to this feature dim
    if opt.face_feature_dim != 512:
        opt.with_fc = True
    else:
        opt.with_fc = False
    face_net = builder.build_facenet(
        opt=opt,
        pool_type=opt.visual_pool,
        fc_out=opt.face_feature_dim,
        with_fc=opt.with_fc,
        weights=opt.weights_facenet)
    # resume from checkpoint (optional)
    if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'facenet_latest.pth')):
        ckpt_path = osp.join(opt.checkpoints_dir, opt.name, 'facenet_latest.pth')
        face_net.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        if opt.rank == 0:
            print(f'resume face_net from {ckpt_path}')
    nets.append(face_net)

    if opt.visual_feature_type == 'both':
        visual_feature_dim = opt.lip_feature_dim + opt.face_feature_dim
    elif opt.visual_feature_type == 'lip':
        visual_feature_dim = opt.lip_feature_dim
    else:  # face
        visual_feature_dim = opt.face_feature_dim
    unet = builder.build_unet(
        opt=opt,
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
        audioVisual_feature_dim=opt.unet_ngf * 8 + visual_feature_dim,
        weights=opt.weights_unet)
    # resume from checkpoint (optional)
    if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'unet_latest.pth')):
        ckpt_path = osp.join(opt.checkpoints_dir, opt.name, 'unet_latest.pth')
        unet.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        if opt.rank == 0:
            print(f'resume unet from {ckpt_path}')
    nets.append(unet)

    FRNet = builder.build_FRNet(
        opt=opt,
        num_layers=opt.FRNet_layers,
        weights=opt.weights_FRNet
    )
    # resume from checkpoint (optional)
    if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'FRNet_latest.pth')):
        ckpt_path = osp.join(opt.checkpoints_dir, opt.name, 'FRNet_latest.pth')
        FRNet.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        if opt.rank == 0:
            print(f'resume FRNet from {ckpt_path}')
    nets.append(FRNet)

    # construct our audio-visual model
    model = BFRNet(nets, opt)
    model.to(opt.device)
    model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
    return model


def create_optimizer(model, opt):
    """create optimizer for each sub-network."""
    model2 = model.module
    lip_net, face_net, unet, FRNet = model2.lip_net, model2.face_net, model2.unet, model2.FRNet
    param_groups = [{'params': lip_net.parameters(), 'lr': opt.lr_lipnet},
                    {'params': face_net.parameters(), 'lr': opt.lr_facenet},
                    {'params': unet.parameters(), 'lr': opt.lr_unet},
                    {'params': FRNet.parameters(), 'lr': opt.lr_FRNet}]
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
    if opt.resume and osp.exists(osp.join(opt.checkpoints_dir, opt.name, 'resume_latest.pth')):
        ckpt = osp.join(opt.checkpoints_dir, opt.name, 'resume_latest.pth')
        optimizer.load_state_dict(torch.load(ckpt, map_location='cpu')['optim_state_dict'])
    return optimizer


def create_loss(opt):
    """create loss function si-snr."""
    crit = {}
    loss_sisnr = criterion.SISNRLoss()
    loss_sisnr.to(opt.device)
    crit['loss_sisnr'] = loss_sisnr
    return crit


def decrease_learning_rate(optimizer, decay_factor=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor


def _reduce_tensor(tensor):
    """reduce the loss tensor between gpus."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def evaluate(model, crit, writer, batch_index, data_loader_val, epoch, opt):
    """
    evaluate the model.
    :param model:             model
    :param crit:              criterion
    :param writer:            logger
    :param batch_index:       cumulative batch index
    :param data_loader_val:   data loader
    :param epoch:             current epoch
    :param opt:
    """
    data_loader_val.set_epoch(epoch)
    # display the evaluation progress in process 0
    if opt.rank == 0:
        pb = ProgressBar(len(data_loader_val), start=False)
        pb.start()
    window = opt.window
    batch_loss = []
    sdrs_dict = defaultdict(list)

    with torch.no_grad():

        time.sleep(5)

        for i, val_data in enumerate(data_loader_val):
            output = model.forward(val_data)
            loss = calculate_loss(opt, output, crit['loss_sisnr']) * opt.sisnr_loss_weight
            reduced_loss = _reduce_tensor(loss.data)
            batch_loss.append(reduced_loss.item())
            try:
                sdr_d = calculate_sdr(output, window, opt)
                for key in sdr_d.keys():
                    sdrs_dict[key] += sdr_d[key]
            except Exception as e:
                pass
            if opt.rank == 0:
                pb.update()
            dist.barrier()
    for key in sdrs_dict.keys():
        sdrs_dict[key] = sum(sdrs_dict[key]) / len(sdrs_dict[key])
    # print the loss
    if opt.rank == 0:
        avg_loss = sum(batch_loss) / len(batch_loss)
        writer.add_scalar('data/val_sisnr_loss', avg_loss, batch_index)
        print('sisnr loss: %.5f, ' % avg_loss, end='')
        for key in sdrs_dict.keys():
            writer.add_scalar(f'data/val_{key}', sdrs_dict[key], batch_index)
    return sdrs_dict['sdr2']


def calculate_loss(opt, output, loss_func):
    """
    Calculate si-snr loss
    :param output:       output of model
    :param loss_func:    loss function
    """
    # obtain ground truth and predicted spectrograms
    gt_specs = output['audio_specs']  # (B, 257, 256, 2)    ground truth spectrogram
    pred_specs_pre = output['pred_specs_pre']  # (B, 257, 256, 2)    predicted spectrogram before FRNet
    pred_specs_aft = output['pred_specs_aft']  # (B, 257, 256, 2)    predicted spectrogram after FRNet

    # obtain the wavform by spectrogram using ISTFT (inverse short-time fourier transform)
    pred_audios_pre = torch.istft(pred_specs_pre, n_fft=opt.n_fft, hop_length=opt.hop_size, win_length=opt.window_size, window=opt.window, center=True)  # (B, L)
    pred_audios_aft = torch.istft(pred_specs_aft, n_fft=opt.n_fft, hop_length=opt.hop_size, win_length=opt.window_size, window=opt.window, center=True)  # (B, L)
    gt_audios = torch.istft(gt_specs, n_fft=opt.n_fft, hop_length=opt.hop_size, win_length=opt.window_size, window=opt.window, center=True)  # (B, L)

    # calculate the loss
    sisnr_loss_pre = loss_func(pred_audios_pre, gt_audios) * opt.lamda
    sisnr_loss_aft = loss_func(pred_audios_aft, gt_audios) * (1 - opt.lamda)
    sisnr_loss = sisnr_loss_pre + sisnr_loss_aft
    return sisnr_loss


def _get_metrics(src, dst):
    """
    calculate the evaluation metric SDR
    :param src:   prediction        batch_size * num_speakers * length
    :param dst:   ground truth      batch_size * num_speakers * length
    :return:  SDR
    """
    sdr, _, _, _ = bss_eval_sources(dst, src, compute_permutation=False)
    return torch.mean(sdr)


def calculate_sdr(output, window, opt):
    """calculate sdr."""
    with torch.no_grad():
        # fetch data and predictions
        gt_specs = output['audio_specs']  # ground truth
        pred_specs_aft = output['pred_specs_aft']  # prediction
        num_speakers = output['num_speakers']

        # obtain the wavform by spectrogram using ISTFT (inverse short-time fourier transform)
        pred_audios = torch.istft(pred_specs_aft, n_fft=opt.n_fft, hop_length=opt.hop_size,
                                  win_length=opt.window_size, window=window, center=True)  # (B, L)

        # ground truth wavforms
        gt_audios = torch.istft(gt_specs, n_fft=opt.n_fft, hop_length=opt.hop_size,
                                win_length=opt.window_size, window=window, center=True)  # (B, L)

        sdr_dict = defaultdict(list)
        pre_num = 0
        # calculate sdr for each kind of mixture
        for i in range(len(num_speakers)):
            num_spk = num_speakers[i].item()  # 2, 3, 4, 5
            cur_num = pre_num + num_spk
            # if num_spk != 5:
            sdr = _get_metrics(pred_audios[pre_num: cur_num].unsqueeze(0), gt_audios[pre_num: cur_num].unsqueeze(0))
            if not math.isnan(sdr) and not math.isinf(sdr):
                sdr_dict[f'sdr{num_spk}'].append(sdr.item())
            pre_num = cur_num

    return sdr_dict


def _save_checkpoints(opt, model, state):
    torch.save(model.module.lip_net.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, f'lipnet_{state}.pth'))
    torch.save(model.module.face_net.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, f'facenet_{state}.pth'))
    torch.save(model.module.unet.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, f'unet_{state}.pth'))
    torch.save(model.module.FRNet.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, f'FRNet_{state}.pth'))


def train(opt, model, data_loader, data_loader_val, optimizer, crit, ):
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

    if opt.rank == 0:
        pb = ProgressBar(total_batches, start=False)
        pb.start()
        for pbt in range(cumsum_batch):
            pb.update()

    data_loading_time = []
    model_forward_time = []
    model_backward_time = []
    batch_loss = []
    for epoch in range(start_epoch, opt.epochs):

        data_loader.set_epoch(epoch)
        iter_start_time = time.time()

        time.sleep(5)

        # resume from the last batch
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
            output = model.forward(data)

            iter_data_forwarded_time = time.time()

            # calculate loss
            loss = calculate_loss(opt, output, crit['loss_sisnr']) * opt.sisnr_loss_weight
            reduced_loss = _reduce_tensor(loss.data)
            batch_loss.append(reduced_loss.item())

            loss.backward()
            optimizer.step()

            iter_model_backward_time = time.time()

            if opt.rank == 0:
                data_loading_time.append(iter_data_loaded_time - iter_start_time)
                model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
                model_backward_time.append(iter_model_backward_time - iter_data_forwarded_time)

            if (batch + 1) % opt.display_freq == 0 and opt.rank == 0:
                print(f'Epoch %d, Batch %d: ' % (epoch, batch), end='')
                avg_loss = sum(batch_loss) / len(batch_loss)
                print('sisnr loss: %.5f, ' % avg_loss, end='')

                batch_loss = []

                if opt.tensorboard:
                    opt.writer.add_scalar('data/lipnet_lr', optimizer.state_dict()['param_groups'][0]['lr'],
                                          cumsum_batch)
                    opt.writer.add_scalar('data/face_lr', optimizer.state_dict()['param_groups'][1]['lr'],
                                          cumsum_batch)
                    opt.writer.add_scalar('data/unet_lr', optimizer.state_dict()['param_groups'][2]['lr'], cumsum_batch)
                    opt.writer.add_scalar('data/FRNet_lr', optimizer.state_dict()['param_groups'][3]['lr'],
                                          cumsum_batch)
                    opt.writer.add_scalar('data/sisnr_loss', avg_loss, cumsum_batch)
                print('data load: %.3f s, ' % (sum(data_loading_time) / len(data_loading_time)), end='')
                print('forward: %.3f s, ' % (sum(model_forward_time) / len(model_forward_time)), end='')
                print('backward: %.3f s' % (sum(model_backward_time) / len(model_backward_time)))
                data_loading_time = []
                model_forward_time = []
                model_backward_time = []

            if (batch + 1) % opt.validation_freq == 0 and opt.validation_on:
                model.eval()
                opt.mode = 'val'
                if opt.rank == 0:
                    print('Begin evaluate at Epoch %d, Batch %d: ' % (epoch, batch), end='')
                val_sdr = evaluate(model, crit, opt.writer, cumsum_batch, data_loader_val, epoch, opt)
                model.train()
                opt.mode = 'train'
                # save the model that achieves the smallest validation error
                if val_sdr > best_sdr or math.isnan(best_sdr) or math.isinf(best_sdr):
                    best_sdr = val_sdr
                    if opt.rank == 0:
                        print('saving the best model (epoch %d, batch %d) with validation sdr %.3f\n' % (
                        epoch, batch, val_sdr))
                        _save_checkpoints(opt, model, "best")

            if (batch + 1) % opt.save_latest_freq == 0 and opt.rank == 0:
                print('saving the latest model (epoch %d, batch %d)' % (epoch, batch))
                _save_checkpoints(opt, model, "latest")
                ckpt_dict = {'optim_state_dict': optimizer.state_dict(), 'epoch': epoch, 'batch': batch + 1,
                             'cumsum_batch': cumsum_batch + 1, 'best_sdr': best_sdr}
                torch.save(ckpt_dict, os.path.join('.', opt.checkpoints_dir, opt.name, 'resume_latest.pth'))

            iter_start_time = time.time()

            cumsum_batch += 1

            if opt.rank == 0:
                pb.update()

            dist.barrier()

        # save checkpoints and evaluate model
        if opt.rank == 0:  # save the latest model for each epoch
            print('saving the latest model (epoch %d)' % epoch)
            _save_checkpoints(opt, model, "latest")
            ckpt_dict = {'optim_state_dict': optimizer.state_dict(), 'epoch': epoch + 1, 'batch': 0,
                         'cumsum_batch': cumsum_batch, 'best_sdr': best_sdr}
            torch.save(ckpt_dict, os.path.join('.', opt.checkpoints_dir, opt.name, 'resume_latest.pth'))

        if (epoch + 1) == opt.epochs:  # the last epoch, evaluate model
            model.eval()
            opt.mode = 'val'
            if opt.rank == 0:
                print('Begin evaluate at last Epoch %d, Batch %d: ' % (epoch, len(data_loader)), end='')
            val_sdr = evaluate(model, crit, opt.writer, cumsum_batch, data_loader_val, epoch, opt)
            model.train()
            opt.mode = 'train'
            # save the model that achieves the smallest validation error
            if val_sdr > best_sdr:
                best_sdr = val_sdr
                if opt.rank == 0:
                    print('saving the best model (epoch %d, batch %d) with validation sdr %.3f\n' % (epoch, len(data_loader), val_sdr))
                    _save_checkpoints(opt, model, "best")

        # decrease learning rate
        if (epoch + 1) in opt.lr_steps:
            decrease_learning_rate(optimizer, opt.decay_factor)

            if opt.rank == 0:
                print('decreased learning rate by ', opt.decay_factor)


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

    # create optimizer
    optimizer = create_optimizer(model, opt)
    # create loss
    crit = create_loss(opt)
    opt.window = torch.hann_window(opt.window_size).cuda()

    cudnn.benchmark = True

    train(opt, model, data_loader, data_loader_val, optimizer, crit)


if __name__ == '__main__':
    main()
