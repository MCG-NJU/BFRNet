#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import io
# import math
import math
import os
import os.path as osp
from scipy.io import wavfile
import numpy as np
import random
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from options.testMany_options import TestOptions
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data.audioVisual_dataset import get_preprocessing_pipelines, load_frame
from facenet_pytorch import MTCNN
# from torch_mir_eval import bss_eval_sources
from mir_eval import separation
from pypesq import pesq
from pystoi import stoi
from mmcv import ProgressBar
from petrel_client.client import Client
import h5py
import soundfile as sf
import cv2
from utils.utils import collate_fn


vision_transform_list = [transforms.ToTensor()]
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
vision_transform_list.append(normalize)
vision_transform = transforms.Compose(vision_transform_list)
lipreading_preprocessing_func = get_preprocessing_pipelines()['test']


# def getSeparationMetrics(audio_pred, audio_gt):
#     # audio_pred, audio_gt: N, L
#     (sdr, sir, sar, perm) = bss_eval_sources(audio_gt, audio_pred, False)
#     # print(sdr, sir, sar, perm)
#     return torch.mean(sdr), torch.mean(sir), torch.mean(sar)


def getSeparationMetrics_np(audio_pred, audio_gt):
    (sdr, sir, sar, perm) = separation.bss_eval_sources(audio_gt, audio_pred, False)
    # return sdr, sir, sar
    return np.mean(sdr), np.mean(sir), np.mean(sar)


def get_sisnr(pred_audios, gt_audios):
    assert gt_audios.size() == pred_audios.size()  # (B, L)
    # Step 1. Zero-mean norm
    gt_audios = gt_audios - torch.mean(gt_audios, axis=-1, keepdim=True)
    pred_audios = pred_audios - torch.mean(pred_audios, axis=-1, keepdim=True)
    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(gt_audios ** 2, axis=-1, keepdim=True) + 1e-6
    proj = torch.sum(gt_audios * pred_audios, axis=-1, keepdim=True) * gt_audios / ref_energy
    # e_noise = s' - s_target
    noise = pred_audios - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis=-1) / (torch.sum(noise ** 2, axis=-1) + 1e-6)
    sisnr = 10 * torch.log10(ratio + 1e-6)
    return torch.mean(sisnr)


def audio_normalize(samples, desired_rms=0.1, eps=1e-4):
    # samples: batch, num_speakers, L
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2, axis=-1)))
    samples = samples * np.expand_dims(desired_rms / rms, axis=-1)
    return samples


def supp_audio(audio, minimum_length):
    # B, L
    if len(audio[0]) >= minimum_length:
        return audio[:, :minimum_length]
    else:
        return np.tile(audio, (1, (minimum_length + len(audio[0]) - 1) // len(audio[0])))[:, :minimum_length]


def supp_mouth(mouth, minimum_length):
    # (B, L, 88, 88)
    if len(mouth[0]) >= minimum_length:
        return mouth[:, :minimum_length]
    else:
        return np.tile(mouth, (1, (minimum_length + len(mouth[0]) - 1) // len(mouth[0]), 1, 1))[:, :minimum_length]


def generate_spectrogram(audio):
    spec = torch.stft(audio, n_fft=512, hop_length=160, win_length=400, window=torch.hann_window(400).cuda()).cuda()
    return spec


class dataset(data.Dataset):
    def __init__(self, opt, mixture_path, audio_direc, mouth_direc, visual_direc, mtcnn):
        self.opt = opt
        self.audio_direc = audio_direc
        self.mouth_direc = mouth_direc
        self.visual_direc = visual_direc
        self.mtcnn = mtcnn

        if opt.ceph == "true":
            self.client = Client()
            with io.BytesIO(self.client.get(mixture_path, update_cache=True)) as mp:
                self.mix_lst = [d.decode('utf-8').strip() for d in mp.readlines()]
        else:
            with open(mixture_path, "r", encoding="utf-8") as f:
                self.mix_lst = f.read().splitlines()

        self.window = torch.hann_window(400)

    def __len__(self):
        return len(self.mix_lst)

    def _audio_augment(self, audio):
        audio = audio * (random.random() * 1.5 + 0.5)  # 0.5 - 2.0
        return audio

    def process_wav(self, tokens):
        num_speakers = len(tokens)
        # audio
        audios = []
        audio_length = []
        for n in range(num_speakers):
            audio_path = os.path.join(self.audio_direc, tokens[n]) + ".wav"
            if self.opt.ceph == "true":
                with io.BytesIO(self.client.get(audio_path, update_cache=True)) as ap:
                    _, audio = wavfile.read(ap)
            else:
                _, audio = wavfile.read(audio_path)
            audio = audio / 32768
            audios.append(audio)
            audio_length.append(len(audio))
        target_length = min(audio_length)

        for i in range(num_speakers):  # 截断audio并进行normalize
            audios[i] = audios[i][:target_length]
        audios = np.array(audios)
        # 将target_length补到大于等于target_length且是40800的倍数的最小值
        margin = int(2.55 * 16000)
        target = ((target_length + margin - 1) // margin) * margin
        seg = target // margin

        # 补充
        audios = supp_audio(audios, target)  # num_speakers, target
        # 变成seg段2.55秒的长度
        audios = audios.reshape(num_speakers, seg, margin).transpose((1, 0, 2))  # seg, num_speakers, margin
        audios = audio_normalize(audios) / num_speakers
        audios = torch.FloatTensor(audios)  # seg, num_speakers, margin

        # mixture
        audio_mix = torch.FloatTensor(torch.sum(audios, dim=1))  # seg, margin

        audio_mix_spec = self.generate_spectrogram_complex(audio_mix, 512, 160, 400)  # seg, 2, 257, 256
        audio_mix_spec = audio_mix_spec.unsqueeze(1).repeat((1, num_speakers, 1, 1, 1))  # seg, speakers, 2, 257, 256

        audio_mix = audio_mix.unsqueeze(1).repeat(1, num_speakers, 1)  # seg, num_speakers, margin

        return audios, audio_mix, audio_mix_spec, target_length

    def process_mouth(self, tokens, seg):
        num_speakers = len(tokens)
        # mouth
        mouthrois = []
        mouth_length = []
        for n in range(num_speakers):
            mouth_path = os.path.join(self.mouth_direc, tokens[n]) + "." + self.opt.mouthroi_format
            if self.opt.ceph == "true":
                with io.BytesIO(self.client.get(mouth_path, update_cache=True)) as path:
                    if self.opt.mouthroi_format == "h5":
                        mouth_roi = h5py.File(path, 'r')['data'][...]
                    elif self.opt.mouthroi_format == "npz":
                        mouth_roi = np.load(path)['data']
                    else:
                        mouth_roi = np.load(path)
            else:
                if self.opt.mouthroi_format == "h5":
                    mouth_roi = h5py.File(mouth_path, 'r')['data'][...]
                elif self.opt.mouthroi_format == "npz":
                    mouth_roi = np.load(mouth_path)['data']
                else:
                    mouth_roi = np.load(mouth_path)
            mouth_length.append(len(mouth_roi))
            mouthrois.append(mouth_roi)
        mouth_length = min(mouth_length)
        for n in range(num_speakers):
            mouthrois[n] = mouthrois[n][:mouth_length]  # T, 96, 96
        mouthrois = np.array(mouthrois)  # num_speakers, T, 96, 96

        # sup
        margin = 64
        target = seg * margin
        mouthrois = supp_mouth(mouthrois, target)  # num_speakers, target, 96, 96

        # preprocess
        tmp_mouthrois = []  # num_speakers, target, 88, 88
        for n in range(num_speakers):
            tmp_mouthrois.append(lipreading_preprocessing_func(mouthrois[n]))
        mouthrois = torch.FloatTensor(tmp_mouthrois).reshape(num_speakers, seg, margin, 88, 88).permute(1, 0, 2, 3, 4).unsqueeze(2)
        # (seg, num_speakers, 1, margin, 88, 88)
        return mouthrois

    def process_frame(self, tokens, seg):
        num_speakers = len(tokens)
        frames = []
        # scores = []
        for n in range(num_speakers):
            best_score = 0
            video_path = osp.join(self.visual_direc, tokens[n]) + ".mp4"
            for i in range(10):
                if self.opt.ceph == "true":
                    with io.BytesIO(self.client.get(video_path, update_cache=True)) as path:
                        frame = load_frame(path)
                else:
                    frame = load_frame(video_path)
                try:
                    boxes, scores = self.mtcnn.detect(frame)
                    if scores[0] > best_score:
                        best_frame = frame
                        best_score = scores[0]
                except:
                    pass
            # scores.append(best_score)
            try:
                best_frame = torch.tensor(np.array(best_frame))  # H, W, C
                best_frame = best_frame.permute(2, 0, 1).unsqueeze(0)  # 1, C, H, W
                _, C, H, W = best_frame.shape
                best_frame = torch.nn.functional.interpolate(best_frame.float(), scale_factor=224 / H, mode='bilinear') \
                    .reshape(1, C, 224, 224).squeeze(0).permute(1, 2, 0).byte()  # H, W, C
                frames.append(vision_transform(best_frame.numpy()).squeeze())
            except:
                if self.opt.ceph == "true":
                    with io.BytesIO(self.client.get(video_path, update_cache=True)) as path:
                        frame = load_frame(path)
                else:
                    frame = load_frame(video_path)
                frame = torch.tensor(np.array(frame))  # H, W, C
                frame = frame.permute(2, 0, 1).unsqueeze(0)  # 1, C, H, W
                _, C, H, W = frame.shape
                frame = torch.nn.functional.interpolate(frame.float(), scale_factor=224 / H, mode='bilinear') \
                    .reshape(1, C, 224, 224).squeeze(0).permute(1, 2, 0).byte()  # H, W, C
                frames.append(vision_transform(frame.numpy()).squeeze())
        frames = torch.stack(frames, dim=0).unsqueeze(0).repeat(seg, 1, 1, 1, 1)  # seg, num_speakers, 3, 224, 224
        return frames

    def generate_spectrogram_complex(self, audio, n_fft, hop_length, win_length):
        # audio: N, L
        spec = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=self.window, center=True)
        # spec.shape: N, F, T, 2
        return spec.permute(0, 3, 1, 2)  # N, 2, 257, 256

    def __getitem__(self, index):
        tokens = self.mix_lst[index].split(' ')

        assert self.opt.mix_number == len(tokens)

        try:
            audios, audio_mix, audio_mix_spec, target_length = self.process_wav(tokens)  # seg, num_speakers, margin
            seg = len(audios)
        except Exception as e:
            return self.__getitem__(index - 1)

        try:
            mouthrois = self.process_mouth(tokens, seg)  # seg, num_speakers, 1, 64, 88, 88
        except Exception as e:
            return self.__getitem__(index - 1)

        try:
            frames = self.process_frame(tokens, seg)  # seg, num_speakers, 3, 224, 224
        except Exception as e:
            return self.__getitem__(index - 1)

        data = {}
        data['frames'] = frames  # seg,num_speakers,3,224,224
        data['audios'] = audios  # seg, num_speakers, 40800
        data['audio_mix'] = audio_mix  # seg, num_speakers, 40800
        data['mouthrois'] = mouthrois  # seg, num_speakers, 1, 64, 88, 88
        data['audio_spec_mix'] = audio_mix_spec  # seg, num_speakers, 2, 257, 256
        data['seg'] = torch.IntTensor([seg])
        data['target_length'] = torch.IntTensor([target_length])

        return data


def visualize(i, sdr, num_speaker, audio_mix, audios, sep_audios, mouthrois, frames):
    mouthrois = mouthrois.detach().cpu().numpy()  # (n, 1, num, 88, 88)
    mouthrois = ((mouthrois * 0.165 + 0.421) * 255).astype(np.uint8)

    frames = np.transpose(frames.detach().cpu().numpy(), (0, 2, 3, 1))  # (n, 224, 224, 3)
    frames[:, :, :, 0] = frames[:, :, :, 0] * 0.229 + 0.485
    frames[:, :, :, 1] = frames[:, :, :, 1] * 0.224 + 0.456
    frames[:, :, :, 2] = frames[:, :, :, 2] * 0.225 + 0.406
    frames = (frames * 255).astype(np.uint8)

    # 保存audio_mix, audios, mouthrois, frames, sep_audios
    os.makedirs(f"sep/num{i}_{round(float(sdr), 2)}", exist_ok=True)
    dirname = f"sep/num{i}_{round(float(sdr), 2)}"
    for n in range(num_speaker):
        sf.write(f"{dirname}/audio_mix{n}.wav", audio_mix[n], 16000)
        sf.write(f"{dirname}/audios{n}.wav", audios[n], 16000)
        sf.write(f"{dirname}/sep_audios{n}.wav", sep_audios[n], 16000)

        dim = (88, 88)
        fourcc = cv2.VideoWriter_fourcc(*'FMP4')
        speaker_video = cv2.VideoWriter(f"{dirname}/mouthrois{n}.mp4", fourcc, 25.0, dim, isColor=False)
        for mouth in mouthrois[n][0]:
            speaker_video.write(mouth)
        speaker_video.release()

        cv2.imwrite(f"{dirname}/frames{n}.jpg", frames[n])


def save_statistics(file, sdr, num_speaker, audio_mix, audios, sep_audios, mouthrois, frames, scores):
    for n in range(num_speaker):
        file.write(f"{round(sdr[n], 2)}\t{len(audio_mix[n])}\t{np.mean(audio_mix[n])}\t{np.mean(audios[n])}\t{np.mean(sep_audios[n])}\t{torch.mean(mouthrois[n])}\t{torch.mean(frames[n])}\t{scores[n].item()}\n")


def process_mixture(opt, model, data_loader):
    model.eval()
    sdr_list, sir_list, sar_list = [], [], []
    sdri_list, siri_list = [], []
    sisnr_list = []
    pesq_list = []
    stoi_list = []
    pb = ProgressBar(len(data_loader), start=False)
    pb.start()
    window = torch.hann_window(400).cuda()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs = {}
            total_seg, num_speakers = data['frames'].shape[:2]
            inputs['frames'] = data['frames'].reshape(total_seg * num_speakers, 3, 224, 224).clone().detach().cuda()
            inputs['mouthrois'] = data['mouthrois'].reshape(total_seg * num_speakers, 1, 64, 88, 88).clone().detach().cuda()
            inputs['audio_spec_mix'] = data['audio_spec_mix'].reshape(total_seg * num_speakers, 2, 257, 256).clone().detach().cuda()
            inputs['num_speakers'] = num_speakers

            output = model(inputs)

            pred_spec = output['pred_specs_aft']  # total_seg * num_speakers, 257, 256, 2
            pred_audio = torch.istft(pred_spec, n_fft=512, hop_length=160, win_length=400, window=window, center=True)
            pred_audio = pred_audio.reshape(total_seg, num_speakers, -1)  # total_seg, num_speakers, L
            # total_seg, num_speakers, L
            seg_list = data['seg']
            cumsum = 0
            for idx, seg in enumerate(seg_list):
                tmp_pred_audio = pred_audio[cumsum: cumsum + seg]  # seg, num_speakers, L
                tmp_pred_audio = tmp_pred_audio.permute(1, 0, 2).reshape(num_speakers, -1)[:, :data['target_length'][idx]].cpu()  # num_speakers, target_length

                tmp_audio = data['audios'][cumsum: cumsum + seg]  # seg, num_speakers, L
                tmp_audio = tmp_audio.permute(1, 0, 2).reshape(num_speakers, -1)[:, :data['target_length'][idx]]  # num_speakers, target_length

                tmp_audio_mix = data['audio_mix'][cumsum: cumsum + seg]  # seg, num_speakers, L
                tmp_audio_mix = tmp_audio_mix.permute(1, 0, 2).reshape(num_speakers, -1)[:, :data['target_length'][idx]]  # num_speakers, target_length

                cumsum += seg

                sisnr = get_sisnr(tmp_pred_audio, tmp_audio).item()

                tmp_pred_audio = tmp_pred_audio.detach().numpy()
                tmp_audio = tmp_audio.numpy()
                tmp_audio_mix = tmp_audio_mix.numpy()

                sdr, sir, sar = getSeparationMetrics_np(tmp_pred_audio, tmp_audio)
                sdr_mix, sir_mix, sar_mix = getSeparationMetrics_np(tmp_audio_mix, tmp_audio)

                sdr = np.mean(sdr)
                sir = np.mean(sir)
                sar = np.mean(sar)
                sdr_mix = np.mean(sdr_mix)
                sir_mix = np.mean(sir_mix)

                sdri = sdr - sdr_mix
                siri = sir - sir_mix

                sdr_list.append(sdr)
                sdri_list.append(sdri)
                sir_list.append(sir)
                siri_list.append(siri)
                sar_list.append(sar)
                sisnr_list.append(sisnr)
                for n in range(num_speakers):
                    pesq_score_ = pesq(tmp_pred_audio[n], tmp_audio[n], 16000)
                    if math.isnan(pesq_score_) or math.isinf(pesq_score_):
                        print(f"tmp_pred_audio: {tmp_pred_audio[n]}", flush=True)
                        print(f"tmp_audio: {tmp_audio[n]}", flush=True)
                    else:
                        pesq_list.append(pesq_score_)
                    stoi_score_ = stoi(tmp_audio[n], tmp_pred_audio[n], 16000, extended=False)
                    stoi_list.append(stoi_score_)

            pb.update()

    avg_sdr = sum(sdr_list) / len(sdr_list)
    avg_sdri = sum(sdri_list) / len(sdri_list)

    avg_sir = sum(sir_list) / len(sir_list)
    avg_siri = sum(siri_list) / len(siri_list)

    avg_sar = sum(sar_list) / len(sar_list)

    avg_pesq = sum(pesq_list) / len(pesq_list)
    avg_stoi = sum(stoi_list) / len(stoi_list)

    avg_sisnr = sum(sisnr_list) / len(sisnr_list)

    print('SDR: %.2f' % avg_sdr)
    print('SIR: %.2f' % avg_sir)
    print('SAR: %.2f' % avg_sar)
    print('PESQ: %.2f' % avg_pesq)
    print('STOI: %.2f' % avg_stoi)
    print('SDRi: %.2f' % avg_sdri)
    print('SIRi: %.2f' % avg_siri)
    print('SISNR: %.2f' % avg_sisnr)


def main():
    #load test arguments
    opt = TestOptions().parse()
    opt.device = torch.device("cuda")
    opt.mode = 'test'
    opt.rank = 0

    # Network Builders
    builder = ModelBuilder()
    net_lipreading = builder.build_lipreadingnet(
        opt=opt,
        config_path=opt.lipreading_config_path,
        weights=opt.weights_lipreadingnet,
        extract_feats=opt.lipreading_extract_feature)
    #if identity feature dim is not 512, for resnet reduce dimension to this feature dim
    if opt.identity_feature_dim != 512:
        opt.with_fc = True
    else:
        opt.with_fc = False
    net_facial_attributes = builder.build_facial(
            opt=opt,
            pool_type=opt.visual_pool,
            fc_out=opt.identity_feature_dim,
            with_fc=opt.with_fc,
            weights=opt.weights_facial)
    net_unet = builder.build_unet(
            opt=opt,
            ngf=opt.unet_ngf,
            input_nc=opt.unet_input_nc,
            output_nc=opt.unet_output_nc,
            weights=opt.weights_unet)
    net_refine = builder.build_refine_net(
        opt=opt,
        num_layers=opt.refine_num_layers,
        weights=opt.weights_refine
    )

    nets = (net_lipreading, net_facial_attributes, net_unet, net_refine)

    # construct our audio-visual model
    model = AudioVisualModel(nets, opt).cuda()
    model.eval()
    mtcnn = MTCNN(keep_all=True, device=opt.device)

    data_set = dataset(opt, opt.test_file, opt.audio_root, opt.mouth_root, opt.mp4_root, mtcnn)
    data_loader = data.DataLoader(
        data_set,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=opt.nThreads,
        collate_fn=collate_fn
    )
    process_mixture(opt, model, data_loader)


if __name__ == '__main__':
    # test
    main()
