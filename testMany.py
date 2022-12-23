#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path as osp

from scipy.io import wavfile
import numpy as np
import h5py
import traceback
import io
from petrel_client.client import Client
from mmcv import ProgressBar
import bisect
import subprocess

import torchvision.transforms as transforms
import torch
import torch.nn.functional as F

from facenet_pytorch import MTCNN
import soundfile as sf

from options.testMany_options import TestOptions
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data.audioVisual_dataset import generate_spectrogram_complex, get_preprocessing_pipelines, load_frame, charToIx40, charToIx32
from utils.text_decoder import ctc_greedy_decode, compute_cer

import mir_eval.separation
from pypesq import pesq
from pystoi import stoi


client = Client()


# def _init_slurm(opt):
#     proc_id = int(os.environ['SLURM_PROCID'])
#     ntasks = int(os.environ['SLURM_NTASKS'])
#     node_list = os.environ['SLURM_NODELIST']
#     num_gpus = torch.cuda.device_count()
#     torch.cuda.set_device(proc_id % num_gpus)
#     addr = subprocess.getoutput(
#         f'scontrol show hostname {node_list} | head -n1')
#     # specify master port
#     if opt.port is not None:
#         os.environ['MASTER_PORT'] = str(opt.port)
#     elif 'MASTER_PORT' in os.environ:
#         pass  # use MASTER_PORT in the environment variable
#     else:
#         # 29500 is torch.distributed default port
#         os.environ['MASTER_PORT'] = '29500'
#     # use MASTER_ADDR in the environment variable if it already exists
#     if 'MASTER_ADDR' not in os.environ:
#         os.environ['MASTER_ADDR'] = addr
#     os.environ['WORLD_SIZE'] = str(ntasks)
#     os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
#     os.environ['RANK'] = str(proc_id)
#     dist.init_process_group(backend='nccl')
#     opt.rank = proc_id
#     opt.local_rank = proc_id % num_gpus
#     opt.device = torch.device("cuda", proc_id % num_gpus)


def load_mouthroi(mouthroi_path, mouthroi_format):
    if mouthroi_format == "npz":
        with io.BytesIO(client.get(mouthroi_path)) as mp1:
            mouthroi = np.load(mp1)["data"]
    else:  # h5
        with io.BytesIO(client.get(mouthroi_path)) as mp1:
            mouthroi = h5py.File(mp1, "r")["data"][...]
    return mouthroi


def load_audio(audio_path):
    with io.BytesIO(client.get(audio_path)) as ap:
        _, audio = wavfile.read(ap)
    return audio


def load_word(anno_path):
    with io.BytesIO(client.get(anno_path)) as ap:
        lines = ap.readlines()[4:]
    words, ends = [], []
    for line in lines:
        word, _, end, _ = line.strip().split()
        words.append(word)
        ends.append(end)
    words = np.array(words)
    ends = np.array(ends)
    return words, ends


def create_model(opt):
    if opt.visual_feature_type == 'both':
        visual_feature_dim = 512 + opt.identity_feature_dim
    elif opt.visual_feature_type == 'lipmotion':
        visual_feature_dim = 512
    else:  # identity
        visual_feature_dim = opt.identity_feature_dim
    # Network Builders
    builder = ModelBuilder()
    net_lipreading = builder.build_lipreadingnet(
        opt=opt,
        config_path=opt.lipreading_config_path,
        weights=opt.weights_lipreadingnet,
        extract_feats=opt.lipreading_extract_feature)
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
    net_unet = builder.build_unet(
        opt=opt,
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
        audioVisual_feature_dim=opt.unet_ngf * 8 + visual_feature_dim,
        identity_feature_dim=opt.identity_feature_dim,
        weights=opt.weights_unet)
    # if opt.with_recognition:
    #     # net_feature_extractor = builder.build_feature_extractor(
    #     #     opt=opt,
    #     #     ckpt_path=opt.extractor_ckpt_path,
    #     #     weights=opt.weights_extractor
    #     # )
    #     # if 'hubert_large' in opt.extractor_ckpt_path:
    #     #     feature_dim = net_feature_extractor.w2v_encoder.dimension
    #     # elif 'hubert_base' in opt.extractor_ckpt_path:
    #     #     feature_dim = net_feature_extractor.dimension
    #     # else:
    #     #     feature_dim = 768
    #     net_recognizer = builder.build_recognizer(
    #         opt=opt,
    #         ckpt_path=opt.recog_ckpt_path,
    #         weights=opt.weights_recognizer)
    #     if opt.recog_origin_fc:
    #         nets = (net_lipreading, net_facial_attribtes, net_unet, net_recognizer)
    #     else:
    #         net_recog_fc = builder.build_recog_fc(
    #             opt=opt,
    #             feature_dim=net_recognizer.feature_dim,
    #             num_classes=opt.recg_num_classes,
    #             weights=opt.weights_recog_fc)
    #         nets = (net_lipreading, net_facial_attribtes, net_unet, net_recognizer, net_recog_fc)
    # else:
    nets = (net_lipreading, net_facial_attribtes, net_unet)

    # construct our audio-visual model
    model = AudioVisualModel(nets, opt)
    model.to(opt.device)

    return model


def getSeparationMetrics_2mix(audio1, audio2, audio1_gt, audio2_gt):
    reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0)), axis=0)
    estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, False)
    #print(sdr, sir, sar, perm)
    return np.mean(sdr), np.mean(sir), np.mean(sar)


def getSeparationMetrics_3mix(audio1, audio2, audio3, audio1_gt, audio2_gt, audio3_gt):
    reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0),
                                        np.expand_dims(audio3_gt, axis=0)), axis=0)
    estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0),
                                        np.expand_dims(audio3, axis=0)), axis=0)
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, False)
    #print(sdr, sir, sar, perm)
    return np.mean(sdr), np.mean(sir), np.mean(sar)


def getSeparationMetrics_4mix(audio1, audio2, audio3, audio4, audio1_gt, audio2_gt, audio3_gt, audio4_gt):
    reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0),
                                        np.expand_dims(audio3_gt, axis=0), np.expand_dims(audio4_gt, axis=0)), axis=0)
    estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0),
                                        np.expand_dims(audio3, axis=0), np.expand_dims(audio4, axis=0)), axis=0)
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, False)
    #print(sdr, sir, sar, perm)
    return np.mean(sdr), np.mean(sir), np.mean(sar)


def getSeparationMetrics_5mix(audio1, audio2, audio3, audio4, audio5,
                              audio1_gt, audio2_gt, audio3_gt, audio4_gt, audio5_gt):
    reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0),
                                        np.expand_dims(audio3_gt, axis=0), np.expand_dims(audio4_gt, axis=0),
                                        np.expand_dims(audio5_gt, axis=0)), axis=0)
    estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0),
                                        np.expand_dims(audio3, axis=0), np.expand_dims(audio4, axis=0),
                                        np.expand_dims(audio5, axis=0)), axis=0)
    (sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, False)
    #print(sdr, sir, sar, perm)
    return np.mean(sdr), np.mean(sir), np.mean(sar)


def audio_normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return rms / desired_rms, samples


def supplement_audio(audio, multiple, minimum_length):
    audio_ = audio
    for i in range(multiple - 1):
        audio_ = np.concatenate((audio_, audio))
    audio = audio_[:minimum_length]
    return audio


def supplement_mouth(mouthroi, multiple, length):
    mouthroi_ = mouthroi
    for i in range(multiple - 1):
        mouthroi_ = np.concatenate((mouthroi_, mouthroi))
    mouthroi = mouthroi_[:length]
    return mouthroi


def supplement_word(words, ends, multiple, total_second, target_second):
    words_ = words
    ends_ = ends
    for i in range(multiple - 1):
        words_ = np.concatenate((words_, words))
        ends_ = np.concatenate((ends_, ends + total_second * (i + 1)))
    index = bisect.bisect_right(ends_, target_second)
    words = words[:index]
    sentence = " ".join(words)
    return sentence


def clip_audio(audio):
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio


def get_separated_audio(outputs, window, opt):
    # fetch data and predictions
    pred_spec1 = outputs['pred_spec1'][0]
    pred_spec2 = outputs['pred_spec2'][0]

    pred_audio1 = torch.istft(pred_spec1, n_fft=opt.n_fft, hop_length=opt.hop_size, win_length=opt.window_size,
                              window=window, center=True).detach().cpu().numpy()
    pred_audio2 = torch.istft(pred_spec2, n_fft=opt.n_fft, hop_length=opt.hop_size, win_length=opt.window_size,
                              window=window, center=True).detach().cpu().numpy()

    return pred_audio1, pred_audio2


def forward_asr(eos, blank, net_feature_extractor, net_recognizer, pred_audio, trgt, trgtLen):
    if 'hubert_large' in net_feature_extractor.name:
        recg_pred = net_recognizer(net_feature_extractor(**dict(source=pred_audio, padding_mask=None)))
    else:  # base
        recg_pred = net_recognizer(net_feature_extractor.extract_features(**dict(source=pred_audio, padding_mask=None)))
    recg_pred = F.log_softmax(recg_pred, dim=1).transpose(0, 1)  # (L, N, C)
    inputLen1 = len(recg_pred)
    prediction1, predictionLen1 = ctc_greedy_decode(recg_pred, inputLen1, eos, blank)
    cer = compute_cer(prediction1, trgt, predictionLen1, trgtLen)
    return cer


def process_2_mixture(model, opt, v1, v2, mtcnn, lipreading_preprocessing_func, vision_transform, window):
    # load data
    mouthroi_path1 = osp.join(opt.mouth_root, v1 + "." + opt.mouthroi_format)
    mouthroi_path2 = osp.join(opt.mouth_root, v2 + "." + opt.mouthroi_format)
    mouthroi_1 = load_mouthroi(mouthroi_path1, opt.mouthroi_format)
    mouthroi_2 = load_mouthroi(mouthroi_path2, opt.mouthroi_format)

    audio_path1 = osp.join(opt.audio_root, v1 + '.wav')
    audio_path2 = osp.join(opt.audio_root, v2 + '.wav')
    audio1 = load_audio(audio_path1)
    audio2 = load_audio(audio_path2)
    audio1 = audio1 / 32768
    audio2 = audio2 / 32768

    with_recog = opt.with_recognition
    opt.with_recognition = False

    if with_recog:
        words1, ends1 = load_word(osp.join(opt.anno_root, v1 + ".txt"))
        words2, ends2 = load_word(osp.join(opt.anno_root, v2 + ".txt"))

    # make sure the two audios are of the same length and then mix them
    audio_length = min(len(audio1), len(audio2))
    minimum_length = int(opt.audio_sampling_rate * opt.audio_length)

    audio_second1 = len(audio1) / opt.audio_sampling_rate
    audio_second2 = len(audio2) / opt.audio_sampling_rate
    target_second = audio_length / opt.audio_sampling_rate if audio_length >= minimum_length else opt.audio_length  # second
    multiple = 1

    if audio_length < minimum_length:
        multiple = (minimum_length + audio_length - 1) // audio_length
        audio1 = supplement_audio(audio1, multiple, minimum_length)
        audio2 = supplement_audio(audio2, multiple, minimum_length)
        audio_length = minimum_length

    if with_recog:
        trgt1 = supplement_word(words1, ends1, multiple, audio_second1, target_second)
        trgt2 = supplement_word(words2, ends2, multiple, audio_second2, target_second)
        if opt.recg_num_classes == 40:
            trgt1 = [charToIx40[c] for c in trgt1] + charToIx40["<EOS>"]
            trgt2 = [charToIx40[c] for c in trgt2] + charToIx40["<EOS>"]
        else: # recg_num_classes==32
            trgt1 = [charToIx32[c] if c in charToIx32 else charToIx32['<unk>'] for c in trgt1]
            trgt2 = [charToIx32[c] if c in charToIx32 else charToIx32['<unk>'] for c in trgt2]
        trgtLen1 = len(trgt1)
        trgtLen2 = len(trgt2)

    video_length = min(len(mouthroi_1), len(mouthroi_2))
    if video_length < opt.num_frames:
        multiple = (opt.num_frames + video_length - 1) // video_length
        mouthroi_1 = supplement_mouth(mouthroi_1, multiple, opt.num_frames)
        mouthroi_2 = supplement_mouth(mouthroi_2, multiple, opt.num_frames)

    audio1 = clip_audio(audio1[:audio_length])
    audio2 = clip_audio(audio2[:audio_length])
    audio_mix = (audio1 + audio2) / 2.0

    if opt.reliable_face:
        best_score_1 = 0
        best_score_2 = 0
        for i in range(10):
            frame_1 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v1 + '.mp4'))))
            frame_2 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v2 + '.mp4'))))
            try:
                boxes, scores = mtcnn.detect(frame_1)
                if scores[0] > best_score_1:
                    best_frame_1 = frame_1
                    best_score_1 = scores[0]
            except:
                pass
            try:
                boxes, scores = mtcnn.detect(frame_2)
                if scores[0] > best_score_2:
                    best_frame_2 = frame_2
                    best_score_2 = scores[0]
            except:
                pass
        frames_1 = vision_transform(best_frame_1).squeeze().unsqueeze(0)
        frames_2 = vision_transform(best_frame_2).squeeze().unsqueeze(0)
    else:
        frame_1_list = []
        frame_2_list = []
        for i in range(opt.number_of_identity_frames):
            frame_1 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v1 + '.mp4'))))
            frame_2 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v2 + '.mp4'))))
            frame_1 = vision_transform(frame_1)
            frame_2 = vision_transform(frame_2)
            frame_1_list.append(frame_1)
            frame_2_list.append(frame_2)
        frames_1 = torch.stack(frame_1_list).squeeze().unsqueeze(0)
        frames_2 = torch.stack(frame_2_list).squeeze().unsqueeze(0)

    # perform separation over the whole audio using a sliding window approach
    overlap_count = np.zeros((audio_length))
    sep_audio1 = np.zeros((audio_length))
    sep_audio2 = np.zeros((audio_length))
    sliding_window_start = 0
    data = {}
    avged_sep_audio1 = np.zeros((audio_length))
    avged_sep_audio2 = np.zeros((audio_length))

    samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
    while sliding_window_start + samples_per_window < audio_length:
        sliding_window_end = sliding_window_start + samples_per_window

        # get audio spectrogram
        segment1_audio = audio1[sliding_window_start:sliding_window_end]
        segment2_audio = audio2[sliding_window_start:sliding_window_end]

        if opt.audio_normalization:
            normalizer1, segment1_audio = audio_normalize(segment1_audio)
            normalizer2, segment2_audio = audio_normalize(segment2_audio)
        else:
            normalizer1 = 1
            normalizer2 = 1

        audio_segment = (segment1_audio + segment2_audio) / 2
        audio_mix_spec = generate_spectrogram_complex(audio_segment, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_1 = generate_spectrogram_complex(segment1_audio, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_2 = generate_spectrogram_complex(segment2_audio, opt.window_size, opt.hop_size, opt.n_fft)

        # get mouthroi
        frame_index_start = int(sliding_window_start / opt.audio_sampling_rate * 25)
        frame_index_end = frame_index_start + opt.num_frames
        if frame_index_end > len(mouthroi_1) or frame_index_end > len(mouthroi_2):
            gap = frame_index_end - min(len(mouthroi_1), len(mouthroi_2))
            frame_index_start -= gap
            frame_index_end -= gap
        segment1_mouthroi = mouthroi_1[frame_index_start:frame_index_end, :, :]
        segment2_mouthroi = mouthroi_2[frame_index_start:frame_index_end, :, :]

        # transform mouthrois
        segment1_mouthroi = lipreading_preprocessing_func(segment1_mouthroi)
        segment2_mouthroi = lipreading_preprocessing_func(segment2_mouthroi)

        data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
        data['mouthroi1'] = torch.FloatTensor(segment1_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['mouthroi2'] = torch.FloatTensor(segment2_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['audio_spec1'] = torch.FloatTensor(audio_spec_1).cuda().unsqueeze(0)
        data['audio_spec2'] = torch.FloatTensor(audio_spec_2).cuda().unsqueeze(0)
        data['frame1'] = frames_1.cuda()
        data['frame2'] = frames_2.cuda()

        try:
            outputs = model.forward(data)
        except Exception as e:
            traceback.print_exc()
            exit(-1)
        reconstructed_signal_1, reconstructed_signal_2 = get_separated_audio(outputs, window, opt)
        reconstructed_signal_1 = reconstructed_signal_1 * normalizer1
        reconstructed_signal_2 = reconstructed_signal_2 * normalizer2
        sep_audio1[sliding_window_start:sliding_window_end] =\
            sep_audio1[sliding_window_start:sliding_window_end] + reconstructed_signal_1
        sep_audio2[sliding_window_start:sliding_window_end] = \
            sep_audio2[sliding_window_start:sliding_window_end] + reconstructed_signal_2
        # update overlap count
        overlap_count[sliding_window_start:sliding_window_end] =\
            overlap_count[sliding_window_start:sliding_window_end] + 1
        sliding_window_start = sliding_window_start + int(opt.hop_second * opt.audio_sampling_rate)

    # deal with the last segment
    # get audio spectrogram
    segment1_audio = audio1[-samples_per_window:]
    segment2_audio = audio2[-samples_per_window:]

    if opt.audio_normalization:
        normalizer1, segment1_audio = audio_normalize(segment1_audio)
        normalizer2, segment2_audio = audio_normalize(segment2_audio)
    else:
        normalizer1 = 1
        normalizer2 = 1

    audio_segment = (segment1_audio + segment2_audio) / 2
    audio_mix_spec = generate_spectrogram_complex(audio_segment, opt.window_size, opt.hop_size, opt.n_fft)
    # get mouthroi
    frame_index_start = int((len(audio1) - samples_per_window) / opt.audio_sampling_rate * 25)
    frame_index_end = frame_index_start + opt.num_frames
    if frame_index_end > len(mouthroi_1) or frame_index_end > len(mouthroi_2):
        gap = frame_index_end - min(len(mouthroi_1), len(mouthroi_2))
        frame_index_start -= gap
        frame_index_end -= gap
    segment1_mouthroi = mouthroi_1[frame_index_start:frame_index_end, :, :]
    segment2_mouthroi = mouthroi_2[frame_index_start:frame_index_end, :, :]
    # transform mouthrois
    segment1_mouthroi = lipreading_preprocessing_func(segment1_mouthroi)
    segment2_mouthroi = lipreading_preprocessing_func(segment2_mouthroi)
    audio_spec_1 = generate_spectrogram_complex(segment1_audio, opt.window_size, opt.hop_size, opt.n_fft)
    audio_spec_2 = generate_spectrogram_complex(segment2_audio, opt.window_size, opt.hop_size, opt.n_fft)

    data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
    data['mouthroi1'] = torch.FloatTensor(segment1_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['mouthroi2'] = torch.FloatTensor(segment2_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['audio_spec1'] = torch.FloatTensor(audio_spec_1).cuda().unsqueeze(0)
    data['audio_spec2'] = torch.FloatTensor(audio_spec_2).cuda().unsqueeze(0)
    data['frame1'] = frames_1.cuda()
    data['frame2'] = frames_2.cuda()

    try:
        outputs = model.forward(data)
    except Exception as e:
        traceback.print_exc()
        exit(-1)
    reconstructed_signal_1, reconstructed_signal_2 = get_separated_audio(outputs, window, opt)
    reconstructed_signal_1 = reconstructed_signal_1 * normalizer1
    reconstructed_signal_2 = reconstructed_signal_2 * normalizer2
    sep_audio1[-samples_per_window:] = sep_audio1[-samples_per_window:] + reconstructed_signal_1
    sep_audio2[-samples_per_window:] = sep_audio2[-samples_per_window:] + reconstructed_signal_2
    # update overlap count
    overlap_count[-samples_per_window:] = overlap_count[-samples_per_window:] + 1

    # divide the aggregated predicted audio by the overlap count
    avged_sep_audio1 = avged_sep_audio1 + clip_audio(np.divide(sep_audio1, overlap_count))
    avged_sep_audio2 = avged_sep_audio2 + clip_audio(np.divide(sep_audio2, overlap_count))

    if opt.save_output == "true":
        output_dir = osp.join(opt.output_dir_root, v1 + '@' + v2)
        if not osp.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        sf.write(osp.join(output_dir, v1 + '_separated.wav'), avged_sep_audio1, opt.audio_sampling_rate)
        sf.write(osp.join(output_dir, v2 + '_separated.wav'), avged_sep_audio2, opt.audio_sampling_rate)

    sdr, sir, sar = getSeparationMetrics_2mix(avged_sep_audio1, avged_sep_audio2, audio1, audio2)
    sdr_mix, sir_mix, sar_mix = getSeparationMetrics_2mix(audio_mix, audio_mix, audio1, audio2)
    # PESQ
    pesq_score1 = pesq(avged_sep_audio1, audio1, opt.audio_sampling_rate)
    pesq_score2 = pesq(avged_sep_audio2, audio2, opt.audio_sampling_rate)
    pesq_score = (pesq_score1 + pesq_score2) / 2
    # STOI
    stoi_score1 = stoi(audio1, avged_sep_audio1, opt.audio_sampling_rate, extended=False)
    stoi_score2 = stoi(audio2, avged_sep_audio2, opt.audio_sampling_rate, extended=False)
    stoi_score = (stoi_score1 + stoi_score2) / 2

    if with_recog:
        opt.with_recognition = True
        net_feature_extractor, net_recognizer = model.net_feature_extractor, model.net_recognizer
        eos = charToIx40["<EOS>"] if opt.recg_num_classes == 40 else charToIx32["</s>"]
        cer1 = forward_asr(eos, opt.blank_label, net_feature_extractor, net_recognizer, avged_sep_audio1.unsqueeze(0), trgt1, trgtLen1)
        cer2 = forward_asr(eos, opt.blank_label, net_feature_extractor, net_recognizer, avged_sep_audio2.unsqueeze(0), trgt2, trgtLen2)
        return sdr, sir, sar, pesq_score, stoi_score, \
               sdr - sdr_mix, sir - sir_mix, sar - sar_mix,\
               (cer1 + cer2) / 2

    return sdr, sir, sar, pesq_score, stoi_score, \
           sdr - sdr_mix, sir - sir_mix, sar - sar_mix


def process_3_mixture(model, opt, v1, v2, v3, mtcnn, lipreading_preprocessing_func, vision_transform, window):
    # load data
    mouthroi_path1 = osp.join(opt.mouth_root, v1 + "." + opt.mouthroi_format)
    mouthroi_path2 = osp.join(opt.mouth_root, v2 + "." + opt.mouthroi_format)
    mouthroi_path3 = osp.join(opt.mouth_root, v3 + "." + opt.mouthroi_format)
    mouthroi_1 = load_mouthroi(mouthroi_path1, opt.mouthroi_format)
    mouthroi_2 = load_mouthroi(mouthroi_path2, opt.mouthroi_format)
    mouthroi_3 = load_mouthroi(mouthroi_path3, opt.mouthroi_format)

    audio_path1 = osp.join(opt.audio_root, v1 + '.wav')
    audio_path2 = osp.join(opt.audio_root, v2 + '.wav')
    audio_path3 = osp.join(opt.audio_root, v3 + '.wav')
    audio1 = load_audio(audio_path1)
    audio2 = load_audio(audio_path2)
    audio3 = load_audio(audio_path3)
    audio1 = audio1 / 32768
    audio2 = audio2 / 32768
    audio3 = audio3 / 32768

    with_recog = opt.with_recognition
    opt.with_recognition = False

    if with_recog:
        words1, ends1 = load_word(osp.join(opt.anno_root, v1 + ".txt"))
        words2, ends2 = load_word(osp.join(opt.anno_root, v2 + ".txt"))
        words3, ends3 = load_word(osp.join(opt.anno_root, v3 + ".txt"))

    # make sure the two audios are of the same length and then mix them
    audio_length = min(len(audio1), len(audio2), len(audio3))
    minimum_length = int(opt.audio_sampling_rate * opt.audio_length)

    audio_second1 = len(audio1) / opt.audio_sampling_rate
    audio_second2 = len(audio2) / opt.audio_sampling_rate
    audio_second3 = len(audio3) / opt.audio_sampling_rate
    target_second = audio_length / opt.audio_sampling_rate if audio_length >= minimum_length else opt.audio_length  # second
    multiple = 1

    if audio_length < minimum_length:
        multiple = (minimum_length + audio_length - 1) // audio_length
        audio1 = supplement_audio(audio1, multiple, minimum_length)
        audio2 = supplement_audio(audio2, multiple, minimum_length)
        audio3 = supplement_audio(audio3, multiple, minimum_length)
        audio_length = minimum_length

    if with_recog:
        trgt1 = supplement_word(words1, ends1, multiple, audio_second1, target_second)
        trgt2 = supplement_word(words2, ends2, multiple, audio_second2, target_second)
        trgt3 = supplement_word(words3, ends3, multiple, audio_second3, target_second)
        if opt.recg_num_classes == 40:
            trgt1 = [charToIx40[c] for c in trgt1] + charToIx40["<EOS>"]
            trgt2 = [charToIx40[c] for c in trgt2] + charToIx40["<EOS>"]
            trgt3 = [charToIx40[c] for c in trgt3] + charToIx40["<EOS>"]
        else:  # recg_num_classes==32
            trgt1 = [charToIx32[c] if c in charToIx32 else charToIx32['<unk>'] for c in trgt1]
            trgt2 = [charToIx32[c] if c in charToIx32 else charToIx32['<unk>'] for c in trgt2]
            trgt3 = [charToIx32[c] if c in charToIx32 else charToIx32['<unk>'] for c in trgt3]
        trgtLen1 = len(trgt1)
        trgtLen2 = len(trgt2)
        trgtLen3 = len(trgt3)

    video_length = min(len(mouthroi_1), len(mouthroi_2), len(mouthroi_3))
    if video_length < opt.num_frames:
        multiple = (opt.num_frames + video_length - 1) // video_length
        mouthroi_1 = supplement_mouth(mouthroi_1, multiple, opt.num_frames)
        mouthroi_2 = supplement_mouth(mouthroi_2, multiple, opt.num_frames)
        mouthroi_3 = supplement_mouth(mouthroi_3, multiple, opt.num_frames)

    audio1 = clip_audio(audio1[:audio_length])
    audio2 = clip_audio(audio2[:audio_length])
    audio3 = clip_audio(audio3[:audio_length])
    audio_mix = (audio1 + audio2 + audio3) / 3.0

    if opt.reliable_face:
        best_score_1 = 0
        best_score_2 = 0
        best_score_3 = 0
        for i in range(10):
            frame_1 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v1 + '.mp4'))))
            frame_2 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v2 + '.mp4'))))
            frame_3 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v3 + '.mp4'))))
            try:
                boxes, scores = mtcnn.detect(frame_1)
                if scores[0] > best_score_1:
                    best_frame_1 = frame_1
                    best_score_1 = scores[0]
            except:
                pass
            try:
                boxes, scores = mtcnn.detect(frame_2)
                if scores[0] > best_score_2:
                    best_frame_2 = frame_2
                    best_score_2 = scores[0]
            except:
                pass
            try:
                boxes, scores = mtcnn.detect(frame_3)
                if scores[0] > best_score_3:
                    best_frame_3 = frame_3
                    best_score_3 = scores[0]
            except:
                pass
        frames_1 = vision_transform(best_frame_1).squeeze().unsqueeze(0)
        frames_2 = vision_transform(best_frame_2).squeeze().unsqueeze(0)
        frames_3 = vision_transform(best_frame_3).squeeze().unsqueeze(0)
    else:
        frame_1_list = []
        frame_2_list = []
        frame_3_list = []
        for i in range(opt.number_of_identity_frames):
            frame_1 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v1 + '.mp4'))))
            frame_2 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v2 + '.mp4'))))
            frame_3 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v3 + '.mp4'))))
            frame_1 = vision_transform(frame_1)
            frame_2 = vision_transform(frame_2)
            frame_3 = vision_transform(frame_3)
            frame_1_list.append(frame_1)
            frame_2_list.append(frame_2)
            frame_3_list.append(frame_3)
        frames_1 = torch.stack(frame_1_list).squeeze().unsqueeze(0)
        frames_2 = torch.stack(frame_2_list).squeeze().unsqueeze(0)
        frames_3 = torch.stack(frame_3_list).squeeze().unsqueeze(0)

    # perform separation over the whole audio using a sliding window approach
    overlap_count = np.zeros((audio_length))
    sep_audio1 = np.zeros((audio_length))
    sep_audio2 = np.zeros((audio_length))
    sep_audio3 = np.zeros((audio_length))
    sliding_window_start = 0
    data = {}
    avged_sep_audio1 = np.zeros((audio_length))
    avged_sep_audio2 = np.zeros((audio_length))
    avged_sep_audio3 = np.zeros((audio_length))

    samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
    while sliding_window_start + samples_per_window < audio_length:
        sliding_window_end = sliding_window_start + samples_per_window

        # get audio spectrogram
        segment1_audio = audio1[sliding_window_start:sliding_window_end]
        segment2_audio = audio2[sliding_window_start:sliding_window_end]
        segment3_audio = audio3[sliding_window_start:sliding_window_end]

        if opt.audio_normalization:
            normalizer1, segment1_audio = audio_normalize(segment1_audio)
            normalizer2, segment2_audio = audio_normalize(segment2_audio)
            normalizer3, segment3_audio = audio_normalize(segment3_audio)
        else:
            normalizer1 = 1
            normalizer2 = 1
            normalizer3 = 1

        audio_segment = (segment1_audio + segment2_audio + segment3_audio) / 3
        audio_mix_spec = generate_spectrogram_complex(audio_segment, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_1 = generate_spectrogram_complex(segment1_audio, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_2 = generate_spectrogram_complex(segment2_audio, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_3 = generate_spectrogram_complex(segment3_audio, opt.window_size, opt.hop_size, opt.n_fft)

        # get mouthroi
        frame_index_start = int(sliding_window_start / opt.audio_sampling_rate * 25)
        frame_index_end = frame_index_start + opt.num_frames
        if frame_index_end > len(mouthroi_1) or frame_index_end > len(mouthroi_2) or frame_index_end > len(mouthroi_3):
            gap = frame_index_end - min(len(mouthroi_1), len(mouthroi_2), len(mouthroi_3))
            frame_index_start -= gap
            frame_index_end -= gap
        segment1_mouthroi = mouthroi_1[frame_index_start:frame_index_end, :, :]
        segment2_mouthroi = mouthroi_2[frame_index_start:frame_index_end, :, :]
        segment3_mouthroi = mouthroi_3[frame_index_start:frame_index_end, :, :]

        # transform mouthrois
        segment1_mouthroi = lipreading_preprocessing_func(segment1_mouthroi)
        segment2_mouthroi = lipreading_preprocessing_func(segment2_mouthroi)
        segment3_mouthroi = lipreading_preprocessing_func(segment3_mouthroi)

        # separate 1 and 2
        data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
        data['mouthroi1'] = torch.FloatTensor(segment1_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['mouthroi2'] = torch.FloatTensor(segment2_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['audio_spec1'] = torch.FloatTensor(audio_spec_1).cuda().unsqueeze(0)
        data['audio_spec2'] = torch.FloatTensor(audio_spec_2).cuda().unsqueeze(0)
        data['frame1'] = frames_1.cuda()
        data['frame2'] = frames_2.cuda()

        try:
            outputs = model.forward(data)
        except Exception as e:
            traceback.print_exc()
            exit(-1)
        reconstructed_signal_1, reconstructed_signal_2 = get_separated_audio(outputs, window, opt)

        # separate 2 and 3
        data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
        data['mouthroi1'] = torch.FloatTensor(segment3_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['mouthroi2'] = torch.FloatTensor(segment2_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['audio_spec1'] = torch.FloatTensor(audio_spec_3).cuda().unsqueeze(0)
        data['audio_spec2'] = torch.FloatTensor(audio_spec_2).cuda().unsqueeze(0)
        data['frame1'] = frames_3.cuda()
        data['frame2'] = frames_2.cuda()

        try:
            outputs = model.forward(data)
        except Exception as e:
            traceback.print_exc()
            exit(-1)
        reconstructed_signal_3, _ = get_separated_audio(outputs, window, opt)

        reconstructed_signal_1 = reconstructed_signal_1 * normalizer1
        reconstructed_signal_2 = reconstructed_signal_2 * normalizer2
        reconstructed_signal_3 = reconstructed_signal_3 * normalizer3

        sep_audio1[sliding_window_start:sliding_window_end] = \
            sep_audio1[sliding_window_start:sliding_window_end] + reconstructed_signal_1
        sep_audio2[sliding_window_start:sliding_window_end] = \
            sep_audio2[sliding_window_start:sliding_window_end] + reconstructed_signal_2
        sep_audio3[sliding_window_start:sliding_window_end] = \
            sep_audio3[sliding_window_start:sliding_window_end] + reconstructed_signal_3

        overlap_count[sliding_window_start:sliding_window_end] = \
            overlap_count[sliding_window_start:sliding_window_end] + 1

        sliding_window_start = sliding_window_start + int(opt.hop_second * opt.audio_sampling_rate)

    # deal with the last segment
    # get audio spectrogram
    segment1_audio = audio1[-samples_per_window:]
    segment2_audio = audio2[-samples_per_window:]
    segment3_audio = audio3[-samples_per_window:]

    if opt.audio_normalization:
        normalizer1, segment1_audio = audio_normalize(segment1_audio)
        normalizer2, segment2_audio = audio_normalize(segment2_audio)
        normalizer3, segment3_audio = audio_normalize(segment3_audio)
    else:
        normalizer1 = 1
        normalizer2 = 1
        normalizer3 = 1

    audio_segment = (segment1_audio + segment2_audio + segment3_audio) / 2
    audio_mix_spec = generate_spectrogram_complex(audio_segment, opt.window_size, opt.hop_size, opt.n_fft)
    # get mouthroi
    frame_index_start = int((len(audio1) - samples_per_window) / opt.audio_sampling_rate * 25)
    frame_index_end = frame_index_start + opt.num_frames
    if frame_index_end > len(mouthroi_1) or frame_index_end > len(mouthroi_2) or frame_index_end > len(mouthroi_3):
        gap = frame_index_end - min(len(mouthroi_1), len(mouthroi_2), len(mouthroi_3))
        frame_index_start -= gap
        frame_index_end -= gap
    segment1_mouthroi = mouthroi_1[frame_index_start:frame_index_end, :, :]
    segment2_mouthroi = mouthroi_2[frame_index_start:frame_index_end, :, :]
    segment3_mouthroi = mouthroi_3[frame_index_start:frame_index_end, :, :]
    # transform mouthrois
    segment1_mouthroi = lipreading_preprocessing_func(segment1_mouthroi)
    segment2_mouthroi = lipreading_preprocessing_func(segment2_mouthroi)
    segment3_mouthroi = lipreading_preprocessing_func(segment3_mouthroi)
    audio_spec_1 = generate_spectrogram_complex(segment1_audio, opt.window_size, opt.hop_size, opt.n_fft)
    audio_spec_2 = generate_spectrogram_complex(segment2_audio, opt.window_size, opt.hop_size, opt.n_fft)
    audio_spec_3 = generate_spectrogram_complex(segment3_audio, opt.window_size, opt.hop_size, opt.n_fft)

    # separate 1 and 2
    data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
    data['mouthroi1'] = torch.FloatTensor(segment1_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['mouthroi2'] = torch.FloatTensor(segment2_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['audio_spec1'] = torch.FloatTensor(audio_spec_1).cuda().unsqueeze(0)
    data['audio_spec2'] = torch.FloatTensor(audio_spec_2).cuda().unsqueeze(0)
    data['frame1'] = frames_1.cuda()
    data['frame2'] = frames_2.cuda()

    try:
        outputs = model.forward(data)
    except Exception as e:
        traceback.print_exc()
        exit(-1)
    reconstructed_signal_1, reconstructed_signal_2 = get_separated_audio(outputs, window, opt)

    # separate 3 and 2
    data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
    data['mouthroi1'] = torch.FloatTensor(segment3_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['mouthroi2'] = torch.FloatTensor(segment2_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['audio_spec1'] = torch.FloatTensor(audio_spec_3).cuda().unsqueeze(0)
    data['audio_spec2'] = torch.FloatTensor(audio_spec_2).cuda().unsqueeze(0)
    data['frame1'] = frames_3.cuda()
    data['frame2'] = frames_2.cuda()

    try:
        outputs = model.forward(data)
    except Exception as e:
        traceback.print_exc()
        exit(-1)
    reconstructed_signal_3, _ = get_separated_audio(outputs, window, opt)

    reconstructed_signal_1 = reconstructed_signal_1 * normalizer1
    reconstructed_signal_2 = reconstructed_signal_2 * normalizer2
    reconstructed_signal_3 = reconstructed_signal_3 * normalizer3

    sep_audio1[-samples_per_window:] = sep_audio1[-samples_per_window:] + reconstructed_signal_1
    sep_audio2[-samples_per_window:] = sep_audio2[-samples_per_window:] + reconstructed_signal_2
    sep_audio3[-samples_per_window:] = sep_audio3[-samples_per_window:] + reconstructed_signal_3

    # divide the aggregated predicted audio by the overlap count
    overlap_count[-samples_per_window:] = overlap_count[-samples_per_window:] + 1
    avged_sep_audio1 = avged_sep_audio1 + clip_audio(np.divide(sep_audio1, overlap_count))
    avged_sep_audio2 = avged_sep_audio2 + clip_audio(np.divide(sep_audio2, overlap_count))
    avged_sep_audio3 = avged_sep_audio3 + clip_audio(np.divide(sep_audio3, overlap_count))

    if opt.save_output == "true":
        output_dir = osp.join(opt.output_dir_root, v1 + '@' + v2 + "@" + v3)
        if not osp.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        sf.write(osp.join(output_dir, v1 + '_separated.wav'), avged_sep_audio1, opt.audio_sampling_rate)
        sf.write(osp.join(output_dir, v2 + '_separated.wav'), avged_sep_audio2, opt.audio_sampling_rate)
        sf.write(osp.join(output_dir, v3 + '_separated.wav'), avged_sep_audio3, opt.audio_sampling_rate)
        sf.write(osp.join(output_dir, 'noise.wav'), audio_segment - avged_sep_audio1 - avged_sep_audio2 - avged_sep_audio3, opt.audio_sampling_rate)

    sdr, sir, sar = getSeparationMetrics_3mix(avged_sep_audio1, avged_sep_audio2, avged_sep_audio3,
                                              audio1, audio2, audio3)
    sdr_mix, sir_mix, sar_mix = getSeparationMetrics_3mix(audio_mix, audio_mix, audio_mix, audio1, audio2, audio3)
    pesq_score1 = pesq(avged_sep_audio1, audio1, opt.audio_sampling_rate)
    pesq_score2 = pesq(avged_sep_audio2, audio2, opt.audio_sampling_rate)
    pesq_score3 = pesq(avged_sep_audio3, audio3, opt.audio_sampling_rate)
    pesq_score = (pesq_score1 + pesq_score2 + pesq_score3) / 3
    # STOI
    stoi_score1 = stoi(audio1, avged_sep_audio1, opt.audio_sampling_rate, extended=False)
    stoi_score2 = stoi(audio2, avged_sep_audio2, opt.audio_sampling_rate, extended=False)
    stoi_score3 = stoi(audio3, avged_sep_audio3, opt.audio_sampling_rate, extended=False)
    stoi_score = (stoi_score1 + stoi_score2 + stoi_score3) / 3

    if with_recog:
        opt.with_recognition = True
        net_feature_extractor, net_recognizer = model.net_feature_extractor, model.net_recognizer
        eos = charToIx40["<EOS>"] if opt.recg_num_classes == 40 else charToIx32["</s>"]
        cer1 = forward_asr(eos, opt.blank_label, net_feature_extractor, net_recognizer, avged_sep_audio1.unsqueeze(0), trgt1, trgtLen1)
        cer2 = forward_asr(eos, opt.blank_label, net_feature_extractor, net_recognizer, avged_sep_audio2.unsqueeze(0), trgt2, trgtLen2)
        cer3 = forward_asr(eos, opt.blank_label, net_feature_extractor, net_recognizer, avged_sep_audio3.unsqueeze(0), trgt3, trgtLen3)
        return sdr, sir, sar, pesq_score, stoi_score, \
               sdr - sdr_mix, sir - sir_mix, sar - sar_mix, \
               (cer1 + cer2 + cer3) / 3

    return sdr, sir, sar, pesq_score, stoi_score, \
           sdr - sdr_mix, sir - sir_mix, sar - sar_mix


def process_4_mixture(model, opt, v1, v2, v3, v4, mtcnn, lipreading_preprocessing_func, vision_transform, window):
    # load data
    mouthroi_path1 = osp.join(opt.mouth_root, v1 + "." + opt.mouthroi_format)
    mouthroi_path2 = osp.join(opt.mouth_root, v2 + "." + opt.mouthroi_format)
    mouthroi_path3 = osp.join(opt.mouth_root, v3 + "." + opt.mouthroi_format)
    mouthroi_path4 = osp.join(opt.mouth_root, v4 + "." + opt.mouthroi_format)
    mouthroi_1 = load_mouthroi(mouthroi_path1, opt.mouthroi_format)
    mouthroi_2 = load_mouthroi(mouthroi_path2, opt.mouthroi_format)
    mouthroi_3 = load_mouthroi(mouthroi_path3, opt.mouthroi_format)
    mouthroi_4 = load_mouthroi(mouthroi_path4, opt.mouthroi_format)

    audio_path1 = osp.join(opt.audio_root, v1 + '.wav')
    audio_path2 = osp.join(opt.audio_root, v2 + '.wav')
    audio_path3 = osp.join(opt.audio_root, v3 + '.wav')
    audio_path4 = osp.join(opt.audio_root, v4 + '.wav')
    audio1 = load_audio(audio_path1)
    audio2 = load_audio(audio_path2)
    audio3 = load_audio(audio_path3)
    audio4 = load_audio(audio_path4)
    audio1 = audio1 / 32768
    audio2 = audio2 / 32768
    audio3 = audio3 / 32768
    audio4 = audio4 / 32768

    with_recog = opt.with_recognition
    opt.with_recognition = False

    if with_recog:
        words1, ends1 = load_word(osp.join(opt.anno_root, v1 + ".txt"))
        words2, ends2 = load_word(osp.join(opt.anno_root, v2 + ".txt"))
        words3, ends3 = load_word(osp.join(opt.anno_root, v3 + ".txt"))
        words4, ends4 = load_word(osp.join(opt.anno_root, v4 + ".txt"))

    # make sure the two audios are of the same length and then mix them
    audio_length = min(len(audio1), len(audio2), len(audio3), len(audio4))
    minimum_length = int(opt.audio_sampling_rate * opt.audio_length)

    audio_second1 = len(audio1) / opt.audio_sampling_rate
    audio_second2 = len(audio2) / opt.audio_sampling_rate
    audio_second3 = len(audio3) / opt.audio_sampling_rate
    audio_second4 = len(audio4) / opt.audio_sampling_rate
    target_second = audio_length / opt.audio_sampling_rate if audio_length >= minimum_length else opt.audio_length  # second
    multiple = 1

    if audio_length < minimum_length:
        multiple = (minimum_length + audio_length - 1) // audio_length
        audio1 = supplement_audio(audio1, multiple, minimum_length)
        audio2 = supplement_audio(audio2, multiple, minimum_length)
        audio3 = supplement_audio(audio3, multiple, minimum_length)
        audio4 = supplement_audio(audio4, multiple, minimum_length)
        audio_length = minimum_length

    if with_recog:
        trgt1 = supplement_word(words1, ends1, multiple, audio_second1, target_second)
        trgt2 = supplement_word(words2, ends2, multiple, audio_second2, target_second)
        trgt3 = supplement_word(words3, ends3, multiple, audio_second3, target_second)
        trgt4 = supplement_word(words4, ends4, multiple, audio_second4, target_second)
        if opt.recg_num_classes == 40:
            trgt1 = [charToIx40[c] for c in trgt1] + charToIx40["<EOS>"]
            trgt2 = [charToIx40[c] for c in trgt2] + charToIx40["<EOS>"]
            trgt3 = [charToIx40[c] for c in trgt3] + charToIx40["<EOS>"]
            trgt4 = [charToIx40[c] for c in trgt4] + charToIx40["<EOS>"]
        else:  # recg_num_classes==32
            trgt1 = [charToIx32[c] if c in charToIx32 else charToIx32['<unk>'] for c in trgt1]
            trgt2 = [charToIx32[c] if c in charToIx32 else charToIx32['<unk>'] for c in trgt2]
            trgt3 = [charToIx32[c] if c in charToIx32 else charToIx32['<unk>'] for c in trgt3]
            trgt4 = [charToIx32[c] if c in charToIx32 else charToIx32['<unk>'] for c in trgt4]
        trgtLen1 = len(trgt1)
        trgtLen2 = len(trgt2)
        trgtLen3 = len(trgt3)
        trgtLen4 = len(trgt4)

    video_length = min(len(mouthroi_1), len(mouthroi_2), len(mouthroi_3), len(mouthroi_4))
    if video_length < opt.num_frames:
        multiple = (opt.num_frames + video_length - 1) // video_length
        mouthroi_1 = supplement_mouth(mouthroi_1, multiple, opt.num_frames)
        mouthroi_2 = supplement_mouth(mouthroi_2, multiple, opt.num_frames)
        mouthroi_3 = supplement_mouth(mouthroi_3, multiple, opt.num_frames)
        mouthroi_4 = supplement_mouth(mouthroi_4, multiple, opt.num_frames)

    audio1 = clip_audio(audio1[:audio_length])
    audio2 = clip_audio(audio2[:audio_length])
    audio3 = clip_audio(audio3[:audio_length])
    audio4 = clip_audio(audio4[:audio_length])
    audio_mix = (audio1 + audio2 + audio3 + audio4) / 4.0

    if opt.reliable_face:
        best_score_1 = 0
        best_score_2 = 0
        best_score_3 = 0
        best_score_4 = 0
        for i in range(10):
            frame_1 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v1 + '.mp4'))))
            frame_2 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v2 + '.mp4'))))
            frame_3 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v3 + '.mp4'))))
            frame_4 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v4 + '.mp4'))))
            try:
                boxes, scores = mtcnn.detect(frame_1)
                if scores[0] > best_score_1:
                    best_frame_1 = frame_1
            except:
                pass
            try:
                boxes, scores = mtcnn.detect(frame_2)
                if scores[0] > best_score_2:
                    best_frame_2 = frame_2
            except:
                pass
            try:
                boxes, scores = mtcnn.detect(frame_3)
                if scores[0] > best_score_3:
                    best_frame_3 = frame_3
            except:
                pass
            try:
                boxes, scores = mtcnn.detect(frame_4)
                if scores[0] > best_score_4:
                    best_frame_4 = frame_4
            except:
                pass
        frames_1 = vision_transform(best_frame_1).squeeze().unsqueeze(0)
        frames_2 = vision_transform(best_frame_2).squeeze().unsqueeze(0)
        frames_3 = vision_transform(best_frame_3).squeeze().unsqueeze(0)
        frames_4 = vision_transform(best_frame_4).squeeze().unsqueeze(0)
    else:
        frame_1_list = []
        frame_2_list = []
        frame_3_list = []
        frame_4_list = []
        for i in range(opt.number_of_identity_frames):
            frame_1 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v1 + '.mp4'))))
            frame_2 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v2 + '.mp4'))))
            frame_3 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v3 + '.mp4'))))
            frame_4 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v4 + '.mp4'))))
            frame_1 = vision_transform(frame_1)
            frame_2 = vision_transform(frame_2)
            frame_3 = vision_transform(frame_3)
            frame_4 = vision_transform(frame_4)
            frame_1_list.append(frame_1)
            frame_2_list.append(frame_2)
            frame_3_list.append(frame_3)
            frame_4_list.append(frame_4)
        frames_1 = torch.stack(frame_1_list).squeeze().unsqueeze(0)
        frames_2 = torch.stack(frame_2_list).squeeze().unsqueeze(0)
        frames_3 = torch.stack(frame_3_list).squeeze().unsqueeze(0)
        frames_4 = torch.stack(frame_4_list).squeeze().unsqueeze(0)

    # perform separation over the whole audio using a sliding window approach
    overlap_count = np.zeros((audio_length))
    sep_audio1 = np.zeros((audio_length))
    sep_audio2 = np.zeros((audio_length))
    sep_audio3 = np.zeros((audio_length))
    sep_audio4 = np.zeros((audio_length))
    sliding_window_start = 0
    data = {}
    avged_sep_audio1 = np.zeros((audio_length))
    avged_sep_audio2 = np.zeros((audio_length))
    avged_sep_audio3 = np.zeros((audio_length))
    avged_sep_audio4 = np.zeros((audio_length))

    samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
    while sliding_window_start + samples_per_window < audio_length:
        sliding_window_end = sliding_window_start + samples_per_window

        # get audio spectrogram
        segment1_audio = audio1[sliding_window_start:sliding_window_end]
        segment2_audio = audio2[sliding_window_start:sliding_window_end]
        segment3_audio = audio3[sliding_window_start:sliding_window_end]
        segment4_audio = audio4[sliding_window_start:sliding_window_end]

        if opt.audio_normalization:
            normalizer1, segment1_audio = audio_normalize(segment1_audio)
            normalizer2, segment2_audio = audio_normalize(segment2_audio)
            normalizer3, segment3_audio = audio_normalize(segment3_audio)
            normalizer4, segment4_audio = audio_normalize(segment4_audio)
        else:
            normalizer1 = 1
            normalizer2 = 1
            normalizer3 = 1
            normalizer4 = 1

        audio_segment = (segment1_audio + segment2_audio + segment3_audio + segment4_audio) / 4
        audio_mix_spec = generate_spectrogram_complex(audio_segment, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_1 = generate_spectrogram_complex(segment1_audio, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_2 = generate_spectrogram_complex(segment2_audio, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_3 = generate_spectrogram_complex(segment3_audio, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_4 = generate_spectrogram_complex(segment4_audio, opt.window_size, opt.hop_size, opt.n_fft)

        # get mouthroi
        frame_index_start = int(sliding_window_start / opt.audio_sampling_rate * 25)
        frame_index_end = frame_index_start + opt.num_frames
        if frame_index_end > len(mouthroi_1) or frame_index_end > len(mouthroi_2) \
                or frame_index_end > len(mouthroi_3) or frame_index_end > len(mouthroi_4):
            gap = frame_index_end - min(len(mouthroi_1), len(mouthroi_2), len(mouthroi_3), len(mouthroi_4))
            frame_index_start -= gap
            frame_index_end -= gap
        segment1_mouthroi = mouthroi_1[frame_index_start:frame_index_end, :, :]
        segment2_mouthroi = mouthroi_2[frame_index_start:frame_index_end, :, :]
        segment3_mouthroi = mouthroi_3[frame_index_start:frame_index_end, :, :]
        segment4_mouthroi = mouthroi_4[frame_index_start:frame_index_end, :, :]

        # transform mouthrois
        segment1_mouthroi = lipreading_preprocessing_func(segment1_mouthroi)
        segment2_mouthroi = lipreading_preprocessing_func(segment2_mouthroi)
        segment3_mouthroi = lipreading_preprocessing_func(segment3_mouthroi)
        segment4_mouthroi = lipreading_preprocessing_func(segment4_mouthroi)

        # separate 1 and 2
        data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
        data['mouthroi1'] = torch.FloatTensor(segment1_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['mouthroi2'] = torch.FloatTensor(segment2_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['audio_spec1'] = torch.FloatTensor(audio_spec_1).cuda().unsqueeze(0)
        data['audio_spec2'] = torch.FloatTensor(audio_spec_2).cuda().unsqueeze(0)
        data['frame1'] = frames_1.cuda()
        data['frame2'] = frames_2.cuda()

        try:
            outputs = model.forward(data)
        except Exception as e:
            traceback.print_exc()
            exit(-1)
        reconstructed_signal_1, reconstructed_signal_2 = get_separated_audio(outputs, window, opt)
        reconstructed_signal_1 = reconstructed_signal_1 * normalizer1
        reconstructed_signal_2 = reconstructed_signal_2 * normalizer2
        sep_audio1[sliding_window_start:sliding_window_end] = \
            sep_audio1[sliding_window_start:sliding_window_end] + reconstructed_signal_1
        sep_audio2[sliding_window_start:sliding_window_end] = \
            sep_audio2[sliding_window_start:sliding_window_end] + reconstructed_signal_2
        # update overlap count
        overlap_count[sliding_window_start:sliding_window_end] = overlap_count[
                                                                 sliding_window_start:sliding_window_end] + 1

        # separate 3 and 4
        data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
        data['mouthroi1'] = torch.FloatTensor(segment3_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['mouthroi2'] = torch.FloatTensor(segment4_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['audio_spec1'] = torch.FloatTensor(audio_spec_3).cuda().unsqueeze(0)
        data['audio_spec2'] = torch.FloatTensor(audio_spec_4).cuda().unsqueeze(0)
        data['frame1'] = frames_3.cuda()
        data['frame2'] = frames_4.cuda()

        try:
            outputs = model.forward(data)
        except Exception as e:
            traceback.print_exc()
            exit(-1)
        reconstructed_signal_3, reconstructed_signal_4 = get_separated_audio(outputs, window, opt)
        reconstructed_signal_3 = reconstructed_signal_3 * normalizer3
        reconstructed_signal_4 = reconstructed_signal_4 * normalizer4
        sep_audio3[sliding_window_start:sliding_window_end] = \
            sep_audio3[sliding_window_start:sliding_window_end] + reconstructed_signal_3
        sep_audio4[sliding_window_start:sliding_window_end] = \
            sep_audio4[sliding_window_start:sliding_window_end] + reconstructed_signal_4

        sliding_window_start = sliding_window_start + int(opt.hop_second * opt.audio_sampling_rate)

    # deal with the last segment
    # get audio spectrogram
    segment1_audio = audio1[-samples_per_window:]
    segment2_audio = audio2[-samples_per_window:]
    segment3_audio = audio3[-samples_per_window:]
    segment4_audio = audio4[-samples_per_window:]

    if opt.audio_normalization:
        normalizer1, segment1_audio = audio_normalize(segment1_audio)
        normalizer2, segment2_audio = audio_normalize(segment2_audio)
        normalizer3, segment3_audio = audio_normalize(segment3_audio)
        normalizer4, segment4_audio = audio_normalize(segment4_audio)
    else:
        normalizer1 = 1
        normalizer2 = 1
        normalizer3 = 1
        normalizer4 = 1

    audio_segment = (segment1_audio + segment2_audio + segment3_audio + segment4_audio) / 4
    audio_mix_spec = generate_spectrogram_complex(audio_segment, opt.window_size, opt.hop_size, opt.n_fft)
    # get mouthroi
    frame_index_start = int((len(audio1) - samples_per_window) / opt.audio_sampling_rate * 25)
    frame_index_end = frame_index_start + opt.num_frames
    if frame_index_end > len(mouthroi_1) or frame_index_end > len(mouthroi_2) \
            or frame_index_end > len(mouthroi_3) or frame_index_end > len(mouthroi_4):
        gap = frame_index_end - min(len(mouthroi_1), len(mouthroi_2), len(mouthroi_3), len(mouthroi_4))
        frame_index_start -= gap
        frame_index_end -= gap
    segment1_mouthroi = mouthroi_1[frame_index_start:frame_index_end, :, :]
    segment2_mouthroi = mouthroi_2[frame_index_start:frame_index_end, :, :]
    segment3_mouthroi = mouthroi_3[frame_index_start:frame_index_end, :, :]
    segment4_mouthroi = mouthroi_4[frame_index_start:frame_index_end, :, :]
    # transform mouthrois
    segment1_mouthroi = lipreading_preprocessing_func(segment1_mouthroi)
    segment2_mouthroi = lipreading_preprocessing_func(segment2_mouthroi)
    segment3_mouthroi = lipreading_preprocessing_func(segment3_mouthroi)
    segment4_mouthroi = lipreading_preprocessing_func(segment4_mouthroi)
    audio_spec_1 = generate_spectrogram_complex(segment1_audio, opt.window_size, opt.hop_size, opt.n_fft)
    audio_spec_2 = generate_spectrogram_complex(segment2_audio, opt.window_size, opt.hop_size, opt.n_fft)
    audio_spec_3 = generate_spectrogram_complex(segment3_audio, opt.window_size, opt.hop_size, opt.n_fft)
    audio_spec_4 = generate_spectrogram_complex(segment4_audio, opt.window_size, opt.hop_size, opt.n_fft)

    # separate 1 and 2
    data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
    data['mouthroi1'] = torch.FloatTensor(segment1_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['mouthroi2'] = torch.FloatTensor(segment2_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['audio_spec1'] = torch.FloatTensor(audio_spec_1).cuda().unsqueeze(0)
    data['audio_spec2'] = torch.FloatTensor(audio_spec_2).cuda().unsqueeze(0)
    data['frame1'] = frames_1.cuda()
    data['frame2'] = frames_2.cuda()

    try:
        outputs = model.forward(data)
    except Exception as e:
        traceback.print_exc()
        exit(-1)
    reconstructed_signal_1, reconstructed_signal_2 = get_separated_audio(outputs, window, opt)
    reconstructed_signal_1 = reconstructed_signal_1 * normalizer1
    reconstructed_signal_2 = reconstructed_signal_2 * normalizer2
    sep_audio1[-samples_per_window:] = sep_audio1[-samples_per_window:] + reconstructed_signal_1
    sep_audio2[-samples_per_window:] = sep_audio2[-samples_per_window:] + reconstructed_signal_2
    # update overlap count
    overlap_count[-samples_per_window:] = overlap_count[-samples_per_window:] + 1
    # divide the aggregated predicted audio by the overlap count
    avged_sep_audio1 = avged_sep_audio1 + clip_audio(np.divide(sep_audio1, overlap_count))
    avged_sep_audio2 = avged_sep_audio2 + clip_audio(np.divide(sep_audio2, overlap_count))

    # separate 3 and 4
    data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
    data['mouthroi1'] = torch.FloatTensor(segment3_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['mouthroi2'] = torch.FloatTensor(segment4_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['audio_spec1'] = torch.FloatTensor(audio_spec_3).cuda().unsqueeze(0)
    data['audio_spec2'] = torch.FloatTensor(audio_spec_4).cuda().unsqueeze(0)
    data['frame1'] = frames_3.cuda()
    data['frame2'] = frames_4.cuda()

    try:
        outputs = model.forward(data)
    except Exception as e:
        traceback.print_exc()
        exit(-1)
    reconstructed_signal_3, reconstructed_signal_4 = get_separated_audio(outputs, window, opt)
    reconstructed_signal_3 = reconstructed_signal_3 * normalizer3
    reconstructed_signal_4 = reconstructed_signal_4 * normalizer4
    sep_audio3[-samples_per_window:] = sep_audio3[-samples_per_window:] + reconstructed_signal_3
    sep_audio4[-samples_per_window:] = sep_audio4[-samples_per_window:] + reconstructed_signal_4
    # divide the aggregated predicted audio by the overlap count
    avged_sep_audio3 = avged_sep_audio3 + clip_audio(np.divide(sep_audio3, overlap_count))
    avged_sep_audio4 = avged_sep_audio4 + clip_audio(np.divide(sep_audio4, overlap_count))

    if opt.save_output == "true":
        output_dir = osp.join(opt.output_dir_root, v1 + '@' + v2 + "@" + v3 + "@" + v4)
        if not osp.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        sf.write(osp.join(output_dir, v1 + '_separated.wav'), avged_sep_audio1, opt.audio_sampling_rate)
        sf.write(osp.join(output_dir, v2 + '_separated.wav'), avged_sep_audio2, opt.audio_sampling_rate)
        sf.write(osp.join(output_dir, v3 + '_separated.wav'), avged_sep_audio3, opt.audio_sampling_rate)
        sf.write(osp.join(output_dir, v4 + '_separated.wav'), avged_sep_audio4, opt.audio_sampling_rate)

    sdr, sir, sar = getSeparationMetrics_4mix(avged_sep_audio1, avged_sep_audio2, avged_sep_audio3, avged_sep_audio4,
                                              audio1, audio2, audio3, audio4)
    sdr_mix, sir_mix, sar_mix = getSeparationMetrics_4mix(audio_mix, audio_mix, audio_mix, audio_mix,
                                              audio1, audio2, audio3, audio4)
    pesq_score1 = pesq(avged_sep_audio1, audio1, opt.audio_sampling_rate)
    pesq_score2 = pesq(avged_sep_audio2, audio2, opt.audio_sampling_rate)
    pesq_score3 = pesq(avged_sep_audio3, audio3, opt.audio_sampling_rate)
    pesq_score4 = pesq(avged_sep_audio4, audio4, opt.audio_sampling_rate)
    pesq_score = (pesq_score1 + pesq_score2 + pesq_score3 + pesq_score4) / 4
    # STOI
    stoi_score1 = stoi(audio1, avged_sep_audio1, opt.audio_sampling_rate, extended=False)
    stoi_score2 = stoi(audio2, avged_sep_audio2, opt.audio_sampling_rate, extended=False)
    stoi_score3 = stoi(audio3, avged_sep_audio3, opt.audio_sampling_rate, extended=False)
    stoi_score4 = stoi(audio4, avged_sep_audio4, opt.audio_sampling_rate, extended=False)
    stoi_score = (stoi_score1 + stoi_score2 + stoi_score3 + stoi_score4) / 4

    if with_recog:
        opt.with_recognition = True
        net_feature_extractor, net_recognizer = model.net_feature_extractor, model.net_recognizer
        eos = charToIx40["<EOS>"] if opt.recg_num_classes == 40 else charToIx32["</s>"]
        cer1 = forward_asr(eos, opt.blank_label, net_feature_extractor, net_recognizer, avged_sep_audio1.unsqueeze(0), trgt1, trgtLen1)
        cer2 = forward_asr(eos, opt.blank_label, net_feature_extractor, net_recognizer, avged_sep_audio2.unsqueeze(0), trgt2, trgtLen2)
        cer3 = forward_asr(eos, opt.blank_label, net_feature_extractor, net_recognizer, avged_sep_audio3.unsqueeze(0), trgt3, trgtLen3)
        cer4 = forward_asr(eos, opt.blank_label, net_feature_extractor, net_recognizer, avged_sep_audio4.unsqueeze(0), trgt4, trgtLen4)
        return sdr, sir, sar, pesq_score, stoi_score, \
               sdr - sdr_mix, sir - sir_mix, sar - sar_mix, \
               (cer1 + cer2 + cer3 + cer4) / 4

    return sdr, sir, sar, pesq_score, stoi_score, \
           sdr - sdr_mix, sir - sir_mix, sar - sar_mix


def process_5_mixture(model, opt, v1, v2, v3, v4, v5, mtcnn, lipreading_preprocessing_func, vision_transform, window):
    # load data
    mouthroi_path1 = osp.join(opt.mouth_root, v1 + "." + opt.mouthroi_format)
    mouthroi_path2 = osp.join(opt.mouth_root, v2 + "." + opt.mouthroi_format)
    mouthroi_path3 = osp.join(opt.mouth_root, v3 + "." + opt.mouthroi_format)
    mouthroi_path4 = osp.join(opt.mouth_root, v4 + "." + opt.mouthroi_format)
    mouthroi_path5 = osp.join(opt.mouth_root, v5 + "." + opt.mouthroi_format)
    mouthroi_1 = load_mouthroi(mouthroi_path1, opt.mouthroi_format)
    mouthroi_2 = load_mouthroi(mouthroi_path2, opt.mouthroi_format)
    mouthroi_3 = load_mouthroi(mouthroi_path3, opt.mouthroi_format)
    mouthroi_4 = load_mouthroi(mouthroi_path4, opt.mouthroi_format)
    mouthroi_5 = load_mouthroi(mouthroi_path5, opt.mouthroi_format)

    audio_path1 = osp.join(opt.audio_root, v1 + '.wav')
    audio_path2 = osp.join(opt.audio_root, v2 + '.wav')
    audio_path3 = osp.join(opt.audio_root, v3 + '.wav')
    audio_path4 = osp.join(opt.audio_root, v4 + '.wav')
    audio_path5 = osp.join(opt.audio_root, v5 + '.wav')
    audio1 = load_audio(audio_path1)
    audio2 = load_audio(audio_path2)
    audio3 = load_audio(audio_path3)
    audio4 = load_audio(audio_path4)
    audio5 = load_audio(audio_path5)
    audio1 = audio1 / 32768
    audio2 = audio2 / 32768
    audio3 = audio3 / 32768
    audio4 = audio4 / 32768
    audio5 = audio5 / 32768

    with_recog = opt.with_recognition
    opt.with_recognition = False

    if with_recog:
        words1, ends1 = load_word(osp.join(opt.anno_root, v1 + ".txt"))
        words2, ends2 = load_word(osp.join(opt.anno_root, v2 + ".txt"))
        words3, ends3 = load_word(osp.join(opt.anno_root, v3 + ".txt"))
        words4, ends4 = load_word(osp.join(opt.anno_root, v4 + ".txt"))
        words5, ends5 = load_word(osp.join(opt.anno_root, v5 + ".txt"))

    # make sure the two audios are of the same length and then mix them
    audio_length = min(len(audio1), len(audio2), len(audio3), len(audio4), len(audio5))
    minimum_length = int(opt.audio_sampling_rate * opt.audio_length)

    audio_second1 = len(audio1) / opt.audio_sampling_rate
    audio_second2 = len(audio2) / opt.audio_sampling_rate
    audio_second3 = len(audio3) / opt.audio_sampling_rate
    audio_second4 = len(audio4) / opt.audio_sampling_rate
    audio_second5 = len(audio5) / opt.audio_sampling_rate
    target_second = audio_length / opt.audio_sampling_rate if audio_length >= minimum_length else opt.audio_length  # second
    multiple = 1

    if audio_length < minimum_length:
        multiple = (minimum_length + audio_length - 1) // audio_length
        audio1 = supplement_audio(audio1, multiple, minimum_length)
        audio2 = supplement_audio(audio2, multiple, minimum_length)
        audio3 = supplement_audio(audio3, multiple, minimum_length)
        audio4 = supplement_audio(audio4, multiple, minimum_length)
        audio5 = supplement_audio(audio5, multiple, minimum_length)
        audio_length = minimum_length

    if with_recog:
        trgt1 = supplement_word(words1, ends1, multiple, audio_second1, target_second)
        trgt2 = supplement_word(words2, ends2, multiple, audio_second2, target_second)
        trgt3 = supplement_word(words3, ends3, multiple, audio_second3, target_second)
        trgt4 = supplement_word(words4, ends4, multiple, audio_second4, target_second)
        trgt5 = supplement_word(words5, ends5, multiple, audio_second5, target_second)
        if opt.recg_num_classes == 40:
            trgt1 = [charToIx40[c] for c in trgt1] + charToIx40["<EOS>"]
            trgt2 = [charToIx40[c] for c in trgt2] + charToIx40["<EOS>"]
            trgt3 = [charToIx40[c] for c in trgt3] + charToIx40["<EOS>"]
            trgt4 = [charToIx40[c] for c in trgt4] + charToIx40["<EOS>"]
            trgt5 = [charToIx40[c] for c in trgt5] + charToIx40["<EOS>"]
        else:  # recg_num_classes==32
            trgt1 = [charToIx32[c] if c in charToIx32 else charToIx32['<unk>'] for c in trgt1]
            trgt2 = [charToIx32[c] if c in charToIx32 else charToIx32['<unk>'] for c in trgt2]
            trgt3 = [charToIx32[c] if c in charToIx32 else charToIx32['<unk>'] for c in trgt3]
            trgt4 = [charToIx32[c] if c in charToIx32 else charToIx32['<unk>'] for c in trgt4]
            trgt5 = [charToIx32[c] if c in charToIx32 else charToIx32['<unk>'] for c in trgt5]
        trgtLen1 = len(trgt1)
        trgtLen2 = len(trgt2)
        trgtLen3 = len(trgt3)
        trgtLen4 = len(trgt4)
        trgtLen5 = len(trgt5)

    video_length = min(len(mouthroi_1), len(mouthroi_2), len(mouthroi_3), len(mouthroi_4), len(mouthroi_5))
    if video_length < opt.num_frames:
        multiple = (opt.num_frames + video_length - 1) // video_length
        mouthroi_1 = supplement_mouth(mouthroi_1, multiple, opt.num_frames)
        mouthroi_2 = supplement_mouth(mouthroi_2, multiple, opt.num_frames)
        mouthroi_3 = supplement_mouth(mouthroi_3, multiple, opt.num_frames)
        mouthroi_4 = supplement_mouth(mouthroi_4, multiple, opt.num_frames)
        mouthroi_5 = supplement_mouth(mouthroi_5, multiple, opt.num_frames)

    audio1 = clip_audio(audio1[:audio_length])
    audio2 = clip_audio(audio2[:audio_length])
    audio3 = clip_audio(audio3[:audio_length])
    audio4 = clip_audio(audio4[:audio_length])
    audio5 = clip_audio(audio5[:audio_length])
    audio_mix = (audio1 + audio2 + audio3 + audio4 + audio5) / 5.0

    if opt.reliable_face:
        best_score_1 = 0
        best_score_2 = 0
        best_score_3 = 0
        best_score_4 = 0
        best_score_5 = 0
        for i in range(10):
            frame_1 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v1 + '.mp4'))))
            frame_2 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v2 + '.mp4'))))
            frame_3 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v3 + '.mp4'))))
            frame_4 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v4 + '.mp4'))))
            frame_5 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v5 + '.mp4'))))
            try:
                boxes, scores = mtcnn.detect(frame_1)
                if scores[0] > best_score_1:
                    best_frame_1 = frame_1
            except:
                pass
            try:
                boxes, scores = mtcnn.detect(frame_2)
                if scores[0] > best_score_2:
                    best_frame_2 = frame_2
            except:
                pass
            try:
                boxes, scores = mtcnn.detect(frame_3)
                if scores[0] > best_score_3:
                    best_frame_3 = frame_3
            except:
                pass
            try:
                boxes, scores = mtcnn.detect(frame_4)
                if scores[0] > best_score_4:
                    best_frame_4 = frame_4
            except:
                pass
            try:
                boxes, scores = mtcnn.detect(frame_5)
                if scores[0] > best_score_5:
                    best_frame_5 = frame_5
            except:
                pass
        frames_1 = vision_transform(best_frame_1).squeeze().unsqueeze(0)
        frames_2 = vision_transform(best_frame_2).squeeze().unsqueeze(0)
        frames_3 = vision_transform(best_frame_3).squeeze().unsqueeze(0)
        frames_4 = vision_transform(best_frame_4).squeeze().unsqueeze(0)
        frames_5 = vision_transform(best_frame_5).squeeze().unsqueeze(0)
    else:
        frame_1_list = []
        frame_2_list = []
        frame_3_list = []
        frame_4_list = []
        frame_5_list = []
        for i in range(opt.number_of_identity_frames):
            frame_1 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v1 + '.mp4'))))
            frame_2 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v2 + '.mp4'))))
            frame_3 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v3 + '.mp4'))))
            frame_4 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v4 + '.mp4'))))
            frame_5 = load_frame(io.BytesIO(client.get(osp.join(opt.mp4_root, v5 + '.mp4'))))
            frame_1 = vision_transform(frame_1)
            frame_2 = vision_transform(frame_2)
            frame_3 = vision_transform(frame_3)
            frame_4 = vision_transform(frame_4)
            frame_5 = vision_transform(frame_5)
            frame_1_list.append(frame_1)
            frame_2_list.append(frame_2)
            frame_3_list.append(frame_3)
            frame_4_list.append(frame_4)
            frame_5_list.append(frame_5)
        frames_1 = torch.stack(frame_1_list).squeeze().unsqueeze(0)
        frames_2 = torch.stack(frame_2_list).squeeze().unsqueeze(0)
        frames_3 = torch.stack(frame_3_list).squeeze().unsqueeze(0)
        frames_4 = torch.stack(frame_4_list).squeeze().unsqueeze(0)
        frames_5 = torch.stack(frame_5_list).squeeze().unsqueeze(0)

    # perform separation over the whole audio using a sliding window approach
    overlap_count = np.zeros((audio_length))
    sep_audio1 = np.zeros((audio_length))
    sep_audio2 = np.zeros((audio_length))
    sep_audio3 = np.zeros((audio_length))
    sep_audio4 = np.zeros((audio_length))
    sep_audio5 = np.zeros((audio_length))
    sliding_window_start = 0
    data = {}
    avged_sep_audio1 = np.zeros((audio_length))
    avged_sep_audio2 = np.zeros((audio_length))
    avged_sep_audio3 = np.zeros((audio_length))
    avged_sep_audio4 = np.zeros((audio_length))
    avged_sep_audio5 = np.zeros((audio_length))

    samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
    while sliding_window_start + samples_per_window < audio_length:
        sliding_window_end = sliding_window_start + samples_per_window

        # get audio spectrogram
        segment1_audio = audio1[sliding_window_start:sliding_window_end]
        segment2_audio = audio2[sliding_window_start:sliding_window_end]
        segment3_audio = audio3[sliding_window_start:sliding_window_end]
        segment4_audio = audio4[sliding_window_start:sliding_window_end]
        segment5_audio = audio5[sliding_window_start:sliding_window_end]

        if opt.audio_normalization:
            normalizer1, segment1_audio = audio_normalize(segment1_audio)
            normalizer2, segment2_audio = audio_normalize(segment2_audio)
            normalizer3, segment3_audio = audio_normalize(segment3_audio)
            normalizer4, segment4_audio = audio_normalize(segment4_audio)
            normalizer5, segment5_audio = audio_normalize(segment5_audio)
        else:
            normalizer1 = 1
            normalizer2 = 1
            normalizer3 = 1
            normalizer4 = 1
            normalizer5 = 1

        audio_segment = (segment1_audio + segment2_audio + segment3_audio + segment4_audio + segment5_audio) / 5
        audio_mix_spec = generate_spectrogram_complex(audio_segment, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_1 = generate_spectrogram_complex(segment1_audio, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_2 = generate_spectrogram_complex(segment2_audio, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_3 = generate_spectrogram_complex(segment3_audio, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_4 = generate_spectrogram_complex(segment4_audio, opt.window_size, opt.hop_size, opt.n_fft)
        audio_spec_5 = generate_spectrogram_complex(segment5_audio, opt.window_size, opt.hop_size, opt.n_fft)

        # get mouthroi
        frame_index_start = int(sliding_window_start / opt.audio_sampling_rate * 25)
        frame_index_end = frame_index_start + opt.num_frames
        if frame_index_end > len(mouthroi_1) or frame_index_end > len(mouthroi_2) or frame_index_end > len(mouthroi_3)\
                or frame_index_end > len(mouthroi_4) or frame_index_end > len(mouthroi_5):
            gap = frame_index_end - min(len(mouthroi_1), len(mouthroi_2), len(mouthroi_3), len(mouthroi_4), len(mouthroi_5))
            frame_index_start -= gap
            frame_index_end -= gap
        segment1_mouthroi = mouthroi_1[frame_index_start:frame_index_end, :, :]
        segment2_mouthroi = mouthroi_2[frame_index_start:frame_index_end, :, :]
        segment3_mouthroi = mouthroi_3[frame_index_start:frame_index_end, :, :]
        segment4_mouthroi = mouthroi_4[frame_index_start:frame_index_end, :, :]
        segment5_mouthroi = mouthroi_5[frame_index_start:frame_index_end, :, :]

        # transform mouthrois
        segment1_mouthroi = lipreading_preprocessing_func(segment1_mouthroi)
        segment2_mouthroi = lipreading_preprocessing_func(segment2_mouthroi)
        segment3_mouthroi = lipreading_preprocessing_func(segment3_mouthroi)
        segment4_mouthroi = lipreading_preprocessing_func(segment4_mouthroi)
        segment5_mouthroi = lipreading_preprocessing_func(segment5_mouthroi)

        # separate 1 and 2
        data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
        data['mouthroi1'] = torch.FloatTensor(segment1_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['mouthroi2'] = torch.FloatTensor(segment2_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['audio_spec1'] = torch.FloatTensor(audio_spec_1).cuda().unsqueeze(0)
        data['audio_spec2'] = torch.FloatTensor(audio_spec_2).cuda().unsqueeze(0)
        data['frame1'] = frames_1.cuda()
        data['frame2'] = frames_2.cuda()
        try:
            outputs = model.forward(data)
        except Exception as e:
            traceback.print_exc()
            exit(-1)
        reconstructed_signal_1, reconstructed_signal_2 = get_separated_audio(outputs, window, opt)
        reconstructed_signal_1 = reconstructed_signal_1 * normalizer1
        reconstructed_signal_2 = reconstructed_signal_2 * normalizer2
        sep_audio1[sliding_window_start:sliding_window_end] = \
            sep_audio1[sliding_window_start:sliding_window_end] + reconstructed_signal_1
        sep_audio2[sliding_window_start:sliding_window_end] = \
            sep_audio2[sliding_window_start:sliding_window_end] + reconstructed_signal_2
        # update overlap count
        overlap_count[sliding_window_start:sliding_window_end] = overlap_count[
                                                                 sliding_window_start:sliding_window_end] + 1

        # separate 3 and 4
        data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
        data['mouthroi1'] = torch.FloatTensor(segment3_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['mouthroi2'] = torch.FloatTensor(segment4_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['audio_spec1'] = torch.FloatTensor(audio_spec_3).cuda().unsqueeze(0)
        data['audio_spec2'] = torch.FloatTensor(audio_spec_4).cuda().unsqueeze(0)
        data['frame1'] = frames_3.cuda()
        data['frame2'] = frames_4.cuda()
        try:
            outputs = model.forward(data)
        except Exception as e:
            traceback.print_exc()
            exit(-1)
        reconstructed_signal_3, reconstructed_signal_4 = get_separated_audio(outputs, window, opt)
        reconstructed_signal_3 = reconstructed_signal_3 * normalizer3
        reconstructed_signal_4 = reconstructed_signal_4 * normalizer4
        sep_audio3[sliding_window_start:sliding_window_end] = \
            sep_audio3[sliding_window_start:sliding_window_end] + reconstructed_signal_3
        sep_audio4[sliding_window_start:sliding_window_end] = \
            sep_audio4[sliding_window_start:sliding_window_end] + reconstructed_signal_4

        # separate 5 and 4
        data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
        data['mouthroi1'] = torch.FloatTensor(segment5_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['mouthroi2'] = torch.FloatTensor(segment4_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
        data['audio_spec1'] = torch.FloatTensor(audio_spec_5).cuda().unsqueeze(0)
        data['audio_spec2'] = torch.FloatTensor(audio_spec_4).cuda().unsqueeze(0)
        data['frame1'] = frames_5.cuda()
        data['frame2'] = frames_4.cuda()
        try:
            outputs = model.forward(data)
        except Exception as e:
            traceback.print_exc()
            exit(-1)
        reconstructed_signal_5, _ = get_separated_audio(outputs, window, opt)
        reconstructed_signal_5 = reconstructed_signal_5 * normalizer5
        sep_audio5[sliding_window_start:sliding_window_end] = \
            sep_audio5[sliding_window_start:sliding_window_end] + reconstructed_signal_5

        sliding_window_start = sliding_window_start + int(opt.hop_second * opt.audio_sampling_rate)

    # deal with the last segment
    # get audio spectrogram
    segment1_audio = audio1[-samples_per_window:]
    segment2_audio = audio2[-samples_per_window:]
    segment3_audio = audio3[-samples_per_window:]
    segment4_audio = audio4[-samples_per_window:]
    segment5_audio = audio5[-samples_per_window:]

    if opt.audio_normalization:
        normalizer1, segment1_audio = audio_normalize(segment1_audio)
        normalizer2, segment2_audio = audio_normalize(segment2_audio)
        normalizer3, segment3_audio = audio_normalize(segment3_audio)
        normalizer4, segment4_audio = audio_normalize(segment4_audio)
        normalizer5, segment5_audio = audio_normalize(segment5_audio)
    else:
        normalizer1 = 1
        normalizer2 = 1
        normalizer3 = 1
        normalizer4 = 1
        normalizer5 = 1

    audio_segment = (segment1_audio + segment2_audio + segment3_audio + segment4_audio + segment5_audio) / 5
    audio_mix_spec = generate_spectrogram_complex(audio_segment, opt.window_size, opt.hop_size, opt.n_fft)
    # get mouthroi
    frame_index_start = int((len(audio1) - samples_per_window) / opt.audio_sampling_rate * 25)
    frame_index_end = frame_index_start + opt.num_frames
    if frame_index_end > len(mouthroi_1) or frame_index_end > len(mouthroi_2) or frame_index_end > len(mouthroi_3) \
            or frame_index_end > len(mouthroi_4) or frame_index_end > len(mouthroi_5):
        gap = frame_index_end - min(len(mouthroi_1), len(mouthroi_2), len(mouthroi_3), len(mouthroi_4), len(mouthroi_5))
        frame_index_start -= gap
        frame_index_end -= gap
    segment1_mouthroi = mouthroi_1[frame_index_start:frame_index_end, :, :]
    segment2_mouthroi = mouthroi_2[frame_index_start:frame_index_end, :, :]
    segment3_mouthroi = mouthroi_3[frame_index_start:frame_index_end, :, :]
    segment4_mouthroi = mouthroi_4[frame_index_start:frame_index_end, :, :]
    segment5_mouthroi = mouthroi_5[frame_index_start:frame_index_end, :, :]
    # transform mouthrois
    segment1_mouthroi = lipreading_preprocessing_func(segment1_mouthroi)
    segment2_mouthroi = lipreading_preprocessing_func(segment2_mouthroi)
    segment3_mouthroi = lipreading_preprocessing_func(segment3_mouthroi)
    segment4_mouthroi = lipreading_preprocessing_func(segment4_mouthroi)
    segment5_mouthroi = lipreading_preprocessing_func(segment5_mouthroi)
    audio_spec_1 = generate_spectrogram_complex(segment1_audio, opt.window_size, opt.hop_size, opt.n_fft)
    audio_spec_2 = generate_spectrogram_complex(segment2_audio, opt.window_size, opt.hop_size, opt.n_fft)
    audio_spec_3 = generate_spectrogram_complex(segment3_audio, opt.window_size, opt.hop_size, opt.n_fft)
    audio_spec_4 = generate_spectrogram_complex(segment4_audio, opt.window_size, opt.hop_size, opt.n_fft)
    audio_spec_5 = generate_spectrogram_complex(segment5_audio, opt.window_size, opt.hop_size, opt.n_fft)

    # separate 1 and 2
    data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
    data['mouthroi1'] = torch.FloatTensor(segment1_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['mouthroi2'] = torch.FloatTensor(segment2_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['audio_spec1'] = torch.FloatTensor(audio_spec_1).cuda().unsqueeze(0)
    data['audio_spec2'] = torch.FloatTensor(audio_spec_2).cuda().unsqueeze(0)
    data['frame1'] = frames_1.cuda()
    data['frame2'] = frames_2.cuda()
    try:
        outputs = model.forward(data)
    except Exception as e:
        traceback.print_exc()
        exit(-1)
    reconstructed_signal_1, reconstructed_signal_2 = get_separated_audio(outputs, window, opt)
    reconstructed_signal_1 = reconstructed_signal_1 * normalizer1
    reconstructed_signal_2 = reconstructed_signal_2 * normalizer2
    sep_audio1[-samples_per_window:] = sep_audio1[-samples_per_window:] + reconstructed_signal_1
    sep_audio2[-samples_per_window:] = sep_audio2[-samples_per_window:] + reconstructed_signal_2
    # update overlap count
    overlap_count[-samples_per_window:] = overlap_count[-samples_per_window:] + 1
    # divide the aggregated predicted audio by the overlap count
    avged_sep_audio1 = avged_sep_audio1 + clip_audio(np.divide(sep_audio1, overlap_count))
    avged_sep_audio2 = avged_sep_audio2 + clip_audio(np.divide(sep_audio2, overlap_count))

    # separate 3 and 4
    data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
    data['mouthroi1'] = torch.FloatTensor(segment3_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['mouthroi2'] = torch.FloatTensor(segment4_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['audio_spec1'] = torch.FloatTensor(audio_spec_3).cuda().unsqueeze(0)
    data['audio_spec2'] = torch.FloatTensor(audio_spec_4).cuda().unsqueeze(0)
    data['frame1'] = frames_3.cuda()
    data['frame2'] = frames_4.cuda()
    try:
        outputs = model.forward(data)
    except Exception as e:
        traceback.print_exc()
        exit(-1)
    reconstructed_signal_3, reconstructed_signal_4 = get_separated_audio(outputs, window, opt)
    reconstructed_signal_3 = reconstructed_signal_3 * normalizer3
    reconstructed_signal_4 = reconstructed_signal_4 * normalizer4
    sep_audio3[-samples_per_window:] = sep_audio3[-samples_per_window:] + reconstructed_signal_3
    sep_audio4[-samples_per_window:] = sep_audio4[-samples_per_window:] + reconstructed_signal_4
    # divide the aggregated predicted audio by the overlap count
    avged_sep_audio3 = avged_sep_audio3 + clip_audio(np.divide(sep_audio3, overlap_count))
    avged_sep_audio4 = avged_sep_audio4 + clip_audio(np.divide(sep_audio4, overlap_count))

    # separate 5 and 4
    data['audio_spec_mix'] = torch.FloatTensor(audio_mix_spec).cuda().unsqueeze(0)
    data['mouthroi1'] = torch.FloatTensor(segment5_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['mouthroi2'] = torch.FloatTensor(segment4_mouthroi).cuda().unsqueeze(0).unsqueeze(0)
    data['audio_spec1'] = torch.FloatTensor(audio_spec_5).cuda().unsqueeze(0)
    data['audio_spec2'] = torch.FloatTensor(audio_spec_4).cuda().unsqueeze(0)
    data['frame1'] = frames_5.cuda()
    data['frame2'] = frames_4.cuda()
    try:
        outputs = model.forward(data)
    except Exception as e:
        traceback.print_exc()
        exit(-1)
    reconstructed_signal_5, _ = get_separated_audio(outputs, window, opt)
    reconstructed_signal_5 = reconstructed_signal_5 * normalizer5
    sep_audio5[-samples_per_window:] = sep_audio5[-samples_per_window:] + reconstructed_signal_5
    # divide the aggregated predicted audio by the overlap count
    avged_sep_audio5 = avged_sep_audio5 + clip_audio(np.divide(sep_audio5, overlap_count))

    if opt.save_output == "true":
        output_dir = osp.join(opt.output_dir_root, v1 + '@' + v2 + "@" + v3 + "@" + v4 + "@" + v5)
        if not osp.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        sf.write(osp.join(output_dir, v1 + '_separated.wav'), avged_sep_audio1, opt.audio_sampling_rate)
        sf.write(osp.join(output_dir, v2 + '_separated.wav'), avged_sep_audio2, opt.audio_sampling_rate)
        sf.write(osp.join(output_dir, v3 + '_separated.wav'), avged_sep_audio3, opt.audio_sampling_rate)
        sf.write(osp.join(output_dir, v4 + '_separated.wav'), avged_sep_audio4, opt.audio_sampling_rate)
        sf.write(osp.join(output_dir, v5 + '_separated.wav'), avged_sep_audio5, opt.audio_sampling_rate)

    sdr, sir, sar = getSeparationMetrics_5mix(avged_sep_audio1, avged_sep_audio2, avged_sep_audio3, avged_sep_audio4,
                                              avged_sep_audio5, audio1, audio2, audio3, audio4, audio5)
    sdr_mix, sir_mix, sar_mix = getSeparationMetrics_5mix(audio_mix, audio_mix, audio_mix, audio_mix, audio_mix,
                                                          audio1, audio2, audio3, audio4, audio5)
    pesq_score1 = pesq(avged_sep_audio1, audio1, opt.audio_sampling_rate)
    pesq_score2 = pesq(avged_sep_audio2, audio2, opt.audio_sampling_rate)
    pesq_score3 = pesq(avged_sep_audio3, audio3, opt.audio_sampling_rate)
    pesq_score4 = pesq(avged_sep_audio4, audio4, opt.audio_sampling_rate)
    pesq_score5 = pesq(avged_sep_audio5, audio5, opt.audio_sampling_rate)
    pesq_score = (pesq_score1 + pesq_score2 + pesq_score3 + pesq_score4 + pesq_score5) / 5
    # STOI
    stoi_score1 = stoi(audio1, avged_sep_audio1, opt.audio_sampling_rate, extended=False)
    stoi_score2 = stoi(audio2, avged_sep_audio2, opt.audio_sampling_rate, extended=False)
    stoi_score3 = stoi(audio3, avged_sep_audio3, opt.audio_sampling_rate, extended=False)
    stoi_score4 = stoi(audio4, avged_sep_audio4, opt.audio_sampling_rate, extended=False)
    stoi_score5 = stoi(audio5, avged_sep_audio5, opt.audio_sampling_rate, extended=False)
    stoi_score = (stoi_score1 + stoi_score2 + stoi_score3 + stoi_score4 + stoi_score5) / 5

    if with_recog:
        opt.with_recognition = True
        net_feature_extractor, net_recognizer = model.net_feature_extractor, model.net_recognizer
        eos = charToIx40["<EOS>"] if opt.recg_num_classes == 40 else charToIx32["</s>"]
        cer1 = forward_asr(eos, opt.blank_label, net_feature_extractor, net_recognizer, avged_sep_audio1.unsqueeze(0), trgt1, trgtLen1)
        cer2 = forward_asr(eos, opt.blank_label, net_feature_extractor, net_recognizer, avged_sep_audio2.unsqueeze(0), trgt2, trgtLen2)
        cer3 = forward_asr(eos, opt.blank_label, net_feature_extractor, net_recognizer, avged_sep_audio3.unsqueeze(0), trgt3, trgtLen3)
        cer4 = forward_asr(eos, opt.blank_label, net_feature_extractor, net_recognizer, avged_sep_audio4.unsqueeze(0), trgt4, trgtLen4)
        cer5 = forward_asr(eos, opt.blank_label, net_feature_extractor, net_recognizer, avged_sep_audio5.unsqueeze(0), trgt5, trgtLen5)
        return sdr, sir, sar, pesq_score, stoi_score, \
               sdr - sdr_mix, sir - sir_mix, sar - sar_mix, \
               (cer1 + cer2 + cer3 + cer4 + cer5) / 5

    return sdr, sir, sar, pesq_score, stoi_score, \
           sdr - sdr_mix, sir - sir_mix, sar - sar_mix


def main():
    #load test arguments
    opt = TestOptions().parse()
    opt.device = torch.device("cuda")
    opt.rank = 0

    for x in vars(opt):
        value = getattr(opt, x)
        if value == "true":
            setattr(opt, x, True)
        elif value == "false":
            setattr(opt, x, False)

    model = create_model(opt)
    model.eval()

    mtcnn = MTCNN(keep_all=True, device=opt.device)

    lipreading_preprocessing_func = get_preprocessing_pipelines()['test']
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    vision_transform_list = [transforms.ToTensor()]
    if opt.normalization:
        vision_transform_list.append(normalize)
    vision_transform = transforms.Compose(vision_transform_list)

    window = torch.hann_window(opt.window_size).cuda()

    sdrs, sirs, sars, pesq_scores, stoi_scores = [], [], [], [], []
    sdris, siris, saris = [], [], []
    cers = []
    with open(opt.test_file, 'r', encoding='utf-8') as pair_file:
        lines = pair_file.readlines()
        pb = ProgressBar(len(lines), start=False)
        pb.start()
        for line in lines:
            if opt.mix_number == 2:
                v1, v2 = line.strip().split(' ')
                results = process_2_mixture(model, opt, v1, v2, mtcnn, lipreading_preprocessing_func, vision_transform, window)
            elif opt.mix_number == 3:
                v1, v2, v3 = line.strip().split(' ')
                results = process_3_mixture(model, opt, v1, v2, v3, mtcnn, lipreading_preprocessing_func, vision_transform, window)
            elif opt.mix_number == 4:  # opt.mix_number ==4
                v1, v2, v3, v4 = line.strip().split(' ')
                results = process_4_mixture(model, opt, v1, v2, v3, v4, mtcnn, lipreading_preprocessing_func, vision_transform, window)
            else:
                v1, v2, v3, v4, v5 = line.strip().split(' ')
                results = process_5_mixture(model, opt, v1, v2, v3, v4, v5, mtcnn, lipreading_preprocessing_func,
                                            vision_transform, window)
            if opt.with_recognition:
                sdr, sir, sar, pesq_score, stoi_score, sdri, siri, sari, cer = results
            else:
                sdr, sir, sar, pesq_score, stoi_score, sdri, siri, sari = results
            sdrs.append(sdr)
            sirs.append(sir)
            sars.append(sar)
            pesq_scores.append(pesq_score)
            stoi_scores.append(stoi_score)
            sdris.append(sdri)
            siris.append(siri)
            saris.append(sari)
            if opt.with_recognition:
                cers.append(cer)

            pb.update()
    print()
    print(f"SDR: {round(float(np.mean(sdrs)), 3)}")
    print(f"SIR: {round(float(np.mean(sirs)), 3)}")
    print(f"SAR: {round(float(np.mean(sars)), 3)}")
    print(f"PESQ: {round(float(np.mean(pesq_scores)), 3)}")
    print(f"STOI: {round(float(np.mean(stoi_scores)), 3)}")
    print(f"SDRi: {round(float(np.mean(sdris)), 3)}")
    print(f"SIRi: {round(float(np.mean(siris)), 3)}")
    print(f"SARi: {round(float(np.mean(saris)), 3)}")

    if opt.with_recognition:
        print(f"CER: {round(float(np.mean(cers)), 3)}")


if __name__ == '__main__':
    # test
    main()
