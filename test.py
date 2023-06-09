import io
import math
import os
import h5py
import numpy as np
import os.path as osp
from scipy.io import wavfile
from facenet_pytorch import MTCNN
from mir_eval import separation
from pypesq import pesq
from mmengine import ProgressBar

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from options.test_options import TestOptions
from models.build_models import ModelBuilder
from models.audioVisual_model import BFRNet
from dataset.audioVisual_dataset import get_preprocessing_pipelines, load_frame
from utils.utils import collate_fn


vision_transform_list = [transforms.ToTensor()]
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
vision_transform_list.append(normalize)
vision_transform = transforms.Compose(vision_transform_list)
lipreading_preprocessing_func = get_preprocessing_pipelines()['test']


def getSeparationMetrics(audio_pred, audio_gt):
    """
    calculate the evaluation metric SDR
    :param audio_pred:     prediction             batch_size * length
    :param audio_gt:       ground truth           batch_size * length
    :return:   SDR
    """
    sdr, _, _, _ = separation.bss_eval_sources(audio_gt, audio_pred, False)
    return np.mean(sdr)


def audio_normalize(samples, desired_rms=0.1, eps=1e-4):
    """
    normalize the samples to have the desired root_mean_square
    :param samples:                   B*N*L
    :param desired_rms:    desired root_mean_square
    :param eps:
    :return: normalized samples
    """
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2, axis=-1)))
    samples = samples * np.expand_dims(desired_rms / rms, axis=-1)
    return samples


def supp_audio(audio, minimum_length):
    """
    repeat the audio to the target length
    :param audio:        B*L
    :param minimum_length:
    :return:            B*L
    """
    if len(audio[0]) >= minimum_length:
        return audio[:, :minimum_length]
    else:
        return np.tile(audio, (1, (minimum_length + len(audio[0]) - 1) // len(audio[0])))[:, :minimum_length]


def supp_mouth(mouth, minimum_length):
    """
    repeat the vision to the target length
    :param mouth:              B*L*88*88
    :param minimum_length:
    :return:                   B*L*88*88
    """
    if len(mouth[0]) >= minimum_length:
        return mouth[:, :minimum_length]
    else:
        return np.tile(mouth, (1, (minimum_length + len(mouth[0]) - 1) // len(mouth[0]), 1, 1))[:, :minimum_length]


class dataset(data.Dataset):
    def __init__(self, opt, mixture_path, audio_direc, mouth_direc, visual_direc, mtcnn):
        self.opt = opt
        self.audio_direc = audio_direc
        self.mouth_direc = mouth_direc
        self.visual_direc = visual_direc
        self.mtcnn = mtcnn

        if opt.ceph == "true":
            from petrel_client.client import Client
            self.client = Client()
            with io.BytesIO(self.client.get(mixture_path, update_cache=True)) as mp:
                self.mix_lst = [d.decode('utf-8').strip() for d in mp.readlines()]
        else:
            with open(mixture_path, "r", encoding="utf-8") as f:
                self.mix_lst = f.read().splitlines()

        self.window = torch.hann_window(400)

    def __len__(self):
        return len(self.mix_lst)

    # def preprocess(self):

    def process_audio(self, tokens):
        num_speakers = len(tokens)

        audios = []
        audio_length = []
        for n in range(num_speakers):
            audio_path = os.path.join(self.audio_direc, tokens[n]) + ".wav"
            if self.opt.ceph == "true":
                with io.BytesIO(self.client.get(audio_path, update_cache=True)) as ap:
                    _, audio = wavfile.read(ap)
            else:
                _, audio = wavfile.read(audio_path)
            audio = audio / 32768  # normalize the int16 data to [-1,1]
            audios.append(audio)
            audio_length.append(len(audio))
        target_length = min(audio_length)

        for i in range(num_speakers):
            audios[i] = audios[i][:target_length]
        audios = np.array(audios)
        slen = int(2.55 * 16000)
        target = ((target_length + slen - 1) // slen) * slen
        seg = target // slen

        audios = supp_audio(audios, target)
        audios = audios.reshape(num_speakers, seg, slen).transpose((1, 0, 2))  # seg, num_speakers, slen
        audios = audio_normalize(audios) / num_speakers
        audios = torch.FloatTensor(audios)  # seg, num_speakers, slen

        # mixture
        audio_mix = torch.FloatTensor(torch.sum(audios, dim=1))  # seg, num_speakers, slen

        audio_mix_spec = self.generate_spectrogram_complex(audio_mix, 512, 160, 400)  # seg, 2, 257, 256
        audio_mix_spec = audio_mix_spec.unsqueeze(1).repeat((1, num_speakers, 1, 1, 1))  # seg, speakers, 2, 257, 256
        audio_mix = audio_mix.unsqueeze(1).repeat(1, num_speakers, 1)  # seg, num_speakers, slen

        return audios, audio_mix, audio_mix_spec, target_length

    def process_mouth(self, tokens, seg):
        num_speakers = len(tokens)

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

        slen = 64
        target = seg * slen
        mouthrois = supp_mouth(mouthrois, target)  # num_speakers, target, 96, 96

        # preprocess
        tmp_mouthrois = []  # num_speakers, target, 88, 88
        for n in range(num_speakers):
            tmp_mouthrois.append(lipreading_preprocessing_func(mouthrois[n]))
        mouthrois = torch.FloatTensor(tmp_mouthrois).reshape(num_speakers, seg, slen, 88, 88).permute(1, 0, 2, 3, 4).unsqueeze(2)

        return mouthrois  # (seg, num_speakers, 1, slen, 88, 88)

    def process_frame(self, tokens, seg):
        # sample a frame from a video
        num_speakers = len(tokens)
        frames = []
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
            audios, audio_mix, audio_mix_spec, target_length = self.process_audio(tokens)  # seg, num_speakers, slen
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


def inference(model, data_loader):
    model.eval()

    # evaluation metrics
    sdr_list, pesq_list = [], []

    # progress bar
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

            # feed to network
            output = model(inputs)

            pred_spec = output['pred_specs_aft']  # total_seg * num_speakers, 257, 256, 2
            pred_audio = torch.istft(pred_spec, n_fft=512, hop_length=160, win_length=400, window=window, center=True)
            pred_audio = pred_audio.reshape(total_seg, num_speakers, -1)  # total_seg, num_speakers, L
            seg_list = data['seg']

            cumsum = 0
            for idx, seg in enumerate(seg_list):
                tmp_pred_audio = pred_audio[cumsum: cumsum + seg]  # seg, num_speakers, L
                tmp_pred_audio = tmp_pred_audio.permute(1, 0, 2).reshape(num_speakers, -1)[:, :data['target_length'][idx]].cpu()  # num_speakers, target_length

                tmp_audio = data['audios'][cumsum: cumsum + seg]  # seg, num_speakers, L
                tmp_audio = tmp_audio.permute(1, 0, 2).reshape(num_speakers, -1)[:, :data['target_length'][idx]]  # num_speakers, target_length

                cumsum += seg

                tmp_pred_audio = tmp_pred_audio.detach().numpy()
                tmp_audio = tmp_audio.numpy()

                # calculate SDR
                sdr = getSeparationMetrics(tmp_pred_audio, tmp_audio)
                # sdr = np.mean(sdr)
                sdr_list.append(sdr)

                # calculate pesq for each speaker
                for n in range(num_speakers):
                    pesq_score_ = pesq(tmp_pred_audio[n], tmp_audio[n], 16000)
                    pesq_list.append(pesq_score_)

            pb.update()

    avg_sdr = sum(sdr_list) / len(sdr_list)
    avg_pesq = sum(pesq_list) / len(pesq_list)

    print('SDR: %.2f' % avg_sdr)
    print('PESQ: %.2f' % avg_pesq)


def main():
    # load test configuration
    opt = TestOptions().parse()
    opt.device = torch.device("cuda:0")
    opt.mode = 'test'
    opt.rank = 0

    # Network Builders
    builder = ModelBuilder()
    lip_net = builder.build_lipnet(
        opt=opt,
        config_path=opt.lipnet_config_path,
        weights=opt.weights_lipnet)
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
    unet = builder.build_unet(
            opt=opt,
            ngf=opt.unet_ngf,
            input_nc=opt.unet_input_nc,
            output_nc=opt.unet_output_nc,
            weights=opt.weights_unet)
    FRNet = builder.build_FRNet(
        opt=opt,
        num_layers=opt.FRNet_layers,
        weights=opt.weights_FRNet
    )

    nets = (lip_net, face_net, unet, FRNet)
    model = BFRNet(nets, opt).cuda()
    model.eval()
    mtcnn = MTCNN(keep_all=True, device=opt.device)

    # build data loader
    data_set = dataset(opt, opt.test_file, opt.audio_root, opt.mouth_root, opt.mp4_root, mtcnn)
    data_loader = data.DataLoader(
        data_set,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=opt.nThreads,
        collate_fn=collate_fn
    )
    inference(model, data_loader)


if __name__ == '__main__':
    # test
    main()
