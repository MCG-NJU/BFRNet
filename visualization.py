import io
import os
import cv2
import h5py
import random
import numpy as np
import os.path as osp
import soundfile as sf
from scipy.io import wavfile

import librosa
import librosa.display
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from mmcv import ProgressBar
from petrel_client.client import Client

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN

from options.testMany_options import TestOptions
from models.models import ModelBuilder
from utils.utils import collate_fn
from models.audioVisual_model import AudioVisualModel
from data.audioVisual_dataset import get_preprocessing_pipelines, load_frame


vision_transform_list = [transforms.ToTensor()]
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
vision_transform_list.append(normalize)
vision_transform = transforms.Compose(vision_transform_list)
lipreading_preprocessing_func = get_preprocessing_pipelines()['test']


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
    return sisnr  # B


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
    # audio: N, L
    spec = torch.stft(audio, n_fft=512, hop_length=160, win_length=400, window=torch.hann_window(400), center=True)
    return spec.permute(0, 3, 1, 2)  # N, 2, F, T


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

        # audio_spec
        audio_specs = self.generate_spectrogram_complex(audios.reshape(seg * num_speakers, margin), 512, 160, 400)  #seg*speaker, 2, 257, 256
        audio_specs = audio_specs.reshape(seg, num_speakers, 2, 257, 256)  # seg, num_speakers, 2, 257, 256

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

        audios, audio_mix, audio_mix_spec, target_length = self.process_wav(tokens)  # seg, num_speakers, margin
        # audio_specs = self.generate_spectrogram_complex(audios.reshape())
        seg = len(audios)

        mouthrois = self.process_mouth(tokens, seg)  # seg, num_speakers, 1, 64, 88, 88

        frames = self.process_frame(tokens, seg)  # seg, num_speakers, 3, 224, 224

        data = {}
        data['frames'] = frames  # seg, num_speakers, 3, 224, 224
        data['audios'] = audios  # seg, num_speakers, 40800
        data['audio_mix'] = audio_mix  # seg, num_speakers, 40800
        data['mouthrois'] = mouthrois  # seg, num_speakers, 1, 64, 88, 88
        data['audio_spec_mix'] = audio_mix_spec  # seg, num_speakers, 2, 257, 256
        data['seg'] = torch.IntTensor([seg])
        data['target_length'] = torch.IntTensor([target_length])
        data['indexes'] = torch.IntTensor([index])

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


def draw_spec(spec, name, vmin, vmax):
    # spec: 2, 257, 256
    plt.cla()
    plt.clf()
    plt.close()
    amp = np.linalg.norm(spec, axis=0)  # 257, 256
    spec = librosa.amplitude_to_db(amp)
    norm = Normalize(vmin=vmin, vmax=vmax)
    img = librosa.display.specshow(spec, y_axis='log', sr=16000, hop_length=160, cmap=plt.cm.coolwarm, norm=norm)
    cb = plt.colorbar(img, format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.savefig(name)


def save_wav(audio, name):
    sf.write(name, audio, 16000)


def save_results(opt, tokens, spec_gt, spec_pre, spec_aft, spec_filter, audio_gt, audio_pre, audio_aft, audio_filter):
    # spec: num_speakers, 2, 257, 256
    # audio: num_speakers, L
    path = "&".join([''.join(p.split('/')[1:]) for p in tokens])
    save_dir = os.path.join(opt.save_dir, path)
    os.makedirs(save_dir, exist_ok=True)
    vmin = np.min([np.min(spec_gt), np.min(spec_pre), np.min(spec_aft), np.min(spec_filter)])
    vmax = np.max([np.max(spec_gt), np.max(spec_pre), np.max(spec_aft), np.max(spec_filter)])
    for i in range(len(tokens)):
        spec_gt_path = os.path.join(save_dir, f"spec_gt_{i+1}.png")
        if not os.path.exists(spec_gt_path):
            draw_spec(spec_gt[i], spec_gt_path, vmin, vmax)

        spec_pre_path = os.path.join(save_dir, f"spec_pre_{i+1}.png")
        if not os.path.exists(spec_pre_path):
            draw_spec(spec_pre[i], spec_pre_path, vmin, vmax)

        spec_aft_path = os.path.join(save_dir, f"spec_aft_{i+1}.png")
        if not os.path.exists(spec_aft_path):
            draw_spec(spec_aft[i], spec_aft_path, vmin, vmax)

        spec_filter_path = os.path.join(save_dir, f"spec_filter_{i + 1}.png")
        if not os.path.exists(spec_filter_path):
            draw_spec(spec_filter[i], spec_filter_path, vmin, vmax)

        audio_gt_path = os.path.join(save_dir, f"audio_gt_{i+1}.wav")
        if not os.path.exists(audio_gt_path):
            save_wav(audio_gt[i], audio_gt_path)

        audio_pre_path = os.path.join(save_dir, f"audio_pre_{i+1}.wav")
        if not os.path.exists(audio_pre_path):
            save_wav(audio_pre[i], audio_pre_path)

        audio_aft_path = os.path.join(save_dir, f"audio_aft_{i+1}.wav")
        if not os.path.exists(audio_aft_path):
            save_wav(audio_aft[i], audio_aft_path)

        audio_filter_path = os.path.join(save_dir, f"audio_filter_{i + 1}.wav")
        if not os.path.exists(audio_filter_path):
            save_wav(audio_filter[i], audio_filter_path)


def normalize2(audio, desired_rms=0.1, eps=1e-4):
    # samples: num_speakers, L
    rms = torch.maximum(torch.tensor(eps), torch.sqrt(torch.mean(audio ** 2, dim=-1)))
    audio = audio * (desired_rms / rms).unsqueeze(1)
    return audio


def process_mixture(opt, model, data_loader):
    model.eval()
    sisnri_list = []
    pb = ProgressBar(len(data_loader), start=False)
    pb.start()
    window = torch.hann_window(400).cuda()

    if opt.ceph == "true":
        client = Client()
        with io.BytesIO(client.get(opt.test_file, update_cache=True)) as mp:
            mix_lst = [d.decode('utf-8').strip() for d in mp.readlines()]
    else:
        with open(opt.test_file, "r", encoding="utf-8") as f:
            mix_lst = f.read().splitlines()

    with torch.no_grad():
        for _, data in enumerate(data_loader):
            inputs = {}
            total_seg, num_speakers = data['frames'].shape[:2]
            inputs['frames'] = data['frames'].reshape(total_seg * num_speakers, 3, 224, 224).clone().detach().cuda()
            inputs['mouthrois'] = data['mouthrois'].reshape(total_seg * num_speakers, 1, 64, 88, 88).clone().detach().cuda()
            inputs['audio_spec_mix'] = data['audio_spec_mix'].reshape(total_seg * num_speakers, 2, 257, 256).clone().detach().cuda()
            inputs['num_speakers'] = num_speakers

            output = model(inputs)

            pred_spec_pre = output['pred_specs_pre']  # total_seg * num_speakers, 257, 256, 2
            pred_spec_aft = output['pred_specs_aft']  # total_seg * num_speakers, 257, 256, 2
            filter_specs = output['filter_specs']  # total_seg * num_speakers, 257, 256, 2

            pred_audio_pre = torch.istft(pred_spec_pre, n_fft=512, hop_length=160, win_length=400, window=window, center=True)
            pred_audio_aft = torch.istft(pred_spec_aft, n_fft=512, hop_length=160, win_length=400, window=window, center=True)
            filter_audio = torch.istft(filter_specs, n_fft=512, hop_length=160, win_length=400, window=window, center=True)
            pred_audio_pre = pred_audio_pre.reshape(total_seg, num_speakers, -1)  # total_seg, num_speakers, L
            pred_audio_aft = pred_audio_aft.reshape(total_seg, num_speakers, -1)  # total_seg, num_speakers, L
            filter_audio = filter_audio.reshape(total_seg, num_speakers, -1)  # total_seg, num_speakers, L
            # total_seg, num_speakers, L
            seg_list = data['seg']
            indexes = data['indexes']  # batch size

            cumsum = 0
            for idx, seg in enumerate(seg_list):  # 每一个人
                tmp_audio = data['audios'][cumsum: cumsum + seg]  # seg, num_speakers, L
                tmp_audio = tmp_audio.permute(1, 0, 2).reshape(num_speakers, -1)[:, :data['target_length'][idx]]  # num_speakers, target_length
                tmp_audio = normalize2(tmp_audio)

                tmp_pred_audio_pre = pred_audio_pre[cumsum: cumsum + seg]  # seg, num_speakers, L
                tmp_pred_audio_pre = tmp_pred_audio_pre.permute(1, 0, 2).reshape(num_speakers, -1)[:, :data['target_length'][idx]].cpu()  # num_speakers, target_length
                tmp_pred_audio_pre = normalize2(tmp_pred_audio_pre)

                tmp_pred_audio_aft = pred_audio_aft[cumsum: cumsum + seg]  # seg, num_speakers, L
                tmp_pred_audio_aft = tmp_pred_audio_aft.permute(1, 0, 2).reshape(num_speakers, -1)[:, :data['target_length'][idx]].cpu()  # num_speakers, target_length
                tmp_pred_audio_aft = normalize2(tmp_pred_audio_aft)

                tmp_filter_audio = filter_audio[cumsum: cumsum + seg]  # seg, num_speakers, L
                tmp_filter_audio = tmp_filter_audio.permute(1, 0, 2).reshape(num_speakers, -1)[:, :data['target_length'][idx]].cpu()  # num_speakers, target_length
                tmp_filter_audio = normalize2(tmp_filter_audio)

                cumsum += seg

                sisnr_pre = get_sisnr(tmp_pred_audio_pre, tmp_audio)  # num_speakers
                sisnr_aft = get_sisnr(tmp_pred_audio_aft, tmp_audio)  # num_speakers
                sisnri = sisnr_aft - sisnr_pre  # num_speakers
                sisnri = sisnri.numpy().tolist()

                specs_gt = generate_spectrogram(tmp_audio).numpy()  # num_speakers, 2, F, T
                specs_pre = generate_spectrogram(tmp_pred_audio_pre).numpy()  # num_speakers, 2, F, T
                specs_aft = generate_spectrogram(tmp_pred_audio_aft).numpy()  # num_speakers, 2, F, T
                specs_filter = generate_spectrogram(tmp_filter_audio).numpy()  # num_speakers, 2, F, T

                index = indexes[idx]
                tokens = mix_lst[index].split(" ")  # num_speakers

                # sisnri_list.append((tokens[]))
                for n in range(num_speakers):
                    sisnri_list.append((tokens[n], sisnri[n]))

                tmp_audio = tmp_audio.numpy()
                tmp_pred_audio_pre = tmp_pred_audio_pre.numpy()
                tmp_pred_audio_aft = tmp_pred_audio_aft.numpy()

                save_results(opt, tokens, specs_gt, specs_pre, specs_aft, specs_filter, tmp_audio, tmp_pred_audio_pre, tmp_pred_audio_aft, tmp_filter_audio)

            pb.update()

    sisnri_list = sorted(sisnri_list, key=lambda x: x[1], reverse=True)
    avg_sisnri = sum([sisnri[1] for sisnri in sisnri_list]) / len(sisnri_list)
    print('SISNRi: %.2f' % avg_sisnri)

    sisnri_list = [" ".join((sisnri[0], str(round(sisnri[1], 2)))) for sisnri in sisnri_list]
    with open(osp.join(osp.dirname(opt.save_dir), f"{opt.name}_sisnri.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sisnri_list))


def main():
    #load test arguments
    opt = TestOptions().parse()
    opt.device = torch.device("cuda")
    opt.mode = 'test'
    opt.rank = 0
    opt.save_dir = osp.join(opt.output_dir_root, osp.basename(osp.dirname(opt.weights_unet)), opt.name)

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
            identity_feature_dim=opt.identity_feature_dim,
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
