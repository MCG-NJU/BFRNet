import os.path
import librosa
from scipy.io import wavfile
from data.base_dataset import BaseDataset
import h5py
import random
from random import randrange
from PIL import Image, ImageEnhance
import numpy as np
import torchvision.transforms as transforms
import torch
from utils.lipreading_preprocess import *
from utils.video_reader import VideoReader
import bisect
from petrel_client.client import Client
import io


def normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples


def get_preprocessing_pipelines():
    # -- preprocess for the video stream
    preprocessing = {}
    # -- LRW config
    crop_size = (88, 88)
    (mean, std) = (0.421, 0.165)
    preprocessing['train'] = Compose([
        Normalize(0.0, 255.0),
        RandomCrop(crop_size),
        HorizontalFlip(0.5),
        Normalize(mean, std)])
    preprocessing['val'] = Compose([
        Normalize(0.0, 255.0),
        CenterCrop(crop_size),
        Normalize(mean, std)])
    preprocessing['test'] = preprocessing['val']
    return preprocessing


def load_frame(clip_path):
    video_reader = VideoReader(clip_path, 1)
    start_pts, time_base, total_num_frames = video_reader._compute_video_stats()
    end_frame_index = total_num_frames - 1
    if end_frame_index < 0:
        clip, _ = video_reader.read(start_pts, 1)
    else:
        clip, _ = video_reader.read(random.randint(0, end_frame_index) * time_base, 1)
    frame = Image.fromarray(np.uint8(clip[0].to_rgb().to_ndarray())).convert('RGB')
    return frame


def get_mouthroi_audio_pair_on_integral_word(second, anno_path, mouthroi, audio, window, num_of_mouthroi_frames, audio_sampling_rate):
    # 计算从每个单词开始的<=2.56s的最大持续时间: durations
    with open(anno_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[4:]
    number = len(lines)
    starts = [float(line.split(' ')[1]) for line in lines]
    ends = [float(line.split(' ')[2]) for line in lines]
    durations = []
    pairs = []
    e = 0
    for s in range(number):
        while e < number and ends[e] - starts[s] < second:
            e += 1
        e -= 1
        pairs.append([s, e])
        durations.append(ends[e] - starts[s])
    # 以长度为正相关概率来选择start-end pair
    # probs = durations / sum(durations)
    durations = np.array(durations)
    probs = np.div(durations, np.sum(durations))
    index = random.choices(list(range(number)), weights=probs)
    # dur =

    audio_start = randrange(0, audio.shape[0] - window + 1)
    audio_sample = audio[audio_start:(audio_start+window)]
    frame_index_start = int(round(audio_start / audio_sampling_rate * 25))
    mouthroi = mouthroi[frame_index_start:(frame_index_start + num_of_mouthroi_frames), :, :]
    return mouthroi, audio_sample


def get_mouthroi_audio_text_random(lines, mouthroi, audio, window, num_of_mouthroi_frames, audio_sampling_rate, num_classes):
    # based on the premise: duration of the video is larger than 2.56s
    starts, ends, words = [], [], []
    for line in lines:
        tokens = line.strip().split(' ')
        starts.append(float(tokens[1]))
        ends.append(float(tokens[2]))
        words.append(tokens[0])

    # audio sample
    audio_start = randrange(0, audio.shape[0] - window + 1)
    audio_end = audio_start + window
    audio_sample = audio[audio_start:audio_end]

    # start time
    start_time = audio_start / audio_sampling_rate
    end_time = audio_end / audio_sampling_rate

    # iou >= 0.5, add into word list
    sindex = bisect.bisect_left(starts, start_time)
    if sindex > 0 and start_time < ends[sindex - 1]:
        if (ends[sindex - 1] - start_time) / (ends[sindex - 1] - starts[sindex - 1]) >= 0.5:
            sindex = sindex - 1
    eindex = bisect.bisect_right(ends, end_time)
    if eindex < len(words) and end_time > starts[eindex]:
        if (end_time - starts[eindex]) / (ends[eindex] - starts[eindex]) < 0.5:
            eindex = eindex - 1
    words = words[sindex:eindex]

    # translate language to index
    sentence = ' '.join(words)
    if num_classes == 32:
        trgt = [charToIx32["<s>"]] + [charToIx32[char] if char in charToIx32 else charToIx32['<unk>'] for char in sentence] + [charToIx32["</s>"]]
    else:  # num_classes = 40
        trgt = [charToIx40[char] for char in sentence] + [charToIx40["<EOS>"]]
    trgt = np.array(trgt)
    trgtLen = len(trgt)

    # frame sample
    frame_index_start = int(round(start_time * 25))
    frame_index_end = frame_index_start + num_of_mouthroi_frames
    if frame_index_end > len(mouthroi):
        rollback = frame_index_end - len(mouthroi)
        frame_index_start -= rollback
        frame_index_end -= rollback
    mouthroi = mouthroi[frame_index_start:frame_index_end, :, :]
    return mouthroi, audio_sample, trgt, trgtLen


def get_mouthroi_audio(mouthroi, audio, window, num_of_mouthroi_frames, audio_sampling_rate):
    # audio sample
    audio_start = randrange(0, audio.shape[0] - window + 1)
    audio_end = audio_start + window
    audio_sample = audio[audio_start:audio_end]

    # start time
    start_time = audio_start / audio_sampling_rate

    # frame sample
    frame_index_start = int(round(start_time * 25))
    frame_index_end = frame_index_start + num_of_mouthroi_frames
    if frame_index_end > len(mouthroi):
        rollback = frame_index_end - len(mouthroi)
        frame_index_start -= rollback
        frame_index_end -= rollback
    mouthroi = mouthroi[frame_index_start:frame_index_end, :, :]
    return mouthroi, audio_sample


def generate_spectrogram_magphase(audio, stft_frame, stft_hop, n_fft):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=n_fft, win_length=stft_frame, center=True)
    spectro_mag, spectro_phase = librosa.core.magphase(spectro)  # 幅度, 模长
    spectro_mag = np.expand_dims(spectro_mag, axis=0)  # 相位
    if with_phase:
        spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)  # 相角
        return spectro_mag, spectro_phase
    else:
        return spectro_mag


def generate_spectrogram_complex(audio, stft_frame, stft_hop, n_fft):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=n_fft, win_length=stft_frame, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel


def augment_image(image):
    if(random.random() < 0.5):
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.random()*0.6 + 0.7)
    return image


def augment_audio(audio):
    audio = audio * (random.random() * 0.2 + 0.9) # 0.9 - 1.1
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio


class AudioVisualDataset(BaseDataset):

    def initialize(self, opt):
        self.opt = opt
        self.audio_window = int(opt.audio_length * opt.audio_sampling_rate)
        random.seed(opt.seed)
        self.lipreading_preprocessing_func = get_preprocessing_pipelines()[opt.mode]
        
        # load videos path
        if opt.mode == 'train':
            anno_file = opt.train_file
            noise_file = opt.noise_file
            noise_root = opt.noise_root
        else:
            anno_file = opt.val_file
            noise_file = None
            noise_root = None

        self.client = Client(backend='petrel')

        with io.BytesIO(self.client.get(anno_file)) as af:
            self.videos_path = [d.decode('utf-8').strip() for d in af.readlines()]
        self.length = len(self.videos_path)

        if opt.audio_augmentation and noise_file is not None and noise_root is not None:
            with io.BytesIO(self.client.get(noise_file)) as nf:
                self.noise_path = [os.path.join(noise_root, d.decode('utf-8').strip()) for d in nf.readlines()]
        else:
            self.noise_path = None

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        vision_transform_list = [transforms.Resize(224), transforms.ToTensor()]
        if self.opt.normalization:
            vision_transform_list.append(normalize)
        self.vision_transform = transforms.Compose(vision_transform_list)

    def _audio_augment(self, audio):
        audio = audio * (random.random() * 1.5 + 0.5)  # 0.5 - 2.0
        return audio

    # def _audio_augment(self, audio):
    #     audio = audio * (random.random() * 0.2 + 0.9)  # 0.9 - 1.1
    #
    #     noise_name = np.random.choice(self.noise_path)
    #     with io.BytesIO(self.client.get(noise_name)) as nf:
    #         _, noise = wavfile.read(nf)
    #     noise = noise / 32768
    #     start = randrange(0, len(noise) - self.audio_window + 1)
    #     end = start + self.audio_window
    #     noise = noise[start:end]
    #     noise = normalize(noise, desired_rms=0.01)
    #
    #     audio = audio + noise
    #     audio[audio > 1.] = 1.
    #     audio[audio < -1.] = -1.
    #
    #     return audio

    def _get_one(self, index):
        # paths
        video_path = os.path.join(self.opt.mp4_root, self.videos_path[index] + '.mp4')
        audio_path = os.path.join(self.opt.audio_root, self.videos_path[index] + '.wav')
        mouthroi_path = os.path.join(self.opt.mouth_root, self.videos_path[index] + '.' + self.opt.mouthroi_format)

        # load audio
        try:
            with io.BytesIO(self.client.get(audio_path)) as ap:
                _, audio = wavfile.read(ap)
            audio = audio / 32768
        except Exception as e:
            return self._get_one(np.random.randint(self.length))

        # load mouth roi
        if self.opt.mouthroi_format == "npz":
            with io.BytesIO(self.client.get(mouthroi_path)) as mp:
                mouthroi = np.load(mp)["data"]
        else:  # h5
            with io.BytesIO(self.client.get(mouthroi_path)) as mp:
                mouthroi = h5py.File(mp, "r")["data"][...]

        if not (len(audio) >= self.audio_window and len(mouthroi) >= self.opt.num_frames):
            return self._get_one(np.random.randint(self.length))

        mouthroi, audio = get_mouthroi_audio(mouthroi, audio, self.audio_window, self.opt.num_frames, self.opt.audio_sampling_rate)
        # transform mouthrois and audios
        mouthroi = self.lipreading_preprocessing_func(mouthroi)
        # audio normalize
        if self.opt.audio_normalization:
            audio = normalize(audio)
        # audio augment
        if self.opt.audio_augmentation and self.opt.mode == 'train':
            audio = self._audio_augment(audio)

        frame_list = []
        for i in range(self.opt.number_of_identity_frames):
            try:
                frame = load_frame(io.BytesIO(self.client.get(video_path)))
            except:
                print(f'error video: {video_path}', flush=True)
                return self._get_one(np.random.randint(self.length))
            if self.opt.mode == 'train':
                frame = augment_image(frame)
            frame = self.vision_transform(frame)
            frame_list.append(frame)
        frames = torch.stack(frame_list).squeeze()

        return frames, audio, mouthroi

    def __getitem__(self, index_batches):
        # print(f"rank:{dist.get_rank()}, getitem", flush=True)
        if self.client is None:
            self.client = Client(backend='petrel')

        num_batch = len(index_batches)

        frames_batch = []
        audios_spec_batch = []
        mouthrois_batch = []
        audio_spec_mix_batch = []
        num_speakers_batch = []
        indexes_batch = []

        for b in range(num_batch):
            indexes = index_batches[b]
            num_speakers = len(indexes)

            frames = []
            audios = []
            mouthrois = []
            audio_specs = []

            for i in range(num_speakers):
                frame, audio, mouthroi = self._get_one(indexes[i])
                audio = audio / num_speakers
                audio_spec = generate_spectrogram_complex(audio, self.opt.window_size, self.opt.hop_size, self.opt.n_fft)
                frames.append(frame)
                audios.append(audio)
                mouthrois.append(mouthroi)
                audio_specs.append(audio_spec)
            aud_mix = np.sum(audios, axis=0)
            audio_spec_mix = generate_spectrogram_complex(aud_mix, self.opt.window_size, self.opt.hop_size, self.opt.n_fft)

            frames = torch.stack(frames, dim=0)  # num_speakers, 3, 224, 224
            mouthrois = torch.FloatTensor(mouthrois).unsqueeze(1)  # num_speakers, 1, 64, 88, 88
            audio_specs = torch.FloatTensor(audio_specs)  # num_speakers, 2, 257, 256
            audio_spec_mix = torch.FloatTensor(audio_spec_mix).unsqueeze(0).repeat(num_speakers, 1, 1, 1)  # num_speakers, 2, 257, 256
            num_speakers = torch.IntTensor([num_speakers])
            indexes = torch.IntTensor(indexes)

            frames_batch.append(frames)
            audios_spec_batch.append(audio_specs)
            mouthrois_batch.append(mouthrois)
            audio_spec_mix_batch.append(audio_spec_mix)
            num_speakers_batch.append(num_speakers)
            indexes_batch.append(indexes)

        data = dict()
        data['frames'] = torch.cat(frames_batch)
        data['audio_specs'] = torch.cat(audios_spec_batch)
        data['mouthrois'] = torch.cat(mouthrois_batch)
        data['audio_spec_mix'] = torch.cat(audio_spec_mix_batch)
        data['num_speakers'] = torch.cat(num_speakers_batch)
        data['indexes'] = torch.cat(indexes_batch)

        # print(f"rank:{dist.get_rank()} finish getitem", flush=True)

        return data

    def __len__(self):  # number of iterations
        return self.length

    def name(self):
        return 'AudioVisualDataset'
