import math
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import librosa


def compute_receptive_size(ks, ds, ss):
    # ks: list of kernel size;  ds: list of dilation size;  ss: list of stride size
    layers = len(ks)
    rs = [1]  # list of receptive size
    s = 1
    ss.insert(0, 1)
    for i in range(layers):
        s = s * ss[i]
        rs.append(rs[i] + ds[i] * (ks[i] - 1) * s)
    return rs


def scatter(seen, unseen):
    sdrs_seen = []
    audio_lengths_seen = []
    audio_mix_mean_seen = []
    audio_mean_seen = []
    sep_audio_mean_seen = []
    mouth_mean_seen = []
    frame_mean_seen = []
    score_seen = []
    with open(seen, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 0:
                sdr, audio_len, audio_mix_mean, audio_mean, sep_audio_mean, mouth_mean, frame_mean, score = line.strip().split("\t")
                sdrs_seen.append(float(sdr))
                audio_lengths_seen.append(int(audio_len))
                audio_mix_mean_seen.append(float(audio_mix_mean))
                audio_mean_seen.append(float(audio_mean))
                sep_audio_mean_seen.append(float(sep_audio_mean))
                mouth_mean_seen.append(float(mouth_mean))
                frame_mean_seen.append(float(frame_mean))
                score_seen.append(int(score))

    sdrs_unseen = []
    audio_lengths_unseen = []
    audio_mix_mean_unseen = []
    audio_mean_unseen = []
    sep_audio_mean_unseen = []
    mouth_mean_unseen = []
    frame_mean_unseen = []
    score_unseen = []
    with open(unseen, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 0:
                sdr, audio_len, audio_mix_mean, audio_mean, sep_audio_mean, mouth_mean, frame_mean, score = line.strip().split("\t")
                sdrs_unseen.append(float(sdr))
                audio_lengths_unseen.append(int(audio_len))
                audio_mix_mean_unseen.append(float(audio_mix_mean))
                audio_mean_unseen.append(float(audio_mean))
                sep_audio_mean_unseen.append(float(sep_audio_mean))
                mouth_mean_unseen.append(float(mouth_mean))
                frame_mean_unseen.append(float(frame_mean))
                score_unseen.append(int(score))

    plt.plot(audio_mix_mean_seen, sdrs_seen, '.', c="green")
    plt.plot(audio_mix_mean_unseen, sdrs_unseen, '.', c="blue")
    plt.show()


def readidnum(id_num_file):
    id_num = dict()
    with open(id_num_file, "r", encoding="utf-8") as f:
        for line in f:
            id, num = line.strip().split("\t")
            id_num[id] = int(num)
    return id_num


def statistic(sdr_file, path_file, id_num, start, end):
    # sdr与id出现次数的关系
    sdr_list = []
    with open(sdr_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 0:
                sdr = line.strip().split("\t")[0]
                sdr_list.append(float(sdr))

    num_list = []
    with open(path_file, "r", encoding="utf-8") as f:
        for line in f:
            two_clips = line.strip().split(" ")
            num_list.append(id_num[two_clips[0][start:end]])
            num_list.append(id_num[two_clips[1][start:end]])

    return sdr_list, num_list


def sdr2num(seen_sdr_file, unseen_sdr_file, seen_mix_file, unseen_mix_file, id_num_file):
    id_num = readidnum(id_num_file)
    seen_sdr_list, seen_num_list = statistic(seen_sdr_file, seen_mix_file, id_num, 6, 13)
    # unseen_sdr_list, unseen_num_list = statistic(unseen_sdr_file, unseen_mix_file, id_num, 4, 11)

    num_sdr = defaultdict(list)
    sdr_mean_list = []
    for sdr, num in zip(seen_sdr_list, seen_num_list):
        num_sdr[num].append(sdr)
    for num in sorted(list(num_sdr.keys())):
        sdr_mean_list.append(np.mean(num_sdr[num]))
    # import pdb; pdb.set_trace()

    # unseen_sdr_list, unseen_num_list = statistic(unseen_sdr_file, unseen_mix_file, id_num)
    plt.plot(seen_num_list, seen_sdr_list, '.', c="green")
    plt.plot(sorted(list(num_sdr.keys())), sdr_mean_list, '.', c="blue")
    plt.show()


def draw_wave():

    def block23(data1, data2, data3, length, color1, color2, color3):
        data1_ = np.copy(data1)

        idx1_ = np.ones(length)
        idx1 = np.random.choice(length, 10, replace=False)
        for i in range(2000):
            idx1_[np.clip(idx1 + i, 0, 70216)] = 0
            data1[np.clip(idx1 + i, 0, 70216)] = 0
        data1_[np.nonzero(idx1_)[0]] = 0

        idx2 = np.random.choice(length, 7, replace=False)
        for i in range(15000):
            data2[np.clip(idx2 + i, 0, 70216)] = 0

        idx3 = np.random.choice(length, 7, replace=False)
        for i in range(15000):
            data3[np.clip(idx3 + i, 0, 70216)] = 0

        time = np.arange(0, length) * (1.0 / 16000)
        print("time:", len(time))
        pl.plot(time, data1, alpha=0.7, c=color1)
        pl.plot(time, data1_, alpha=0.7, c="gray")
        pl.plot(time, data2, alpha=0.5, c=color2)
        pl.plot(time, data3, alpha=0.5, c=color3)
        pl.xlabel("time (seconds)")
        pl.show()

    import pylab as pl
    import numpy as np
    from scipy.io import wavfile
    _, wave_data1 = wavfile.read("videos/00012.wav")
    _, wave_data2 = wavfile.read("videos/00008.wav")
    _, wave_data3 = wavfile.read("videos/00010.wav")
    wave_data1 = wave_data1 / 32768 / 1.5
    wave_data2 = wave_data2 / 32768 / 2
    wave_data3 = wave_data3 / 32768 / 2
    wave_data1 = np.clip(wave_data1, -0.2, 0.2)
    wave_data2 = np.clip(wave_data2, -0.2, 0.2)
    wave_data3 = np.clip(wave_data3, -0.2, 0.2)
    length = 40800
    wave_data1 = wave_data1[:length]
    wave_data2 = wave_data2[:length]
    wave_data2[17000:19000] = np.clip(wave_data2[17000:19000], -0.1, 0.1)
    wave_data3 = wave_data3[:length]

    time = np.arange(0, length) * (1.0 / 16000)
    print("time:", len(time))

    # # speaker 1
    # wave_data1[6000:8000] = 0
    # wave_data1[10000:12000] = 0
    # wave_data1[32000:36000] = 0
    #
    wave_data2_to_1 = np.zeros(length)
    wave_data2_to_1[17000:19000] = wave_data2[17000:19000]
    wave_data2_to_1 = np.clip(wave_data2_to_1, -0.1, 0.1)
    wave_data2_to_1[19800:21500] = wave_data2[19800:21500]
    #
    wave_data3_to_1 = np.zeros(length)
    wave_data3_to_1[28000:33000] = wave_data3[28000:33000]


    # # speaker 2
    wave_data2[32000:33000] = wave_data3[1500:2500]
    wave_data2[33000:34000] = wave_data1[15000:16000]
    wave_data2[34000:35000] = wave_data3[15000:16000]
    wave_data2[35000:36000] = wave_data1[25000:26000]
    # wave_data2[8200:11300] = 0
    # wave_data2[14800:18000] = 0
    #
    wave_data1_to_2 = np.zeros(length)
    wave_data1_to_2[5800:16000] = wave_data1[5800:16000]
    #
    # wave_data3_to_2 = np.zeros(length)
    # wave_data3_to_2[29000:38000] = wave_data3[29000:38000]


    # speaker 3
    # wave_data3[31000:34000] = 0
    #
    wave_data1_to_3 = np.zeros(length)
    wave_data1_to_3[5000:7000] = wave_data1[5000:7000]
    wave_data1_to_3[26000:37000] = wave_data1[26000:37000]
    #
    # wave_data2_to_3 = np.zeros(length)
    # wave_data2_to_3[6000:18000] = wave_data2[6000:18000]


    # speaker 1
    # pl.plot(time[:6000], wave_data1[:6000], alpha=0.7, c="royalblue")
    # pl.plot(time[8000:10000], wave_data1[8000:10000], alpha=0.7, c="royalblue")
    # pl.plot(time[12000:32000], wave_data1[12000:32000], alpha=0.7, c="royalblue")
    # pl.plot(time[36000:], wave_data1[36000:], alpha=0.7, c="royalblue")
    # to speaker 1
    # pl.plot(time[17000:19000], wave_data2_to_1[17000:19000], alpha=0.7, c="green")
    # pl.plot(time[19800:21500], wave_data2_to_1[19800:21500], alpha=0.7, c="green")
    # pl.plot(time[28000:33000], wave_data3_to_1[28000:33000], alpha=0.7, c="orange")

    pl.plot(time[:5800], np.zeros(5800), alpha=0.7, c="white")
    pl.plot(time[12000:32000], np.zeros(32000-12000), alpha=0.7, c="white")
    pl.plot(time[36000:], np.zeros(length - 36000), alpha=0.7, c="white")

    # speaker 2
    # pl.plot(time, wave_data2, alpha=0.7, c="green")
    # pl.plot(time[:8200], wave_data2[:8200], alpha=0.7, c="green")
    # pl.plot(time[11300:14800], wave_data2[11300:14800], alpha=0.7, c="green")
    # pl.plot(time[18000:], wave_data2[18000:], alpha=0.7, c="green")
    pl.plot(time[5800:12000], wave_data1_to_2[5800:12000], alpha=0.7, c="royalblue")
    # pl.plot(time[29000:38000], wave_data3_to_2[29000:38000], alpha=0.7, c="orange")


    # speaker 3
    # pl.plot(time[:31000], wave_data3[:31000], alpha=0.7, c="orange")
    # pl.plot(time[34000:], wave_data3[34000:], alpha=0.7, c="orange")
    # pl.plot(time[5000:7000], wave_data1_to_3[5000:7000], alpha=0.7, c="royalblue")
    pl.plot(time[32000:36000], wave_data1_to_3[32000:36000], alpha=0.7, c="royalblue")
    # pl.plot(time[6000:18000], wave_data2_to_3[6000:18000], alpha=0.7, c="green")

    pl.show()

    # block23(wave_data3, wave_data1, wave_data2, length, "orange", "royalblue", "green")


def draw_mask():
    # shape: 256, 256
    from scipy.io import wavfile
    _, audio1 = wavfile.read("videos/00007.wav")
    _, audio2 = wavfile.read("videos/00002.wav")
    _, audio3 = wavfile.read("videos/00012.wav")
    audio1 = audio1 / 32768
    audio2 = audio2 / 32768
    audio3 = audio3 / 32768
    # length = min(len(audio1), len(audio2))
    length = 40800
    audio1 = audio1[:length]
    audio2 = audio2[:length]
    audio3 = audio3[:length]
    mixture = audio1 + audio2 + audio3

    spectro1 = librosa.core.stft(audio1, hop_length=160, n_fft=512, win_length=400, center=True)
    spectro1 = np.expand_dims(np.real(spectro1), axis=0)

    spectro_mix = librosa.core.stft(mixture, hop_length=160, n_fft=512, win_length=400, center=True)
    spectro_mix = np.expand_dims(np.real(spectro_mix), axis=0)

    mask = spectro1 / spectro_mix
    mask = (np.clip(mask, -1, 1) + 1) / 2
    mask = mask[0]
    mask = np.where(mask > 0.8, 1, 0)

    # plt.matshow(mask, cmap=plt.get_cmap('Greens'), alpha=0.5)  # , alpha=0.3

    plt.matshow(np.random.randn(256, 256), cmap=plt.cm.gray, alpha=1.0)  # , alpha=0.3
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()

    # plt.scatter(x, y, s=20, c='b')
    # plt.show()


def draw_trends():
    x = ["2mix", "3mix", "4mix", "5mix"]
    # lavse_seen = [5.54, 1.45, -0.72, -2.31]
    # lavse_unseen = [5.38, 1.53, -0.6, -2.13]
    lavse = [5.46, 1.49, -0.66, -2.22]
    # visualvoice_seen = [8,61, 3.96, 1.39, -0.4]
    # visualvoice_unseen = [8.97, 4.59, 2.05, 0.25]
    visualvoice = [8.79, 4.275, 1.72, -0.075]
    # vovit_seen = [9.91, 5.29, 2.17, -0.11]
    # vovit_unseen = [9.62, 5.08, 2.06, -0.21]
    vovit = [9.765, 5.185, 2.115, -0.16]
    # ours_seen = [11.18, 7.4, 4.81, 2.77]
    # ours_unseen = [10.95, 7.41, 5.04, 3.18]
    ours = [11.065, 7.405, 4.925, 2.975]

    # lavse_fr_seen = [8.61, 4.56, 1.97, 0.05]
    # lavse_fr_unseen = [8.35, 4.52, 2.06, 0.25]
    lavse_fr = [8.48, 4.54, 2.015, 0.15]
    # visualvoice_fr_seen = [10.8, 6.91, 4.34, 2.37]
    # visualvoice_fr_unseen = [10.64, 7.01, 4.61, 2.76]
    visualvoice_fr = [10.72, 6.96, 4.475, 2.565]
    # vovit_fr_seen = [10.95, 6.93, 3.93, 1.48]
    # vovit_fr_unseen = [10.63, 6.69, 3.78, 1.37]
    vovit_fr = [10.79, 6.81, 3.855, 1.425]

    plt.plot(x, lavse, 'o', color='g', linestyle='-', label="LAVSE")
    # plt.plot(x, lavse_fr, 'o', color='g', linestyle='--', label="LAVSE+FR")

    plt.plot(x, visualvoice, 'o', color='deepskyblue', linestyle='-', label="VisualVoice")
    # plt.plot(x, visualvoice_fr, 'o', color='deepskyblue', linestyle='--', label="VisualVoice+FR")

    plt.plot(x, vovit, 'o', color='mediumpurple', linestyle='-', label="VoViT")
    # plt.plot(x, vovit_fr, 'o', color='mediumpurple', linestyle='--', label="VoViT+FR")

    plt.plot(x, ours, 'o', color='orange', linestyle='-', label="Ours")

    plt.ylabel("SDR (dB)")
    plt.legend()

    plt.show()


def draw_bar():
    name_list = ["LAVSE", "VisualVoice", "VoViT", "ours"]
    x = np.array(list(range(len(name_list))))
    total_width, n = 0.6, len(x)
    gap = 0.05
    width = total_width / n
    x_lavse = np.array([x[0], x[0] + (width + gap), x[0] + 2 * (width + gap), x[0] + 3 * (width + gap)])
    x_vv = np.array([x[1], x[1] + (width + gap), x[1] + 2 * (width + gap), x[1] + 3 * (width + gap)])
    x_vovit = np.array([x[2], x[2] + (width + gap), x[2] + 2 * (width + gap), x[2] + 3 * (width + gap)])
    x_ours = np.array([x[3], x[3] + (width + gap), x[3] + 2 * (width + gap), x[3] + 3 * (width + gap)])
    LAVSE = [5.46, 1.49, -0.66, -2.22]
    VisualVoice = [8.79, 4.275, 1.72, -0.075]
    VoViT = [9.765, 5.185, 2.115, -0.16]
    ours = [10.985, 7.42, 5.035, 3.115]
    plt.bar(x_lavse, LAVSE, width=width, label="LAVSE", fc="y")
    plt.bar(x_vv, VisualVoice, width=width, label="VisualVoice", fc="r")
    plt.bar(x_vovit, VoViT, width=width, label="VoViT", fc="b")
    plt.bar(x_ours, ours, width=width, label="ours", fc="g")
    plt.xticks(x + (width + gap) * 1.5, name_list)
    plt.legend()
    plt.show()


def draw_bar2():
    # fig = plt.figure()
    # ax = plt.subplot(111)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    bottom = -3
    name_list = ["LAVSE", "VisualVoice", "VoViT", "BaSe"]
    x = np.array(list(range(len(name_list))))
    print(x)
    total_width, n = 0.6, len(x)
    gap = 0.05
    width = total_width / n
    x_lavse = np.array([x[0], x[0] + (width + gap), x[0] + 2 * (width + gap), x[0] + 3 * (width + gap)])
    x_vv = np.array([x[1], x[1] + (width + gap), x[1] + 2 * (width + gap), x[1] + 3 * (width + gap)])
    x_vovit = np.array([x[2], x[2] + (width + gap), x[2] + 2 * (width + gap), x[2] + 3 * (width + gap)])
    x_ours = np.array([x[3], x[3] + (width + gap), x[3] + 2 * (width + gap), x[3] + 3 * (width + gap)])
    x_mix2 = x
    x_mix3 = x + (width + gap)
    x_mix4 = x + 2 * (width + gap)
    x_mix5 = x + 3 * (width + gap)
    # mix2 = np.array([5.38, 8.97, 9.62, 11.06])
    mix2 = np.array([5.38, 8.97, 9.62, 10.03])
    # mix3 = np.array([1.53, 4.59, 5.08, 7.48])
    mix3 = np.array([1.53, 4.59, 5.08, 5.97])
    # mix4 = np.array([-0.6, 2.05, 2.06, 5.13])
    mix4 = np.array([-0.6, 2.05, 2.06, 3.39])
    # mix5 = np.array([-2.13, 0.25, -0.21, 3.26])
    # mix5 = np.array([10.03, 5.97, 3.39, 1.45])
    mix5 = np.array([-2.13, 0.25, -0.21, 1.45])

    # lavse_fr = [8.48, 4.54, 2.015, 0.15]
    # vv_fr = [10.72, 6.96, 4.475, 2.565]
    # vovit_fr = [10.79, 6.81, 3.855, 1.425]
    mix2_delta = np.array([2.97, 1.67, 1.01, 1.03])
    mix3_delta = np.array([2.99, 2.42, 1.61, 1.51])
    mix4_delta = np.array([2.66, 2.56, 1.72, 1.74])
    # mix5_delta = np.array([2.38, 2.51, 1.58])
    mix5_delta = np.array([2.38, 2.51, 1.58, 1.81])
    LAVSE = [5.46, 1.49, -0.66, -2.22]
    VisualVoice = [8.79, 4.275, 1.72, -0.075]
    VoViT = [9.765, 5.185, 2.115, -0.16]
    ours = [10.985, 7.42, 5.035, 3.115]
    plt.bar(x_mix2, mix2 - bottom, width=width, bottom=bottom, label="2mix", fc="orange", alpha=0.5)
    plt.bar(x_mix3, mix3 - bottom, width=width, bottom=bottom, label="3mix", fc="dodgerblue", alpha=0.5)
    plt.bar(x_mix4, mix4 - bottom, width=width, bottom=bottom, label="4mix", fc="darkviolet", alpha=0.5)
    plt.bar(x_mix5, mix5 - bottom, width=width, bottom=bottom, label="5mix", fc="limegreen", alpha=0.5)
    # plt.plot(x_lavse, LAVSE, '.', color='g', linestyle='-', label="LAVSE")
    # plt.plot(x_vv, VisualVoice, '.', color='g', linestyle='-', label="LAVSE")
    # plt.plot(x_vovit, VoViT, '.', color='g', linestyle='-', label="LAVSE")
    # plt.plot(x_ours, ours, '.', color='g', linestyle='-', label="LAVSE")
    # plt.bar(x_mix2[:3], mix2_delta, bottom=mix2[:3], width=width, fc="orange", alpha=0.5)
    # plt.bar(x_mix3[:3], mix3_delta, bottom=mix3[:3], width=width, fc="dodgerblue", alpha=0.5)
    # plt.bar(x_mix4[:3], mix4_delta, bottom=mix4[:3], width=width, fc="darkviolet", alpha=0.5)
    # plt.bar(x_mix5[:3], mix5_delta, bottom=mix5[:3], width=width, fc="limegreen", alpha=0.5)
    plt.errorbar(x_mix2, mix2 + (mix2_delta / 2), yerr=(mix2_delta / 2), fmt="|", ecolor="black", mec="black", elinewidth=2, capsize=6.5)
    plt.errorbar(x_mix3, mix3 + (mix3_delta / 2), yerr=(mix3_delta / 2), fmt="|", ecolor="black", mec="black", elinewidth=2, capsize=6.5)
    plt.errorbar(x_mix4, mix4 + (mix4_delta / 2), yerr=(mix4_delta / 2), fmt="|", ecolor="black", mec="black", elinewidth=2, capsize=6.5)
    plt.errorbar(x_mix5, mix5 + (mix5_delta / 2), yerr=(mix5_delta / 2), fmt="|", ecolor="black", mec="black", elinewidth=2, capsize=6.5)
    plt.xticks(x + (width + gap) * 1.5, name_list)
    for i, x_ in enumerate(x_mix2):
        plt.text(x_, mix2_delta[i] + mix2[i] + 0.2, "+" + str(mix2_delta[i]), ha='center', fontsize=7)
    for i, x_ in enumerate(x_mix3):
        plt.text(x_, mix3_delta[i] + mix3[i] + 0.2, "+" + str(mix3_delta[i]), ha='center', fontsize=7)
    for i, x_ in enumerate(x_mix4):
        plt.text(x_, mix4_delta[i] + mix4[i] + 0.2, "+" + str(mix4_delta[i]), ha='center', fontsize=7)
    for i, x_ in enumerate(x_mix5):
        plt.text(x_, mix5_delta[i] + mix5[i] + 0.2, "+" + str(mix5_delta[i]), ha='center', fontsize=7)
    for i, x_ in enumerate(x_mix2):
        plt.text(x_ - 0.17, mix2[i] - 0.16, str(mix2[i]), ha='center', fontsize=7)
    for i, x_ in enumerate(x_mix3):
        plt.text(x_ - 0.17, mix3[i] - 0.16, str(mix3[i]), ha='center', fontsize=7)
    for i, x_ in enumerate(x_mix4):
        plt.text(x_ - 0.18, mix4[i] - 0.16, str(mix4[i]), ha='center', fontsize=7)
    for i, x_ in enumerate(x_mix5):
        plt.text(x_ - 0.18, mix5[i] - 0.16, str(mix5[i]), ha='center', fontsize=7)
    plt.ylabel("SDR (dB)")
    plt.legend()
    plt.show()


def foo5():
    from petrel_client.client import Client
    import io
    client = Client()
    # with io.BytesIO(client.get("s3://chy/sh1986/data/voxceleb2/unseen_test/unseen_5mix.txt"))

def calculate():
    values = [2.72, 2.23, 1.90, 1.65]
    # 10.8 6.91 4.34 2.37
    result = np.mean(values)
    print(np.around(result, 2))


if __name__ == '__main__':
    # ks = [5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    # ss = [1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
    # ds = [1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8]
    #
    # rs = compute_receptive_size(ks, ds, ss)
    # print(rs)

    # sdr2num("foo/sep_seen_heard.txt", "foo/sep_unseen_unheard.txt",
    #         "foo/seen_heard_test_2mix.txt", "foo/unseen_unheard_test_2mix.txt",
    #         "foo/id_num.txt")

    # scatter("foo/sep_seen_heard.txt", "foo/sep_unseen_unheard.txt")

    calculate()
