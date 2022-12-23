#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# https://github.com/vBaiCai/python-pesq
# https://github.com/mpariente/pystoi
import io
import os
import os.path as osp
import librosa
import argparse
import numpy as np
import mir_eval.separation
# import time
from pypesq import pesq
from pystoi import stoi
# from petrel_client.client import Client
from mmcv import ProgressBar


def getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt):
	reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0)), axis=0)
	estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
	(sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources, False)
	#print(sdr, sir, sar, perm)
	return np.mean(sdr), np.mean(sir), np.mean(sar)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--results_dir', type=str, required=True)
	parser.add_argument('--audios_dir', type=str, required=True)
	parser.add_argument('--audio_sampling_rate', type=int, default=16000)
	parser.add_argument('--dataset', type=str, choices=["lrs2", "voxceleb2"], required=True)
	args = parser.parse_args()

	sdr_list = []
	sir_list = []
	sar_list = []
	pesq_score_list = []
	stoi_score_list = []

	results_list = os.listdir(args.results_dir)
	pb = ProgressBar(len(results_list), start=False)
	pb.start()
	for result in results_list:
		v1, v2 = result.split('@')
		if args.dataset == "lrs2":
			v1p = v1.replace("_", "/")
			v2p = v2.replace("_", "/")
		else:
			v1p = "/".join([v1[:7], v1[8:-6], v1[-5:]])
			v2p = "/".join([v2[:7], v2[8:-6], v2[-5:]])

		audio1_gt, _ = librosa.load(osp.join(args.audios_dir, v1p + ".wav"), sr=args.audio_sampling_rate)
		audio2_gt, _ = librosa.load(osp.join(args.audios_dir, v2p + ".wav"), sr=args.audio_sampling_rate)
		audio1, _ = librosa.load(osp.join(args.results_dir, result, v1 + "_separated.wav"), sr=args.audio_sampling_rate)
		audio2, _ = librosa.load(osp.join(args.results_dir, result, v2 + "_separated.wav"), sr=args.audio_sampling_rate)

		length = min(len(audio1), len(audio2), len(audio1_gt), len(audio2_gt))
		audio1_gt = audio1_gt[:length]
		audio2_gt = audio2_gt[:length]
		audio1 = audio1[:length]
		audio2 = audio2[:length]

		sdr, sir, sar = getSeparationMetrics(audio1, audio2, audio1_gt, audio2_gt)
		# PESQ
		pesq_score1 = pesq(audio1, audio1_gt, args.audio_sampling_rate)
		pesq_score2 = pesq(audio2, audio2_gt, args.audio_sampling_rate)
		pesq_score = (pesq_score1 + pesq_score2) / 2

		# STOI
		stoi_score1 = stoi(audio1_gt, audio1, args.audio_sampling_rate, extended=False)
		stoi_score2 = stoi(audio2_gt, audio2, args.audio_sampling_rate, extended=False)
		stoi_score = (stoi_score1 + stoi_score2) / 2

		sdr_list.append(sdr)
		sir_list.append(sir)
		sar_list.append(sar)
		pesq_score_list.append(pesq_score)
		stoi_score_list.append(stoi_score)
		pb.update()

	print(np.mean(sdr_list))
	print(np.mean(sir_list))
	print(np.mean(sar_list))
	print(np.mean(pesq_score_list))
	print(np.mean(stoi_score_list))


if __name__ == '__main__':
	main()
