#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
# import os
# import torch
# from utils import utils


class BaseOptions:
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False
		self.mode = 'train'

	def initialize(self):
		self.parser.add_argument('--port', type=int, default=29660)

		self.parser.add_argument('--mp4_root', type=str, required=True, help='path to dataset')
		self.parser.add_argument('--audio_root', type=str, required=True, help='path to dataset')
		self.parser.add_argument('--mouth_root', type=str, required=True, help='path to dataset')

		self.parser.add_argument('--mouthroi_format', type=str, choices=["npz", "h5", "npy"], required=True)
		self.parser.add_argument('--name', type=str, default='audioVisual', help='name of the experiment. It decides where to store models')
		self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/', help='models are saved here')
		self.parser.add_argument('--model', type=str, default='audioVisual', help='chooses how datasets are loaded.')
		self.parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
		self.parser.add_argument('--nThreads', default=32, type=int, help='# threads for loading data')
		self.parser.add_argument('--seed', default=0, type=int, help='random seed')

		# arguments
		self.parser.add_argument('--num_frames', default=64, type=int, help='number of frames used for lipreading')
		self.parser.add_argument('--audio_length', default=2.55, type=float, help='audio segment length')
		self.parser.add_argument('--audio_sampling_rate', default=16000, type=int, help='sound sampling rate')
		self.parser.add_argument('--window_size', default=400, type=int, help="stft window length")
		self.parser.add_argument('--hop_size', default=160, type=int, help="stft hop length")
		self.parser.add_argument('--n_fft', default=512, type=int, help="stft hop length")
		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		self.opt.mode = self.mode

		return self.opt
