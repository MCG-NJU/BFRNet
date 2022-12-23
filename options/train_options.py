#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--resume', type=str, choices=["true", "false"], default="true", help='whether to resume if the checkpoint dir is not empty.')

		self.parser.add_argument('--train_file', type=str, required=True, help='train file')
		self.parser.add_argument('--val_file', type=str, required=True, help='val file')
		self.parser.add_argument('--noise_file', type=str, help='noise file')
		self.parser.add_argument('--noise_root', type=str, help='noise root')

		self.parser.add_argument('--sampler_type', type=str, choices=["normal", "curriculum", "curriculum2"], help='data sample strategy')
		self.parser.add_argument('--curriculum_sample', nargs='+', type=int, default=[1, 2, 3], help='the epochs to add 3mix, 4mix, 5mix')

		self.parser.add_argument('--display_freq', type=int, default=20, help='frequency of displaying average loss and accuracy')
		self.parser.add_argument('--save_latest_freq', type=int, default=500, help='frequency of saving the latest results')
		self.parser.add_argument('--tensorboard', type=str, choices=["true", "false"], default="true", help='use tensorboard to visualize loss change ')
		self.parser.add_argument('--validation_on', type=str, choices=["true", "false"], default="true", help='whether to test on validation set during training')
		self.parser.add_argument('--validation_freq', type=int, default=200, help='frequency of testing on validation set')

		# model arguments
		self.parser.add_argument('--visual_pool', type=str, default='maxpool', help='avg or max pool for visual stream feature')
		self.parser.add_argument('--audio_pool', type=str, default='maxpool', help="avg or max pool for audio stream feature")
		self.parser.add_argument('--weights_facial', type=str, default='./pretrained_models/cross-modal-pretraining/facial.pth', help="weights for facial attributes net")
		self.parser.add_argument('--weights_unet', type=str, default='', help="weights for unet")
		self.parser.add_argument('--weights_refine', type=str, default='', help="weights for refine net")
		self.parser.add_argument('--weights_lipreadingnet', type=str, default='', help="weights for lipreading net")
		self.parser.add_argument('--weights_vocal', type=str, default='', help="weights for vocal net")
		self.parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
		self.parser.add_argument('--unet_input_nc', type=int, default=2, help="input spectrogram number of channels")
		self.parser.add_argument('--unet_output_nc', type=int, default=2, help="output spectrogram number of channels")
		self.parser.add_argument('--lipreading_config_path', type=str, default='configs/lrw_snv1x_tcn2x.json', help='path to the config file of lipreading')
		self.parser.add_argument('--identity_feature_dim', type=int, default=128, help="dimension of identity feature map")

		# refine model arguments
		self.parser.add_argument('--refine_num_layers', type=int, default=1, help="number of the encoder layers in refine model")
		self.parser.add_argument('--residual_last', type=str, choices=["true", "false"], help="whether to use residual in the last layer of refine model")
		self.parser.add_argument('--refine_kernel_size', type=int, choices=[1, 3], help="the kernel size of the av-convolution in refine module")

		self.parser.add_argument('--visual_feature_type', default='both', type=str, choices=('lipmotion', 'identity', 'both'), help='type of visual feature to use')
		self.parser.add_argument('--lipreading_extract_feature', type=str, choices=["true", "false"], default="true", help="whether use features extracted from 3d conv")
		self.parser.add_argument('--number_of_identity_frames', type=int, default=1, help="number of identity frames to use")
		self.parser.add_argument('--compression_type', type=str, default='none', choices=('hyperbolic', 'sigmoidal', 'none'), help="type of compression on masks")
		self.parser.add_argument('--hyperbolic_compression_K', type=int, default=10, help="hyperbolic compression K")
		self.parser.add_argument('--hyperbolic_compression_C', type=float, default=0.1, help="hyperbolic compression C")
		self.parser.add_argument('--sigmoidal_compression_a', type=float, default=0.1, help="sigmoidal compression a")
		self.parser.add_argument('--sigmoidal_compression_b', type=int, default=0, help="sigmoidal compression b")
		self.parser.add_argument('--mask_clip_threshold', type=int, default=5, help="mask_clip_threshold")
		self.parser.add_argument('--l2_feature_normalization', type=str, choices=["true", "false"], default="false", help="whether l2 nomalizing identity/audio features")
		# self.parser.add_argument('--gt_percentage', type=float, default=0.5, help="percentage to use gt embeddings")

		# preprocessing
		self.parser.add_argument('--scale_w', nargs='+', help='Scale width of the video', default=[128], type=int)
		self.parser.add_argument('--scale_h', nargs='+', help='Scale height oft the video', default=[128], type=int)
		self.parser.add_argument("--crop_size", type=int, default=112, help="Final image scale")
		self.parser.add_argument('--normalization', type=str, choices=["true", "false"], default="true", help="Should we use input normalization?")
		self.parser.add_argument('--audio_augmentation', type=str, choices=["true", "false"], default="false", help='whether to augment input audio')
		self.parser.add_argument('--audio_normalization', type=str, choices=["true", "false"], default="true", help="whether to normalize audio?")

		# whether to use loss
		self.parser.add_argument('--use_mixandseparate_loss', default="true", type=str, choices=["true", "false"], help='whether to use mix-and-separate loss')
		self.parser.add_argument('--use_sisnr_loss', default="true", type=str, choices=["true", "false"], help='whether to use sisnr loss')
		self.parser.add_argument('--use_contrast_loss', default="true", type=str, choices=["true", "false"], help='whether to use contrast loss')
		# loss type
		self.parser.add_argument('--mask_loss_type', default='L2', type=str, choices=('L1', 'L2', 'BCE'), help='type of loss on mask')
		self.parser.add_argument('--weighted_mask_loss', action='store_true', help="weighted loss")
		self.parser.add_argument('--contrast_loss_type', default='TripletLossCosine', type=str, choices=('TripletLossCosine', 'NCELoss', 'TripletLossCosine2', 'NCELoss2'), help='type of contrasitve loss')

		# loss weight
		self.parser.add_argument('--mixandseparate_loss_weight', default=1, type=float, help='weight for reconstruction loss')
		self.parser.add_argument('--sisnr_loss_weight', default=1, type=float, help='weight for sisnr loss')
		self.parser.add_argument('--contrast_loss_weight', default=1e-2, type=float, help='weight for contrast loss')
		self.parser.add_argument('--after_refine_ratio', default=0.5, type=float, help='the ratio of after refine loss of the total')

		# contrast params
		self.parser.add_argument('--contrast_margin', default=0.5, type=float, help='margin for triplet loss')
		self.parser.add_argument('--contrast_temp', default=0.2, type=float, help='temperature for NCELoss')

		# optimizer arguments
		self.parser.add_argument('--lr_lipreading', type=float, default=1e-4, help='learning rate for lipreading stream')
		self.parser.add_argument('--lr_facial_attributes', type=float, default=1e-5, help='learning rate for identity stream')
		self.parser.add_argument('--lr_unet', type=float, default=1e-4, help='learning rate for unet')
		self.parser.add_argument('--lr_refine', type=float, default=1e-4, help='learning rate for refine net')
		self.parser.add_argument('--lr_vocal', type=float, default=1e-4, help='learning rate for vocal')

		self.parser.add_argument('--epochs', type=int, default=1, help='# of epochs to train, set to 1 because we are doing random sampling from the whole dataset')
		self.parser.add_argument('--lr_steps', nargs='+', type=int, default=[6, 8], help='steps to drop LR in training samples')
		self.parser.add_argument('--optimizer', default='adam', type=str, help='adam or sgd for optimization')
		self.parser.add_argument('--decay_factor', type=float, default=0.1, help='decay factor for learning rate')
		self.parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam')
		self.parser.add_argument('--weight_decay', default=1e-4, type=float, help='weights regularizer')

		self.mode = 'train'
