from .base_options import BaseOptions


class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--resume', type=str, choices=["true", "false"], default="true", help='whether to resume if the checkpoint dir is not empty.')

		self.parser.add_argument('--train_file', type=str, required=True, help='train file')
		self.parser.add_argument('--val_file', type=str, required=True, help='val file')

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
		self.parser.add_argument('--weights_facenet', type=str, default='./pretrained_models/cross-modal-pretraining/facial.pth', help="weights for facial attributes net")
		self.parser.add_argument('--weights_unet', type=str, default='', help="weights for unet")
		self.parser.add_argument('--weights_FRNet', type=str, default='', help="weights for FRNet")
		self.parser.add_argument('--weights_lipnet', type=str, default='', help="weights for lipnet")
		self.parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
		self.parser.add_argument('--unet_input_nc', type=int, default=2, help="input spectrogram number of channels")
		self.parser.add_argument('--unet_output_nc', type=int, default=2, help="output spectrogram number of channels")
		self.parser.add_argument('--lipnet_config_path', type=str, default='configs/lrw_snv1x_tcn2x.json', help='path to the config file of lipreading')
		self.parser.add_argument('--lip_feature_dim', type=int, default=512, help="dimension of lip feature map")
		self.parser.add_argument('--face_feature_dim', type=int, default=128, help="dimension of face feature map")

		# refine model arguments
		self.parser.add_argument('--FRNet_layers', type=int, default=1, help="number of layers in FRNet")

		self.parser.add_argument('--visual_feature_type', default='both', type=str, choices=('lip', 'face', 'both'), help='type of visual feature to use')
		self.parser.add_argument('--number_of_face_frames', type=int, default=1, help="number of face frames to use")
		self.parser.add_argument('--compression_type', type=str, default='none', choices=('hyperbolic', 'sigmoidal', 'none'), help="type of compression on masks")
		self.parser.add_argument('--hyperbolic_compression_K', type=int, default=10, help="hyperbolic compression K")
		self.parser.add_argument('--hyperbolic_compression_C', type=float, default=0.1, help="hyperbolic compression C")
		self.parser.add_argument('--sigmoidal_compression_a', type=float, default=0.1, help="sigmoidal compression a")
		self.parser.add_argument('--sigmoidal_compression_b', type=int, default=0, help="sigmoidal compression b")
		self.parser.add_argument('--mask_clip_threshold', type=int, default=5, help="mask_clip_threshold")
		self.parser.add_argument('--l2_feature_normalization', type=str, choices=["true", "false"], default="false", help="whether l2 nomalizing face/audio features")

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
		# loss type
		self.parser.add_argument('--mask_loss_type', default='L2', type=str, choices=('L1', 'L2', 'BCE'), help='type of loss on mask')
		self.parser.add_argument('--weighted_mask_loss', action='store_true', help="weighted loss")

		# loss weight
		self.parser.add_argument('--mixandseparate_loss_weight', default=1, type=float, help='weight for reconstruction loss')
		self.parser.add_argument('--sisnr_loss_weight', default=1, type=float, help='weight for sisnr loss')
		self.parser.add_argument('--lamda', default=0.5, type=float, help='the factor to control the ratio of the losses.')

		# optimizer arguments
		self.parser.add_argument('--lr_lipnet', type=float, default=1e-4, help='learning rate for lipreading stream')
		self.parser.add_argument('--lr_facenet', type=float, default=1e-5, help='learning rate for face stream')
		self.parser.add_argument('--lr_unet', type=float, default=1e-4, help='learning rate for unet')
		self.parser.add_argument('--lr_FRNet', type=float, default=1e-4, help='learning rate for refine net')

		self.parser.add_argument('--epochs', type=int, default=1, help='# of epochs to train, set to 1 because we are doing random sampling from the whole dataset')
		self.parser.add_argument('--lr_steps', nargs='+', type=int, default=[6, 8], help='steps to drop LR in training samples')
		self.parser.add_argument('--optimizer', default='adam', type=str, help='adam or sgd for optimization')
		self.parser.add_argument('--decay_factor', type=float, default=0.1, help='decay factor for learning rate')
		self.parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam')
		self.parser.add_argument('--weight_decay', default=1e-4, type=float, help='weights regularizer')

		self.mode = 'train'
