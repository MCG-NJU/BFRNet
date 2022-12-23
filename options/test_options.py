from .base_options import BaseOptions


#test by mix and separate two videos
class TestOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--mouthroi1_path', type=str, required=True)
		self.parser.add_argument('--mouthroi2_path', type=str, required=True)
		self.parser.add_argument('--audio1_path', type=str, required=True)
		self.parser.add_argument('--audio2_path', type=str, required=True)
		self.parser.add_argument('--video1_path', type=str, required=True)
		self.parser.add_argument('--video2_path', type=str, required=True)

		self.parser.add_argument('--output_dir_root', type=str, default='output')
		self.parser.add_argument('--hop_length', default=0.04, type=float, help='the hop length to perform audio separation in a sliding window approach')
		self.parser.add_argument('--subtract_mean', default=True, type=bool, help='subtract channelwise mean from input image')
		self.parser.add_argument('--enable_data_augmentation', type=bool, default=False, help='whether to augment input audio/image')
		self.parser.add_argument('--spectrogram_type', type=str, default='magonly', choices=('complex', 'magonly'), help='whether to use magonly or complex spectrogram')
		self.parser.add_argument('--mask_to_use', type=str, default='pred', choices=('gt', 'pred'), help='what phase to use')
		self.parser.add_argument('--visualize_spectrogram', action='store_true', help='whether to use discriminator')

		#model specification
		self.parser.add_argument('--visual_pool', type=str, default='maxpool', help='avg or max pool for visual stream feature')
		self.parser.add_argument('--audio_pool', type=str, default='maxpool', help="avg or max pool for audio stream feature")
		self.parser.add_argument('--weights_facial', type=str, default='', help="weights for facial attributes net")
		self.parser.add_argument('--weights_unet', type=str, default='', help="weights for unet")
		self.parser.add_argument('--weights_vocal', type=str, default='', help="weights for vocal attributes net")
		self.parser.add_argument('--weights_lipreadingnet', type=str, default='', help="weights for lipreading net")
		self.parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
		self.parser.add_argument('--unet_input_nc', type=int, default=2, help="input spectrogram number of channels")
		self.parser.add_argument('--unet_output_nc', type=int, default=2, help="output spectrogram number of channels")
		self.parser.add_argument('--weighted_mask_loss', action='store_true', help="weighted loss")
		self.parser.add_argument('--lipreading_config_path', type=str, help='path to the config file of lipreading')
		self.parser.add_argument('--visual_feature_type', default='both', type=str, choices=('lipmotion', 'identity', 'both'), help='type of visual feature to use')
		self.parser.add_argument('--audioVisual_feature_dim', type=int, default=1280, help="dimension of audioVisual feature map")
		self.parser.add_argument('--identity_feature_dim', type=int, default=64, help="dimension of identity feature map")
		self.parser.add_argument('--lipreading_extract_feature',default=False,action='store_true',help="whether use features extracted from 3d conv")
		self.parser.add_argument('--number_of_identity_frames', type=int, default=1, help="number of identity frames to use")
		self.parser.add_argument('--compression_type', type=str, default='hyperbolic', choices=('hyperbolic', 'sigmoidal', 'none'), help="type of compression on masks")
		self.parser.add_argument('--hyperbolic_compression_K', type=int, default=10, help="hyperbolic compression K")
		self.parser.add_argument('--hyperbolic_compression_C', type=float, default=0.1, help="hyperbolic compression C")
		self.parser.add_argument('--sigmoidal_compression_a', type=float, default=0.1, help="sigmoidal compression a")
		self.parser.add_argument('--sigmoidal_compression_b', type=int, default=0, help="sigmoidal compression b")
		self.parser.add_argument('--mask_clip_threshold', type=int, default=5, help="mask_clip_threshold")
		self.parser.add_argument('--l2_feature_normalization',default=False,action='store_true',help="whether l2 nomalizing identity/audio features")
		self.parser.add_argument('--reliable_face',default=False,action='store_true',help="whether to use the face that has high detection score")

		#preprocessing
		self.parser.add_argument('--scale_w',nargs='+',help='Scale width of the video',default=[128],type=int)
		self.parser.add_argument('--scale_h',nargs='+',help='Scale height oft the video',default=[128],type=int)
		self.parser.add_argument("--crop_size",type=int,default=112,help="Final image scale",)
		self.parser.add_argument('--normalization',default=False,action='store_true',help="Should we use input normalization?")
		self.parser.add_argument('--audio_normalization',default=False,action='store_true',help="whether to normalize audio?")

		#include test related hyper parameters here
		self.mode = "test"
