from .base_options import BaseOptions


# test by mix and separate two videos
class VisualOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ceph', type=str, choices=['true', 'false'], required=True)

        self.parser.add_argument('--test_file', type=str, required=True)
        self.parser.add_argument('--mix_number', type=int, default=2, required=True)
        self.parser.add_argument('--output_dir_root', type=str, required=True)

        # model specification
        self.parser.add_argument('--visual_pool', type=str, default='maxpool',
                                 help='avg or max pool for visual stream feature')
        self.parser.add_argument('--weights_facenet', type=str, required=True, help="weights for face net")
        self.parser.add_argument('--weights_unet', type=str, required=True, help="weights for unet")
        self.parser.add_argument('--weights_lipnet', type=str, required=True, help="weights for lip net")
        self.parser.add_argument('--weights_FRNet', type=str, required=True, help="weights for FRNet")
        self.parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
        self.parser.add_argument('--unet_input_nc', type=int, default=2, help="input spectrogram number of channels")
        self.parser.add_argument('--unet_output_nc', type=int, default=2, help="output spectrogram number of channels")
        self.parser.add_argument('--lipnet_config_path', type=str, help='path to the config file of lip net')
        self.parser.add_argument('--visual_feature_type', default='both', type=str, choices=('lip', 'face', 'both'),
                                 help='type of visual feature to use')

        self.parser.add_argument('--face_feature_dim', type=int, default=128, help="dimension of face feature map")
        self.parser.add_argument('--number_of_face_frames', type=int, default=1, help="number of face frames to use")
        self.parser.add_argument('--compression_type', type=str, default='hyperbolic',
                                 choices=('hyperbolic', 'sigmoidal', 'none'), help="type of compression on masks")
        self.parser.add_argument('--hyperbolic_compression_K', type=int, default=10, help="hyperbolic compression K")
        self.parser.add_argument('--hyperbolic_compression_C', type=float, default=0.1, help="hyperbolic compression C")
        self.parser.add_argument('--sigmoidal_compression_a', type=float, default=0.1, help="sigmoidal compression a")
        self.parser.add_argument('--sigmoidal_compression_b', type=int, default=0, help="sigmoidal compression b")
        self.parser.add_argument('--mask_clip_threshold', type=int, default=5, help="mask_clip_threshold")
        self.parser.add_argument('--l2_feature_normalization', default="false", choices=["false", "true"],
                                 help="whether l2 nomalizing identity/audio features")

        self.parser.add_argument('--FRNet_layers', type=int, default=1, help='number of refine model layers')

        # preprocessing
        self.parser.add_argument('--scale_w', nargs='+', help='Scale width of the video', default=[128], type=int)
        self.parser.add_argument('--scale_h', nargs='+', help='Scale height oft the video', default=[128], type=int)
        self.parser.add_argument("--crop_size", type=int, default=112, help="Final image scale", )

        # include test related hyper parameters here
        self.mode = "test"
