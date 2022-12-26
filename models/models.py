import torch
import torchvision
from .networks import Resnet18, VisualVoiceUNet, weights_init, FRModel
from .lipreading_model import Lipreading
import os
import sys
sys.path.insert(0, '..')
from utils.utils import load_json


class ModelBuilder:
    # builder for face net
    def build_facenet(self, opt, pool_type='maxpool', input_channel=3, fc_out=512, with_fc=False, weights=''):
        pretrained = False
        original_resnet = torchvision.models.resnet18(pretrained)
        net = Resnet18(original_resnet, pool_type=pool_type, with_fc=with_fc, fc_in=512, fc_out=fc_out)

        if len(weights) > 0 and os.path.exists(weights):
            if opt.rank == 0:
                print(f'Loading weights for face net: {weights}')
            pretrained_state = torch.load(weights, map_location='cpu')
            model_state = net.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            net.load_state_dict(model_state)
        return net

    # builder for lipreading stream
    def build_lipnet(self, opt, config_path, weights='', extract_feats=True):
        if os.path.exists(config_path):
            args_loaded = load_json(config_path)
            if opt.rank == 0:
            # if dist.get_rank() == 0:
                print(f'Lipreading configuration file loaded: {config_path}')
            tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                            'kernel_size': args_loaded['tcn_kernel_size'],
                            'dropout': args_loaded['tcn_dropout'],
                            'dwpw': args_loaded['tcn_dwpw'],
                            'width_mult': args_loaded['tcn_width_mult']}                 
        net = Lipreading(tcn_options=tcn_options,
                         backbone_type=args_loaded['backbone_type'],
                         relu_type=args_loaded['relu_type'],
                         width_mult=args_loaded['width_mult'],
                         extract_feats=extract_feats)

        if len(weights) > 0 and os.path.exists(weights):
            if opt.rank == 0:
            # if dist.get_rank() == 0:
                print(f'Loading weights for lipreading stream: {weights}')
            net.load_state_dict(torch.load(weights, map_location='cpu'))
        return net

    # builder for audio stream
    def build_unet(self, opt, ngf=64, input_nc=1, output_nc=1, audioVisual_feature_dim=1280, weights=''):
        net = VisualVoiceUNet(ngf, input_nc, output_nc, audioVisual_feature_dim)
        net.apply(weights_init)

        if len(weights) > 0 and os.path.exists(weights):
            if opt.rank == 0:
                print(f'Loading weights for UNet: {weights}')
            net.load_state_dict(torch.load(weights, map_location='cpu'))
        return net

    def build_vocal(self, opt, pool_type='maxpool', input_channel=1, with_fc=False, fc_out=64, weights=''):
        pretrained = False
        original_resnet = torchvision.models.resnet18(pretrained)
        net = Resnet18(original_resnet, pool_type=pool_type, input_channel=input_channel,
                       with_fc=with_fc, fc_in=512, fc_out=fc_out)

        if len(weights) > 0 and os.path.exists(weights):
            if opt.rank == 0:
                print('Loading weights for vocal net')
            pretrained_state = torch.load(weights)
            model_state = net.state_dict()
            pretrained_state = { k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size() }
            model_state.update(pretrained_state)
            net.load_state_dict(model_state)
        return net

    def build_FRNet(self, opt, num_layers, weights=''):
        net = FRModel(num_layers)
        if len(weights) > 0 and os.path.exists(weights):
            if opt.rank == 0:
                print(f'Loading weights for FRNet: {weights}')
            net.load_state_dict(torch.load(weights, map_location='cpu'))
        return net
