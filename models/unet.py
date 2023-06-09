import torch
import torch.nn as nn

from .networks import MultiHeadAttention, unet_conv, unet_upconv, conv_block, up_conv


class FusionModule(nn.Module):
    def __init__(self, audio_dim=512, visual_dim=640, dropout=0.1):
        super(FusionModule, self).__init__()
        self.conv_a = nn.Conv1d(audio_dim, audio_dim, kernel_size=3, stride=1, padding=1)
        self.conv_v = nn.Conv1d(visual_dim, audio_dim, kernel_size=3, stride=1, padding=1)

        self.crsatt_av = MultiHeadAttention(audio_dim, dropout)
        self.norm_av = nn.LayerNorm(audio_dim)

    def forward(self, audio, visual):
        # audio: (N, 512, 1, 64), visual: (N, 640, 1, 64)
        audio = self.conv_a(audio.squeeze(2)).transpose(1, 2)  # (N, 64, 512), (N, T, C)
        visual = self.conv_v(visual.squeeze(2)).transpose(1, 2)  # (N, 64, 512), (N, T, C)
        feat = self.norm_av(visual + self.crsatt_av(audio, audio, visual)).transpose(1, 2).unsqueeze(2)  # (N, 512, 1, 64)
        return feat


class Unet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2, audioVisual_feature_dim=1152):
        super(Unet, self).__init__()
        # initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = conv_block(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = conv_block(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = conv_block(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = conv_block(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = conv_block(ngf * 8, ngf * 8)
        self.audionet_convlayer8 = conv_block(ngf * 8, ngf * 8)
        self.frequency_pool = nn.MaxPool2d([2, 1])
        self.crsatt = FusionModule(ngf * 8, visual_dim=640)  # fuse audio and visual features
        self.audionet_upconvlayer1 = up_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer2 = up_conv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = up_conv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = up_conv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer5 = up_conv(ngf * 16, ngf * 4)
        self.audionet_upconvlayer6 = up_conv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer8 = unet_upconv(ngf * 2, output_nc, True, outpad=1)
        self.Sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

    def forward(self, audio_mix_spec, visual_feat, activation='Sigmoid'):
        audio_conv1feature = self.audionet_convlayer1(audio_mix_spec[:, :, :-1, :])
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv3feature = self.frequency_pool(audio_conv3feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv4feature = self.frequency_pool(audio_conv4feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv5feature = self.frequency_pool(audio_conv5feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv6feature = self.frequency_pool(audio_conv6feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)
        audio_conv7feature = self.frequency_pool(audio_conv7feature)
        audio_conv8feature = self.audionet_convlayer8(audio_conv7feature)
        audio_conv8feature = self.frequency_pool(audio_conv8feature)

        # audioVisual_feature = torch.cat((visual_feat, audio_conv8feature), dim=1)
        audioVisual_feature = self.crsatt(audio_conv8feature, visual_feat)

        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv7feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv6feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv5feature), dim=1))
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv4feature), dim=1))
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audio_conv3feature), dim=1))
        audio_upconv7feature = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audio_conv2feature), dim=1))
        pred_mask = self.audionet_upconvlayer8(torch.cat((audio_upconv7feature, audio_conv1feature), dim=1))  # (N, 2, 257, 256)
        if activation == 'Sigmoid':
            pred_mask = self.Sigmoid(pred_mask)
        elif activation == 'Tanh':
            pred_mask = self.Tanh(pred_mask)
        return pred_mask
