import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])


def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d, kernel_size=4, outpad=0):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1, output_padding=(outpad, 0))
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv])


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, outermost=False):
        super(up_conv, self).__init__()
        if not outermost:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=(2., 1.)),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
                )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=(2., 1.)),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Sigmoid()
                )

    def forward(self, x):
        x = self.up(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)


class Resnet18(nn.Module):
    def __init__(self, original_resnet, pool_type='maxpool', input_channel=3, with_fc=False, fc_in=512, fc_out=512):
        super(Resnet18, self).__init__()
        self.pool_type = pool_type
        self.input_channel = input_channel
        self.with_fc = with_fc

        # customize first convolution layer to handle different number of channels for images and spectrograms
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = [self.conv1]
        layers.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers)  # features before pooling

        if with_fc:
            self.fc = nn.Linear(fc_in, fc_out)
            self.fc.apply(weights_init)

    def forward(self, x):
        x = self.feature_extraction(x)

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(x, 1)
        else:
            return x

        if self.with_fc:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x.view(x.size(0), -1, 1, 1)
        else:
            return x.view(x.size(0), -1, 1, 1)

    def forward_multiframe(self, x, pool=True):
        (B, T, C, H, W) = x.size()
        x = x.contiguous()
        x = x.view(B * T, C, H, W)
        x = self.feature_extraction(x)

        (_, C, H, W) = x.size()
        x = x.view(B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        if not pool:
            return x 

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool3d(x, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool3d(x, 1)
        
        if self.with_fc:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x.view(x.size(0), -1, 1, 1)
        else:
            return x.view(x.size(0), -1, 1, 1)
        # return x


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim=512, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.att_fc_q = nn.Linear(input_dim, input_dim)  # attention开始前的fc层
        self.att_fc_k = nn.Linear(input_dim, input_dim)
        self.att_fc_v = nn.Linear(input_dim, input_dim)
        self.att_fc = nn.Linear(input_dim, input_dim)  # attention最后的fc层
        self.att_drop = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # query: (N, T, C), (N, T', C),  N-head
        q = self.att_fc_q(query)  # (N, T, C)
        k = self.att_fc_k(key)  # (N, T', C)
        v = self.att_fc_v(value)  # (N, T', C)
        _, _, C = q.shape
        att = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(C), dim=-1)  # (N, T, T')
        result = self.att_drop(self.att_fc(torch.matmul(att, v)))  # (N, T, C)
        return result


class FeedForward(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.ffn_fc1 = nn.Linear(input_dim, input_dim)
        self.ffn_fc2 = nn.Linear(input_dim, input_dim)
        self.ffn_drop1 = nn.Dropout(dropout)
        self.ffn_drop2 = nn.Dropout(dropout)
        self.att_norm = nn.LayerNorm(input_dim)
        self.ffn_norm = nn.LayerNorm(input_dim)
        self.act_ffn = nn.ReLU()

    def forward(self, x):
        # x: T, N, C
        out = self.ffn_drop2(self.ffn_fc2(self.ffn_drop1(self.act_ffn(self.ffn_fc1(x)))))
        return out


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


class VisualVoiceUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2, audioVisual_feature_dim=1152):
        super(VisualVoiceUNet, self).__init__()
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
        self.crsatt = FusionModule(ngf * 8, visual_dim=640)
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


class FilterLayer(nn.Module):
    def __init__(self, input_dim=514, dropout=0.1):
        super(FilterLayer, self).__init__()
        self.input_dim = input_dim
        self.dropout = dropout
        self.ca = MultiHeadAttention(input_dim, dropout)
        self.ffn = FeedForward(input_dim, dropout)

        self.norm_ca = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

    def forward(self, query, key, value, residual):
        # S, T, C
        result = self.norm_ca(residual + self.ca(query, key, value))  # (S, T, C)
        result = self.norm_ffn(result + self.ffn(result))
        return result


class Filter(nn.Module):
    def __init__(self, num_layers, audio_dim=514, dropout=0.1):
        super(Filter, self).__init__()
        self.num_layers = num_layers
        self.audio_dim = audio_dim
        self.dropout = dropout

        layers = []
        for n in range(num_layers):
            layers.append(FilterLayer(audio_dim, dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, visual, audio):
        # S, T, C
        for n in range(self.num_layers):
            audio = self.layers[n](visual, audio, audio, audio)
        return audio


class RecoveryLayer(nn.Module):
    def __init__(self, audio_dim=514, dropout=0.1):
        super(RecoveryLayer, self).__init__()
        self.dropout = dropout
        self.sa = MultiHeadAttention(audio_dim, dropout)
        self.ca = MultiHeadAttention(audio_dim, dropout)
        self.ffn = FeedForward(audio_dim, dropout)

        self.norm_sa = nn.LayerNorm(audio_dim)
        self.norm_ca = nn.LayerNorm(audio_dim)
        self.norm_ffn = nn.LayerNorm(audio_dim)

    def forward(self, query, key, value):
        # S, T, C
        query = self.norm_sa(query + self.sa(query, query, query)).permute(1, 0, 2)  # (T, S, C)
        key = key.permute(1, 0, 2)  # T, S, C
        value = value.permute(1, 0, 2)  # T, S, C

        num_speakers = len(query[0])
        ca_output = []
        for n in range(num_speakers):
            query_n = query[:, n:(n + 1), :]  # T, 1, C
            ca_output.append(self.norm_ca(query_n + self.ca(query_n, key[:, 1:, :], value[:, 1:, :])))
            key = torch.roll(key, -1, 1)
            value = torch.roll(value, -1, 1)
        ca_output = torch.cat(ca_output, dim=1).permute(1, 0, 2)  # S, T, C

        result = self.norm_ffn(ca_output + self.ffn(ca_output))
        return result


class Recovery(nn.Module):
    def __init__(self, num_layers, audio_dim=514, dropout=0.1):
        super(Recovery, self).__init__()
        self.num_layers = num_layers
        self.audio_dim = audio_dim
        self.dropout = dropout

        layers = []
        for n in range(num_layers):
            layers.append(RecoveryLayer(audio_dim, dropout))
        self.layers = nn.ModuleList(layers)

        self.linear = nn.Linear(audio_dim, audio_dim)

    def forward(self, query, key, value):
        # S, T, C
        for n in range(self.num_layers):
            query = self.layers[n](query, key, value)  # key和value保持不变
        result = self.linear(query)  # S, T, C
        return result


class FRModel(nn.Module):
    def __init__(self, num_layers, audio_time=256, visual_time=64, audio_dim=514, visual_dim=640, dropout=0.1):
        super(FRModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        kernel = audio_time // visual_time
        self.conv_v = nn.ConvTranspose1d(visual_dim, audio_dim, kernel, stride=kernel, padding=0)
        self.filter = Filter(num_layers, audio_dim, dropout)
        self.recovery = Recovery(num_layers, audio_dim, dropout)

    def forward(self, audio, visual):
        # audio: (N, 2, 257, 256),  visual: (N, 640, 64)->(N, C', T')
        N, L, C, T = audio.shape
        audio = audio.reshape(N, L * C, T).permute(0, 2, 1)  # N, T, L*C

        # filter
        visual = self.conv_v(visual).permute(0, 2, 1)  # (N, T, L*C)
        audio_filter = self.filter(visual, audio)  # (N, T, L*C)
        # recovery
        audio_recovery = self.recovery(audio_filter, audio, audio)  # N, T, L*C

        result = audio_filter + audio_recovery  # N, T, L*C
        result = result.permute(0, 2, 1).reshape(N, L, C, T)  # (N, L, C, T)

        return result
