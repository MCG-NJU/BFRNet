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
        self.att_fc_q = nn.Linear(input_dim, input_dim)
        self.att_fc_k = nn.Linear(input_dim, input_dim)
        self.att_fc_v = nn.Linear(input_dim, input_dim)
        self.att_fc = nn.Linear(input_dim, input_dim)
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




