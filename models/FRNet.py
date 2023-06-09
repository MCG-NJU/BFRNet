import torch
import torch.nn as nn

from .networks import MultiHeadAttention, FeedForward


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
            query = self.layers[n](query, key, value)
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
