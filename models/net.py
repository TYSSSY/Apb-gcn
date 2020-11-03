from abc import ABC

import torch.nn as nn
import torch.nn.functional as fn
from .layers import HGAConv
from third_party.models import SelfAttention
from einops import rearrange


class DualGraphTransformer(nn.Module, ABC):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers,
                 sequential=True):
        super(DualGraphTransformer, self).__init__()
        self.sequential = sequential
        channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.spatial_layers = nn.ModuleList([
            HGAConv(in_channels=channels[i],
                    out_channels=channels[i + 1]) for i in range(num_layers)
        ])
        self.temporal_layers = nn.ModuleList([
            SelfAttention(dim=channels[i + 1],  # TODO ??? potential dimension problem
                          nb_features=channels[i]) for i in range(num_layers)
        ])
        self.bottle_neck = nn.Linear(in_features=out_channels,
                                     out_features=out_channels)

    def forward(self, t):
        if self.sequential:
            for i in range(len(self.num_layers)):
                t = rearrange(fn.relu(self.spatial_layers[i](t)),
                              'b n c -> n b c')
                t = rearrange(self.temporal_layers[i](t),
                              'n b c -> b n c')
        else:
            s = t
            t_ = rearrange(t, 'b n c -> n b c')
            for i in range(len(self.num_layers)):
                s = fn.relu(self.spatial_layers[i](s))
                t_ = fn.relu(self.temporal_layers[i](t_))
