from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as fn
from .layers import HGAConv


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
            HGAConv(in_channels=channels[i],
                    out_channels=channels[i + 1]) for i in range(num_layers)
        ])
        self.bottle_neck = nn.Linear(in_features=out_channels,
                                     out_features=out_channels)

    def forward(self, t):
        if self.sequential:
            for i in range(len(self.num_layers)):
                t = self.temporal_layers[i](
                    fn.relu(self.spatial_layers[i](t)))
        else:
            s = t
            for i in range(len(self.num_layers)):
                s = fn.relu(self.spatial_layers[i](s))
                t = fn.relu(self.temporal_layers[i](t))
