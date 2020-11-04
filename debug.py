import torch
from data.dataset import SkeletonDataset
from torch_geometric.data import DataLoader
from models.layers import HGAConv
from third_party.performer import SelfAttention
from einops import rearrange


ds = SkeletonDataset(root='dataset',
                     name='ntu')
loader = DataLoader(ds, batch_size=4)
b = next(iter(loader))
ly = HGAConv(in_channels=7,
             out_channels=16,
             heads=8)
t = ly(b.x, adj=ds.skeleton_)

t = rearrange(t, 'b n c -> n b c')
lt = SelfAttention(dim=128,
                   nb_features=7,
                   causal=True)

t = lt(t)
