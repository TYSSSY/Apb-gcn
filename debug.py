import torch
from data.dataset import SkeletonDataset
from torch_geometric.data import DataLoader
from models.layers import HGAConv

ds = SkeletonDataset(root='dataset',
                     name='ntu')
loader = DataLoader(ds, batch_size=4)
b = next(iter(loader))
ly = HGAConv(in_channels=7,
             out_channels=16,
             heads=8)
b = ly(b.x, adj=ds.skeleton_)
