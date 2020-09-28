import os
import numpy as np

import sys
import pickle
import argparse
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import DataLoader
import torch
from torch_geometric.data import Dataset
from prepare_ntu import *

import torch.utils.data
from torch._six import container_abcs, string_classes, int_classes


# NTURGBD dataset joints connection list
# 1-base of the spine
# 2-middle of the spine
# 3-neck
# 4-head
# 5-left shoulder
# 6-left elbow
# 7-left wrist
# 8-left hand
# 9-right shoulder
# 10-right elbow
# 11-right wrist
# 12-right hand
# 13-left hip
# 14-left knee
# 15-left ankle
# 16-left foot
# 17-right hip
# 18-right knee
# 19-right ankle
# 20-right foot
# 21-spine
# 22-tip of the left hand
# 23-left thumb
# 24-tip of the right hand
# 25-right thumb

# edge index is based on the list above
edge_index_plus_one = torch.tensor([(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)],dtype = torch.long)

edge_index = torch.tensor([(i - 1, j - 1) for (i, j) in edge_index_plus_one],dtype = torch.long)

num_joint = 25

#benchmark = ['cs', 'cv']
#part = ['train', 'val']

class NTUDataset(Dataset):
    def __init__(self, root, batch_size, transform=None, pre_transform=None, benchmark='cv', part='val', ignored_sample_path=None):
        self.batch_size = batch_size
        self.raw_path = root + "/raw"
        self.ignored_sample_path = ignored_sample_path
        self.benchmark = benchmark
        self.part = part
        self.out_path = os.path.join(os.path.join(root, 'processed'), self.benchmark,self.part)
        
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.rawFileNames = dataSample(
                                self.raw_path,
                                self.out_path,
                                ignored_sample_path=self.ignored_sample_path,
                                benchmark=self.benchmark,
                                part=self.part
        )
        super(NTUDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.rawFileNames[0]

    @property
    def processed_file_names(self):

        processedData = []
        for i in range(len(self.rawFileNames[0])):
            processedData.append('data_{}.pt'.format(i))
        return processedData

    def process(self):
        gendata(
                self.raw_path,
                self.out_path,
                ignored_sample_path = self.ignored_sample_path,
                benchmark=self.benchmark,
                part=self.part)
            
    def len(self):
        return len(self.raw_paths)

    def get(self, idx):
        data = torch.load(os.path.join(self.out_path, 'data_{}.pt'.format(idx)))
        data.num_nodes = num_joint
        return data

if __name__ == '__main__':

    n_batch_size = 2
    ntu_dataset = NTUDataset(
                            "/home/lawbuntu/Downloads/pytorch_geometric-master/docker", 
                            batch_size=n_batch_size, 
                            benchmark='cv',
                            part='val',
                            ignored_sample_path=None)
    
    ntu_dataloader = DataLoader(ntu_dataset, batch_size=n_batch_size, shuffle=True)
    
    count = 0
    i = 0
    batch = None
    for b in (ntu_dataloader):
        batch = b
        print('Index', count)
        print(len(batch))
        print(batch)
        count += 1
    print(count)
    