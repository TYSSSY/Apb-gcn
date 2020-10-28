from abc import ABC

import os
import os.path as osp

import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
from .skeleton import process_skeleton


class SkeletonDataset(Dataset, ABC):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        if 'ntu' in name:
            self.num_joints = 25
        else:
            self.num_joints = 31
        if not osp.exists(osp.join(root, "raw")):
            os.mkdir(osp.join(root, "raw"))
        super(Dataset, self).__init__(root, transform, pre_transform)
        path = osp.join(self.processed_dir, self.processed_file_names)
        self.data, self.labels = torch.load(path)

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir)]

    @property
    def processed_file_names(self):
        return '{}.pt'.format(self.name)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        fs = self.raw_file_names
        progress_bar = tqdm(fs)
        skeletons, labels = [], []
        for f in progress_bar:
            # Read data from `raw_path`.
            data = Data(...)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data, label = process_skeleton(f, self.num_joints)
            skeletons.append(data)
            labels.append(label)

        torch.save([skeletons, labels],
                   osp.join(self.processed_dir,
                            self.processed_file_names))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
