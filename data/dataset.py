from abc import ABC

import os
import os.path as osp
from time import sleep

import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
from data.skeleton import process_skeleton, skeleton_parts


class SkeletonDataset(Dataset, ABC):
    def __init__(self,
                 root,
                 name,
                 use_motion_vector=True,
                 transform=None,
                 pre_transform=None):
        self.name = name
        if 'ntu' in name:
            self.num_joints = 25
        elif 'hdm' in name:
            self.num_joints = 31
        else:
            raise ValueError(self.name + " not supported")

        self.skeleton_ = skeleton_parts()
        self.use_motion_vector = use_motion_vector

        if not osp.exists(osp.join(root, "raw")):
            os.mkdir(osp.join(root, "raw"))
        super(SkeletonDataset, self).__init__(root, transform, pre_transform)
        path = osp.join(self.processed_dir, self.processed_file_names)
        self.data, self.labels = torch.load(path)

    @property
    def raw_file_names(self):
        fp = lambda x: osp.join(self.root, 'raw', x)
        return [fp(f) for f in os.listdir(self.raw_dir)]  # if osp.isfile(fp(f))]

    @property
    def processed_file_names(self):
        return '{}.pt'.format(self.name)

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        progress_bar = tqdm(self.raw_file_names)
        skeletons, labels = [], torch.zeros(len(self.raw_file_names))
        i = 0
        for f in progress_bar:
            # Read data from `raw_path`.
            sleep(1e-4)
            progress_bar.set_description("processing %s" % f)
            data, label = process_skeleton(f,
                                           self.num_joints,
                                           use_motion_vector=self.use_motion_vector)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data = Data(x=data)  # , edge_index=self.skeleton_)
            skeletons.append(data)
            labels[i] = label
            i += 1

        torch.save([skeletons, labels],
                   osp.join(self.processed_dir,
                            self.processed_file_names))

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


def test():
    from torch_geometric.data import DataLoader
    ds = SkeletonDataset(root='../dataset',
                         name='ntu')
    loader = DataLoader(ds, batch_size=4)
    for b in loader:
        print(b.batch)


if __name__ == "__main__":
    test()
