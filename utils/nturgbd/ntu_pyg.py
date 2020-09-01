import os
import numpy as np

import sys
import pickle
import argparse
from tqdm import tqdm
import math
from torch_geometric.data import Data, Batch
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import DataLoader
import torch
from torch_geometric.data import Dataset


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

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 2
num_joint = 25
max_frame = 300
benchmark='cv'
part='val'


def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {'numBody': int(f.readline()), 'bodyInfo': []}
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((7, seq_info['numFrame'], num_joint, max_body))
    # print(seq_info['frameInfo'])
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:3, n, j, m] = [v['x'], v['y'], v['z']]
                    if n > 0:
                        motionVector = data[:, n - 1, j, m] - data[:, n, j, m]
                        x = motionVector[0]
                        y = motionVector[1]
                        z = motionVector[2]
                        magnitude = math.sqrt(x ** 2 + y ** 2 + z ** 2)
                        if magnitude > 0:
                            xyAngle = math.acos(z / magnitude)
                            yzAngle = math.acos(x / magnitude)
                            xzAngle = math.acos(y / magnitude)
                        if magnitude == 0:
                            xyAngle = 0
                            yzAngle = 0
                            xzAngle = 0
                        data[3:, n - 1, j, m] = [xyAngle, yzAngle, xzAngle, magnitude]

                else:
                    pass
    return data

class NTUDataset(Dataset):
    def __init__(self, root, batch_size, transform=None, pre_transform=None):
        super(NTUDataset, self).__init__(root, transform, pre_transform)
        self.index = 0
        self.batch_size = batch_size

        # self.raw_paths = "/app/nturgb+d_skeletons"
        # self.processed_dir = "/save_path"

    @property
    def raw_file_names(self):
        lst = []
        for f in os.listdir('/app/raw'):
            lst.append(f)
        return lst

    @property
    def processed_file_names(self):
        return ['train.pt', 'validate.pt']

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     # Already downloaded
    #     # Please read https://github.com/shahroudy/NTURGB-D for instructions
    #     pass

    def process(self):
        i = 0
        sample_name = []
        sample_label = []
        training_data = []
        validating_data = []
        self.size = len(self.raw_paths)
        self.data = []
        
        for filename in self.raw_paths:
            # if filename in ignored_samples:
            #     continue
            action_class = int(
                filename[filename.find('A') + 1:filename.find('A') + 4])
            subject_id = int(
                filename[filename.find('P') + 1:filename.find('P') + 4])
            camera_id = int(
                filename[filename.find('C') + 1:filename.find('C') + 4])


            if benchmark == 'cv':
                istraining = (camera_id in training_cameras)
            elif benchmark == 'cs':
                istraining = (subject_id in training_subjects)
            else:
                raise ValueError()

            if part == 'train':
                issample = istraining
            elif part == 'val':
                issample = not (istraining)
            else:
                raise ValueError()

            if issample:
                sample_name.append(filename)
                sample_label.append(action_class - 1)


            pyg_x = np.zeros((num_joint, 7, max_frame, max_body, len(sample_name)))
            pyg_y = np.zeros(len(sample_name))

            for i in tqdm(range(len(sample_name))):
                s = sample_name[i]
                data = read_xyz(
                    os.path.join(filename, s), max_body=max_body, num_joint=num_joint)
                modified_data = np.transpose(data, [2, 0, 1, 3])

                # For saving purposes
                pyg_x[:, :, 0:data.shape[1], :, i] = modified_data
                pyg_y[i] = sample_label[i] 
                
                # For creating batches
                data_pyg = Data(x=pyg_x, edge_index=edge_index.t().contiguous(), y=sample_label[i])
                data_pyg.num_nodes = 25
                self.data.append(data_pyg)

                

            #data = Data(x=pyg_x, edge_index=edge_index.t().contiguous(), y=pyg_y)
            #data.num_nodes = 25

            if self.pre_filter is not None and not self.pre_filter(modified_data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(modified_data)
            
            if issample:
                training_data.append(modified_data)
            else:
                validating_data.append(modified_data)
            i += 1
        torch.save(training_data, os.path.join(self.processed_dir, 'train.pt'))
        torch.save(validating_data, os.path.join(self.processed_dir, 'validate.pt'))

    def len(self):
        return len(self.raw_paths)

    def get(self, idx):
        data = self.data[idx]
        return data

    # def get_batch(self):
    #     batch = []
    #     for i in range(self.index, self.index+self.batch_size):
    #         if i < self.size:
    #             data = self.get(i)
    #             batch.append(data)
    #             self.index += 1
    #         else:
    #             break
    #     if not batch:
    #         return
    #     batch = Batch.from_data_list(batch,  follow_batch=[])
    #     return batch


if __name__ == '__main__':

    # n_batch_size should be a divisor of number of pt files.
    n_batch_size = 2
    ntu_dataset = NTUDataset("/app", batch_size=n_batch_size)


    ntu_dataloader = DataLoader(ntu_dataset, batch_size=n_batch_size, shuffle=False)

    count = 0
    i = 0
    batch = None
    for b in (ntu_dataloader):
        batch = b
        # break
        print('Index', count)
        print(len(batch))
        print(batch)
        count += 1
    # # batch = ntu_dataloader[0]
    # # print(count)