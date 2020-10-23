import os
import sys
import pickle
from tqdm import tqdm
import argparse
import numpy as np
import torch
from read_skeleton import *
from torch_geometric.data import Data

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

def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='cv',
            part='val'):

    sample_name,sample_label = dataSample(
                                        data_path,
                                        out_path,
                                        ignored_sample_path,
                                        benchmark,
                                        part)

    pyg_x = np.zeros((num_joint, 7, max_frame, max_body, len(sample_name)))
    pyg_y = np.zeros(len(sample_name))

    for i in tqdm(range(len(sample_name))):
        s = sample_name[i]
        data = read_xyz(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        modified_data = np.transpose(data, [2, 0, 1, 3]) #[max_frame,num_joint,num_features,max_body,clips]
        pyg_x[:, :, 0:data.shape[1], :, i] = modified_data #data
        pyg_y[i] = sample_label[i] #label
        pyg_data = Data(x=pyg_x[:,:,:,:,i], edge_index=edge_index.t().contiguous(), y=pyg_y[i])
        torch.save(pyg_data, os.path.join(out_path, 'data_{}.pt'.format(i)))
    
def dataSample(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='cv',
            part='val'):

    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
        sample_name = []
        sample_label = []

    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
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
    return [sample_name,sample_label]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTURGB+D Data Converter.')
    parser.add_argument(
        '--data_path', default='/home/lawbuntu/Downloads/pytorch_geometric-master/docker/raw')
    parser.add_argument(
        '--ignored_sample_path',
        default=None)
    parser.add_argument('--out_folder', default='/home/lawbuntu/Downloads/pytorch_geometric-master/docker/processed')

    benchmark = ['cs', 'cv']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)
    
