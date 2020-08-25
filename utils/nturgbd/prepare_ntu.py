import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

# from .read_skeleton import read_xyz

import math

import numpy as np
import torch
from torch_geometric.data import Data

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
edge_index = torch.tensor([(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)],dtype = torch.long)


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

                        xyAngle = math.acos(z / magnitude)
                        yzAngle = math.acos(x / magnitude)
                        xzAngle = math.acos(y / magnitude)
                        data[3:, n - 1, j, m] = [xyAngle, yzAngle, xzAngle, magnitude]

                else:
                    pass
    return data



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

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 7, max_frame, num_joint, max_body))

    for i in tqdm(range(len(sample_name))):
        s = sample_name[i]
        data = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data

def pyg_gendata(data_path,
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

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 7, max_frame, num_joint, max_body))

    pyg_x = np.zeros((num_joint, 7, max_frame,max_body, len(sample_name)))

    for i in tqdm(range(len(sample_name))):
        s = sample_name[i]
        data = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        modified_data = np.transpose(data, [2, 0, 1, 3])
        pyg_x[:, :, 0:data.shape[1], :, i] = modified_data
        fp[i, :, 0:data.shape[1], :, :] = data
    pyg_data = Data(x=pyg_x, edge_index=edge_index.t().contiguous())
    print(pyg_data)
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTURGB+D Data Converter.')
    parser.add_argument(
        '--data_path', default='/app/nturgb+d_skeletons')
    parser.add_argument(
        '--ignored_sample_path',
        default='/app/Apb-gcn-master/data/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='/app/Apb-gcn-master/data/NTURGB+D')

    benchmark = ['cs', 'cv']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            pyg_gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)
