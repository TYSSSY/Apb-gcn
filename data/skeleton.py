from __future__ import with_statement

import torch
from utils.linalg import power_adj


def skeleton_parts(num_joints=25):
    sk_adj = torch.tensor([
        [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24],
        [1, 20, 20,  2, 20,  4,  5,  6, 20,  8,  9, 10,  0, 12, 13, 14,  0, 16, 17, 18, 22,  7, 24, 11]])
    return torch.cat([sk_adj,
                      power_adj(sk_adj, max(num_joints, max(sk_adj[1]) + 1), 2),
                      power_adj(sk_adj, max(num_joints, max(sk_adj[1]) + 1), 3)], dim=1)


def process_skeleton(path, num_joints=25, num_features=3):
    import os.path as osp
    t = osp.split(path)[-1][-12:-9]
    with open(path, 'r') as f:
        lines = f.readlines()
        num_frames = int(lines[0])
        start = 1
        num_persons = int(lines[1])
        offset = int((len(lines) - 1) / num_frames)
        frames = [lines[start + 3 + i * offset:
                        start + 3 + i * offset + num_joints] for i in range(num_frames)]
        frames = process_frames(frames, num_joints, num_features)
        if num_persons == 2:
            frames_ = [lines[start + (i + 1) * offset - num_joints:
                             start + (i + 1) * offset + num_joints] for i in range(num_frames)]
            frames_ = process_frames(frames_, num_joints, num_features)
            frames = torch.cat([frames, frames_], dim=0)
        return frames, int(t)


def process_frames(frames, num_joints, num_features=3):
    fv = torch.zeros((len(frames), num_joints, num_features))
    for i in range(len(frames)):
        f = frames[i]
        for j in range(num_joints):
            vs = [float(n) for n in f[j].split()]
            fv[i, j, :] = torch.tensor(vs[0: num_features])
    return fv


