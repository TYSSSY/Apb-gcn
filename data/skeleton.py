from __future__ import with_statement

import torch

skeleton_adj = torch.tensor([(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                             (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                             (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                             (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                             (22, 23), (23, 8), (24, 25), (25, 12)],
                            dtype=torch.long).transpose(1, 0) - 1

body_info_key = ['bodyID', 'clippedEdges', 'handLeftConfidence',
                 'handLeftState', 'handRightConfidence', 'handRightState',
                 'isRestricted', 'leanX', 'leanY', 'trackingState'
                 ]

joint_info_key = ['x', 'y', 'z',
                  'depthX', 'depthY',
                  'colorX', 'colorY',
                  'orientationW', 'orientationX', 'orientationY', 'orientationZ',
                  'trackingState'
                  ]


def process_skeleton(path, num_joints=25, num_features=3):
    with open(path, 'r') as f:
        lines = f.readlines()
        num_frames = int(lines[0])
        start = 1
        skeleton_sequence = {'numFrame': num_frames}
        # frames, frames_ = torch.zeros((num_frames, num_joints, num_features)), None
        num_persons = int(lines[1])
        offset = int((len(lines) - 1) / num_frames)
        frames = [lines[start + 3 + i * offset:
                        start + 3 + i * offset + num_joints] for i in range(num_frames)]
        frames = process_frames(frames, num_joints, num_features)
        if num_persons == 2:
            frames_ = [lines[start + (i + 1) * offset - num_joints:
                             start + (i + 1) * offset + num_joints] for i in range(num_frames)]
            frames_ = process_frames(frames_, num_joints, num_features)
        else:
            frames_ = []

        return frames, frames_


def process_frames(frames, num_joints, num_features):
    fv = torch.zeros((len(frames), num_joints, num_features))
    for f in frames:
        pass

