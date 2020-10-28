from __future__ import with_statement

import torch

skeleton_adj = torch.tensor([(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                             (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                             (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                             (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)],
                            dtype=torch.long).transpose(1, 0) - 1

body_info_key = ['bodyID', 'clippedEdges', 'handLeftConfidence',
                 'handLeftState', 'handRightConfidence', 'handRightState',
                 'isRestricted', 'leanX', 'leanY', 'trackingState'
                 ]

joint_info_key = ['x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                  'orientationW', 'orientationX', 'orientationY',
                  'orientationZ', 'trackingState'
                  ]


def process_skeleton_file(path, num_features):
    skeleton_sequence = {}
    with open(path, 'r') as f:
        skeleton_sequence['numFrame'] = int(f.readline())
        