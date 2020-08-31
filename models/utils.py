import numpy as np
import math
import torch
from torch_sparse import spspmm


def conv_init(module):
    n = module.out_channels
    for k in module.kernel_size:
        n *= k
    module.weight.data.normal_(0, math.sqrt(2. / n))


def power_adj(adj, dim, p):
    val = torch.ones(adj.shape[1])
    ic, vc = spspmm(adj, val, adj, val, dim, dim, dim)
    if p > 2:
        for i in range(p - 2):
            ic, vc = spspmm(ic, vc, adj, val, dim, dim, dim)
    return ic


