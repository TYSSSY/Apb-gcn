import numpy as np
import math
from torch_geometric.nn import GATConv


def glorot():
    return


def conv_init(module):
    n = module.out_channels
    for k in module.kernel_size:
        n *= k
    module.weight.data.normal_(0, math.sqrt(2. / n))


