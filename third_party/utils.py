# Helper functions
import math
from abc import ABC

import torch
import torch.nn as nn


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.qr(unstructured_block.cpu(), some=True)
    q, r = map(lambda t: t.to(device), (q, r))

    # proposed by @Parskatt
    # to make sure Q is uniform 
    # https://arxiv.org/pdf/math-ph/0609050.pdf
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows,
                                      nb_columns,
                                      scaling=0,
                                      qr_uniform_q=False,
                                      device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns,
                                    qr_uniform_q=qr_uniform_q,
                                    device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


def linear_attention(q, k, v):
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k.sum(dim=-2))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out


# efficient causal linear attention, created by EPFL
# installation of fast_transformers:
# pip install --user pytorch-fast-transformers
def causal_linear_attention(q, k, v):
    from fast_transformers.causal_product import CausalDotProduct
    return CausalDotProduct.apply(q, k, v)


# inefficient causal linear attention, without cuda code, for reader's reference
# not being used
def causal_linear_attention_noncuda(q, k, v):
    k_cumsum = k.cumsum(dim=-2)
    context = torch.einsum('...nd,...ne->...nde', k, v)
    context = context.cumsum(dim=-3)
    context /= k_cumsum.unsqueeze(dim=-1)
    out = torch.einsum('...nde,...nd->...ne', context, q)
    return out


def generalized_kernel(data, *, projection_matrix,
                       kernel_fn=nn.ReLU(), kernel_epsilon=0.001,
                       normalize_data=True,
                       device=None):
    if normalize_data:
        data_normalizer = 1.0 / (data.shape[-1] ** 0.25)
    else:
        data_normalizer = 1.0

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    data_mod_shape = data.shape[0:len(data.shape) - 2] + projection_matrix.shape
    data_thick_random_matrix = torch.zeros(data_mod_shape, device=device) + projection_matrix

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), data_thick_random_matrix)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime


def softmax_kernel(data, *, projection_matrix, is_query,
                   normalize_data=True, eps=1e-4, device=None):
    if normalize_data:
        data_normalizer = 1.0 / (data.shape[-1] ** 0.25)
    else:
        data_normalizer = 1.0

    ratio = 1.0 / (projection_matrix.shape[0] ** 0.5)

    data_mod_shape = data.shape[:(len(data.shape) - 2)] + projection_matrix.shape
    data_thick_random_matrix = torch.zeros(data_mod_shape, device=device) + projection_matrix

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), data_thick_random_matrix)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data -
                          torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash


# Helper Classes
class ReZero(nn.Module, ABC):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g


class PreScaleNorm(nn.Module, ABC):
    def __init__(self, dim, fn, eps=1e-5):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x, **kwargs):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.g
        return self.fn(x, **kwargs)


class PreLayerNorm(nn.Module, ABC):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Chunk(nn.Module, ABC):
    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)


class FeedForward(nn.Module, ABC):
    def __init__(self, dim, mult=4, dropout=0., activation=None, glu=False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x
