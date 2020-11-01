"""
The code of this file is inspired and extracted from the following link:
https://github.com/lucidrains/performer-pytorch
which implements the Pytorch version of the paper:

@misc{choromanski2020rethinking,
    title   = {Rethinking Attention with Performers},
    author  = {Krzysztof Choromanski and Valerii Likhosherstov and David Dohan and Xingyou Song and Andreea Gane and Tamas Sarlos and Peter Hawkins and Jared Davis and Afroz Mohiuddin and Lukasz Kaiser and David Belanger and Lucy Colwell and Adrian Weller},
    year    = {2020},
    eprint  = {2009.14794},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
"""

from abc import ABC
# from system library
import math
import torch.nn as nn
from functools import partial
# from custom library
from models.third_party.utils import default, gaussian_orthogonal_random_matrix, \
    causal_linear_attention, causal_linear_attention_noncuda, generalized_kernel, \
    softmax_kernel, exists, linear_attention, PreScaleNorm, ReZero, PreLayerNorm, \
    Chunk, FeedForward
from models.third_party.reversible import ReversibleSequence, SequentialSequence
from einops import rearrange


class FastAttention(nn.Module, ABC):
    def __init__(self, dim_heads, nb_features=None, redraw_projection=True, ortho_scaling=0, causal=False,
                 generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.redraw_projection = redraw_projection

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features,
                                         nb_columns=dim_heads, scaling=ortho_scaling, qr_uniform_q=qr_uniform_q)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        if not redraw_projection:
            self.set_projection_matrix()

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = causal_linear_attention
            except ImportError:
                print(
                    'unable to import cuda code for auto-regressive Performer. will default to the memory inefficient '
                    'non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    def set_projection_matrix(self, device):
        projection_matrix = self.create_projection(device=device)
        self.register_buffer('projection_matrix', projection_matrix)

    def forward(self, q, k, v):
        device = q.device

        if self.redraw_projection and not hasattr(self, 'projection_matrix'):
            projection_matrix = self.create_projection(device=device)
        else:
            projection_matrix = self.projection_matrix

        if self.generalized_attention:
            create_kernel = partial(generalized_kernel,
                                    kernel_fn=self.kernel_fn,
                                    projection_matrix=projection_matrix,
                                    device=device)
            q, k = map(create_kernel, (q, k))
        else:
            create_kernel = partial(softmax_kernel,
                                    projection_matrix=projection_matrix,
                                    device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out


class SelfAttention(nn.Module, ABC):
    def __init__(self,
                 dim, heads=8, nb_features=None,
                 causal=True, redraw_projection=True,
                 generalized_attention=False,
                 kernel_fn=nn.ReLU(), qr_uniform_q=False, dropout=0.):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        self.fast_attention = FastAttention(dim // heads, nb_features, redraw_projection, causal=causal,
                                            generalized_attention=generalized_attention, kernel_fn=kernel_fn,
                                            qr_uniform_q=qr_uniform_q)

        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        if exists(mask):
            mask = mask[:, None, :, None]
            k.masked_fill_(~mask, 0)

        out = self.fast_attention(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class Performer(nn.Module, ABC):
    def __init__(self, dim, depth, heads,
                 causal=False, ff_mult=4,
                 nb_features=None, reversible=False, ff_chunks=1,
                 generalized_attention=False, kernel_fn=nn.ReLU(),
                 qr_uniform_q=False, use_scalenorm=False,
                 use_rezero=False, ff_glu=False, ff_dropout=0., attn_dropout=0.):
        super().__init__()
        layers = nn.ModuleList([])

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _ in range(depth):
            layers.append(nn.ModuleList([
                wrapper_fn(SelfAttention(dim, causal=causal, heads=heads, nb_features=nb_features,
                                         generalized_attention=generalized_attention, kernel_fn=kernel_fn,
                                         qr_uniform_q=qr_uniform_q, dropout=attn_dropout)),
                wrapper_fn(
                    Chunk(ff_chunks, FeedForward(dim, mult=ff_mult, dropout=ff_dropout, glu=ff_glu), along_dim=1))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {'mask': route_attn}
        self.net = execute_type(layers, args_route={**attn_route_map})

    def forward(self, x, **kwargs):
        return self.net(x, **kwargs)
