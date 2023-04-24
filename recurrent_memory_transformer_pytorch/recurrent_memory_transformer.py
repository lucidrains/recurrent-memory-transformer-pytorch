import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange

from recurrent_memory_transformer_pytorch.attend import Attend

# helpers

def exists(val):
    return val is not None

# classes

class RecurrentMemoryTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_memory_tokens,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()

    def forward(self, x):
        return x
