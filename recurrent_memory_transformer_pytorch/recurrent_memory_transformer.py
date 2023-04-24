import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

from recurrent_memory_transformer_pytorch.attend import Attend

# helpers

def exists(val):
    return val is not None

def default(vals):
    for val in vals:
        if exists(val):
            return val
    return None

# norms

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Linear(dim_inner, dim)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.attend = Attend(causal = causal, dropout = dropout)

        self.norm = RMSNorm(dim)

        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# classes

class RecurrentMemoryTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        depth,
        num_memory_tokens,
        seq_len,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        assert num_memory_tokens > 0

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)

        self.num_memory_tokens = num_memory_tokens
        self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(
        self,
        x,
        past_memories = None
    ):
        b, n, device, m = *x.shape, x.device, self.num_memory_tokens

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device = device))

        # concat past memories, if needed

        past_length = 0
        if exists(past_memories):
            x = torch.cat((past_memories, x), dim = -2)
            past_length = m

        # concat memories into the future, to be passed onto the next segment

        future_memories = repeat(self.memory_tokens, 'm d -> b m d', b = b)
        x = torch.cat((x, future_memories), dim = -2)

        # attention and feedforward

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        # split out memories

        past_memories, x, memories = x[:, :past_length], x[:, past_length:-m], x[:, -m:]

        # to logits

        return self.to_logits(x), memories
