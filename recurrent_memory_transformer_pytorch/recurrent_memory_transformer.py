import math

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

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

def divisible_by(numer, denom):
    return (numer % denom) == 0

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# rotary embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 32768):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions):
        freqs = torch.einsum('i , j -> i j', positions, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())

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
        dropout = 0.,
        use_flash_attn = False
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.heads = heads

        self.attend = Attend(causal = causal, dropout = dropout, use_flash = use_flash_attn)

        self.norm = RMSNorm(dim)

        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_kv = nn.Linear(dim, dim_inner * 2, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x,
        rotary_emb = None
    ):
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer

class RecurrentMemoryTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        depth,
        num_memory_tokens,
        seq_len,
        causal = True,        
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        use_flash_attn = False
    ):
        super().__init__()
        self.causal = causal
        self.seq_len = seq_len

        assert num_memory_tokens > 0

        self.token_emb = nn.Embedding(num_tokens, dim)

        # positions

        self.pos_emb = nn.Embedding(seq_len, dim)
        self.rotary_pos_emb = RotaryEmbedding(dim_head)

        # memory related

        self.num_memory_tokens = num_memory_tokens

        self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))
        nn.init.normal_(self.memory_tokens, std = 0.02)

        # layers

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim = dim,
                    dim_head = dim_head,
                    causal = causal,
                    heads = heads,
                    use_flash_attn = use_flash_attn
                ),
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
        b, n, device, mem_length = *x.shape, x.device, self.num_memory_tokens

        pos = torch.arange(n, device = device)

        x = self.token_emb(x)
        x = x + self.pos_emb(pos)

        # concat memories into the future, to be passed onto the next segment

        future_memories = repeat(self.memory_tokens, 'm d -> b m d', b = b)
        x = torch.cat((x, future_memories), dim = -2)

        # concat past memories, if needed

        past_length = 0

        if exists(past_memories):
            x = torch.cat((past_memories, x), dim = -2)
            past_length = mem_length

        # rotary embedding - offset main positions by 10000, and keep all memories at position 0

        pos = pos + 10000
        pos = F.pad(pos, (past_length, mem_length), value = 0)

        rotary_emb = self.rotary_pos_emb(pos)

        # attention and feedforward

        for attn, ff in self.layers:
            x = attn(x, rotary_emb = rotary_emb) + x
            x = ff(x) + x

        # split out memories

        past_memories, x, memories = x[:, :past_length], x[:, past_length:-mem_length], x[:, -mem_length:]

        # to logits

        return self.to_logits(x), memories

# wrapper to manage many segments

class RecurrentMemoryTransformerWrapper(nn.Module):
    def __init__(
        self,
        transformer: RecurrentMemoryTransformer
    ):
        super().__init__()
        self.transformer = transformer
        self.seq_len = transformer.seq_len

    @torch.no_grad()
    def generate(
        self,
        prime,
        *,
        length,
        memories = None,
        temperature = 1.,
        filter_thres = 0.9,
    ):
        assert self.transformer.causal, 'only autoregressive transformers can generate'

        start_len = prime.shape[-1]

        output = prime

        for ind in range(length - start_len):

            logits, next_memories = self.forward(output, memories)

            logits = logits[:, -1]

            filtered_logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature)
            sampled = rearrange(sampled, 'b -> b 1')

            output = torch.cat((output, sampled), dim = -1)

            if divisible_by(output.shape[-1] - 1, self.seq_len):
                memories = next_memories

        output = output[:, start_len:]
        return output

    def forward(
        self,
        x,
        memories = None,
        return_loss = False
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        segments = x.split(self.seq_len, dim = -1)

        all_logits = []

        for segment in segments:
            logits, memories = self.transformer(segment, memories)
            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim = -2)

        if return_loss:
            all_logits = rearrange(all_logits, 'b n c -> b c n')
            return F.cross_entropy(all_logits, labels)

        return all_logits, memories
