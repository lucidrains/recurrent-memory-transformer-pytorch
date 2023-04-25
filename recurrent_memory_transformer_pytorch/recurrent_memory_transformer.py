import math

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack

from recurrent_memory_transformer_pytorch.attend import Attend

# helpers

def exists(val):
    return val is not None

def default(*vals):
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
        rotary_emb = None,
        mask = None
    ):
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        out = self.attend(q, k, v, mask = mask)

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
        use_flash_attn = False,
        ignore_index = -1
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

        self.ignore_index = ignore_index

    def forward(
        self,
        x,
        read_memories = None,
        *,
        mask = None,
        labels = None,
    ):
        b, n, device, mem_length, return_loss = *x.shape, x.device, self.num_memory_tokens, exists(labels)

        pos = torch.arange(n, device = device)

        x = self.token_emb(x)
        x = x + self.pos_emb(pos)

        # prepare read and write memories, as in paper

        write_memories = repeat(self.memory_tokens, 'm d -> b m d', b = b)

        read_memories = default(read_memories, x[:, 0:0])
        read_length = read_memories.shape[-2]

        # concat to main sequence using einop's pack

        x, ps = pack([read_memories, x, write_memories], 'b * d')

        # take care of mask

        if exists(mask):
            mask = F.pad(mask, (read_length, mem_length), value = True)

        # rotary embedding - offset main positions by 10000, and keep all memories at position 0

        pos = pos + 10000
        pos = F.pad(pos, (read_length, mem_length), value = 0)

        rotary_emb = self.rotary_pos_emb(pos)

        # attention and feedforward

        for attn, ff in self.layers:
            x = attn(x, mask = mask, rotary_emb = rotary_emb) + x
            x = ff(x) + x

        # split out memories using unpack

        read_memories, x, write_memories = unpack(x, ps, 'b * d')

        # to logits

        logits = self.to_logits(x)

        if not return_loss:
            return logits, write_memories

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss, write_memories

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

        start_len, seq_len = prime.shape[-1], self.seq_len

        assert length >= start_len

        *past_segments, curr_segment = prime.split(seq_len, dim = -1)

        # catch memories up to the current segment

        for past_segment in past_segments:
            _, memories = self.forward(past_segment, memories)

        # sample for the remaining length

        for ind in range(length - start_len):
            logits, next_memories = self.forward(curr_segment, memories)

            logits = logits[:, -1]

            filtered_logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature)
            sampled = rearrange(sampled, 'b -> b 1')

            curr_segment = torch.cat((curr_segment, sampled), dim = -1)

            if divisible_by(curr_segment.shape[-1] - 1, seq_len):
                memories = next_memories
                past_segment, curr_segment = curr_segment[..., :seq_len], curr_segment[..., -1:]
                past_segments.append(past_segment)

        # add current segment to all segments

        past_segments.append(curr_segment)

        # reconcat all segments

        output = torch.cat(past_segments, dim = -1)

        output = output[:, start_len:]
        return output

    def forward(
        self,
        x,
        memories = None,
        *,
        mask = None,
        return_loss = False
    ):
        seq_len = self.seq_len

        labels = None
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # segment input

        segments = x.split(seq_len, dim = -1)
        total_length = x.shape[-1]
        num_segments = len(segments)
        segment_length_frac = tuple(map(lambda t: t.shape[-1] / total_length, segments))

        # take care of labels

        if exists(labels):
            label_segments = labels.split(seq_len, dim = -1)
        else:
            label_segments = (None,) * num_segments

        # take care of the mask

        if exists(mask):
            mask_segments = mask.split(seq_len, dim = -1)
        else:
            mask_segments = (None,) * num_segments

        # forward and get all outputs (can be either loss or logits)

        outputs = []

        for segment, mask_segment, label_segment in zip(segments, mask_segments, label_segments):
            output, memories = self.transformer(segment, memories, mask = mask_segment, labels = label_segment)
            outputs.append(output)

        if not return_loss:
            outputs = torch.cat(outputs, dim = -2)
            return outputs, memories

        weighted_loss = [(loss * weight) for loss, weight in zip(outputs, segment_length_frac)]
        loss = sum(weighted_loss)

        return loss, memories
