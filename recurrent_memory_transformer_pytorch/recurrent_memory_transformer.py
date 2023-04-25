import math
from itertools import zip_longest
from contextlib import nullcontext

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
        use_flash_attn = False,
        use_custom_causal_attn_mask = False
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.heads = heads

        self.attend = Attend(
            causal = causal and not use_custom_causal_attn_mask,
            dropout = dropout,
            use_flash = use_flash_attn
        )

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
        ignore_index = -1,
        memory_not_causal = True # flash attention behaves a bit more optimally if causal mask is not explicitly passed in - but if the memories perform better without a causal mask, it is necessary to have this turned on
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
                    use_flash_attn = use_flash_attn,
                    use_custom_causal_attn_mask = memory_not_causal
                ),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

        self.ignore_index = ignore_index

        # whether to use custom attention mask if causal and memory should not be causal

        self.use_custom_causal_attn_mask = causal and memory_not_causal

    def init_memory(self, batch):
        return repeat(self.memory_tokens, 'm d -> b m d', b = batch)

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

        write_memories = self.init_memory(b)

        read_memories = default(read_memories, x[:, 0:0])
        read_mem_length = read_memories.shape[-2]

        # concat to main sequence using einop's pack

        x, ps = pack([read_memories, x, write_memories], 'b * d')

        # take care of mask

        if exists(mask):
            mask = F.pad(mask, (read_mem_length, mem_length), value = True)

        # custom causal mask, if needed

        if self.use_custom_causal_attn_mask:
            causal_mask = torch.ones((n, n), device = device, dtype = torch.bool).tril()
            causal_mask = F.pad(causal_mask, (read_mem_length, mem_length) * 2, value = True)

            assert not exists(mask)
            mask = rearrange(causal_mask, 'i j -> 1 1 i j')

        # rotary embedding - offset main positions by 10000, and keep all memories at position 0

        pos = pos + 10000
        pos = F.pad(pos, (read_mem_length, mem_length), value = 0)

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
    @eval_decorator
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
        return_loss = False,
        labels = None,
        memory_replay_backprop = False,  # whether to have the class do the backwards pass memory efficiently
        mrbp_loss_weight = 1.            # if using memory replay backprop with gradient accumulation, scale loss by this factor ex. (1. / <num grad accum steps>)
    ):
        seq_len = self.seq_len

        labels = None
        if (return_loss or memory_replay_backprop) and not exists(labels):
            x, labels = x[:, :-1], x[:, 1:]

        # segment input

        segments = x.split(seq_len, dim = -1)
        total_length = x.shape[-1]
        num_segments = len(segments)
        segment_length_frac = tuple(map(lambda t: t.shape[-1] / total_length, segments))

        # default values

        label_segments = mask_segments = (None,)

        # take care of labels

        if exists(labels):
            label_segments = labels.split(seq_len, dim = -1)

        # take care of the mask

        if exists(mask):
            mask_segments = mask.split(seq_len, dim = -1)

        # keep replay buffer

        replay_buffer = [memories]

        # decide context of forward depending on whether doing memory-replay-backprop

        forward_context = nullcontext if not memory_replay_backprop else torch.no_grad

        # forward and get all outputs (can be either loss or logits)

        logits = []
        losses = []

        for segment, mask_segment, label_segment, loss_weight in zip_longest(segments, mask_segments, label_segments, segment_length_frac):

            with forward_context():
                output, memories = self.transformer(segment, memories, mask = mask_segment, labels = label_segment)

            replay_buffer.append(memories)

            if return_loss:
                losses.append(output * loss_weight)
            else:
                logits.append(output)

        # whether to do memory replay backpropagation

        # https://arxiv.org/abs/2010.06891
        # algorithm 1

        if memory_replay_backprop:
            memories_grad = torch.zeros_like(replay_buffer[-1])

            reversed_inputs = zip_longest(*map(reversed, [
                range(num_segments),
                segments,
                replay_buffer[:-1],
                mask_segments,
                label_segments,
                segment_length_frac,
            ]))

            total_loss = 0.

            for i, segment, segment_memories, mask_segment, label_segment, loss_weight in reversed_inputs:
                is_first = i == 0

                if exists(segment_memories):
                    segment_memories.requires_grad_()

                loss, next_segment_memories = self.transformer(segment, segment_memories, mask = mask_segment, labels = label_segment)

                weighted_loss = loss * loss_weight * mrbp_loss_weight

                weighted_loss.backward(retain_graph = True)

                next_segment_memories.backward(memories_grad)

                total_loss += weighted_loss

                if is_first:
                    continue

                memories_grad.copy_(segment_memories.grad.data)

            return total_loss

        # return logits if needed

        if not return_loss:
            logits = torch.cat(logits, dim = -2)
            return logits, memories

        # otherwise return losses

        return sum(losses), memories
