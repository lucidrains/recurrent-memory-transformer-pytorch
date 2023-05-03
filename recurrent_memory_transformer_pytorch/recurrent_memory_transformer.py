import math
from functools import partial
from itertools import zip_longest
from contextlib import nullcontext

from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, pack, unpack

from recurrent_memory_transformer_pytorch.attend import Attend

# constants

Linear = partial(nn.Linear, bias = False)

# helpers

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

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

def token_shift_fn(t, ps):
    read_mem, t, write_mem = unpack(t, ps, 'b * d')
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1), value = 0.)
    t = torch.cat((t, t_shift), dim = -1)
    return torch.cat((read_mem, t, write_mem), dim = -2)

def frac_gradient(t, frac = 1.):
    if frac == 1.:
        return t

    return t * frac + t.detach() * (1. - frac)

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
        Linear(dim, dim_inner * 2, bias = False),
        GEGLU(),
        RMSNorm(dim_inner),
        Linear(dim_inner, dim, bias = False)
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

        self.to_q = Linear(dim, dim_inner)
        self.to_kv = Linear(dim, dim_inner * 2)
        self.to_out = Linear(dim_inner, dim)

    def forward(
        self,
        x,
        rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
        mask = None,
        xl_memories = None
    ):
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        next_xl_memories = torch.stack((k, v))

        if exists(xl_memories):
            kx, vx = xl_memories
            k = torch.cat((kx, k), dim = -2)
            v = torch.cat((vx, v), dim = -2)

            mask = F.pad(mask, (xl_memories.shape[-2], 0), value = True)

        if exists(rotary_emb):
            q_rotary_emb, k_rotary_emb = rotary_emb

            q = apply_rotary_pos_emb(q_rotary_emb, q)
            k = apply_rotary_pos_emb(k_rotary_emb, k)

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), next_xl_memories

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
        abs_pos_emb = True,
        rotary_pos_emb = False,
        token_shift = True,
        use_xl_memories = True,
        xl_mem_len = None,
        enhanced_xl_recurrence = False,     # add simple method for enhancing receptive field of xl memories, from ernie-doc paper
        emb_gradient_frac = 0.1,            # trick from cogview paper that leads to a bit more stability
        memory_not_causal = True,           # flash attention behaves a bit more optimally if causal mask is not explicitly passed in - but if the memories perform better without a causal mask, it is necessary to have this turned on
        norm_write_memories = False,
    ):
        super().__init__()
        self.causal = causal
        self.seq_len = seq_len

        self.emb_gradient_frac = emb_gradient_frac

        assert num_memory_tokens > 0

        self.token_emb = nn.Embedding(num_tokens, dim)

        # positions

        assert any([abs_pos_emb, rotary_pos_emb, token_shift])

        self.pos_emb = nn.Embedding(seq_len, dim) if abs_pos_emb else None

        self.rotary_pos_emb = RotaryEmbedding(dim_head) if rotary_pos_emb else None

        self.maybe_token_shift = token_shift_fn if token_shift else identity

        # memory related

        self.num_memory_tokens = num_memory_tokens

        self.read_memory_emb = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.read_memory_emb, std = 0.02)

        self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))
        nn.init.normal_(self.memory_tokens, std = 0.02)

        self.write_memories_norm = RMSNorm(dim) if norm_write_memories else None

        # xl memories

        xl_mem_len = default(xl_mem_len, seq_len)
        assert xl_mem_len <= seq_len
        self.xl_mem_len = xl_mem_len

        self.use_xl_memories = use_xl_memories
        self.enhanced_xl_recurrence = enhanced_xl_recurrence

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
            nn.Linear(dim, num_tokens)
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
        xl_memories: Optional[List[Tensor]] = None
    ):
        has_xl_memories = exists(xl_memories) and len(xl_memories) > 0

        b, n, device, mem_length, return_loss = *x.shape, x.device, self.num_memory_tokens, exists(labels)

        assert n <= self.seq_len

        pos = torch.arange(n, device = device)

        x = self.token_emb(x)

        # maybe absolute positional embedding

        if exists(self.pos_emb):
            x = x + self.pos_emb(pos)

        # trick from cogview paper

        x = frac_gradient(x, self.emb_gradient_frac)

        # prepare read and write memories, as in paper

        write_memories = self.init_memory(b)

        if exists(read_memories):
            read_mem_length = mem_length
            read_memories = read_memories + self.read_memory_emb
        else:
            read_mem_length = 0
            read_memories = x[:, 0:0]

        # concat to main sequence using einop's pack

        x, ps = pack([read_memories, x, write_memories], 'b * d')

        # take care of mask

        if exists(mask):
            mask = F.pad(mask, (read_mem_length, mem_length), value = True)

        # custom causal mask, if needed

        if self.use_custom_causal_attn_mask:
            causal_mask = torch.ones((n, n), device = device, dtype = torch.bool).tril()

            causal_mask = F.pad(causal_mask, (0, mem_length, read_mem_length, 0), value = False)
            causal_mask = F.pad(causal_mask, (read_mem_length, 0, 0, mem_length), value = True)

            assert not exists(mask)
            mask = rearrange(causal_mask, 'i j -> 1 1 i j')

        # rotary embedding - offset main positions by 10000, and keep all memories at position 0

        rotary_emb = None

        if exists(self.rotary_pos_emb):
            mem_rel_dist = 10000

            q_pos = pos + mem_rel_dist

            if has_xl_memories:
                xl_mem_length = xl_memories[0].shape[-2]
                q_pos += xl_mem_length

            q_pos = F.pad(q_pos, (read_mem_length, mem_length), value = 0)
            q_rotary_emb = self.rotary_pos_emb(q_pos)

            # kind of confusing at the moment
            # but the order of the keys are - [xl memories] [read memories] [main sequence] [ write memories]
            # so the positions are (say xl memory length of 256) - [10001, 10002, 10003 ...] [0, 0, ...] [10256, 10257, ...] [0, 0, ...]

            if has_xl_memories:
                k_pos = torch.arange(xl_mem_length, device = device) + mem_rel_dist
                k_pos = torch.cat((k_pos, q_pos), dim = -1)
                k_rotary_emb = self.rotary_pos_emb(k_pos)
            else:
                k_rotary_emb = q_rotary_emb

            rotary_emb = (q_rotary_emb, k_rotary_emb)

        # maybe token shift function

        shift_fn = partial(self.maybe_token_shift, ps = ps)

        # prepare xl memories

        xl_memories = default(xl_memories, [])
        xl_memories_iter = iter(xl_memories)
        new_xl_memories = []

        if has_xl_memories and self.enhanced_xl_recurrence and len(xl_memories) > 1:  # simply shift all the xl memories down by one, so lower layer gets access to representations from layer above
            xl_memories = [*xl_memories[1:], xl_memories[0]]

        # attention and feedforward

        for attn, ff in self.layers:
            attn_out, xl_memories = attn(shift_fn(x), mask = mask, xl_memories = next(xl_memories_iter, None), rotary_emb = rotary_emb)
            new_xl_memories.append(xl_memories)

            x = x + attn_out

            x = ff(shift_fn(x)) + x

        # whether to return xl memories

        next_xl_memories = None

        if self.use_xl_memories:
            next_xl_memories = list(map(lambda t: torch.detach(t[..., -self.xl_mem_len:, :]), new_xl_memories))

        # split out memories using unpack

        read_memories, x, write_memories = unpack(x, ps, 'b * d')

        # whether to norm the write memories

        if exists(self.write_memories_norm):
            write_memories = self.write_memories_norm(write_memories)

        # to logits

        logits = self.to_logits(x)

        if not return_loss:
            return logits, write_memories, next_xl_memories

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss, write_memories, next_xl_memories

# wrapper to manage many segments

class RecurrentMemoryTransformerWrapper(nn.Module):
    def __init__(
        self,
        transformer: RecurrentMemoryTransformer,
        truncate_at_step = None  # number of steps before detaching memories (truncated bptt). with memory replay checkpointing, there should be no memory issues, but in case of instability, as reported in initial paper
    ):
        super().__init__()
        self.transformer = transformer
        self.seq_len = transformer.seq_len
        self.truncate_at_step = truncate_at_step

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prime,
        *,
        length,
        memories = None,
        xl_memories: Optional[List[Tensor]] = None,
        temperature = 1.,
        filter_thres = 0.9
    ):
        assert self.transformer.causal, 'only autoregressive transformers can generate'

        start_len, seq_len = prime.shape[-1], self.seq_len

        assert length >= start_len

        *past_segments, curr_segment = prime.split(seq_len, dim = -1)

        # catch memories up to the current segment

        for past_segment in past_segments:
            _, memories, xl_memories = self.transformer(past_segment, memories, xl_memories = xl_memories)

        # sample for the remaining length

        for ind in range(length - start_len):
            logits, next_memories, next_xl_memories = self.transformer(curr_segment, memories, xl_memories = xl_memories)

            logits = logits[:, -1]

            filtered_logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature)
            sampled = rearrange(sampled, 'b -> b 1')

            curr_segment = torch.cat((curr_segment, sampled), dim = -1)

            if divisible_by(curr_segment.shape[-1] - 1, seq_len):
                memories = next_memories
                xl_memories = next_xl_memories

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
        xl_memories: Optional[List[Tensor]] = None,
        return_loss = False,
        labels = None,
        truncate_at_step = None,         # if set, this would override the truncate_at_step at init
        memory_replay_backprop = False,  # whether to have the class do the backwards pass memory efficiently
        mrbp_loss_weight = 1.            # if using memory replay backprop with gradient accumulation, scale loss by this factor ex. (1. / <num grad accum steps>)
    ):
        seq_len, truncate_at_step = self.seq_len, default(truncate_at_step, self.truncate_at_step)

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

        # replay buffer for xl memories

        xl_segments = [xl_memories]

        # decide context of forward depending on whether doing memory-replay-backprop

        forward_context = nullcontext if not memory_replay_backprop else torch.no_grad

        # forward and get all outputs (can be either loss or logits)

        logits = []
        losses = []

        for step, (segment, mask_segment, label_segment, loss_weight) in enumerate(zip_longest(segments, mask_segments, label_segments, segment_length_frac)):

            with forward_context():
                output, memories, xl_memories = self.transformer(segment, memories, mask = mask_segment, labels = label_segment)

            if exists(truncate_at_step) and divisible_by(step + 1, truncate_at_step):
                memories = memories.detach()

            replay_buffer.append(memories)

            xl_segments.append(xl_memories)

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
                xl_segments[:-1],
                mask_segments,
                label_segments,
                segment_length_frac,
            ]))

            total_loss = 0.

            for step, segment, segment_memories, segment_xl_memories, mask_segment, label_segment, loss_weight in reversed_inputs:
                is_first = step == 0

                if exists(segment_memories):
                    segment_memories.requires_grad_()

                loss, next_segment_memories, _ = self.transformer(segment, segment_memories, mask = mask_segment, xl_memories = segment_xl_memories, labels = label_segment)

                weighted_loss = loss * loss_weight * mrbp_loss_weight

                weighted_loss.backward(retain_graph = True)

                next_segment_memories.backward(memories_grad)

                total_loss += weighted_loss

                if is_first:
                    continue

                if exists(truncate_at_step) and divisible_by(step, truncate_at_step):
                    memories_grad.zero_()
                else:
                    memories_grad.copy_(segment_memories.grad.data)

            return total_loss

        # return logits if needed

        if not return_loss:
            logits = torch.cat(logits, dim = -2)
            return logits, memories

        # otherwise return losses

        return sum(losses), memories
