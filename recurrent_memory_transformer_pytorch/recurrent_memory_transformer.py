from __future__ import annotations

import math
from functools import partial
from itertools import zip_longest
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from recurrent_memory_transformer_pytorch.attend import Attend

from hyper_connections import get_init_and_expand_reduce_stream_functions

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

def frac_gradient(t, frac = 1.):
    if frac == 1.:
        return t

    return t * frac + t.detach() * (1. - frac)

# rotary embedding

class RotaryEmbedding(Module):
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

# feedforward

class GEGLU(Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_inner = int(dim * mult * 2 / 3)

    return nn.Sequential(
        nn.RMSNorm(dim),
        Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        Linear(dim_inner, dim)
    )

# attention

class Attention(Module):
    def __init__(
        self,
        *,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        accept_value_residual = False,
        use_flash_attn = False,
        use_custom_causal_attn_mask = False
    ):
        super().__init__()
        self.norm = nn.RMSNorm(dim)

        dim_inner = dim_head * heads
        self.heads = heads

        self.attend = Attend(
            causal = causal and not use_custom_causal_attn_mask,
            dropout = dropout,
            use_flash = use_flash_attn
        )

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_q = Linear(dim, dim_inner)
        self.to_kv = Linear(dim, dim_inner * 2)
        self.to_out = Linear(dim_inner, dim)

        # learned value residual mixing

        self.learned_value_residual_mix = None

        if accept_value_residual:
            self.learned_value_residual_mix = nn.Sequential(
                Linear(dim, heads),
                Rearrange('b n h -> b h n 1'),
                nn.Sigmoid()
            )

    def forward(
        self,
        x,
        rotary_emb: tuple[Tensor, Tensor] | None = None,
        mask = None,
        xl_memories = None,
        value_residual = None
    ):
        assert not (exists(value_residual) ^ exists(self.learned_value_residual_mix))

        h = self.heads
        x = self.norm(x)

        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # handle value residual

        orig_v = v

        if exists(self.learned_value_residual_mix):
            mix = self.learned_value_residual_mix(x)
            v = v.lerp(value_residual, mix)

        # add a null key / value
        # to protect against an entirely masked out sequence
        # as well as giving attention ability to attend to nothing

        nk, nv = map(lambda t: repeat(t, 'h d -> b h 1 d', b = x.shape[0]), self.null_kv)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)

        # manage memories

        next_xl_memories = torch.stack((k, v))

        if exists(xl_memories):
            kx, vx = xl_memories
            k = torch.cat((kx, k), dim = -2)
            v = torch.cat((vx, v), dim = -2)

            if exists(mask):
                mask = F.pad(mask, (xl_memories.shape[-2], 0), value = True)

        if exists(rotary_emb):
            q_rotary_emb, k_rotary_emb = rotary_emb

            q = apply_rotary_pos_emb(q_rotary_emb, q)
            k = apply_rotary_pos_emb(k_rotary_emb, k)

        out = self.attend(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), next_xl_memories, orig_v

# transformer

class RecurrentMemoryTransformer(Module):
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
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_flash_attn = False,
        ignore_index = -1,
        abs_pos_emb = True,
        rotary_pos_emb = False,
        use_xl_memories = True,
        xl_mem_len = None,
        enhanced_xl_recurrence = False,      # add simple method for enhancing receptive field of xl memories, from ernie-doc paper
        emb_gradient_frac = 0.1,             # trick from cogview paper that leads to a bit more stability
        memory_not_causal = True,            # flash attention behaves a bit more optimally if causal mask is not explicitly passed in - but if the memories perform better without a causal mask, it is necessary to have this turned on
        add_write_to_next_write_mem = False, # add the write memories of previous step to the next write step - thanks to @IcarusWizard for pointing out this discrepancy
        next_write_mem_stop_grad = True,     # whether to stop gradient of previous read memory -> next write memory
        always_have_read_memories = True,    # whether to always have read memories, even on the first step, so to make the model onnx-able
        num_residual_streams = 4             # number of residual streams for hyper connections
    ):
        super().__init__()
        self.causal = causal
        self.seq_len = seq_len

        self.emb_gradient_frac = emb_gradient_frac

        assert num_memory_tokens > 0

        self.token_emb = nn.Embedding(num_tokens, dim)

        # positions

        assert any([abs_pos_emb, rotary_pos_emb])

        self.pos_emb = nn.Embedding(seq_len, dim) if abs_pos_emb else None

        self.rotary_pos_emb = RotaryEmbedding(dim_head) if rotary_pos_emb else None

        # memory related

        self.num_memory_tokens = num_memory_tokens

        self.read_memory_emb = nn.Parameter(torch.zeros(num_memory_tokens, dim))
        nn.init.normal_(self.read_memory_emb, std = 0.02)

        self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))
        nn.init.normal_(self.memory_tokens, std = 0.02)

        # xl memories

        xl_mem_len = default(xl_mem_len, seq_len)
        assert xl_mem_len <= seq_len
        self.xl_mem_len = xl_mem_len

        self.use_xl_memories = use_xl_memories
        self.enhanced_xl_recurrence = enhanced_xl_recurrence

        # hyper connections

        init_hyper_conn, self.expand_streams, self.reduce_streams = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        # layers

        self.layers = ModuleList([])

        for layer_index in range(depth):
            is_first = layer_index == 0

            self.layers.append(ModuleList([
                init_hyper_conn(dim = dim, branch = Attention(
                    dim = dim,
                    dim_head = dim_head,
                    causal = causal,
                    heads = heads,
                    use_flash_attn = use_flash_attn,
                    accept_value_residual = not is_first,
                    use_custom_causal_attn_mask = memory_not_causal,
                    dropout = attn_dropout
                )),
                init_hyper_conn(dim = dim, branch = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)),
            ]))

        self.norm = nn.RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)

        self.ignore_index = ignore_index

        # whether to use custom attention mask if causal and memory should not be causal

        self.use_custom_causal_attn_mask = causal and memory_not_causal

        # in the paper, they actually also use the previous write memories for the next write memories

        self.add_write_to_next_write_mem = add_write_to_next_write_mem
        self.next_write_mem_stop_grad = next_write_mem_stop_grad

        # allow for attending to raw read memory positional embeddings on first step
        # hack to make it onnx-able and should not hurt

        self.always_have_read_memories = always_have_read_memories

    def init_memory(self, batch):
        return repeat(self.memory_tokens, 'm d -> b m d', b = batch)

    def forward(
        self,
        x,
        read_memories = None,
        *,
        mask = None,
        labels = None,
        xl_memories: list[Tensor] | None = None,
        mask_out_read_memories = False   # in the case one is passing in 0s for read memories, for onnx-able model
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

        # prepare write memories, as in paper

        write_memories = self.init_memory(b)

        if exists(read_memories) and self.add_write_to_next_write_mem:
            maybe_detach = torch.detach if self.next_write_mem_stop_grad else identity
            write_memories = write_memories + maybe_detach(read_memories)

        # prepare read memories

        if exists(read_memories):
            if read_memories.ndim == 2:
                read_memories = repeat(read_memories, 'n d -> b n d', b = b)

            read_mem_length = mem_length
            read_memories = read_memories + self.read_memory_emb
        elif self.always_have_read_memories:
            read_mem_length = mem_length
            read_memories = repeat(self.read_memory_emb, 'n d -> b n d', b = b)
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

            causal_mask = rearrange(causal_mask, 'i j -> 1 1 i j')

            if exists(mask):
                mask = rearrange(mask, 'b j -> b 1 1 j')
                mask = mask & causal_mask
            else:
                mask = causal_mask

        # masking out read memories, either for passing in 0s for read memories on first step, or if you are doing some regularization game on the memories

        if read_mem_length > 0 and mask_out_read_memories:
            read_mem_mask = torch.arange(x.shape[-2], device = device) < read_mem_length

            if exists(mask):
                mask = mask & ~read_mem_mask
            else:
                mask = read_mem_mask

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
            else:
                k_pos = q_pos

            # account for null key / value

            k_pos = F.pad(k_pos, (1, 0), value = mem_rel_dist - 1) # give a null memory token, to allow for attending to nothing

            k_rotary_emb = self.rotary_pos_emb(k_pos)

            rotary_emb = (q_rotary_emb, k_rotary_emb)

        # prepare xl memories

        xl_memories = default(xl_memories, [])
        xl_memories_iter = iter(xl_memories)
        new_xl_memories = []

        if has_xl_memories and self.enhanced_xl_recurrence and len(xl_memories) > 1:  # simply shift all the xl memories down by one, so lower layer gets access to representations from layer above
            xl_memories = [*xl_memories[1:], xl_memories[0]]

        # value residual

        value_residual = None

        # expand streams for hyper connections

        x = self.expand_streams(x)

        # attention and feedforward

        for attn, ff in self.layers:
            x, xl_memories, attn_values = attn(x, mask = mask, xl_memories = next(xl_memories_iter, None), rotary_emb = rotary_emb, value_residual = value_residual)

            value_residual = default(value_residual, attn_values)
            new_xl_memories.append(xl_memories)

            x = ff(x)

        # reduce streams for hyper connections

        x = self.reduce_streams(x)

        # final norm

        x = self.norm(x)

        # whether to return xl memories

        next_xl_memories = None

        if self.use_xl_memories:
            next_xl_memories = list(map(lambda t: torch.detach(t[..., -self.xl_mem_len:, :]), new_xl_memories))

        # split out memories using unpack

        read_memories, x, write_memories = unpack(x, ps, 'b * d')

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

class RecurrentMemoryTransformerWrapper(Module):
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
        xl_memories: list[Tensor] | None = None,
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
        xl_memories: list[Tensor] | None = None,
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
