<img src="./rmt.png" width="450px"></img>

## Recurrent Memory Transformer - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2207.06881">Recurrent Memory Transformer</a> <a href="https://openreview.net/forum?id=Uynr3iPhksa">(openreview)</a> in Pytorch. They had <a href="https://arxiv.org/abs/2304.11062">a short follow up paper</a> recently that demonstrated it was able to copy information across 1 million tokens at the very least.

## Appreciation

- <a href="https://stability.ai/">Stability</a> and <a href="https://huggingface.co/">🤗 Huggingface</a> for their generous sponsorships to work on and open source cutting edge artificial intelligence research

## Install

```bash
$ pip install recurrent-memory-transformer-pytorch
```

## Usage

```python
import torch
from recurrent_memory_transformer_pytorch import RecurrentMemoryTransformer

model = RecurrentMemoryTransformer(
    num_tokens = 20000,               # number of tokens
    num_memory_tokens = 128,          # number of memory tokens, this will determine the bottleneck for information being passed to the future
    dim = 512,                        # model dimensions
    depth = 6,                        # transformer depth
    causal = True,                    # autoregressive or not
    dim_head = 64,                    # dimension per head
    heads = 8,                        # heads
    seq_len = 1024,                   # sequence length of a segment
    use_flash_attn = True             # whether to use flash attention
)

x = torch.randint(0, 256, (1, 1024))

logits1, mem1 = model(x)        # (1, 1024, 20000), (1, 128, 512)
logits2, mem2 = model(x, mem1)  # (1, 1024, 20000), (1, 128, 512)
logits3, mem3 = model(x, mem2)  # (1, 1024, 20000), (1, 128, 512)

# and so on ...

```

## Todo

- [ ] add an axial attention down the past memories axis as an option
- [ ] for autoregressive, run experiments between future memories being causal masked or not

- [x] add the memory replay backprop technique from memformer paper
- [x] relative positional encoding

## Alternatives

- <a href="https://github.com/lucidrains/block-recurrent-transformer-pytorch">Block Recurrent Transformer</a>

- <a href="https://github.com/lucidrains/memformer">Memformer</a>

## Citations

```bibtex
@inproceedings{bulatov2022recurrent,
  title     = {Recurrent Memory Transformer},
  author    = {Aydar Bulatov and Yuri Kuratov and Mikhail Burtsev},
  booktitle = {Advances in Neural Information Processing Systems},
  editor    = {Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year      = {2022},
  url       = {https://openreview.net/forum?id=Uynr3iPhksa}
}
```

```bibtex
@misc{bulatov2023scaling,
  title     = {Scaling Transformer to 1M tokens and beyond with RMT},
  author    = {Aydar Bulatov and Yuri Kuratov and Mikhail S. Burtsev},
  year      = {2023},
  eprint    = {2304.11062},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
```

```bibtex
@inproceedings{dao2022flashattention,
  title     = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author    = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022}
}
```

```bibtex
@misc{shazeer2020glu,
    title   = {GLU Variants Improve Transformer},
    author  = {Noam Shazeer},
    year    = {2020},
    url     = {https://arxiv.org/abs/2002.05202}
}
```

```bibtex
@misc{su2021roformer,
    title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
    author  = {Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
    year    = {2021},
    eprint  = {2104.09864},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

```bibtex
@inproceedings{Wu2020MemformerAM,
    title   = {Memformer: A Memory-Augmented Transformer for Sequence Modeling},
    author  = {Qingyang Wu and Zhenzhong Lan and Kun Qian and Jing Gu and Alborz Geramifard and Zhou Yu},
    booktitle = {AACL/IJCNLP},
    year    = {2020}
}
```
