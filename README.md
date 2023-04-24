<img src="./rmt.png" width="450px"></img>

## Recurrent Memory Transformer - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2207.06881">Recurrent Memory Transformer</a> in Pytorch. They had <a href="https://arxiv.org/abs/2304.11062">a short follow up paper</a> recently that demonstrated it was able to copy information across 1 million tokens at the very least.

## Appreciation

- <a href="https://stability.ai/">Stability</a> and <a href="https://huggingface.co/">🤗 Huggingface</a> for their generous sponsorships to work on and open source cutting edge artificial intelligence research

## Alternatives

- <a href="https://github.com/lucidrains/block-recurrent-transformer-pytorch">Block Recurrent Transformer</a>

- <a href="https://github.com/lucidrains/memformer">Memformer</a>

## Citations

```bibtex
@inproceedings{bulatov2022recurrent,
  title   = {Recurrent Memory Transformer},
  author  = {Aydar Bulatov and Yuri Kuratov and Mikhail Burtsev},
  booktitle = {Advances in Neural Information Processing Systems},
  editor  = {Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year    = {2022},
  url     = {https://openreview.net/forum?id=Uynr3iPhksa}
}
```

```bibtex
@misc{bulatov2023scaling,
  title   = {Scaling Transformer to 1M tokens and beyond with RMT}, 
  author  = {Aydar Bulatov and Yuri Kuratov and Mikhail S. Burtsev},
  year    = {2023},
  eprint  = {2304.11062},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
```

```bibtex
@inproceedings{dao2022flashattention,
  title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle = {Advances in Neural Information Processing Systems},
  year    = {2022}
}
```
