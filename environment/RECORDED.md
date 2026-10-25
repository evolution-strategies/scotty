# Recorded environment

The frozen run records PyTorch `2.14.1+cu130`, BF16 autocast, CUDA, and an NVIDIA
GB10. Training used gradient checkpointing and full encoder fine-tuning.

The source imports Python's standard library plus NumPy, PyTorch, Transformers,
the Laya package, and project helper modules listed in `REPRODUCIBILITY.md`.
Exact versions other than PyTorch were not preserved in the frozen reports, so
this file does not claim unsupported pins.

For paper compilation, use a current TeX Live distribution with Latin Modern,
amsmath, amssymb, array, booktabs, graphicx, url, and hyperref.
