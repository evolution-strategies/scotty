# Environment evidence

## Frozen evidence

The frozen run records PyTorch `2.14.1+cu130`, BF16 autocast, CUDA, and an NVIDIA
GB10. The frozen encoder configuration records Transformers `5.19.0`. Training
used gradient checkpointing and full encoder fine-tuning.

## Recovered project virtual environment

On 25 October 2026, the project-local virtual environment reported:

- Python `3.12.3` on Linux/aarch64 with glibc 2.39;
- NumPy `2.5.3`;
- PyTorch `2.14.1+cu130`;
- Transformers `5.19.0`;
- Safetensors `0.8.0`;
- Hugging Face Hub `1.33.0`;
- Tokenizers `0.23.3`;
- CUDA runtime `13.0`, cuDNN `9.24.0`, NVIDIA GB10.

The complete recovered `pip freeze` is in `recovered-project-venv.txt`. It is a
post-run snapshot, not a lock file created at training time. The installed
`laya==0.4.2` entry was not the source selected by the launch wrappers: their
`PYTHONPATH` placed the vendored checkout at commit
`eba3b0367643498e478ed02e870fa53737da5061` first.

Exact versions for secondary packages were not preserved in the frozen run
reports. Consequently, bit-for-bit environment reconstruction remains an
explicit gap even though the later project environment is documented.
