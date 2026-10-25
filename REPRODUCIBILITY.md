# Reproducibility guide

## Verified without external artifacts

These commands inspect only files in this repository:

```bash
python3 tools/verify_repository.py
python3 -m unittest discover -s tests -v
python3 tools/show_results.py
```

The repository verifier checks the six-page paper PDF's SHA-256 against
[`provenance/artifacts.json`](provenance/artifacts.json). Paper source is not
included in this repository.

## Source setup

The eight scripts in the frozen snapshot and their four project helper modules
are in [`reproduction/scripts/`](reproduction/scripts/). The Apache-2.0 Laya
checkout actually selected by the launch wrappers is in
[`reproduction/vendor/laya_checkout/`](reproduction/vendor/laya_checkout/).
Machine-specific root assignments were replaced by the required
`SCOTTY_REPRO_ROOT` environment variable; provenance and before/after hashes are
recorded in [`provenance/artifacts.json`](provenance/artifacts.json).

From the repository root:

```bash
export SCOTTY_REPRO_ROOT=/absolute/path/to/reproduction-root
export PYTHONPATH="$PWD/reproduction/vendor/laya_checkout:$PWD/reproduction/scripts"
python3 -m compileall -q reproduction/scripts reproduction/vendor/laya_checkout/laya
```

`SCOTTY_REPRO_ROOT` is not this Git repository by default. It is the external
artifact tree expected by the historical scripts.

## Full training and evaluation: external artifacts required

The source dependency closure is now included, but execution still needs:

1. The Beam-generated and Typed Decisions split files whose hashes are recorded
   in `provenance/artifacts.json`.
2. For evaluation/inference, the frozen checkpoint plus tokenizer with the
   recorded model hash.
3. A CUDA-capable environment compatible with the recorded run.

Then invoke the appropriate script with `--help` or inspect its top-level path
constants before running. Training is intentionally not run by the repository
verification target.

### What was tested

- Repository and artifact hash verification.
- Python syntax parsing/compilation for every included script and vendored Laya
  module.
- Aggregate-result and split invariants.
- Paper PDF hash, page count, and visual inspection performed during packaging.

Checkpoint loading, inference, training, and full evaluation were **not** run
during packaging because the weights and raw rows are intentionally excluded
and training was out of scope. The Python files are therefore verified source,
not a claim of an end-to-end runnable release from a clean checkout.

## Recorded environment

- Python: `3.12.3` (recovered project virtual environment)
- PyTorch: `2.14.1+cu130` (frozen run report)
- Transformers: `5.19.0` (frozen encoder configuration)
- accelerator: NVIDIA GB10
- precision: BF16 autocast
- base model revision: `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`
- seed: `0`

The full current project-venv snapshot is preserved in
[`environment/recovered-project-venv.txt`](environment/recovered-project-venv.txt).
It was captured after the run and is not immutable proof that every secondary
package had the same version during training. See
[`environment/RECORDED.md`](environment/RECORDED.md) for the evidence boundary.
