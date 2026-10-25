# Reproducibility guide

## Verified without external artifacts

These commands inspect only files in this repository:

```bash
python3 tools/verify_repository.py
python3 -m unittest discover -s tests -v
python3 tools/show_results.py
```

The paper was rebuilt with TeX Live by running:

```bash
make paper
```

It compiles twice from `paper/source/scotty1.tex` and the supplied unchanged
`esannV2.cls`. The resulting PDF is six A4 pages. The repository verifier checks
its SHA-256 against [`provenance/artifacts.json`](provenance/artifacts.json).

## Full training and evaluation: external artifacts required

The scripts in [`reproduction/scripts/`](reproduction/scripts/) are the exact
available frozen scripts except that the old machine-specific root assignment
was replaced by the required `SCOTTY_REPRO_ROOT` environment variable. They are
not a self-contained release. Before they can run, the root must contain:

1. The Laya source tree and its `laya` Python package.
2. Project helper modules imported by the scripts: `beam_one_common`,
   `smoke_beam_one`, `train_laya_stage_compare`, and
   `evaluate_beam_three_typed_test`.
3. The Beam-generated and Typed Decisions split files whose hashes are recorded
   in `provenance/artifacts.json`.
4. For evaluation/inference, the frozen checkpoint plus tokenizer with the
   recorded model hash.
5. A CUDA-capable environment compatible with the recorded run.

Set the root explicitly:

```bash
export SCOTTY_REPRO_ROOT=/absolute/path/to/reproduction-root
```

Then invoke the appropriate script with `--help` or inspect its top-level path
constants before running. Training is intentionally not run by the repository
verification target.

## Recorded environment

- PyTorch: `2.14.1+cu130`
- accelerator: NVIDIA GB10
- precision: BF16 autocast
- base model revision: `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`
- seed: `0`

The frozen reports did not record every package version. This repository does
not invent missing pins; exact bit-for-bit environment reconstruction remains a
gap. See [`environment/RECORDED.md`](environment/RECORDED.md).
