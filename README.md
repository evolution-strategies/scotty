# Scotty-1

Scotty-1 is a compact typed-decision model: it reads a state, an instruction,
and a set of candidate answers, then scores the candidates in one encoder pass.
It combines a fresh `answerdotai/ModernBERT-large` backbone with a newly
initialized Laya-format typed head. It does **not** load pretrained Laya weights.

This private research repository packages the paper, frozen aggregate evidence,
model and training configuration, and the source that was available at freeze
time. Large checkpoints, raw datasets, and row-level predictions are deliberately
excluded. Their identities are recorded by hash in [`provenance/artifacts.json`](provenance/artifacts.json).

## Headline results

| Evaluation | Scotty-1 | Matched-update Typed-only control | Other context |
|---|---:|---:|---:|
| Protected Beam family holdout (raw) | **83.03%** | 56.34% | Released root Laya: 53.51% |
| Protected Beam family holdout (position-balanced) | **76.39%** | 51.64% | Released root Laya: 48.56% |
| Official Typed test (hard) | 75.45% | **78.35%** | Published specialist Laya: 76.6% |

The control shows a coverage-specialization tradeoff, not blanket superiority.
This is a single-seed study, and both tests were inspected retrospectively.
See [`RESULTS.md`](RESULTS.md) for definitions and caveats.

## Repository tour

- [`paper/scotty-1-paper.pdf`](paper/scotty-1-paper.pdf) - six-page Scotty-1 manuscript.
- [`configs/`](configs/) - architecture and training settings.
- [`results/`](results/) - machine-readable aggregate metrics and exact split counts.
- [`reproduction/`](reproduction/) - frozen scripts, recovered helpers, and the licensed Laya source used by the run.
- [`MODEL_CARD.md`](MODEL_CARD.md) and [`DATA_CARD.md`](DATA_CARD.md) - intended use, data provenance, and limitations.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) - verified commands and external-artifact requirements.
- [`LICENSE_AUDIT.md`](LICENSE_AUDIT.md) - upstream terms evidence and unresolved release questions.
- [`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) - work required before any public release.

## Decision interface

Conceptually, one example is:

```json
{
  "state": "The shipment is sealed, labelled, and the manifest is signed.",
  "instruction": "Is this shipment ready for dispatch?",
  "decision_type": "boolean",
  "options": ["false", "true"]
}
```

Scotty-1 jointly serializes the instruction, every option, and the state. Each
option has a `[MASK]` marker; the shared head scores those marker states and a
softmax returns one probability per option. Candidates are never scored in
independent chunks. The full input limit is 1,024 tokens, the instruction/options
budget is 256, and each option is capped at 48 tokens.

## Quick verification

The commands below are runnable with Python 3 and do not download data or models:

```bash
python3 tools/verify_repository.py
python3 -m unittest discover -s tests -v
python3 tools/show_results.py
```

Training, checkpoint inference, and full evaluation still need the excluded
checkpoint and hashed datasets. Follow
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md); source and syntax paths are verified,
but packaging did not rerun training or checkpoint inference.

## Use an authorized local checkpoint

The repository does not host weights or invent a download location. If you
already have the authorized Laya-format checkpoint, verify it without loading
the model:

```bash
python3 tools/run_local_checkpoint.py \
  --checkpoint /absolute/path/to/selected_checkpoint \
  --verify-only
```

Then run one joint candidate decision:

```bash
python3 tools/run_local_checkpoint.py \
  --checkpoint /absolute/path/to/selected_checkpoint \
  --device cuda \
  --state "The shipment is sealed, labelled, and the manifest is signed." \
  --instruction "Is this shipment ready for dispatch?" \
  --option false \
  --option true
```

The command refuses a checkpoint whose `model.safetensors` does not match the
recorded SHA-256. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the required
directory layout and the environment evidence boundary.

## License and release status

This repository is intended to remain **private** until a separate competition
release decision. Original Scotty-1 software and intended checkpoint weights
are under Apache-2.0; the manuscript, datasets, and third-party material are
outside that grant. See [`LICENSE_SCOPE.md`](LICENSE_SCOPE.md),
[`THIRD_PARTY.md`](THIRD_PARTY.md), and
[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md).
