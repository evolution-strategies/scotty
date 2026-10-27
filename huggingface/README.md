---
license: apache-2.0
library_name: custom
tags:
  - beam
  - build-with-beam
  - modernbert
  - decision-model
  - candidate-scoring
  - synthetic-supervision
---

# Scotty-1

Scotty-1 is a **Build with Beam** competition entry: a compact, local typed-
decision model trained with synthetic supervision from Reflection Beam. Given a
state, an instruction, and a finite set of candidates, it scores every candidate
jointly in one encoder pass.

- Source, paper, reproducibility, and full results:
  <https://github.com/evolution-strategies/scotty-1>
- Competition call:
  <https://huggingface.co/reflection/Beam-1.0-501B-A23B/discussions/5>

## How Beam was used

Reflection Beam 1.0 501B-A23B was used through Hugging Face Inference Providers
as the synthetic teacher. It generated 19,878 typed decisions across 18 task
families. Scotty-1 trained on 13,292 of those decisions together with 4,800
Typed Decisions examples; 3,327 Beam-generated decisions and 1,200 Typed
Decisions examples were used for validation. Three entire Beam-generated task
families, containing 3,259 decisions, were protected as a separate holdout.

Beam supplied supervision, not model weights. Scotty-1 starts from a fresh
`answerdotai/ModernBERT-large` encoder and a newly initialized two-layer
Laya-format typed head. It does not load pretrained Laya weights or a local Beam
checkpoint.

## Why it is useful

Large generative models are powerful decision teachers, but many downstream
workflows only need a fast, auditable choice among known options. Scotty-1
turns that supervision into a 421,293,827-parameter local scorer with a shared
head over option `[MASK]` positions. It supports boolean, categorical, and
ordered candidate decisions without generating free-form text.

This is a research artifact rather than a calibrated autonomous decision-maker.
Its intended value is reproducible study of compact decision interfaces and
controlled local experiments.

## Results

| Evaluation | Scotty-1 | Matched-update Typed-only control |
|---|---:|---:|
| Protected Beam-family holdout, raw | **83.03%** | 56.34% |
| Protected Beam-family holdout, position-balanced | **76.39%** | 51.64% |
| Official Typed Decisions test, hard | 75.45% | **78.35%** |

The control demonstrates a coverage-specialization tradeoff, not blanket
superiority. Results use one training seed, and the tests were retrospectively
inspected. See the source repository for definitions, confidence intervals,
aggregate reports, and limitations.

## Checkpoint contents

```text
.
├── model.safetensors
├── rl_agent_config.json
├── encoder/config.json
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

`model.safetensors` SHA-256:

```text
b7983f2820f46d58038920a860223a1e18826448c1833541f8bc52ba6e3fb783
```

## Download and run

If repository access is restricted, authenticate with an authorized Hugging
Face account first. Then:

```python
from huggingface_hub import snapshot_download

checkpoint = snapshot_download("DarwinGuru/scotty-1")
print(checkpoint)
```

From the source repository checkout, verify and run the downloaded checkpoint:

```bash
python3 tools/run_local_checkpoint.py \
  --checkpoint /path/printed/by/snapshot_download \
  --verify-only

python3 tools/run_local_checkpoint.py \
  --checkpoint /path/printed/by/snapshot_download \
  --device cuda \
  --state "The shipment is sealed, labelled, and the manifest is signed." \
  --instruction "Is this shipment ready for dispatch?" \
  --option false \
  --option true
```

The exact frozen checkpoint completed this smoke inference successfully during
submission preparation. Scotty-1 is a custom decision-model checkpoint, not a
drop-in Transformers text-classification pipeline.

## Limitations

- One training seed; intervals describe test-sample uncertainty, not training variability.
- Both principal supervision sources are synthetic; unrelated real-world generalization is unknown.
- Teacher labels are not guaranteed truth.
- Probabilities were not deployment-calibrated.
- The recorded 23.36 ms median is one batch-one, 350-token, four-option input on
  an NVIDIA GB10, not a broad throughput claim.

## License and attribution

The original Scotty-1 checkpoint weights are provided under Apache-2.0. That
grant does not automatically license datasets, the manuscript, or third-party
material. The source repository preserves upstream licenses, pinned revisions,
attribution, and the release-readiness audit.
