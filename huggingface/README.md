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

# Scotty

Scotty is a 421-million-parameter model for choosing among supplied options.
It reads a situation, a question, and all candidate answers together, then
scores the candidates in one ModernBERT forward pass. It supports boolean,
categorical, and ordered decisions without generating free-form text.

- Inference code: [`run_scotty.py`](https://huggingface.co/DarwinGuru/scotty/blob/main/run_scotty.py)
- Paper: [`paper/scotty-paper.pdf`](https://huggingface.co/DarwinGuru/scotty/blob/main/paper/scotty-paper.pdf)
- Demo: <https://spark-29fd.taile0f6e8.ts.net> (access-controlled)

## Contact

For research questions or demo access, contact Oliver Kramer at
<oliver.kramer@uni-oldenburg.de>.

## How it works

Scotty combines a pretrained `answerdotai/ModernBERT-large` encoder with a
newly initialized two-layer Laya-format decision head. Each option receives a
`[MASK]` marker. A shared scalar head scores those marker states, and softmax
ranks the options. Candidates are encoded jointly rather than in separate
chunks.

The output values are comparative scores, not calibrated confidence. Scotty
does not produce explanations or hidden reasoning.

## Training data

Reflection Beam 1.0 501B-A23B generated 19,878 typed decisions across 18 task
families. Beam supplied supervision, not weights. Training uses 13,292 of those
decisions plus 4,800 examples from `LocalLLaMA/typed-decisions`.

Every effective batch is split 50:50 between Beam-generated and Typed examples,
with uniform sampling inside each source. The final model uses seed 0, 849
optimizer updates, an effective batch size of 64, BF16 autocast, and full encoder
fine-tuning.

The Beam dataset-generation API calls stayed within the $25 participant credit
budget. This excludes local training and hardware costs.

### How Beam was used

This Build with Beam project uses Reflection Beam 1.0 501B-A23B as the teacher
that generated typed decision examples. Scotty does not contain Beam weights:
it learns from the generated situations, questions, options, and target choices.

## Results

Three training seeds obtain mean validation objective
`0.859532 ± 0.004603` (mean ± sample standard deviation).

| Group-safe source audit | Scotty |
|---|---:|
| Combined objective | 0.841031 |
| Typed hard accuracy | 76.67% (368/480) |
| Beam raw accuracy | 88.62% (1,191/1,344) |
| Beam position-balanced accuracy | 91.54% |

The audit contains withheld groups from the same two source collections. It is
not an external or fresh-distribution benchmark.

## Files

```text
.
├── model.safetensors
├── scotty_config.json
├── run_scotty.py
├── requirements.txt
├── paper/scotty-paper.pdf
├── encoder/config.json
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

`scotty_config.json` records the encoder, decision-head shape, token budgets,
precision, and release provenance. The included `run_scotty.py` adapter gives
Laya 0.3.28 the legacy filename it expects only inside a temporary compatibility
view; the published checkpoint does not use an RL-labelled manifest. Scotty was
trained with supervised learning.

`model.safetensors` SHA-256:

```text
e79b758aea590bcc4504e6b622c206080a7d1fedc49d572d3a185321b8e2eeda
```

## Download and run

```bash
python3 -m pip install huggingface_hub
python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("DarwinGuru/scotty", local_dir="scotty")'
cd scotty
python3 -m pip install -r requirements.txt

python3 run_scotty.py \
  --checkpoint . \
  --verify-only

python3 run_scotty.py \
  --checkpoint . \
  --device cpu \
  --state "The shipment is sealed, labelled, and the manifest is signed." \
  --instruction "Is this shipment ready for dispatch?" \
  --option false \
  --option true
```

Scotty is a custom Laya-format checkpoint, not a drop-in Transformers text-
classification pipeline.

## Limitations

- Both supervision sources are synthetic; teacher errors can be learned.
- Three seeds are a small sample.
- The source audit is not an independent external test.
- Scores require calibration before confidence-based automation.
- No real-world deployment study or broad throughput benchmark is reported.

## License

Original Scotty checkpoint weights are Apache-2.0. That grant does not
automatically license datasets, the paper, or third-party material. This model
card, the checkpoint configuration, and the release files record the relevant
revisions, attribution, and license scope. See
[`LICENSE_SCOPE.md`](https://huggingface.co/DarwinGuru/scotty/blob/main/LICENSE_SCOPE.md)
for the files in this release.

Key upstreams are `answerdotai/ModernBERT-large` revision
`45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`, Laya `0.3.28` (training checkout
`eba3b0367643498e478ed02e870fa53737da5061`), and
`LocalLLaMA/typed-decisions` revision
`468b1461d59e01d6404e1cf37431da962acd43a0`. Each retains its own attribution
and terms.
