# Scotty

Scotty is a compact model for decisions with known options. Give it a situation,
a question, and two or more candidate answers; it scores every candidate in one
ModernBERT forward pass and returns the best-supported option.

It is useful for boolean checks, routing, classification, and ordered ratings.
It does not generate free-form text or hidden reasoning.

- **Model:** <https://huggingface.co/DarwinGuru/scotty>
- **Paper:** [`paper/scotty-paper.pdf`](paper/scotty-paper.pdf)
- **Interactive demo:** <https://spark-29fd.taile0f6e8.ts.net> (access-controlled)

## A small example

```json
{
  "state": "The shipment is sealed, labelled, and the manifest is signed.",
  "question": "Is this shipment ready for dispatch?",
  "options": ["false", "true"]
}
```

Scotty reads the state, question, and all options together. A shared decision
head scores the option markers, and softmax turns those scores into a ranking.
The values are not calibrated probabilities of correctness.

## Install and run

```bash
git clone https://github.com/evolution-strategies/scotty.git
cd scotty

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e reproduction/vendor/laya_checkout
python -m pip install huggingface_hub

python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("DarwinGuru/scotty", local_dir="scotty-checkpoint"))'

python tools/run_local_checkpoint.py \
  --checkpoint scotty-checkpoint \
  --device cpu \
  --state "The shipment is sealed, labelled, and the manifest is signed." \
  --instruction "Is this shipment ready for dispatch?" \
  --option false \
  --option true
```

The loader verifies the checkpoint before loading it. The expected
`model.safetensors` SHA-256 is:

```text
e79b758aea590bcc4504e6b622c206080a7d1fedc49d572d3a185321b8e2eeda
```

## How Scotty was trained

Reflection Beam 1.0 501B-A23B generated typed decision examples across 18 task
families. Scotty was trained with effective batches split equally between that
Beam-generated data and the public Typed Decisions dataset. Sampling is uniform
within each source.

The final run uses a pretrained `answerdotai/ModernBERT-large` encoder, a newly
initialized two-layer typed-decision head, seed 0, 849 optimizer updates, and an
effective batch size of 64. Beam supplied supervision, not model weights.

The Beam dataset-generation API calls stayed within the $25 credit budget
provided to challenge participants. This excludes local training and hardware
costs.

## Results

Across three training seeds, the mean validation objective is
`0.859532 ± 0.004603` (mean ± sample standard deviation).

| Group-safe source audit | Scotty |
|---|---:|
| Combined objective | 0.841031 |
| Typed hard accuracy | 76.67% (368/480) |
| Beam raw accuracy | 88.62% (1,191/1,344) |
| Beam position-balanced accuracy | 91.54% |

The audit uses withheld groups from the same source collections used for
training. It is not an external or fresh-distribution benchmark. See
[`RESULTS.md`](RESULTS.md) for definitions and limitations.

## Repository map

- [`paper/scotty-paper.pdf`](paper/scotty-paper.pdf) — accessible six-page paper.
- [`MODEL_CARD.md`](MODEL_CARD.md) — architecture, intended use, and limitations.
- [`DATA_CARD.md`](DATA_CARD.md) — data sources and split provenance.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — verification and inference steps.
- [`configs/`](configs/) and [`results/`](results/) — machine-readable settings and metrics.
- [`reproduction/`](reproduction/) — archived training/evaluation source and its upstream license.

## License

Original Scotty code and checkpoint weights are Apache-2.0. The paper, datasets,
and third-party material are outside that grant. See
[`LICENSE_SCOPE.md`](LICENSE_SCOPE.md) and [`THIRD_PARTY.md`](THIRD_PARTY.md).
