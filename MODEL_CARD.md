# Model card: Scotty

## Model

Scotty is a compact, one-pass model trained with Beam-generated data to choose
among supplied options. Its 421,293,827-parameter architecture comprises:

- ModernBERT-large encoder: 394,781,696 parameters;
- newly initialized Laya-format head: 26,512,131 parameters;
- two norm-first Transformer head layers;
- learned decision-type embedding;
- one shared scalar scorer over option `[MASK]` positions.

The encoder starts from `answerdotai/ModernBERT-large` revision
`45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`. Candidates are encoded jointly.

## Inputs and outputs

Inputs are a state, an instruction, a decision type, and an ordered candidate
list. Outputs are one logit and one softmax score per candidate. The largest
score selects the answer.

Limits:

- full serialized input: 1,024 tokens;
- instruction plus options: 256 tokens;
- each option: 48 tokens.

Softmax scores are not calibrated confidence, and the model does not generate
reasoning.

## Training

Training uses 13,292 Beam-generated hard targets and 4,800 Typed Decisions soft
targets. Every effective batch contains equal numbers from the two sources,
sampled uniformly within source.

The reported checkpoint uses seed 0, 849 optimizer updates, effective batch 64,
BF16 autocast, gradient checkpointing, and full encoder fine-tuning.

Checkpoint SHA-256:

```text
e79b758aea590bcc4504e6b622c206080a7d1fedc49d572d3a185321b8e2eeda
```

The complete checkpoint is hosted at
[`DarwinGuru/scotty`](https://huggingface.co/DarwinGuru/scotty).

## Intended use

Scotty is intended for research and controlled applications that need a choice
among explicit options: boolean checks, routing, categorization, and ordered
ratings. Applications must validate their own question wording, option sets,
and escalation rules.

## Limitations

- Both primary supervision sources are synthetic.
- Teacher errors and shared biases may be learned.
- The three-seed result is a small sample.
- The group-safe audit uses the same source collections as training and is not
  an external or fresh-distribution test.
- Scores are not calibrated for deployment.
- Long inputs can lose information beyond the 1,024-token limit.
- High-stakes decisions require domain review and human escalation.
