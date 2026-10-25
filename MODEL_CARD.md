# Model card: Scotty-1

## Summary

Scotty-1 is a 421,293,827-parameter typed decision model. Its 394,781,696-
parameter encoder is initialized from `answerdotai/ModernBERT-large` revision
`45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`; its 26,512,131-parameter head is
newly initialized at seed 0. The head count includes a 263,938-parameter
auxiliary action module that is inherited by the format but unused by the
decision loss.

The decision path adds a learned three-type embedding, applies two norm-first
Transformer layers (width 1,024; 16 heads; feed-forward width 4,096; dropout
0.1), and uses one shared scalar scorer over option `[MASK]` positions.

## Inputs and outputs

Inputs are a state, an instruction, a decision type, and an ordered candidate
list. Outputs are logits and a normalized probability distribution across the
candidates. Boolean candidates are canonicalized as false/true. Categorical
outputs use the highest-probability option. Ordered values may use either the
argmax option or the probability-weighted expected value.

Limits from the frozen configuration:

- full serialized input: 1,024 tokens;
- instruction plus options: 256 tokens;
- each option: 48 tokens;
- candidates are encoded jointly, not independently chunked.

## Training

The mixed training set contains 18,092 decisions: 13,292 Beam-generated hard
targets and 4,800 Typed Decisions soft targets. Validation contains 4,527
decisions: 3,327 Beam and 1,200 Typed. AdamW training ran for three epochs and
849 optimizer updates with micro-batches of 8, eight-step gradient accumulation,
BF16 autocast, full encoder fine-tuning, and gradient checkpointing.

The selected checkpoint SHA-256 is
`b7983f2820f46d58038920a860223a1e18826448c1833541f8bc52ba6e3fb783`.
Weights are not stored in Git. The complete checkpoint is hosted privately at
[`DarwinGuru/scotty-1`](https://huggingface.co/DarwinGuru/scotty-1), including
the encoder and tokenizer files required by the Laya-format loader.

## Intended use

Scotty-1 is a research artifact for studying compact, local, candidate-scoring
decision interfaces. It is suitable for reproduction work and controlled
experiments after the private checkpoint and separate data have been obtained
under appropriate terms.

It is not a calibrated autonomous decision-maker. Deployment needs domain
review, threshold calibration, logging, validated escalation, and checks for
harmful failure modes.

## Limitations

- One training seed; intervals describe test-sample uncertainty, not training variability.
- Tests were retrospectively inspected during the broader project.
- Both principal data sources are synthetic; unrelated real-world generalization is unknown.
- Matching optimizer updates does not match tokens or FLOPs.
- Teacher labels are not guaranteed truth.
- Raw expected calibration error is 0.1093; no calibration was fitted.
- The 23.36 ms median measurement is one batch-one, 350-token, four-option case
  on an NVIDIA GB10 and is not a throughput or matched-LLM claim.
