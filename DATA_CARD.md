# Data card

## Beam-generated decisions

The accepted synthetic corpus contains 19,878 decisions across 18 task families.
The teacher was Reflection Beam 1.0 501B-A23B, accessed through Hugging Face
Inference Providers. This was an API-served full model, not a local IQ1_S GGUF.
Requests used temperature 0.65, high reasoning effort and JSON responses; one
initial pilot used maximum effort. The displayed identity is provider-neutral;
detailed routing metadata remains in private frozen provenance outside Git.

The protected holdout contains 3,259 examples from three entire families:

- operational evidence strength: 935;
- packing readiness: 1,146;
- warehouse exception routing: 1,178.

The other 15 families contribute 13,292 training and 3,327 validation examples.
Generation requests and reviewed duplicate links form components constrained to
one split. Allocation used neither targets nor model performance. Validation
shares families with training, while the holdout uses different families.

## Typed Decisions

Source: `LocalLLaMA/typed-decisions`, revision
`468b1461d59e01d6404e1cf37431da962acd43a0`. The official 1,200 training cases
(6,000 decisions) were split by workflow and kept case-disjoint: 960 cases /
4,800 decisions for training and 240 cases / 1,200 decisions for validation.
The official test is separate: 400 cases / 2,000 decisions.

## Exact split table

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| Beam-generated | 13,292 | 3,327 | 3,259 |
| Typed Decisions | 4,800 | 1,200 | 2,000 |
| Total decisions | 18,092 | 4,527 | 5,259 |

## Availability and provenance

Raw rows are not included. Row-level predictions are also excluded because a
future public release needs a separate content and licensing review. Exact
source/split hashes are recorded in [`provenance/artifacts.json`](provenance/artifacts.json).
No private DGX pilot data was used in this model.

The root Apache-2.0 license covers original project software and intended
Scotty-1 weights, not the synthetic rows. The Typed Decisions dataset and
upstream model/code terms must be checked at the exact revisions before
redistribution. Do not infer row-level redistribution rights from hashes,
citations, or this private repository.
