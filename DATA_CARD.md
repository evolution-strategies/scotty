# Data card

## Beam-generated decisions

Reflection Beam 1.0 501B-A23B generated typed decisions for boolean,
categorical, and ordered questions across 18 operational task families. After
filtering, the accepted collection contains 19,878 decisions.

The model uses 13,292 Beam decisions for training and 3,327 for validation.
The group-safe source audit contains 1,344 Beam decisions. Related examples are
kept together when splitting.

Beam was accessed through Hugging Face Inference Providers. Beam supplied
supervision, not model weights. The dataset-generation calls stayed within the
$25 participant credit budget, excluding local training and hardware costs.

## Typed Decisions

Source: `LocalLLaMA/typed-decisions`, revision
`468b1461d59e01d6404e1cf37431da962acd43a0`.

Scotty uses 4,800 decisions for training and 1,200 for validation. The source
audit contains 480 decisions from 96 cases. All decisions from one case remain
in the same group.

Typed targets can assign weight to more than one candidate. Scotty trains
against those soft distributions; Beam targets are one-hot.

## Sampling

Every effective training batch is split equally between the two sources:
50% Beam-generated and 50% Typed Decisions. Sampling is uniform within each
source. This balances source presentations, not token counts or compute.

## Audit boundary

Audit groups are excluded from the final run's training and validation. The
audit nevertheless comes from the same source collections used by the project,
so it is not an external or fresh-distribution benchmark.

## Availability and rights

Raw rows and row-level predictions are not stored in this repository. Exact
source identities and hashes are recorded in
[`provenance/artifacts.json`](provenance/artifacts.json).

The root Apache-2.0 license covers original Scotty code and intended checkpoint
weights, not the datasets. Review the source datasets' exact terms before
redistributing any rows.
