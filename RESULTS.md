# Results

## Primary evaluations

Scotty scores 83.0316047867% raw and 76.3893546632% position-balanced
accuracy on the 3,259-example protected Beam family holdout. Position-balanced
accuracy is the macro-average over observed `(decision type, option count,
converted gold position)` cells.

On the 2,000-decision official Typed test it scores 75.45% hard accuracy and
60.1953550368% soft-teacher agreement. The case-clustered 95% interval for hard
accuracy is 73.30%-77.60% (400 cases, 10,000 bootstrap draws, seed 0).

## Matched-update control

The Typed-only control starts from the same fresh encoder/head seed and uses the
same optimizer and schedule. It cycles 4,800 Typed training decisions to match
849 updates and 54,276 decision exposures. It scores 56.34% raw / 51.64%
position-balanced on Beam and 78.35% hard / 61.11% soft on Typed.

Scotty therefore gains 26.70 percentage points on Beam raw accuracy, while the
control gains 2.90 points on Typed hard accuracy. This is a coverage-
specialization tradeoff. It does not show universal superiority, and matching
updates/exposures does not match tokens or FLOPs.

## Laya context

Released root Laya was rerun on the Beam holdout and scores 53.51% raw / 48.56%
position-balanced. External published Typed results are 36.2% for root Laya and
76.6% for a specialized Laya Typed Decisions model; these are context, not
matched reruns. The historical 74.85% local result was a four-epoch fine-tuned
specialist and must not be described as released-root performance.

Machine-readable values are in [`results/metrics.json`](results/metrics.json).
