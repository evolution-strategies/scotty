# Results

## Three-seed validation

The validation objective gives equal weight to Typed hard accuracy and Beam
position-balanced accuracy:

```text
S = 0.5 × Typed hard accuracy + 0.5 × Beam position-balanced accuracy
```

| Metric | Mean across seeds 0, 1, 2 | Sample SD |
|---|---:|---:|
| Combined objective S | 0.859532 | 0.004603 |
| Typed hard accuracy | 0.798611 | — |
| Beam position-balanced accuracy | 0.920454 | — |

The standard deviation summarizes three observed runs; it is not a confidence
interval or a population guarantee.

## Group-safe source audit

The seed-0 checkpoint was evaluated on 1,344 Beam decisions and 480 Typed
decisions from 96 Typed cases.

| Metric | Scotty |
|---|---:|
| Combined objective S | 0.841031 |
| Typed hard accuracy | 0.766667 (368/480) |
| Beam raw accuracy | 0.886161 (1,191/1,344) |
| Beam position-balanced accuracy | 0.915395 |

Related examples remain grouped, but the audit is drawn from the same source
collections used for training. It is not an external or fresh-distribution
benchmark. Only aggregate audit results were retained, so a case-grouped paired
confidence interval cannot be reconstructed without rerunning the audit.

Machine-readable values are in [`results/metrics.json`](results/metrics.json).
