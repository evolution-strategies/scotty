# Reproduction source

`scripts/` contains the eight Python scripts available in the frozen snapshot
plus the four exact helper modules they import. The only portability change is
replacement of former absolute experiment roots with the required
`SCOTTY_REPRO_ROOT` environment variable. Historical internal identifiers in
filenames and evidence checks are retained because they are part of the frozen
protocol, not a public model-name claim.

`vendor/laya_checkout/` contains the Laya source tree selected by the historical
launch wrappers, including its Apache-2.0 license and the one-line local
`train.py` modification present at run time. See its `PROVENANCE.md`.

The scripts still require the hashed datasets and, for evaluation, the excluded
model checkpoint. See [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) before
running.
