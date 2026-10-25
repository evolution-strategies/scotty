# Reproduction source

`scripts/` contains the eight Python scripts available in the frozen snapshot.
The only portability change is replacement of the former absolute experiment
root with the required `SCOTTY_REPRO_ROOT` environment variable. Historical
internal identifiers in filenames and evidence checks are retained because they
are part of the frozen protocol, not a public model-name claim.

The scripts depend on Laya and project helper modules that are not present here.
They also require the hashed datasets and, for evaluation, the excluded model
checkpoint. See [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) before running.
