# Reproducibility

## Verify the repository

These commands need no model or dataset download:

```bash
python3 tools/verify_repository.py
python3 -m unittest discover -s tests -v
python3 tools/show_results.py
```

They check the paper, configuration, aggregate metrics, included source hashes,
and the rule that large checkpoints stay out of Git.

## Download the checkpoint

```bash
python3 -m pip install huggingface_hub
python3 -c 'from huggingface_hub import snapshot_download; print(snapshot_download("DarwinGuru/scotty", local_dir="scotty-checkpoint"))'
```

Verify the downloaded files before loading:

```bash
python3 tools/run_local_checkpoint.py \
  --checkpoint scotty-checkpoint \
  --verify-only
```

Expected `model.safetensors` SHA-256:

```text
e79b758aea590bcc4504e6b622c206080a7d1fedc49d572d3a185321b8e2eeda
```

Expected layout:

```text
scotty-checkpoint/
├── scotty_config.json
├── model.safetensors
├── encoder/config.json
└── tokenizer/
    ├── tokenizer.json
    └── tokenizer_config.json
```

`scotty_config.json` stores Scotty's checkpoint architecture and runtime
settings. The loader presents it to Laya 0.3.28 through a temporary compatibility
view, because that upstream release still expects a legacy filename. Scotty was
trained with supervised learning, not reinforcement learning.

## Run one decision

Install the included Laya dependency and run the example:

```bash
python3 -m pip install -e reproduction/vendor/laya_checkout

python3 tools/run_local_checkpoint.py \
  --checkpoint scotty-checkpoint \
  --device cpu \
  --state "The shipment is sealed, labelled, and the manifest is signed." \
  --instruction "Is this shipment ready for dispatch?" \
  --option false \
  --option true
```

The command encodes all options jointly and prints the selected answer and
scores as JSON.

## Training configuration

The machine-readable configuration is in [`configs/training.json`](configs/training.json).
The final model uses:

- seed 0;
- 849 optimizer updates;
- effective batch 64;
- 50:50 Beam/Typed sampling, uniform within each source;
- hard Beam targets and soft Typed targets;
- full encoder fine-tuning with BF16 autocast.

Full training still requires the separately licensed data identified in
[`provenance/artifacts.json`](provenance/artifacts.json). Raw rows are not
stored in this repository.

## Source boundary

The exact Apache-2.0 Laya checkout used by the training scripts is included at
[`reproduction/vendor/laya_checkout`](reproduction/vendor/laya_checkout).
The accompanying scripts are preserved for inspection. Some need the external
data tree referenced by `SCOTTY_REPRO_ROOT`; repository tests do not run
training.

The recorded environment is in [`environment/RECORDED.md`](environment/RECORDED.md).
Package versions captured after the run are supporting evidence, not proof that
every secondary dependency was unchanged during training.
