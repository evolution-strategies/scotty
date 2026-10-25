# Vendored Laya provenance

This directory is the Laya Python source selected by the historical launch
wrappers through `PYTHONPATH`.

- Upstream: <https://github.com/NandhaKishorM/laya>
- Git commit: `eba3b0367643498e478ed02e870fa53737da5061`
- Upstream package version in this checkout: `0.3.28`
- License: Apache License 2.0; the upstream `LICENSE` is included unchanged.
- Checkout state: one local modification in `laya/train.py`.

The local modification moves the model to the selected device immediately
before the optional base-checkpoint evaluation inside `finetune`:

```diff
     before_eval = None
     if eval_items:
+        model.to(dev)
         base_records = calibration_records(...)
```

No other tracked Laya file was modified in the source checkout. This vendored
copy is included to preserve the exact source dependency used by the private
research run; it does not change the licensing of any non-Laya file.
