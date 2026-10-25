from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ["SCOTTY_REPRO_ROOT"]).expanduser().resolve()
ANALYSIS = ROOT / "analysis/typed-only-matched-updates-v1"
RUN = ROOT / "runs/typed-only-matched-updates-v1"
TRAINER = ROOT / "train_typed_only_matched_updates.py"

BASE = ROOT / "checkpoints/fresh-modernbert-45bb465/model.safetensors"
BEAM1 = ROOT / "models/beam-two/selected_checkpoint/model.safetensors"
TYPED_TRAIN = ROOT / "analysis/beam-one-laya-stage-v1/train.jsonl"
TYPED_VAL = ROOT / "analysis/beam-one-laya-stage-v1/validation.jsonl"
BEAM_VAL = ROOT / "analysis/plain-modernbert-beam-typed-union-protected-test-v2/beam_validation_laya.jsonl"
BEAM_MAP = ROOT / "analysis/plain-modernbert-beam-typed-union-protected-test-v2/beam_validation_map.jsonl"
TYPED_TEST = ROOT / "data/typed_decisions/test.jsonl"
BEAM_TEST = ROOT / "analysis/beam-one-family-disjoint-v1/test_source.jsonl"

EXPECTED = {
    BASE: "6f0fe404e2b83acd0fd7a949f2ceff0543cf56711d13f4d9c27f7c06d98b1002",
    BEAM1: "b7983f2820f46d58038920a860223a1e18826448c1833541f8bc52ba6e3fb783",
    TYPED_TRAIN: "fc5899226124c2dcf76f271d0caecfc8ab946df3bd046b197a141961dc90ee3d",
    TYPED_VAL: "9194df80f1afe6930e6d12fd3cb8b98c90c0f45771f3208e39a2f44fe6cd095b",
    BEAM_VAL: "6aa496eb50c81b97f3ebb6c42967852720ee939c4ceb1449cd85ed4894952021",
    BEAM_MAP: "985d8475016addd6ab8dbf4e739187580f85602b504ad9393db232ca7b453c93",
    TYPED_TEST: "64043cd6dc7b805f9c4f43bc0f8c6115c207992b9bba7aac451f8bda49c6fad5",
    BEAM_TEST: "30298aad191624cda56a83e1af54702b364c455a709387e61f9bc235ca3988b7",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def describe(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def main() -> int:
    if RUN.exists():
        raise FileExistsError(f"run path already exists: {RUN}")
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"hash mismatch for {path}: {actual} != {expected}")

    train_rows = rows(TYPED_TRAIN)
    val_rows = rows(TYPED_VAL)
    beam_val_rows = rows(BEAM_VAL)
    train_decisions = sum(len(row.get("questions", {})) for row in train_rows)
    val_decisions = sum(len(row.get("questions", {})) for row in val_rows)
    if (len(train_rows), train_decisions, len(val_rows), val_decisions, len(beam_val_rows)) != (
        960, 4800, 240, 1200, 3327
    ):
        raise ValueError("unexpected split counts")

    protocol = {
        "name": "typed-only-matched-updates-v1",
        "state": "frozen_before_training",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "retrospective_status": (
            "Controlled retrospective ablation. Both final tests were inspected in earlier work; "
            "no result from either test may alter this protocol."
        ),
        "initialization": {
            **describe(BASE),
            "kind": "plain answerdotai/ModernBERT-large plus the same fresh seed-0 Laya-format head",
            "encoder_revision": "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
            "released_laya_weights_loaded": False,
            "trained_beam_weights_loaded": False,
            "fresh_head_seed": 0,
        },
        "frozen_reference_beam1": {
            **describe(BEAM1),
            "display_name": "Beam-1",
            "physical_snapshot_name": "Beam Two",
        },
        "data": {
            "typed_train": {**describe(TYPED_TRAIN), "cases": 960, "decisions": 4800},
            "typed_validation": {**describe(TYPED_VAL), "cases": 240, "decisions": 1200},
            "beam_validation": {**describe(BEAM_VAL), "decisions": 3327},
            "beam_validation_map": describe(BEAM_MAP),
            "typed_official_test_frozen_identity": {**describe(TYPED_TEST), "cases": 400, "decisions": 2000},
            "beam_protected_test_frozen_identity": {**describe(BEAM_TEST), "decisions": 3259},
            "beam_training_rows": 0,
            "dgx_rows": 0,
        },
        "training": {
            "optimizer": "AdamW",
            "loss": "soft target cross-entropy",
            "encoder_lr": 2.5e-5,
            "head_lr": 1e-4,
            "weight_decay": 0.01,
            "min_lr": 1e-6,
            "schedule": "CosineAnnealingLR over exactly 849 optimizer updates; no warmup",
            "micro_batch": 8,
            "gradient_accumulation": 8,
            "nominal_effective_batch": 64,
            "gradient_clip": 1.0,
            "autocast": "bfloat16",
            "gradient_scaling": True,
            "gradient_checkpointing": True,
            "seed": 0,
            "max_len": 1024,
            "head_max_len": 256,
            "option_layout": "sequential",
            "option_shuffle": False,
            "label_smoothing": 0.0,
            "optimizer_updates": 849,
            "validation_updates": [283, 566, 849],
            "segment_exposures": 18092,
            "total_item_exposures": 54276,
            "source_training_decisions": 4800,
            "equivalent_source_passes": 54276 / 4800,
            "partial_update_treatment": (
                "Each 283-update segment uses 282 full 64-item updates and one 44-item update "
                "split into microbatches 8,8,8,8,8,4, matching Beam-1's per-epoch exposure and final window."
            ),
        },
        "selection": {
            "formula": "0.5 * typed_validation_soft_agreement + 0.5 * beam_validation_position_balanced_accuracy",
            "ties": ["typed_validation_soft_agreement", "beam_validation_position_balanced_accuracy", "earlier_update"],
            "disclosure": "Beam validation is used for checkpoint selection but contributes no gradient updates.",
        },
        "final_evaluation": {
            "after_selection_only": True,
            "typed_official_test": "score once with frozen existing adapter and metrics",
            "beam_protected_test": "score once with frozen existing adapter and metrics",
            "paired_reference": "Beam-1 frozen predictions",
            "no_calibration_or_threshold_fit": True,
        },
        "prohibitions": {
            "test_driven_revision": True,
            "hyperparameter_search": True,
            "paid_api": True,
            "new_generation": True,
            "publication": True,
        },
        "trainer": describe(TRAINER),
        "training_or_scoring_started": False,
    }

    ANALYSIS.mkdir(parents=True, exist_ok=False)
    target = ANALYSIS / "protocol.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, target)
    print(json.dumps({"protocol": str(target), "sha256": sha256(target)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
