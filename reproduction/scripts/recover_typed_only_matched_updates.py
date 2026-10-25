from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import laya.train as lt
from beam_one_common import prediction_metrics, read_jsonl as read_audit_jsonl
from laya.train import calibration_records, items_from_rows, load_checkpoint, read_jsonl
from smoke_beam_one import bf16_forward
from train_laya_stage_compare import metrics


ROOT = Path(os.environ["SCOTTY_REPRO_ROOT"]).expanduser().resolve()
OUT = ROOT / "runs/typed-only-matched-updates-v1"
TYPED_VAL = ROOT / "analysis/beam-one-laya-stage-v1/validation.jsonl"
BEAM_VAL = ROOT / "analysis/plain-modernbert-beam-typed-union-protected-test-v2/beam_validation_laya.jsonl"
BEAM_MAP = ROOT / "analysis/plain-modernbert-beam-typed-union-protected-test-v2/beam_validation_map.jsonl"
MAX_LEN, HEAD_MAX_LEN = 1024, 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def typed_metrics_from_saved(rows):
    qtype_index = {"choice": 0, "score": 1, "noul": 2}
    records = []
    for row in rows:
        records.append((qtype_index[row["qtype"]], row["logits"], row["target"], row["options"]))
    return metrics(records)


def beam_metrics_from_saved(rows, mapping):
    records = [(0, row["logits"], row["target"], row["options"]) for row in rows]
    return prediction_metrics(records, mapping)


def main() -> None:
    if (OUT / "final_report.json").exists() or (OUT / "recovered_status.json").exists():
        raise FileExistsError("recovery outputs already exist")
    history = json.loads((OUT / "milestone_metrics.json").read_text(encoding="utf-8"))
    best = json.loads((OUT / "best_selection.json").read_text(encoding="utf-8"))
    run_config = json.loads((OUT / "run_config.json").read_text(encoding="utf-8"))
    typed_saved = json.loads((OUT / "typed_validation_predictions.json").read_text(encoding="utf-8"))
    beam_saved = json.loads((OUT / "beam_validation_predictions.json").read_text(encoding="utf-8"))
    beam_map = read_audit_jsonl(BEAM_MAP)
    if len(history) != 3 or [row["optimizer_update"] for row in history] != [283, 566, 849]:
        raise ValueError("milestone history is incomplete")
    if best["optimizer_update"] != 849:
        raise ValueError("unexpected selected update")
    if (len(typed_saved), len(beam_saved), len(beam_map)) != (1200, 3327, 3327):
        raise ValueError("saved validation prediction count mismatch")

    selected_sha = sha256(OUT / "selected_checkpoint/model.safetensors")
    best_sha = sha256(OUT / "checkpoint_update_849/model.safetensors")
    if selected_sha != best_sha:
        raise ValueError("selected checkpoint differs from selected milestone")

    typed_recomputed = typed_metrics_from_saved(typed_saved)
    beam_recomputed = beam_metrics_from_saved(beam_saved, beam_map)
    selection_typed = best["metrics"]["typed_validation"]
    selection_beam = best["metrics"]["beam_validation"]
    saved_checks = {
        "typed_hard": [typed_recomputed["accuracy"], selection_typed["accuracy"]],
        "typed_soft": [typed_recomputed["soft_accuracy"], selection_typed["soft_accuracy"]],
        "beam_raw": [beam_recomputed["accuracy"], selection_beam["accuracy"]],
        "beam_balanced": [beam_recomputed["position_balanced_accuracy"], selection_beam["position_balanced_accuracy"]],
    }
    if abs(saved_checks["typed_hard"][0] - saved_checks["typed_hard"][1]) * 1200 > 2.1:
        raise ValueError({"material_typed_reload_difference": saved_checks})
    if abs(saved_checks["typed_soft"][0] - saved_checks["typed_soft"][1]) >= 1e-3:
        raise ValueError({"material_typed_soft_reload_difference": saved_checks})
    if abs(saved_checks["beam_raw"][0] - saved_checks["beam_raw"][1]) * 3327 > 2.1:
        raise ValueError({"material_beam_reload_difference": saved_checks})
    if abs(saved_checks["beam_balanced"][0] - saved_checks["beam_balanced"][1]) >= 1e-3:
        raise ValueError({"material_beam_balanced_reload_difference": saved_checks})

    # A second read-only reload/evaluation checks that the selected checkpoint is usable.
    model, tokenizer, _ = load_checkpoint(str(OUT / "selected_checkpoint"))
    lt._forward = bf16_forward
    typed_rows = read_jsonl(str(TYPED_VAL))
    beam_rows = read_jsonl(str(BEAM_VAL))
    typed_items, typed_skipped = items_from_rows(tokenizer, typed_rows, MAX_LEN, HEAD_MAX_LEN)
    beam_items, beam_skipped = items_from_rows(tokenizer, beam_rows, MAX_LEN, HEAD_MAX_LEN)
    if typed_skipped or beam_skipped or len(typed_items) != 1200 or len(beam_items) != 3327:
        raise ValueError("validation fixture conversion changed during recovery")
    model.cuda().eval()
    device = torch.device("cuda", 0)
    typed_fresh_records = calibration_records(
        model, tokenizer, typed_items, device, MAX_LEN, HEAD_MAX_LEN, parallel=False
    )
    beam_fresh_records = calibration_records(
        model, tokenizer, beam_items, device, MAX_LEN, HEAD_MAX_LEN, parallel=False
    )
    typed_fresh = metrics(typed_fresh_records)
    beam_fresh = prediction_metrics(beam_fresh_records, beam_map)
    del model
    torch.cuda.empty_cache()

    fresh_checks = {
        "typed_hard": [typed_fresh["accuracy"], typed_recomputed["accuracy"]],
        "typed_soft": [typed_fresh["soft_accuracy"], typed_recomputed["soft_accuracy"]],
        "beam_raw": [beam_fresh["accuracy"], beam_recomputed["accuracy"]],
        "beam_balanced": [beam_fresh["position_balanced_accuracy"], beam_recomputed["position_balanced_accuracy"]],
    }
    if abs(fresh_checks["typed_hard"][0] - fresh_checks["typed_hard"][1]) * 1200 > 2.1:
        raise ValueError({"material_second_typed_reload_difference": fresh_checks})
    if abs(fresh_checks["typed_soft"][0] - fresh_checks["typed_soft"][1]) >= 1e-3:
        raise ValueError({"material_second_typed_soft_reload_difference": fresh_checks})
    if abs(fresh_checks["beam_raw"][0] - fresh_checks["beam_raw"][1]) * 3327 > 2.1:
        raise ValueError({"material_second_beam_reload_difference": fresh_checks})
    if abs(fresh_checks["beam_balanced"][0] - fresh_checks["beam_balanced"][1]) >= 1e-3:
        raise ValueError({"material_second_beam_balanced_reload_difference": fresh_checks})

    completed = datetime.now(timezone.utc).isoformat()
    verification = {
        "recovered_utc": completed,
        "training_and_checkpoints_complete_before_postprocess_failure": True,
        "original_failure": (
            "The saved-prediction verifier incorrectly enumerated QTYPE_NAMES as an iterable and rejected "
            "the valid string decision type 'score'."
        ),
        "training_repeated": False,
        "optimizer_updates": 849,
        "item_exposures": 54276,
        "best_update": 849,
        "checkpoint_reload_verified": True,
        "selected_checkpoint_matches_best_milestone": True,
        "selected_checkpoint_sha256": selected_sha,
        "saved_prediction_recompute": {
            "typed_validation": typed_recomputed,
            "beam_validation": beam_recomputed,
        },
        "fresh_second_reload": {
            "typed_validation": typed_fresh,
            "beam_validation": beam_fresh,
        },
        "metric_checks": {
            "selection_vs_saved_reload": saved_checks,
            "saved_reload_vs_fresh_second_reload": fresh_checks,
        },
        "tests_accessed": False,
    }
    atomic_json(OUT / "verification_report.json", verification)
    report = {
        "status": "training_and_validation_completed",
        "completed_utc": completed,
        "optimizer_updates": 849,
        "item_exposures": 54276,
        "equivalent_source_passes": 54276 / 4800,
        "source_cycles_started": math.ceil(54276 / 4800),
        "best_update": 849,
        "selected_checkpoint": str(OUT / "selected_checkpoint"),
        "selected_checkpoint_sha256": selected_sha,
        "selected_matches_best_milestone": True,
        "reload_verified": True,
        "typed_validation": typed_recomputed,
        "beam_validation": beam_recomputed,
        "fresh_second_reload": {
            "typed_validation": typed_fresh,
            "beam_validation": beam_fresh,
        },
        "independent_saved_prediction_recompute": {
            "typed_validation": typed_recomputed,
            "beam_validation": beam_recomputed,
        },
        "tests_accessed": False,
        "history": history,
        "postprocess_recovery": {
            "required": True,
            "training_repeated": False,
            "verification_report": str(OUT / "verification_report.json"),
        },
        "run_config": run_config,
    }
    atomic_json(OUT / "final_report.json", report)
    atomic_json(OUT / "recovered_status.json", {
        "status": "completed_with_postprocess_recovery",
        "utc": completed,
        "training_process_exit": 1,
        "training_process_exit_reason": "postprocess verifier bug after training, selection, and saved predictions",
        "training_and_checkpoints_complete": True,
        "verification_complete": True,
    })
    print(json.dumps({
        "status": "completed_with_postprocess_recovery",
        "best_update": 849,
        "checkpoint_sha256": selected_sha,
        "typed_hard": typed_recomputed["accuracy"],
        "typed_soft": typed_recomputed["soft_accuracy"],
        "beam_raw": beam_recomputed["accuracy"],
        "beam_balanced": beam_recomputed["position_balanced_accuracy"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
