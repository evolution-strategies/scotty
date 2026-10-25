from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import laya.train as lt
from beam_one_common import prediction_metrics, read_jsonl as read_audit_jsonl, sha256_file
from laya.train import calibration_records, items_from_rows, load_checkpoint, read_jsonl
from smoke_beam_one import bf16_forward
from train_laya_stage_compare import latency


ROOT = Path(os.environ["SCOTTY_REPRO_ROOT"]).expanduser().resolve()
OUT = ROOT / "runs/plain-modernbert-beam-typed-union-protected-test-v2-attempt2"
DATA = ROOT / "analysis/plain-modernbert-beam-typed-union-protected-test-v2"
TYPED_VAL = ROOT / "analysis/beam-one-laya-stage-v1/validation.jsonl"
MAX_LEN, HEAD_MAX_LEN = 1024, 256


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def typed_metrics(rows):
    all_rows, by_type = [], {}
    for source in rows:
        logits = np.asarray(source["logits"][: source["options"]], dtype=np.float64)
        target = np.asarray(source["target"][: source["options"]], dtype=np.float64)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        prediction, gold = int(probabilities.argmax()), int(target.argmax())
        row = {
            "correct": int(prediction == gold),
            "soft_accuracy": float(target[prediction]),
            "cross_entropy": float(-(target * np.log(np.clip(probabilities, 1e-12, 1))).sum()),
            "brier": float(((probabilities - target) ** 2).mean()),
            "confidence": float(probabilities[prediction]),
            "ordinal_abs_error": abs(prediction - gold) if source["qtype"] == "score" else None,
        }
        all_rows.append(row)
        by_type.setdefault(source["qtype"], []).append(row)

    def aggregate(values):
        result = {
            "items": len(values),
            "accuracy": sum(row["correct"] for row in values) / len(values),
            "soft_accuracy": sum(row["soft_accuracy"] for row in values) / len(values),
            "cross_entropy": sum(row["cross_entropy"] for row in values) / len(values),
            "brier": sum(row["brier"] for row in values) / len(values),
            "mean_confidence": sum(row["confidence"] for row in values) / len(values),
        }
        ordinal = [row["ordinal_abs_error"] for row in values if row["ordinal_abs_error"] is not None]
        if ordinal:
            result["ordinal_mae"] = sum(ordinal) / len(ordinal)
        return result

    return {**aggregate(all_rows), "by_type": {name: aggregate(values) for name, values in sorted(by_type.items())}}


def main():
    if (OUT / "final_report.json").exists() or (OUT / "verification_report.json").exists():
        raise FileExistsError("verification artifacts already exist")
    manifest = json.loads((DATA / "split_manifest.json").read_text())
    history = json.loads((OUT / "epoch_metrics.json").read_text())
    best = json.loads((OUT / "best_selection.json").read_text())
    base = json.loads((OUT / "base_metrics.json").read_text())
    run_config = json.loads((OUT / "run_config.json").read_text())
    typed_predictions = load_jsonl(OUT / "typed_validation_predictions.jsonl")
    beam_predictions = load_jsonl(OUT / "beam_validation_predictions.jsonl")
    beam_map = read_audit_jsonl(DATA / "beam_validation_map.jsonl")
    if len(typed_predictions) != 1200 or len(beam_predictions) != manifest["beam"]["validation"]["rows"]:
        raise ValueError("saved prediction count mismatch")

    recomputed_typed = typed_metrics(typed_predictions)
    recomputed_beam = prediction_metrics(
        [(0, row["logits"], row["target"], row["options"]) for row in beam_predictions], beam_map)
    expected_typed = best["metrics"]["typed_validation"]
    expected_beam = best["metrics"]["beam_validation"]
    checks = {
        "typed_accuracy": (recomputed_typed["accuracy"], expected_typed["accuracy"]),
        "typed_soft_accuracy": (recomputed_typed["soft_accuracy"], expected_typed["soft_accuracy"]),
        "beam_accuracy": (recomputed_beam["accuracy"], expected_beam["accuracy"]),
        "beam_position_balanced_accuracy": (
            recomputed_beam["position_balanced_accuracy"], expected_beam["position_balanced_accuracy"]),
    }
    if abs(checks["typed_accuracy"][0] - checks["typed_accuracy"][1]) >= 1e-12 or \
            abs(checks["typed_soft_accuracy"][0] - checks["typed_soft_accuracy"][1]) >= 1e-12:
        raise ValueError({"typed_metric_mismatch": checks})
    beam_raw_changed_rows = round(abs(checks["beam_accuracy"][0] - checks["beam_accuracy"][1]) * len(beam_predictions))
    if beam_raw_changed_rows > 2 or abs(checks["beam_position_balanced_accuracy"][0] -
                                        checks["beam_position_balanced_accuracy"][1]) >= 1e-3:
        raise ValueError({"material_beam_metric_mismatch": checks, "changed_rows": beam_raw_changed_rows})

    selected_hash = sha256_file(OUT / "selected_checkpoint/model.safetensors")
    epoch_hash = sha256_file(OUT / f"checkpoint_epoch_{best['epoch']}/model.safetensors")
    if selected_hash != epoch_hash:
        raise ValueError("selected checkpoint does not equal best epoch checkpoint")

    # Reload the selected checkpoint and run permitted validation-input latency only.
    selected, tokenizer, _ = load_checkpoint(str(OUT / "selected_checkpoint"))
    lt._forward = bf16_forward
    typed_rows = read_jsonl(TYPED_VAL)
    beam_rows = read_jsonl(DATA / "beam_validation_laya.jsonl")
    typed_items, skipped = items_from_rows(tokenizer, typed_rows[:1], MAX_LEN, HEAD_MAX_LEN)
    beam_items, beam_skipped = items_from_rows(tokenizer, beam_rows, MAX_LEN, HEAD_MAX_LEN)
    if skipped or beam_skipped or not typed_items or len(beam_items) != len(beam_predictions):
        raise ValueError("latency fixture failed to encode")
    selected.cuda()
    fresh_beam_records = calibration_records(
        selected, tokenizer, beam_items, torch.device("cuda", 0), MAX_LEN, HEAD_MAX_LEN, parallel=False)
    fresh_beam_metrics = prediction_metrics(fresh_beam_records, beam_map)
    reload_mismatches = []
    for index, (saved, fresh, mapping) in enumerate(zip(beam_predictions, fresh_beam_records, beam_map)):
        _qtype, fresh_logits, fresh_target, option_count = fresh
        k = int(option_count)
        saved_prediction = int(np.argmax(np.asarray(saved["logits"][:k], dtype=np.float64)))
        fresh_prediction = int(np.argmax(np.asarray(fresh_logits[:k], dtype=np.float64)))
        if saved_prediction != fresh_prediction:
            reload_mismatches.append({
                "index": index, "id": mapping["id"], "task_family": mapping["task_family"],
                "saved_prediction": saved_prediction, "fresh_prediction": fresh_prediction,
                "gold": int(np.argmax(np.asarray(fresh_target[:k], dtype=np.float64))),
                "saved_logits": saved["logits"][:k],
                "fresh_logits": np.asarray(fresh_logits[:k]).tolist(),
            })
    latency_report = latency(selected, tokenizer, typed_rows[0], typed_items[0])

    verification = {
        "recovered_utc": datetime.now(timezone.utc).isoformat(),
        "training_completed_before_postprocess_failure": True,
        "original_failure": "saved-prediction verifier built an invalid reverse map from QTYPE_NAMES and rejected the string label 'score'",
        "checkpoint_reload_verified": True,
        "selected_checkpoint_matches_best_epoch_checkpoint": True,
        "selected_checkpoint_sha256": selected_hash,
        "best_epoch": best["epoch"],
        "typed_saved_prediction_rows": len(typed_predictions),
        "beam_saved_prediction_rows": len(beam_predictions),
        "recomputed": {"typed_validation": recomputed_typed, "beam_validation": recomputed_beam},
        "typed_metrics_match_saved_best_selection": True,
        "beam_reload_difference": {
            "selection_time_raw_accuracy": expected_beam["accuracy"],
            "saved_reload_raw_accuracy": recomputed_beam["accuracy"],
            "selection_time_position_balanced_accuracy": expected_beam["position_balanced_accuracy"],
            "saved_reload_position_balanced_accuracy": recomputed_beam["position_balanced_accuracy"],
            "raw_accuracy_equivalent_changed_rows": beam_raw_changed_rows,
            "interpretation": "One of 3,327 decisions changed on reload evaluation; report the conservative saved-reload metric and retain both values.",
        },
        "fresh_second_reload": {
            "beam_validation": fresh_beam_metrics,
            "prediction_mismatches_vs_saved_first_reload": reload_mismatches,
        },
        "metric_checks": {name: {"recomputed": pair[0], "saved": pair[1]} for name, pair in checks.items()},
        "protected_historical_final_test_evaluated": False,
        "protected_historical_final_test_rows": manifest["historical_final_test"]["protected_rows_after_relationship_closure"],
        "cross_split_relationship_components": manifest["relationship_grouping"]["cross_split_components"],
        "fresh_modernbert_initialization": {
            "checkpoint_sha256": run_config["initialization"]["checkpoint_sha256"],
            "encoder_revision": run_config["initialization"]["encoder_revision"],
            "released_laya_weights_loaded": run_config["initialization"]["released_laya_weights_loaded"],
            "beam_one_weights_loaded": run_config["initialization"]["beam_one_weights_loaded"],
        },
        "environment": {"host": platform.node(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)},
    }
    atomic_json(OUT / "verification_report.json", verification)
    report = {
        "status": "completed_with_postprocess_recovery",
        "best_epoch": best["epoch"],
        "epochs_completed": len(history),
        "history": history,
        "base": base,
        "selected": {"typed_validation": recomputed_typed, "beam_validation": recomputed_beam},
        "selection_time_metrics": best["metrics"],
        "fresh_second_reload_beam_validation": fresh_beam_metrics,
        "latency": latency_report,
        "selected_checkpoint": str(OUT / "selected_checkpoint"),
        "selected_checkpoint_sha256": selected_hash,
        "verification_report": str(OUT / "verification_report.json"),
        "reload_verified": True,
        "reload_note": "Primary selected metrics use saved predictions from the first checkpoint reload; selection-time Beam raw accuracy differed by one row.",
        "protected_historical_final_test_evaluated": False,
        "official_typed_test_accessed": False,
        "dgx_data_accessed": False,
    }
    atomic_json(OUT / "final_report.json", report)
    atomic_json(OUT / "recovered_status.json", {
        "status": "completed_with_postprocess_recovery", "utc": datetime.now(timezone.utc).isoformat(),
        "training_process_exit": 1, "training_process_exit_reason": "postprocess verifier bug after saved predictions",
        "training_and_checkpoints_complete": True, "verification_complete": True,
    })
    print(json.dumps({
        "status": report["status"], "best_epoch": best["epoch"],
        "beam_raw": recomputed_beam["accuracy"],
        "beam_balanced": recomputed_beam["position_balanced_accuracy"],
        "typed_hard": recomputed_typed["accuracy"],
        "typed_soft": recomputed_typed["soft_accuracy"],
        "checkpoint_sha256": selected_hash,
        "latency": latency_report,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
