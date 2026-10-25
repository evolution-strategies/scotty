from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import laya.train as lt
from beam_one_common import convert_rows, prediction_metrics, read_jsonl, sha256_file
from laya.train import calibration_records, items_from_rows, load_checkpoint, uses_parallel_layout
from smoke_beam_one import bf16_forward


ROOT = Path(os.environ["SCOTTY_REPRO_ROOT"]).expanduser().resolve()
OUT = ROOT / "runs/shared-protected-test-comparison-v1"
MANIFEST = OUT / "frozen_manifest.json"
MAX_LEN, HEAD_MAX_LEN, BATCH_SIZE = 1024, 256, 16


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def write_predictions(path, records, mappings, source_rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for record, mapping, source in zip(records, mappings, source_rows):
            qtype, logits_raw, target_raw, k_raw = record
            k = int(k_raw)
            logits = [float(value) for value in logits_raw[:k]]
            target = [float(value) for value in target_raw[:k]]
            prediction = max(range(k), key=lambda index: logits[index])
            gold = max(range(k), key=lambda index: target[index])
            row = {
                **mapping,
                "generation_request_id": source["metadata"]["generation_request_id"],
                "qtype_index": int(qtype),
                "logits": logits,
                "target": target,
                "options": k,
                "predicted_position": prediction,
                "gold_position": gold,
                "correct": prediction == gold,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def reread_metrics(path, mappings):
    rows = read_jsonl(path)
    records = [(row["qtype_index"], row["logits"], row["target"], row["options"]) for row in rows]
    return rows, prediction_metrics(records, mappings)


def cluster_bootstrap(values, clusters, seed, iterations=10000):
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[str(cluster)].append(float(value))
    keys = sorted(grouped)
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        picked = [rng.choice(keys) for _ in keys]
        sample = [value for key in picked for value in grouped[key]]
        samples.append(sum(sample) / len(sample))
    samples.sort()
    return {
        "clusters": len(keys), "iterations": iterations,
        "lower_95": samples[int(0.025 * iterations)],
        "upper_95": samples[int(0.975 * iterations) - 1],
    }


def paired_cluster_bootstrap(left, right, clusters, seed, iterations=10000):
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    interval = cluster_bootstrap(differences, clusters, seed, iterations)
    interval.update({
        "estimate": sum(differences) / len(differences),
        "left_only_correct": sum(bool(a) and not bool(b) for a, b in zip(left, right)),
        "right_only_correct": sum(bool(b) and not bool(a) for a, b in zip(left, right)),
    })
    return interval


def main():
    manifest = json.loads(MANIFEST.read_text())
    if manifest["state"] != "frozen_before_scoring":
        raise ValueError("comparison manifest is not frozen")
    if sha256_file(Path(__file__)) != manifest["evaluator_sha256"]:
        raise ValueError("evaluator changed after manifest freeze")
    test_path = Path(manifest["test"]["path"])
    if sha256_file(test_path) != manifest["test"]["sha256"]:
        raise ValueError("protected test changed after manifest freeze")
    if (OUT / "comparison_report.json").exists() or (OUT / "evaluation_state.json").exists():
        raise FileExistsError("comparison already evaluated or started")
    atomic_json(OUT / "evaluation_state.json", {
        "state": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256_file(MANIFEST), "test_access_events": 1,
    })

    test_source = read_jsonl(test_path)
    if len(test_source) != 3259:
        raise ValueError("protected test row count changed")
    test_rows, test_map = convert_rows(test_source)
    lt._forward = bf16_forward
    device = torch.device("cuda", 0)
    model_metrics, prediction_paths = {}, {}
    for model_name, checkpoint in manifest["models"].items():
        path = Path(checkpoint["path"])
        if sha256_file(path / "model.safetensors") != checkpoint["sha256"]:
            raise ValueError(f"{model_name} checkpoint identity changed")
        model, tokenizer, config = load_checkpoint(str(path))
        if uses_parallel_layout(config):
            raise ValueError(f"{model_name} is not sequential layout")
        items, skipped = items_from_rows(
            tokenizer, test_rows, MAX_LEN, HEAD_MAX_LEN, label_smoothing=0.0)
        if skipped or len(items) != len(test_source):
            raise ValueError({"model": model_name, "items": len(items), "skipped": skipped})
        model.to(device)
        records = calibration_records(
            model, tokenizer, items, device, MAX_LEN, HEAD_MAX_LEN,
            batch_size=BATCH_SIZE, parallel=False)
        model_metrics[model_name] = prediction_metrics(records, test_map)
        prediction_path = OUT / f"{model_name}_predictions.jsonl"
        write_predictions(prediction_path, records, test_map, test_source)
        prediction_paths[model_name] = prediction_path
        print("MODEL_COMPLETE " + json.dumps({
            "model": model_name,
            "accuracy": model_metrics[model_name]["accuracy"],
            "position_balanced_accuracy": model_metrics[model_name]["position_balanced_accuracy"],
        }, sort_keys=True), flush=True)
        del records, items, model
        torch.cuda.empty_cache()

    saved_rows, independently_recomputed = {}, {}
    for model_name, path in prediction_paths.items():
        rows, metrics = reread_metrics(path, test_map)
        saved_rows[model_name] = rows
        independently_recomputed[model_name] = metrics
        for field in ("accuracy", "position_balanced_accuracy"):
            if abs(metrics[field] - model_metrics[model_name][field]) > 1e-12:
                raise ValueError({"model": model_name, "metric": field,
                                  "in_memory": model_metrics[model_name][field], "saved": metrics[field]})

    clusters = [row["generation_request_id"] for row in saved_rows["plain_modernbert_union"]]
    correct = {
        model_name: [int(row["correct"]) for row in rows]
        for model_name, rows in saved_rows.items()
    }
    model_intervals = {
        model_name: cluster_bootstrap(values, clusters, 2026102400 + index)
        for index, (model_name, values) in enumerate(sorted(correct.items()))
    }
    pairs = [
        ("plain_modernbert_union", "original_beam_one"),
        ("plain_modernbert_union", "root_laya"),
        ("original_beam_one", "root_laya"),
    ]
    paired = {
        f"{left}_minus_{right}": paired_cluster_bootstrap(
            correct[left], correct[right], clusters, 2026102410 + index)
        for index, (left, right) in enumerate(pairs)
    }

    report = {
        "status": "completed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256_file(MANIFEST),
        "test_rows": len(test_source),
        "test_sha256": manifest["test"]["sha256"],
        "evaluation": {
            "shared_rows": True, "shared_conversion_and_scoring": True,
            "raw_argmax_only": True, "max_len": MAX_LEN,
            "head_max_len": HEAD_MAX_LEN, "batch_size": BATCH_SIZE, "dtype": "bf16",
        },
        "metrics": independently_recomputed,
        "scenario_cluster_confidence_intervals": model_intervals,
        "paired_scenario_cluster_differences": paired,
        "predictions": {
            model_name: {"path": str(path), "sha256": sha256_file(path), "rows": len(saved_rows[model_name])}
            for model_name, path in prediction_paths.items()
        },
        "independent_saved_prediction_recompute": True,
        "checkpoint_selection_after_test": False,
        "training_or_tuning_performed": False,
        "typed_official_test_accessed": False,
        "dgx_data_accessed": False,
        "interpretation": [
            "This is a shared whole-family holdout comparison, not a globally never-before-seen test for the research process.",
            "Original Beam One and released root Laya had historical scores on these same rows before this run; the new plain ModernBERT union model did not.",
            "Only three held-out families are represented, one per decision type; scenario-cluster intervals are more informative than family-level intervals here.",
            "The released root Laya training corpus is not fully documented, so its possible conceptual overlap with this test cannot be established.",
        ],
    }
    atomic_json(OUT / "comparison_report.json", report)
    atomic_json(OUT / "evaluation_state.json", {
        "state": "completed", "completed_utc": report["completed_utc"],
        "manifest_sha256": report["manifest_sha256"], "test_access_events": 1,
        "comparison_report_sha256": sha256_file(OUT / "comparison_report.json"),
    })
    print("SHARED_TEST_COMPLETE " + json.dumps({
        "metrics": {name: {"accuracy": value["accuracy"],
                            "position_balanced_accuracy": value["position_balanced_accuracy"]}
                    for name, value in independently_recomputed.items()},
        "paired": paired,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
