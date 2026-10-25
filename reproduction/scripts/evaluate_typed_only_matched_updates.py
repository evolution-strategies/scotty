from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import evaluate_beam_three_typed_test as typed_core
import laya.train as lt
from beam_one_common import convert_rows, prediction_metrics, read_jsonl as read_audit_jsonl
from laya.train import calibration_records, items_from_rows, load_checkpoint, uses_parallel_layout
from smoke_beam_one import bf16_forward


ROOT = Path(os.environ["SCOTTY_REPRO_ROOT"]).expanduser().resolve()
TRAIN_RUN = ROOT / "runs/typed-only-matched-updates-v1"
CHECKPOINT = TRAIN_RUN / "selected_checkpoint"
OUT = ROOT / "eval/typed-only-matched-updates-v1"
MANIFEST = OUT / "frozen_manifest.json"
TRAIN_PROTOCOL = ROOT / "analysis/typed-only-matched-updates-v1/protocol.json"
BEAM_TEST = ROOT / "analysis/beam-one-family-disjoint-v1/test_source.jsonl"
TYPED_TEST = ROOT / "data/typed_decisions/test.jsonl"
TYPED_META = ROOT / "data/typed_decisions/test_meta.json"
TYPED_TRAIN_FULL = ROOT / "data/typed_decisions/train.jsonl"
BEAM1_BEAM_PRED = ROOT / "runs/shared-protected-test-comparison-v1/plain_modernbert_union_predictions.jsonl"
BEAM1_TYPED_PRED = ROOT / "eval/beam-two-typed-decisions-official-test-v1/predictions.jsonl"
MAX_LEN, HEAD_MAX_LEN, BATCH_SIZE = 1024, 256, 16

EXPECTED = {
    TRAIN_PROTOCOL: "c5a209f5df348225233b944ab40737fe9245d03851fd496b015a2e7b3c5ad465",
    BEAM_TEST: "30298aad191624cda56a83e1af54702b364c455a709387e61f9bc235ca3988b7",
    TYPED_TEST: "64043cd6dc7b805f9c4f43bc0f8c6115c207992b9bba7aac451f8bda49c6fad5",
    TYPED_META: "b0615c7597400b97de0ade4b8403ed8244a5951fc061b67b8193816bcd233cb5",
    TYPED_TRAIN_FULL: "178c52f3d8ce234dd0c5dd5eef48c8152721e53be0e360446ffacf51fb12d2de",
    BEAM1_BEAM_PRED: "4fc3f614bc2236f00a434b89fa8beb7726aa0676a44f5588b604d499e43693c6",
    BEAM1_TYPED_PRED: "9332402c7ecd318e8561fd7d66bbf323c8cf614085e39bb456442d2c09271ee4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


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
        "unit_clusters": len(keys),
        "iterations": iterations,
        "seed": seed,
        "lower_95": samples[int(0.025 * iterations)],
        "upper_95": samples[int(0.975 * iterations) - 1],
    }


def paired_cluster_bootstrap(left, right, clusters, seed, iterations=10000):
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    result = cluster_bootstrap(differences, clusters, seed, iterations)
    result.update({
        "estimate_control_minus_beam1": sum(differences) / len(differences),
        "control_only_correct": sum(bool(a) and not bool(b) for a, b in zip(left, right)),
        "beam1_only_correct": sum(bool(b) and not bool(a) for a, b in zip(left, right)),
        "both_correct": sum(bool(a) and bool(b) for a, b in zip(left, right)),
        "both_wrong": sum(not bool(a) and not bool(b) for a, b in zip(left, right)),
    })
    return result


def freeze() -> None:
    if OUT.exists():
        raise FileExistsError(f"evaluation directory already exists: {OUT}")
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"frozen artifact hash mismatch: {path}: {actual} != {expected}")
    training = json.loads((TRAIN_RUN / "final_report.json").read_text(encoding="utf-8"))
    if training["status"] != "training_and_validation_completed" or not training["reload_verified"]:
        raise ValueError("training did not complete and reload cleanly")
    if training["optimizer_updates"] != 849 or training["item_exposures"] != 54276:
        raise ValueError("matched-update accounting mismatch")
    if training["tests_accessed"]:
        raise ValueError("training report unexpectedly accessed tests")
    checkpoint_sha = sha256(CHECKPOINT / "model.safetensors")
    if checkpoint_sha != training["selected_checkpoint_sha256"]:
        raise ValueError("selected checkpoint hash mismatch")

    beam_rows = read_audit_jsonl(BEAM_TEST)
    typed_rows = typed_core.read_jsonl(TYPED_TEST)
    typed_meta = json.loads(TYPED_META.read_text(encoding="utf-8"))
    if len(beam_rows) != 3259 or len(typed_rows) != 400 or len(typed_meta) != 400:
        raise ValueError("frozen test counts changed")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT / "tokenizer", local_files_only=True)
    typed_items, typed_skipped = items_from_rows(tokenizer, typed_rows, MAX_LEN, HEAD_MAX_LEN)
    typed_preflight = typed_core.preflight(tokenizer, typed_rows, typed_items, typed_skipped)
    if len(typed_core.metadata(typed_meta)) != 2000:
        raise ValueError("Typed decision metadata count changed")

    OUT.mkdir(parents=True, exist_ok=False)
    manifest = {
        "name": "typed-only-matched-updates-v1 frozen final evaluation",
        "state": "frozen_before_scoring",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "retrospective_status": (
            "Both test sets were inspected in earlier research. This is a retrospective controlled ablation, "
            "not a pristine confirmatory evaluation. No test result may alter training, selection, or this protocol."
        ),
        "checkpoint": {
            "path": str(CHECKPOINT),
            "sha256": checkpoint_sha,
            "selected_update": training["best_update"],
            "selection_used_tests": False,
        },
        "training_protocol": {"path": str(TRAIN_PROTOCOL), "sha256": EXPECTED[TRAIN_PROTOCOL]},
        "evaluator": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))},
        "tests": {
            "beam_protected": {"path": str(BEAM_TEST), "sha256": EXPECTED[BEAM_TEST], "decisions": 3259, "clusters": 543},
            "typed_official": {
                "path": str(TYPED_TEST), "sha256": EXPECTED[TYPED_TEST], "cases": 400, "decisions": 2000,
                "metadata_path": str(TYPED_META), "metadata_sha256": EXPECTED[TYPED_META],
                "dataset_revision": "468b1461d59e01d6404e1cf37431da962acd43a0",
                "preflight": typed_preflight,
            },
        },
        "reference_beam1_predictions": {
            "beam_protected": {"path": str(BEAM1_BEAM_PRED), "sha256": EXPECTED[BEAM1_BEAM_PRED]},
            "typed_official": {"path": str(BEAM1_TYPED_PRED), "sha256": EXPECTED[BEAM1_TYPED_PRED]},
            "rescored": False,
        },
        "scoring": {
            "dtype": "bfloat16",
            "temperature": 1.0,
            "calibration": False,
            "batch_size": 16,
            "max_len": 1024,
            "head_max_len": 256,
            "layout": "sequential",
            "beam_metrics": ["raw accuracy", "position-balanced accuracy", "per-type accuracy"],
            "typed_metrics": ["hard accuracy", "soft teacher agreement", "Brier", "raw ECE", "ordinal MAE", "per-type", "per-workflow"],
            "paired_uncertainty": "10,000 cluster bootstrap draws: Beam generation request and Typed case",
        },
        "prohibitions": {
            "training": True,
            "calibration": True,
            "threshold_fit": True,
            "checkpoint_reselection": True,
            "prompt_or_adapter_revision": True,
        },
        "model_scoring_performed": False,
    }
    atomic_json(MANIFEST, manifest)
    (OUT / "MANIFEST_SHA256").write_text(sha256(MANIFEST) + "  frozen_manifest.json\n", encoding="utf-8")
    atomic_json(OUT / "status.json", {
        "state": "frozen_not_scored", "utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(MANIFEST),
    })
    print(json.dumps({"manifest": str(MANIFEST), "sha256": sha256(MANIFEST), "checkpoint_sha256": checkpoint_sha}, sort_keys=True))


def write_beam_predictions(path, records, mappings, source_rows):
    with Path(path).open("x", encoding="utf-8") as handle:
        for record, mapping, source in zip(records, mappings, source_rows):
            qtype, logits_raw, target_raw, k_raw = record
            k = int(k_raw)
            logits = [float(value) for value in logits_raw[:k]]
            target = [float(value) for value in target_raw[:k]]
            pred = max(range(k), key=lambda idx: logits[idx])
            gold = max(range(k), key=lambda idx: target[idx])
            row = {
                **mapping,
                "generation_request_id": source["metadata"]["generation_request_id"],
                "qtype_index": int(qtype), "logits": logits, "target": target, "options": k,
                "predicted_position": pred, "gold_position": gold, "correct": pred == gold,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def score_beam(model, tokenizer):
    source = read_audit_jsonl(BEAM_TEST)
    rows, mappings = convert_rows(source)
    items, skipped = items_from_rows(tokenizer, rows, MAX_LEN, HEAD_MAX_LEN)
    if skipped or len(items) != 3259:
        raise ValueError({"beam_items": len(items), "skipped": skipped})
    records = calibration_records(
        model, tokenizer, items, torch.device("cuda", 0), MAX_LEN, HEAD_MAX_LEN,
        batch_size=BATCH_SIZE, parallel=False,
    )
    path = OUT / "beam_protected_predictions.jsonl"
    write_beam_predictions(path, records, mappings, source)
    saved = read_audit_jsonl(path)
    reread_records = [(row["qtype_index"], row["logits"], row["target"], row["options"]) for row in saved]
    result = prediction_metrics(reread_records, mappings)

    reference = read_audit_jsonl(BEAM1_BEAM_PRED)
    if len(reference) != len(saved):
        raise ValueError("Beam paired count mismatch")
    for left, right in zip(saved, reference):
        if left["id"] != right["id"] or left["answer_id"] != right["answer_id"]:
            raise ValueError("Beam paired row identity mismatch")
    clusters = [row["generation_request_id"] for row in saved]
    paired = paired_cluster_bootstrap(
        [row["correct"] for row in saved], [row["correct"] for row in reference], clusters, 2026102451
    )
    ci = cluster_bootstrap([row["correct"] for row in saved], clusters, 2026102450)
    return {
        "metrics": result,
        "cluster_95pct_ci": ci,
        "paired_vs_beam1": paired,
        "predictions": {"path": str(path), "rows": len(saved), "sha256": sha256(path)},
        "independent_saved_prediction_recompute": True,
    }


def typed_majorities():
    counts = defaultdict(Counter)
    for case in typed_core.read_jsonl(TYPED_TRAIN_FULL):
        for qid, question in case["questions"].items():
            counts[typed_core.schema_key(question)][str(case["gold"][qid]["label"]).lower()] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in counts.items()}


def score_typed(model, tokenizer):
    test_rows = typed_core.read_jsonl(TYPED_TEST)
    test_meta = json.loads(TYPED_META.read_text(encoding="utf-8"))
    meta = typed_core.metadata(test_meta)
    items, skipped = items_from_rows(tokenizer, test_rows, MAX_LEN, HEAD_MAX_LEN)
    if skipped or len(items) != 2000 or len(meta) != 2000:
        raise ValueError({"typed_items": len(items), "meta": len(meta), "skipped": skipped})
    started = time.perf_counter()
    records = calibration_records(
        model, tokenizer, items, torch.device("cuda", 0), MAX_LEN, HEAD_MAX_LEN,
        batch_size=BATCH_SIZE, parallel=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    majorities = typed_majorities()
    predictions = []
    for record, info in zip(records, meta):
        qtype, logits, target, k = record
        raw = np.asarray(logits[:k], dtype=float)
        probs = np.exp(raw - raw.max()); probs /= probs.sum()
        teacher = np.asarray(target[:k], dtype=float); teacher /= teacher.sum()
        pred, gold = int(raw.argmax()), int(teacher.argmax())
        row = {
            **info,
            "qtype_index": int(qtype),
            "predicted_label": info["labels"][pred],
            "hard_label": info["labels"][gold],
            "correct": pred == gold,
            "raw_logits": raw.tolist(), "probabilities": probs.tolist(),
            "teacher_probabilities": teacher.tolist(), "confidence": float(probs.max()),
            "teacher_probability_of_prediction": float(teacher[pred]),
            "teacher_max_probability": float(teacher.max()),
            "brier": float(np.square(probs - teacher).sum()),
            "cross_entropy": float(-(teacher * np.log(np.clip(probs, 1e-12, 1))).sum()),
            "option_count": int(k), "majority_prediction": majorities.get(info["schema_key"]),
        }
        if info["question_type"] == "score":
            model_score = float(np.dot(probs, np.arange(k)))
            teacher_score = float(np.dot(teacher, np.arange(k)))
            row["ordinal_absolute_error"] = abs(pred - gold)
            row["ordinal_model_expected_score"] = model_score
            row["ordinal_teacher_expected_score"] = teacher_score
            row["ordinal_expected_score_absolute_error"] = abs(model_score - teacher_score)
        predictions.append(row)
    path = OUT / "typed_official_predictions.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions), encoding="utf-8")
    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = typed_core.reports(saved)

    reference = [json.loads(line) for line in BEAM1_TYPED_PRED.read_text(encoding="utf-8").splitlines() if line]
    if len(reference) != len(saved):
        raise ValueError("Typed paired count mismatch")
    for left, right in zip(saved, reference):
        identity_left = (left["case_index"], left["question_id"], left["workflow"], left["hard_label"])
        identity_right = (right["case_index"], right["question_id"], right["workflow"], right["hard_label"])
        if identity_left != identity_right or left["teacher_probabilities"] != right["teacher_probabilities"]:
            raise ValueError("Typed paired row identity mismatch")
    clusters = [row["case_index"] for row in saved]
    paired_hard = paired_cluster_bootstrap(
        [row["correct"] for row in saved], [row["correct"] for row in reference], clusters, 2026102453
    )
    paired_soft = cluster_bootstrap(
        [float(left["teacher_probability_of_prediction"]) - float(right["teacher_probability_of_prediction"])
         for left, right in zip(saved, reference)], clusters, 2026102454
    )
    paired_soft["estimate_control_minus_beam1"] = float(np.mean([
        float(left["teacher_probability_of_prediction"]) - float(right["teacher_probability_of_prediction"])
        for left, right in zip(saved, reference)
    ]))
    majority_valid = [row for row in saved if row["majority_prediction"] is not None]
    return {
        "metrics": result,
        "paired_vs_beam1": {"hard_accuracy": paired_hard, "soft_teacher_agreement": paired_soft},
        "baselines": {
            "train_schema_majority_hard_accuracy": float(np.mean([
                str(row["majority_prediction"]).lower() == str(row["hard_label"]).lower()
                for row in majority_valid
            ])),
            "uniform_random_expected_hard_accuracy": float(np.mean([1.0 / row["option_count"] for row in saved])),
            "teacher_expected_agreement_ceiling": result["overall"]["teacher_expected_agreement_ceiling"],
        },
        "inference": {"elapsed_seconds": elapsed, "decisions_per_second_batch16": len(saved) / elapsed},
        "predictions": {"path": str(path), "rows": len(saved), "sha256": sha256(path)},
        "independent_saved_prediction_recompute": True,
    }


def score() -> None:
    status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "frozen_not_scored":
        raise ValueError(f"unexpected evaluation state: {status}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    recorded = (OUT / "MANIFEST_SHA256").read_text(encoding="utf-8").split()[0]
    if sha256(MANIFEST) != recorded or sha256(Path(__file__)) != manifest["evaluator"]["sha256"]:
        raise ValueError("frozen manifest/evaluator changed")
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise ValueError(f"frozen artifact changed: {path}")
    checkpoint_sha = manifest["checkpoint"]["sha256"]
    if sha256(CHECKPOINT / "model.safetensors") != checkpoint_sha:
        raise ValueError("selected checkpoint changed before scoring")
    if (OUT / "final_report.json").exists() or (OUT / "beam_protected_predictions.jsonl").exists() or (OUT / "typed_official_predictions.jsonl").exists():
        raise FileExistsError("evaluation outputs already exist")
    atomic_json(OUT / "status.json", {
        "state": "scoring", "utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "manifest_sha256": recorded,
    })

    model, tokenizer, config = load_checkpoint(str(CHECKPOINT))
    if uses_parallel_layout(config):
        raise ValueError("control checkpoint unexpectedly uses parallel layout")
    lt._forward = bf16_forward
    model.cuda().eval()
    beam = score_beam(model, tokenizer)
    print("BEAM_TEST_COMPLETE " + json.dumps({
        "accuracy": beam["metrics"]["accuracy"],
        "position_balanced_accuracy": beam["metrics"]["position_balanced_accuracy"],
        "paired_vs_beam1": beam["paired_vs_beam1"]["estimate_control_minus_beam1"],
    }, sort_keys=True), flush=True)
    typed = score_typed(model, tokenizer)
    del model
    torch.cuda.empty_cache()
    print("TYPED_TEST_COMPLETE " + json.dumps({
        "hard_accuracy": typed["metrics"]["overall"]["hard_accuracy"],
        "soft_teacher_agreement": typed["metrics"]["overall"]["soft_teacher_agreement"],
        "paired_vs_beam1": typed["paired_vs_beam1"]["hard_accuracy"]["estimate_control_minus_beam1"],
    }, sort_keys=True), flush=True)

    report = {
        "status": "completed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {"path": str(MANIFEST), "sha256": recorded},
        "checkpoint": {"path": str(CHECKPOINT), "sha256": checkpoint_sha},
        "beam_protected_test": beam,
        "typed_official_test": typed,
        "beam1_reference_predictions_rescored": False,
        "training_or_tuning_performed": False,
        "calibration_performed": False,
        "independent_saved_prediction_recompute": True,
        "retrospective_caveat": manifest["retrospective_status"],
    }
    atomic_json(OUT / "final_report.json", report)
    report_sha = sha256(OUT / "final_report.json")
    atomic_json(OUT / "status.json", {
        "state": "completed", "utc": report["completed_utc"], "report_sha256": report_sha,
        "checkpoint_sha256": checkpoint_sha,
    })
    print("EVALUATION_COMPLETE " + json.dumps({
        "report_sha256": report_sha,
        "beam_accuracy": beam["metrics"]["accuracy"],
        "typed_accuracy": typed["metrics"]["overall"]["hard_accuracy"],
    }, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "score"))
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
        return
    try:
        score()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUT / "status.json", {
            "state": "failed", "utc": datetime.now(timezone.utc).isoformat(),
            "error": repr(exc), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
