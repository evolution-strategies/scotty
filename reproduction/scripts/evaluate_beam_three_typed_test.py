from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import laya.train as lt
from laya.common import QTYPE_NAMES, build_sequence, ece_score, encode_text, serialize_state
from laya.train import calibration_records, items_from_rows, load_checkpoint, read_jsonl
from smoke_beam_one import bf16_forward


ROOT = Path(os.environ["SCOTTY_REPRO_ROOT"]).expanduser().resolve()
CHECKPOINT = ROOT / "runs/beam-three-modernbert-beam-only-v1/selected_checkpoint"
BEAM_TRAIN = ROOT / "analysis/plain-modernbert-beam-typed-union-protected-test-v2/beam_train_laya.jsonl"
TYPED = ROOT / "data/typed_decisions"
TEST = TYPED / "test.jsonl"
TEST_META = TYPED / "test_meta.json"
TRAIN = TYPED / "train.jsonl"
OUT = ROOT / "eval/beam-three-typed-decisions-official-test-v1"
PROTOCOL = OUT / "protocol.json"
EXPECTED_CHECKPOINT_SHA = "98511bc7d23a6fc55e61cef5e64985f633d8706a4bfc29969963fbb842f1fa11"
EXPECTED_TEST_SHA = "64043cd6dc7b805f9c4f43bc0f8c6115c207992b9bba7aac451f8bda49c6fad5"
EXPECTED_META_SHA = "b0615c7597400b97de0ade4b8403ed8244a5951fc061b67b8193816bcd233cb5"
EXPECTED_TRAIN_SHA = "178c52f3d8ce234dd0c5dd5eef48c8152721e53be0e360446ffacf51fb12d2de"
DATASET_REVISION = "468b1461d59e01d6404e1cf37431da962acd43a0"
MAX_LEN, HEAD_MAX_LEN = 1024, 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def normalize(text) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def schema_key(question: dict) -> str:
    return json.dumps(question, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def labels_for(question: dict) -> list[str]:
    qtype = question["type"]
    if qtype == "choice":
        return [str(key) for key in question["criteria"].keys()]
    if qtype == "noul":
        return ["false", "true"]
    return [str(index) for index in range(len(question["criteria"]))]


def metadata(test_meta: list[dict]) -> list[dict]:
    output = []
    for case_index, case in enumerate(test_meta):
        for question_id, question in case["questions"].items():
            gold = case["gold"][question_id]
            output.append({
                "case_index": case_index,
                "case_id": case.get("id", ""),
                "workflow": case.get("workflow", ""),
                "question_id": question_id,
                "question_type": question["type"],
                "labels": labels_for(question),
                "schema_key": schema_key(question),
                "hard_label": str(gold["label"]).lower(),
            })
    return output


def training_majorities() -> dict[str, str]:
    counts = defaultdict(Counter)
    for case in read_jsonl(TRAIN):
        for question_id, question in case["questions"].items():
            counts[schema_key(question)][str(case["gold"][question_id]["label"]).lower()] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in counts.items()}


def exact_overlap_report(test_rows: list[dict]) -> dict:
    beam_rows = read_jsonl(BEAM_TRAIN)
    beam_states = {normalize(serialize_state(row["state"])) for row in beam_rows}
    beam_full = set()
    for row in beam_rows:
        for question in row["questions"].values():
            beam_full.add(normalize(serialize_state(row["state"]) + "\n" + str(question["instructions"])))
    state_matches = 0
    full_matches = 0
    decisions = 0
    for row in test_rows:
        state = normalize(serialize_state(row["state"]))
        state_matches += int(state in beam_states)
        for question in row["questions"].values():
            decisions += 1
            full_matches += int(normalize(serialize_state(row["state"]) + "\n" + str(question["instructions"])) in beam_full)
    return {
        "test_cases": len(test_rows),
        "test_decisions": decisions,
        "exact_normalized_state_matches": state_matches,
        "exact_normalized_state_plus_instruction_matches": full_matches,
        "rows_excluded": 0,
    }


def preflight(tok, test_rows: list[dict], items: list[dict], skipped: dict) -> dict:
    if skipped or len(items) != 2000:
        raise ValueError({"items": len(items), "skipped": skipped})
    index = 0
    truncations, encoded_lengths, distinct_failures, marker_failures = [], [], 0, 0
    for case in test_rows:
        state_ids = encode_text(
            tok, serialize_state(case["state"]).replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        for _question_id in case["questions"]:
            item = items[index]
            ids, markers, stats, trunc = build_sequence(
                tok, None, item["q"], MAX_LEN, HEAD_MAX_LEN, state_ids=state_ids,
                return_stats=True, return_truncation_stats=True)
            encoded_lengths.append(len(ids))
            truncations.append(trunc)
            distinct_failures += int(stats["options_distinct"] != item["k"])
            marker_failures += int(len(markers) != item["k"])
            index += 1
    return {
        "items": len(items),
        "skipped": skipped,
        "all_options_distinct": distinct_failures == 0,
        "all_option_markers_fit": marker_failures == 0,
        "distinct_option_failures": distinct_failures,
        "option_marker_failures": marker_failures,
        "encoded_length": {
            "min": min(encoded_lengths),
            "median": float(np.median(encoded_lengths)),
            "p95": float(np.quantile(encoded_lengths, 0.95)),
            "max": max(encoded_lengths),
        },
        "state_truncation": {
            "rows": sum(row["truncated"] for row in truncations),
            "total_tokens_dropped": sum(row["state_tokens_dropped"] for row in truncations),
            "max_tokens_dropped": max(row["state_tokens_dropped"] for row in truncations),
        },
    }


def freeze() -> None:
    if OUT.exists():
        raise FileExistsError(f"official Typed test evaluation directory already exists: {OUT}")
    for path, expected in (
        (CHECKPOINT / "model.safetensors", EXPECTED_CHECKPOINT_SHA),
        (TEST, EXPECTED_TEST_SHA),
        (TEST_META, EXPECTED_META_SHA),
        (TRAIN, EXPECTED_TRAIN_SHA),
    ):
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"hash mismatch for {path}: {actual} != {expected}")
    OUT.mkdir(parents=True, exist_ok=False)
    test_rows = read_jsonl(TEST)
    test_meta = json.loads(TEST_META.read_text())
    if len(test_rows) != 400 or len(test_meta) != 400:
        raise ValueError("official test case count mismatch")
    tok = AutoTokenizer.from_pretrained(CHECKPOINT / "tokenizer", local_files_only=True)
    items, skipped = items_from_rows(tok, test_rows, MAX_LEN, HEAD_MAX_LEN)
    if len(metadata(test_meta)) != 2000:
        raise ValueError("official test decision metadata count mismatch")
    protocol = {
        "name": "Beam Three on LocalLLaMA/typed-decisions official test",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(CHECKPOINT), "sha256": EXPECTED_CHECKPOINT_SHA, "selection": "already frozen by Beam validation only"},
        "dataset": {
            "repo": "LocalLLaMA/typed-decisions", "config": "all", "revision": DATASET_REVISION,
            "split": "test", "cases": 400, "decisions": 2000,
            "test_jsonl": {"path": str(TEST), "sha256": EXPECTED_TEST_SHA},
            "test_metadata": {"path": str(TEST_META), "sha256": EXPECTED_META_SHA},
            "train_jsonl_for_majority_baseline_only": {"path": str(TRAIN), "sha256": EXPECTED_TRAIN_SHA},
        },
        "serialization": {
            "adapter": "pinned laya.train.items_from_rows and canonical sequential option layout",
            "model_input_fields": ["state", "questions"],
            "gold_or_metadata_in_model_input": False,
            "max_length": MAX_LEN, "head_max_length": HEAD_MAX_LEN, "dtype": "bfloat16",
            "temperature": 1.0, "temperature_fitting": False,
        },
        "metrics": {
            "hard_accuracy": "argmax raw model probability versus teacher hard label",
            "soft_teacher_agreement": "teacher target probability assigned to model argmax",
            "teacher_expected_agreement_ceiling": "mean maximum teacher target probability; not hard-label accuracy",
            "brier": "mean sum over options of squared model-minus-teacher probability",
            "ece": "15-bin ECE of raw model max probability versus hard-label correctness; descriptive, not calibrated",
            "ordinal_mae": "absolute difference between predicted and hard-label ordinal indices",
            "ordinal_expected_score_mae": "absolute difference between model and teacher expected ordinal indices",
            "case_cluster_ci": "10,000 bootstrap resamples of 400 cases, seed 0",
            "breakdowns": ["question_type", "workflow"],
        },
        "baselines": ["training-schema majority hard accuracy", "uniform random expected hard accuracy", "teacher expected-agreement ceiling"],
        "prohibitions": {
            "training": False, "calibration": False, "tuning": False, "checkpoint_reselection": False,
            "beam_historical_test_access": False, "beam_two_evaluation": False,
        },
        "preflight": preflight(tok, test_rows, items, skipped),
        "exact_overlap_with_beam_training": exact_overlap_report(test_rows),
        "model_scoring_performed": False,
    }
    atomic_json(PROTOCOL, protocol)
    (OUT / "PROTOCOL_SHA256").write_text(sha256(PROTOCOL) + "  protocol.json\n")
    atomic_json(OUT / "status.json", {"state": "frozen_not_scored", "utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha256(PROTOCOL)})
    print(json.dumps({"protocol": str(PROTOCOL), "protocol_sha256": sha256(PROTOCOL), "preflight": protocol["preflight"], "overlap": protocol["exact_overlap_with_beam_training"]}, sort_keys=True))


def ece(confidences: list[float], correct: list[bool]) -> float:
    return float(ece_score(np.asarray(confidences, dtype=float), np.asarray(correct, dtype=float), bins=15))


def metric_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"items": 0}
    hard = [bool(row["correct"]) for row in rows]
    soft = [float(row["teacher_probability_of_prediction"]) for row in rows]
    briers = [float(row["brier"]) for row in rows]
    ce = [float(row["cross_entropy"]) for row in rows]
    confidence = [float(row["confidence"]) for row in rows]
    result = {
        "items": len(rows),
        "hard_accuracy": float(np.mean(hard)),
        "soft_teacher_agreement": float(np.mean(soft)),
        "teacher_expected_agreement_ceiling": float(np.mean([row["teacher_max_probability"] for row in rows])),
        "brier": float(np.mean(briers)),
        "cross_entropy": float(np.mean(ce)),
        "ece_15bin_descriptive_raw": ece(confidence, hard),
        "mean_confidence": float(np.mean(confidence)),
    }
    ordinal = [row for row in rows if row["question_type"] == "score"]
    if ordinal:
        result["ordinal_items"] = len(ordinal)
        result["ordinal_mae_hard_label"] = float(np.mean([row["ordinal_absolute_error"] for row in ordinal]))
        result["ordinal_expected_score_mae"] = float(np.mean([row["ordinal_expected_score_absolute_error"] for row in ordinal]))
        result["ordinal_within_one_hard_level"] = float(np.mean([row["ordinal_absolute_error"] <= 1 for row in ordinal]))
    return result


def clustered_ci(rows: list[dict], value_key: str, draws: int = 10000, seed: int = 0) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["case_index"])].append(float(row[value_key]))
    keys = sorted(grouped)
    sums = np.asarray([sum(grouped[key]) for key in keys], dtype=float)
    counts = np.asarray([len(grouped[key]) for key in keys], dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=float)
    for i in range(draws):
        sample = rng.integers(0, len(keys), size=len(keys))
        estimates[i] = sums[sample].sum() / counts[sample].sum()
    return {
        "unit": "case", "clusters": len(keys), "draws": draws, "seed": seed,
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def reports(predictions: list[dict]) -> dict:
    by_type = defaultdict(list)
    by_workflow = defaultdict(list)
    for row in predictions:
        by_type[row["question_type"]].append(row)
        by_workflow[row["workflow"]].append(row)
    return {
        "overall": metric_summary(predictions),
        "by_type": {key: metric_summary(value) for key, value in sorted(by_type.items())},
        "by_workflow": {key: metric_summary(value) for key, value in sorted(by_workflow.items())},
        "case_cluster_95pct_ci": {
            "hard_accuracy": clustered_ci(predictions, "correct"),
            "soft_teacher_agreement": clustered_ci(predictions, "teacher_probability_of_prediction"),
        },
    }


def score() -> None:
    status = json.loads((OUT / "status.json").read_text())
    if status.get("state") != "frozen_not_scored":
        raise ValueError(f"unexpected evaluation state: {status}")
    protocol_sha = sha256(PROTOCOL)
    recorded = (OUT / "PROTOCOL_SHA256").read_text().split()[0]
    if protocol_sha != recorded:
        raise ValueError("frozen protocol hash mismatch")
    for path, expected in (
        (CHECKPOINT / "model.safetensors", EXPECTED_CHECKPOINT_SHA),
        (TEST, EXPECTED_TEST_SHA), (TEST_META, EXPECTED_META_SHA), (TRAIN, EXPECTED_TRAIN_SHA)):
        if sha256(path) != expected:
            raise ValueError(f"hash changed before scoring: {path}")
    if (OUT / "predictions.jsonl").exists() or (OUT / "final_report.json").exists():
        raise FileExistsError("official test output already exists")
    atomic_json(OUT / "status.json", {"state": "scoring", "utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "protocol_sha256": protocol_sha})

    test_rows = read_jsonl(TEST)
    test_meta = json.loads(TEST_META.read_text())
    meta = metadata(test_meta)
    model, tok, _ = load_checkpoint(str(CHECKPOINT))
    lt._forward = bf16_forward
    items, skipped = items_from_rows(tok, test_rows, MAX_LEN, HEAD_MAX_LEN)
    if skipped or len(items) != 2000 or len(meta) != 2000:
        raise ValueError({"items": len(items), "meta": len(meta), "skipped": skipped})
    model.cuda().eval()
    start = time.perf_counter()
    records = calibration_records(model, tok, items, torch.device("cuda", 0), MAX_LEN, HEAD_MAX_LEN, batch_size=16, parallel=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    del model
    torch.cuda.empty_cache()

    majorities = training_majorities()
    predictions = []
    for record, info in zip(records, meta):
        qtype, logits, target, k = record
        raw = np.asarray(logits[:k], dtype=float)
        shifted = raw - raw.max()
        probs = np.exp(shifted); probs /= probs.sum()
        teacher = np.asarray(target[:k], dtype=float); teacher /= teacher.sum()
        predicted_index = int(raw.argmax())
        hard_index = int(teacher.argmax())
        labels = info["labels"]
        predicted_label = labels[predicted_index]
        hard_label = labels[hard_index]
        cross_entropy = float(-(teacher * np.log(np.clip(probs, 1e-12, 1.0))).sum())
        row = {
            **info,
            "qtype_index": int(qtype),
            "predicted_label": predicted_label,
            "hard_label": hard_label,
            "correct": predicted_index == hard_index,
            "raw_logits": raw.tolist(),
            "probabilities": probs.tolist(),
            "teacher_probabilities": teacher.tolist(),
            "confidence": float(probs.max()),
            "teacher_probability_of_prediction": float(teacher[predicted_index]),
            "teacher_max_probability": float(teacher.max()),
            "brier": float(np.square(probs - teacher).sum()),
            "cross_entropy": cross_entropy,
            "option_count": int(k),
            "majority_prediction": majorities.get(info["schema_key"]),
        }
        if info["question_type"] == "score":
            model_score = float(np.dot(probs, np.arange(k)))
            teacher_score = float(np.dot(teacher, np.arange(k)))
            row["ordinal_absolute_error"] = abs(predicted_index - hard_index)
            row["ordinal_model_expected_score"] = model_score
            row["ordinal_teacher_expected_score"] = teacher_score
            row["ordinal_expected_score_absolute_error"] = abs(model_score - teacher_score)
        predictions.append(row)
    prediction_path = OUT / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions))

    online = reports(predictions)
    saved = [json.loads(line) for line in prediction_path.read_text().splitlines() if line]
    recomputed = reports(saved)
    if json.dumps(online, sort_keys=True) != json.dumps(recomputed, sort_keys=True):
        raise ValueError("independent saved-prediction metric recomputation mismatch")

    majority_valid = [row for row in saved if row["majority_prediction"] is not None]
    majority_accuracy = float(np.mean([
        str(row["majority_prediction"]).lower() == str(row["hard_label"]).lower() for row in majority_valid]))
    chance = float(np.mean([1.0 / row["option_count"] for row in saved]))
    report = {
        "status": "completed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(CHECKPOINT), "sha256": EXPECTED_CHECKPOINT_SHA},
        "dataset": {"repo": "LocalLLaMA/typed-decisions", "revision": DATASET_REVISION, "split": "test", "cases": 400, "decisions": 2000, "sha256": EXPECTED_TEST_SHA},
        "protocol": {"path": str(PROTOCOL), "sha256": protocol_sha},
        "metrics": online,
        "baselines": {
            "train_schema_majority_hard_accuracy": majority_accuracy,
            "train_schema_majority_covered_decisions": len(majority_valid),
            "uniform_random_expected_hard_accuracy": chance,
            "teacher_expected_agreement_ceiling": online["overall"]["teacher_expected_agreement_ceiling"],
            "note": "Teacher expected-agreement ceiling is the best deterministic expected agreement with the teacher distributions, not a ceiling on hard-label accuracy."
        },
        "inference": {"elapsed_seconds": elapsed, "decisions_per_second_batch16": len(saved) / elapsed, "dtype": "bfloat16", "temperature": 1.0},
        "predictions": {"path": str(prediction_path), "rows": len(saved), "sha256": sha256(prediction_path)},
        "independent_saved_prediction_recomputation_match": True,
        "preflight": json.loads(PROTOCOL.read_text())["preflight"],
        "exact_overlap_with_beam_training": json.loads(PROTOCOL.read_text())["exact_overlap_with_beam_training"],
        "calibration_or_tuning_performed": False,
        "beam_historical_test_accessed": False,
        "beam_two_evaluated": False,
    }
    atomic_json(OUT / "final_report.json", report)
    report_sha = sha256(OUT / "final_report.json")
    atomic_json(OUT / "status.json", {"state": "completed", "utc": datetime.now(timezone.utc).isoformat(), "report_sha256": report_sha, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA})
    print(json.dumps({
        "status": "completed",
        "hard_accuracy": online["overall"]["hard_accuracy"],
        "soft_teacher_agreement": online["overall"]["soft_teacher_agreement"],
        "brier": online["overall"]["brier"],
        "ece": online["overall"]["ece_15bin_descriptive_raw"],
        "ordinal_mae": online["overall"].get("ordinal_mae_hard_label"),
        "report_sha256": report_sha,
    }, sort_keys=True))


def main():
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
        atomic_json(OUT / "status.json", {"state": "failed", "utc": datetime.now(timezone.utc).isoformat(), "error": repr(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
