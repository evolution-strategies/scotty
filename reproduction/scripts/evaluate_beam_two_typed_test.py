from __future__ import annotations

import argparse
import json
import os
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import evaluate_beam_three_typed_test as core
from laya.common import serialize_state
from laya.train import items_from_rows


ROOT = Path(os.environ["SCOTTY_REPRO_ROOT"]).expanduser().resolve()
CHECKPOINT = ROOT / "models/beam-two/selected_checkpoint"
OUT = ROOT / "eval/beam-two-typed-decisions-official-test-v1"
EXPECTED_CHECKPOINT_SHA = "b7983f2820f46d58038920a860223a1e18826448c1833541f8bc52ba6e3fb783"
BEAM3_EVAL = ROOT / "eval/beam-three-typed-decisions-official-test-v1"
UNION_TRAIN = ROOT / "analysis/plain-modernbert-beam-typed-union-protected-test-v2/train_union.jsonl"
BEAM_VALIDATION = ROOT / "analysis/plain-modernbert-beam-typed-union-protected-test-v2/beam_validation_laya.jsonl"
TYPED_STAGE_TRAIN = ROOT / "analysis/beam-one-laya-stage-v1/train.jsonl"
TYPED_STAGE_VALIDATION = ROOT / "analysis/beam-one-laya-stage-v1/validation.jsonl"


core.CHECKPOINT = CHECKPOINT
core.OUT = OUT
core.PROTOCOL = OUT / "protocol.json"
core.EXPECTED_CHECKPOINT_SHA = EXPECTED_CHECKPOINT_SHA


def case_signature(row: dict) -> str:
    return core.normalize(json.dumps({
        "state": row["state"],
        "questions": row["questions"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def decision_signatures(row: dict) -> list[str]:
    state = serialize_state(row["state"])
    return [
        core.normalize(state + "\n" + json.dumps(question, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        for question in row["questions"].values()
    ]


def disjointness_report(test_rows: list[dict]) -> dict:
    test_cases = {case_signature(row) for row in test_rows}
    test_decisions = {signature for row in test_rows for signature in decision_signatures(row)}
    sources = {
        "beam_two_union_training": UNION_TRAIN,
        "beam_validation_used_for_selection": BEAM_VALIDATION,
        "typed_training_partition": TYPED_STAGE_TRAIN,
        "typed_validation_used_for_selection": TYPED_STAGE_VALIDATION,
        "pinned_official_typed_train_full": core.TRAIN,
    }
    report = {}
    for name, path in sources.items():
        rows = core.read_jsonl(path)
        source_cases = {case_signature(row) for row in rows}
        source_decisions = {signature for row in rows for signature in decision_signatures(row)}
        report[name] = {
            "path": str(path),
            "sha256": core.sha256(path),
            "rows": len(rows),
            "exact_case_overlaps_with_official_test": len(test_cases & source_cases),
            "exact_decision_overlaps_with_official_test": len(test_decisions & source_decisions),
        }
        if report[name]["exact_case_overlaps_with_official_test"] or report[name]["exact_decision_overlaps_with_official_test"]:
            raise ValueError(f"official test overlap found in {name}: {report[name]}")
    report["pinned_dataset_split_identity"] = {
        "train_sha256": core.EXPECTED_TRAIN_SHA,
        "test_sha256": core.EXPECTED_TEST_SHA,
        "distinct_hashes": core.EXPECTED_TRAIN_SHA != core.EXPECTED_TEST_SHA,
        "dataset_revision": core.DATASET_REVISION,
    }
    return report


def freeze() -> None:
    if OUT.exists():
        raise FileExistsError(f"Beam Two official Typed test output already exists: {OUT}")
    for path, expected in (
        (CHECKPOINT / "model.safetensors", EXPECTED_CHECKPOINT_SHA),
        (core.TEST, core.EXPECTED_TEST_SHA),
        (core.TEST_META, core.EXPECTED_META_SHA),
        (core.TRAIN, core.EXPECTED_TRAIN_SHA),
    ):
        actual = core.sha256(path)
        if actual != expected:
            raise ValueError(f"hash mismatch for {path}: {actual} != {expected}")
    beam_two_manifest = json.loads((ROOT / "models/beam-two/manifest.json").read_text())
    if beam_two_manifest["selected_checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("Beam Two named snapshot identity mismatch")
    beam3_protocol = json.loads((BEAM3_EVAL / "protocol.json").read_text())
    if core.sha256(BEAM3_EVAL / "final_report.json") != "b4f25e21ab02db9065b8cd2475ab45c60db462a1ceae557160a25c744eb855dc":
        raise ValueError("Beam Three frozen reference report changed")

    OUT.mkdir(parents=True, exist_ok=False)
    test_rows = core.read_jsonl(core.TEST)
    test_meta = json.loads(core.TEST_META.read_text())
    tok = AutoTokenizer.from_pretrained(CHECKPOINT / "tokenizer", local_files_only=True)
    items, skipped = items_from_rows(tok, test_rows, core.MAX_LEN, core.HEAD_MAX_LEN)
    if len(test_rows) != 400 or len(test_meta) != 400 or len(core.metadata(test_meta)) != 2000:
        raise ValueError("official test counts changed")
    protocol = {
        "name": "Beam Two on LocalLLaMA/typed-decisions official test",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(CHECKPOINT), "sha256": EXPECTED_CHECKPOINT_SHA,
            "named_snapshot_manifest": str(ROOT / "models/beam-two/manifest.json"),
            "selection": "already frozen using Beam and Typed validation, never official test",
        },
        "dataset": beam3_protocol["dataset"],
        "serialization": beam3_protocol["serialization"],
        "metrics": beam3_protocol["metrics"],
        "baselines": beam3_protocol["baselines"],
        "matched_to_beam_three_protocol_sha256": core.sha256(BEAM3_EVAL / "protocol.json"),
        "preflight": core.preflight(tok, test_rows, items, skipped),
        "exact_overlap_with_beam_training": core.exact_overlap_report(test_rows),
        "beam_two_training_and_validation_disjointness": disjointness_report(test_rows),
        "prohibitions": {
            "training": False, "calibration": False, "tuning": False, "checkpoint_reselection": False,
            "beam_historical_test_access": False, "other_test_or_model_scoring": False,
        },
        "comparison_caveat": "Beam Two trained on Typed Decisions training data; Beam Three did not. The paired test comparison is not an equal-data or equal-compute architecture comparison.",
        "model_scoring_performed": False,
    }
    core.atomic_json(core.PROTOCOL, protocol)
    (OUT / "PROTOCOL_SHA256").write_text(core.sha256(core.PROTOCOL) + "  protocol.json\n")
    core.atomic_json(OUT / "status.json", {
        "state": "frozen_not_scored", "utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": core.sha256(core.PROTOCOL),
    })
    print(json.dumps({
        "protocol": str(core.PROTOCOL), "protocol_sha256": core.sha256(core.PROTOCOL),
        "preflight": protocol["preflight"], "disjointness": protocol["beam_two_training_and_validation_disjointness"],
    }, sort_keys=True))


def paired_cluster_ci(rows: list[dict], value_key: str, draws: int = 10000, seed: int = 0) -> dict:
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
        "point_difference_beam_two_minus_beam_three": float(sums.sum() / counts.sum()),
        "unit": "case", "clusters": len(keys), "draws": draws, "seed": seed,
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def add_paired_comparison() -> dict:
    beam2 = [json.loads(line) for line in (OUT / "predictions.jsonl").read_text().splitlines() if line]
    beam3 = [json.loads(line) for line in (BEAM3_EVAL / "predictions.jsonl").read_text().splitlines() if line]
    if len(beam2) != 2000 or len(beam3) != 2000:
        raise ValueError("paired prediction count mismatch")
    paired = []
    outcomes = Counter()
    for left, right in zip(beam2, beam3):
        key_left = (left["case_index"], left["question_id"], left["workflow"], left["hard_label"])
        key_right = (right["case_index"], right["question_id"], right["workflow"], right["hard_label"])
        if key_left != key_right or left["teacher_probabilities"] != right["teacher_probabilities"]:
            raise ValueError(f"paired prediction identity mismatch: {key_left} != {key_right}")
        lc, rc = int(left["correct"]), int(right["correct"])
        outcomes[(lc, rc)] += 1
        paired.append({
            "case_index": left["case_index"],
            "hard_difference": lc - rc,
            "soft_difference": float(left["teacher_probability_of_prediction"]) - float(right["teacher_probability_of_prediction"]),
        })
    return {
        "hard_accuracy": paired_cluster_ci(paired, "hard_difference"),
        "soft_teacher_agreement": paired_cluster_ci(paired, "soft_difference"),
        "paired_outcomes": {
            "both_correct": outcomes[(1, 1)],
            "beam_two_only_correct": outcomes[(1, 0)],
            "beam_three_only_correct": outcomes[(0, 1)],
            "both_wrong": outcomes[(0, 0)],
        },
        "beam_three_predictions_sha256": core.sha256(BEAM3_EVAL / "predictions.jsonl"),
        "comparison_caveat": "Beam Two trained on Typed Decisions training data and used 849 optimizer steps; Beam Three was Beam-only and used 624. This is a paired outcome comparison, not an equal-data or equal-compute architecture comparison.",
    }


def score() -> None:
    core.score()
    report_path = OUT / "final_report.json"
    report = json.loads(report_path.read_text())
    report.pop("beam_two_evaluated", None)
    report["paired_comparison_to_frozen_beam_three"] = add_paired_comparison()
    report["beam_three_reference_predictions_used_without_rescoring"] = True
    report["comparison_caveat"] = "Beam Two trained on Typed Decisions training data; Beam Three did not. Training data, update count, and selection objectives differ."
    report["beam_two_training_and_validation_disjointness"] = json.loads(core.PROTOCOL.read_text())["beam_two_training_and_validation_disjointness"]
    core.atomic_json(report_path, report)
    report_sha = core.sha256(report_path)
    core.atomic_json(OUT / "status.json", {
        "state": "completed", "utc": datetime.now(timezone.utc).isoformat(),
        "report_sha256": report_sha, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
    })
    print(json.dumps({
        "status": "completed",
        "hard_accuracy": report["metrics"]["overall"]["hard_accuracy"],
        "soft_teacher_agreement": report["metrics"]["overall"]["soft_teacher_agreement"],
        "beam_two_minus_beam_three_hard_difference": report["paired_comparison_to_frozen_beam_three"]["hard_accuracy"],
        "report_sha256": report_sha,
    }, sort_keys=True))


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
        core.atomic_json(OUT / "status.json", {
            "state": "failed", "utc": datetime.now(timezone.utc).isoformat(),
            "error": repr(exc), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
