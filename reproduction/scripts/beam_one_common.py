from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    payload = "".join(f"{line}\n" for line in lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: row is not an object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], *, exclusive: bool = True) -> None:
    mode = "x" if exclusive else "w"
    with Path(path).open(mode, encoding="utf-8") as handle:
        for row in rows:
            # Candidate/criterion order is model semantics. Never recursively sort keys here.
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _candidate_contract(row: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError(f"{row.get('id')}: candidates must contain at least two entries")
    normalized: list[dict[str, str]] = []
    by_id: dict[str, str] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"{row.get('id')}: candidate is not an object")
        cid = candidate.get("id")
        desc = candidate.get("description")
        if not isinstance(cid, str) or not cid or not isinstance(desc, str) or not desc:
            raise ValueError(f"{row.get('id')}: candidate id/description must be non-empty strings")
        if cid in by_id:
            raise ValueError(f"{row.get('id')}: duplicate candidate id {cid!r}")
        by_id[cid] = desc
        normalized.append({"id": cid, "description": desc})
    return normalized, by_id


def convert_beam_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert one Beam row to one Laya item without copying metadata or rationale into input."""
    row_id = row.get("id")
    family = row.get("task_family")
    situation = row.get("situation")
    instruction = row.get("instruction")
    answer_id = row.get("answer_id")
    metadata = row.get("metadata")
    if not all(isinstance(value, str) and value for value in (row_id, family, situation, instruction, answer_id)):
        raise ValueError(f"{row_id}: required Beam string field is empty or invalid")
    if not isinstance(metadata, dict):
        raise ValueError(f"{row_id}: metadata is not an object")
    decision_type = metadata.get("decision_type")
    candidates, by_id = _candidate_contract(row)
    source_option_ids = [candidate["id"] for candidate in candidates]
    if answer_id not in by_id:
        raise ValueError(f"{row_id}: answer_id is not a candidate")

    if decision_type == "categorical":
        qtype = "choice"
        criteria: Any = {candidate["id"]: candidate["description"] for candidate in candidates}
        expected: Any = answer_id
        converted_option_ids = list(source_option_ids)
    elif decision_type == "boolean":
        if len(candidates) != 2:
            raise ValueError(f"{row_id}: boolean row has {len(candidates)} candidates, expected 2")
        true_id = metadata.get("boolean_true_id")
        if not isinstance(true_id, str) or true_id not in by_id:
            raise ValueError(f"{row_id}: boolean_true_id is missing or not a candidate")
        false_ids = [cid for cid in source_option_ids if cid != true_id]
        if len(false_ids) != 1:
            raise ValueError(f"{row_id}: boolean true/false mapping is not one-to-one")
        false_id = false_ids[0]
        qtype = "noul"
        criteria = {"false": by_id[false_id], "true": by_id[true_id]}
        expected = answer_id == true_id
        converted_option_ids = [false_id, true_id]
    elif decision_type == "ordinal":
        ordinal_order = metadata.get("ordinal_order")
        if not isinstance(ordinal_order, list) or any(not isinstance(cid, str) for cid in ordinal_order):
            raise ValueError(f"{row_id}: ordinal_order must be a list of candidate ids")
        if len(ordinal_order) != len(candidates) or set(ordinal_order) != set(source_option_ids):
            raise ValueError(f"{row_id}: ordinal_order must contain every candidate exactly once")
        qtype = "score"
        criteria = [by_id[cid] for cid in ordinal_order]
        expected = ordinal_order.index(answer_id)
        converted_option_ids = list(ordinal_order)
    else:
        raise ValueError(f"{row_id}: unsupported decision_type {decision_type!r}")

    converted = {
        "state": situation,
        "questions": {
            "decision": {
                "type": qtype,
                "instructions": instruction,
                "criteria": criteria,
            }
        },
        "expected": {"decision": expected},
    }
    converted_gold_index = converted_option_ids.index(answer_id)
    mapping = {
        "id": row_id,
        "task_family": family,
        "decision_type": decision_type,
        "laya_type": qtype,
        "option_count": len(candidates),
        "source_option_ids": source_option_ids,
        "converted_option_ids": converted_option_ids,
        "answer_id": answer_id,
        "source_gold_position": source_option_ids.index(answer_id),
        "converted_gold_position": converted_gold_index,
    }
    return converted, mapping


def convert_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    converted: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        item, mapping = convert_beam_row(row)
        if mapping["id"] in seen:
            raise ValueError(f"duplicate id {mapping['id']!r}")
        seen.add(mapping["id"])
        converted.append(item)
        mappings.append(mapping)
    return converted, mappings


def distribution(mappings: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def counts(key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[key]) for row in mappings).items()))

    return {
        "rows": len(mappings),
        "decision_type": counts("decision_type"),
        "laya_type": counts("laya_type"),
        "task_family": counts("task_family"),
        "option_count": counts("option_count"),
        "source_gold_position": counts("source_gold_position"),
        "converted_gold_position": counts("converted_gold_position"),
    }


def prediction_metrics(records: Sequence[tuple[Any, Any, Any, int]],
                       mappings: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != len(mappings):
        raise ValueError(f"record/mapping length mismatch: {len(records)} != {len(mappings)}")
    total_correct = 0
    ce_total = 0.0
    cells: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    groups: dict[str, dict[str, list[int]]] = {
        "decision_type": defaultdict(list),
        "task_family": defaultdict(list),
        "option_count": defaultdict(list),
    }
    for record, mapping in zip(records, mappings):
        _qtype, logits_raw, target_raw, k_raw = record
        k = int(k_raw)
        logits = [float(value) for value in logits_raw[:k]]
        target = [float(value) for value in target_raw[:k]]
        pred = max(range(k), key=lambda idx: logits[idx])
        gold = max(range(k), key=lambda idx: target[idx])
        if gold != int(mapping["converted_gold_position"]):
            raise ValueError(f"{mapping['id']}: target index disagrees with conversion audit")
        correct = int(pred == gold)
        total_correct += correct
        zmax = max(logits)
        logsumexp = zmax + math.log(sum(math.exp(value - zmax) for value in logits))
        ce_total += logsumexp - logits[gold]
        cell = (str(mapping["decision_type"]), k, gold)
        cells[cell].append(correct)
        groups["decision_type"][str(mapping["decision_type"])].append(correct)
        groups["task_family"][str(mapping["task_family"])].append(correct)
        groups["option_count"][str(k)].append(correct)

    cell_accuracy = {"|".join(map(str, key)): sum(values) / len(values)
                     for key, values in sorted(cells.items())}
    by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for group_name, values_by_key in groups.items():
        by_group[group_name] = {
            key: {"items": len(values), "accuracy": sum(values) / len(values)}
            for key, values in sorted(values_by_key.items())
        }
    n = len(records)
    return {
        "items": n,
        "accuracy": total_correct / n if n else 0.0,
        "cross_entropy": ce_total / n if n else 0.0,
        "position_balanced_accuracy": sum(cell_accuracy.values()) / len(cell_accuracy) if cell_accuracy else 0.0,
        "position_balance_definition": "macro mean over observed (decision_type, option_count, converted_gold_position) cells",
        "position_cells": len(cell_accuracy),
        "position_cell_accuracy": cell_accuracy,
        "by_group": by_group,
    }


def baseline_metrics(mappings: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mappings:
        groups[str(row["decision_type"])].append(row)

    def one(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
        return {
            "items": len(rows),
            "chance_expected_accuracy": sum(1.0 / int(row["option_count"]) for row in rows) / len(rows),
            "converted_first_position_accuracy": sum(int(row["converted_gold_position"]) == 0 for row in rows) / len(rows),
            "source_first_position_accuracy": sum(int(row["source_gold_position"]) == 0 for row in rows) / len(rows),
        }

    return {
        "overall": one(mappings),
        "by_decision_type": {key: one(rows) for key, rows in sorted(groups.items())},
    }
