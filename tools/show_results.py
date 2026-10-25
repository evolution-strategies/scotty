#!/usr/bin/env python3
"""Print the frozen aggregate headline results."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    metrics = json.loads((ROOT / "results/metrics.json").read_text())
    model = metrics["scotty_1"]
    control = metrics["typed_only_matched_updates_control"]
    rows = [
        ("Beam holdout, raw", model["beam_protected_holdout"]["raw_accuracy"], control["beam_protected_holdout"]["raw_accuracy"]),
        ("Beam holdout, balanced", model["beam_protected_holdout"]["position_balanced_accuracy"], control["beam_protected_holdout"]["position_balanced_accuracy"]),
        ("Typed official, hard", model["typed_official_test"]["hard_accuracy"], control["typed_official_test"]["hard_accuracy"]),
        ("Typed official, soft", model["typed_official_test"]["soft_teacher_agreement"], control["typed_official_test"]["soft_teacher_agreement"]),
    ]
    print(f"{'Evaluation':30} {'Scotty-1':>10} {'Typed-only':>10}")
    for label, scotty, typed_only in rows:
        print(f"{label:30} {pct(scotty):>10} {pct(typed_only):>10}")


if __name__ == "__main__":
    main()
