#!/usr/bin/env python3
"""Print Scotty's final validation and source-audit results."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    metrics = json.loads((ROOT / "results/metrics.json").read_text())["scotty"]
    validation = metrics["three_seed_validation"]
    audit = metrics["group_safe_source_audit"]

    print("Three-seed validation")
    print(f"  objective mean ± sample SD: {validation['primary_objective_mean']:.6f} ± {validation['primary_objective_sample_stdev']:.6f}")
    print(f"  Typed hard mean:            {pct(validation['typed_hard_accuracy_mean'])}")
    print(f"  Beam balanced mean:         {pct(validation['beam_position_balanced_accuracy_mean'])}")
    print()
    print("Group-safe source audit")
    print(f"  objective:                  {audit['primary_objective']:.6f}")
    print(f"  Typed hard:                 {pct(audit['typed_hard_accuracy'])} ({audit['typed_correct']}/{audit['typed_decisions']})")
    print(f"  Beam raw:                   {pct(audit['beam_raw_accuracy'])} ({audit['beam_correct']}/{audit['beam_decisions']})")
    print(f"  Beam balanced:              {pct(audit['beam_position_balanced_accuracy'])}")


if __name__ == "__main__":
    main()
