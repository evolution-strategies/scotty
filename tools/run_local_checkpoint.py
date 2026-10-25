#!/usr/bin/env python3
"""Verify and run an authorized local Scotty-1 checkpoint.

The checkpoint is intentionally not downloaded by this script. It must already
exist as a local Laya-format directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
VENDORED_LAYA = ROOT / "reproduction/vendor/laya_checkout"
EXPECTED_MODEL_SHA256 = "b7983f2820f46d58038920a860223a1e18826448c1833541f8bc52ba6e3fb783"
REQUIRED_PATHS = (
    "rl_agent_config.json",
    "model.safetensors",
    "tokenizer",
    "encoder",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(checkpoint: Path, expected_sha256: str = EXPECTED_MODEL_SHA256) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    missing = [name for name in REQUIRED_PATHS if not (checkpoint / name).exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing: {', '.join(missing)}")
    model_path = checkpoint / "model.safetensors"
    actual = sha256(model_path)
    if actual != expected_sha256:
        raise ValueError(
            "model.safetensors SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual}"
        )
    return {
        "checkpoint": str(checkpoint),
        "model_safetensors_sha256": actual,
        "status": "verified",
    }


def choice_question(instruction: str, options: Sequence[str]) -> dict:
    clean = [option.strip() for option in options]
    if not instruction.strip():
        raise ValueError("instruction must not be empty")
    if len(clean) < 2 or any(not option for option in clean):
        raise ValueError("provide at least two non-empty --option values")
    if len(set(clean)) != len(clean):
        raise ValueError("--option values must be unique")
    return {
        "decision": {
            "type": "choice",
            "instructions": instruction,
            "criteria": clean,
        }
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--device", help="Laya device such as cuda, cpu, or mps")
    parser.add_argument("--state", help="State text to score")
    parser.add_argument("--instruction", help="Decision instruction")
    parser.add_argument("--option", action="append", default=[], help="Candidate; repeat for each option")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    verified = verify_checkpoint(args.checkpoint)
    if args.verify_only:
        print(json.dumps(verified, indent=2, sort_keys=True))
        return 0
    if args.state is None or args.instruction is None:
        raise SystemExit("--state and --instruction are required unless --verify-only is used")

    questions = choice_question(args.instruction, args.option)
    sys.path.insert(0, str(VENDORED_LAYA))
    import laya

    agent = laya.load(
        verified["checkpoint"],
        device=args.device,
        expected_sha256={"model.safetensors": EXPECTED_MODEL_SHA256},
    )
    result = agent.predict(args.state, questions)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
