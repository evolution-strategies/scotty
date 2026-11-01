#!/usr/bin/env python3
"""Verify and run an authorized local Scotty checkpoint.

The checkpoint is intentionally not downloaded by this script. It must already
exist as a local Laya-format directory.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
VENDORED_LAYA = ROOT / "reproduction/vendor/laya_checkout"
EXPECTED_MODEL_SHA256 = "e79b758aea590bcc4504e6b622c206080a7d1fedc49d572d3a185321b8e2eeda"
PUBLIC_CONFIG = "scotty_config.json"
LEGACY_CONFIG = "rl_agent_config.json"
REQUIRED_PATHS = (
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
    config = next(
        (checkpoint / name for name in (PUBLIC_CONFIG, LEGACY_CONFIG) if (checkpoint / name).is_file()),
        None,
    )
    if config is None:
        missing.append(f"{PUBLIC_CONFIG} (or legacy {LEGACY_CONFIG})")
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
        "config": config.name,
        "model_safetensors_sha256": actual,
        "status": "verified",
    }


@contextmanager
def laya_checkpoint_view(checkpoint: Path):
    """Expose Scotty's neutral config name through Laya's legacy file layout."""
    legacy = checkpoint / LEGACY_CONFIG
    if legacy.is_file():
        yield checkpoint
        return

    public = checkpoint / PUBLIC_CONFIG
    with tempfile.TemporaryDirectory(prefix="scotty-laya-") as directory:
        view = Path(directory)
        shutil.copy2(public, view / LEGACY_CONFIG)

        model_source = checkpoint / "model.safetensors"
        model_target = view / "model.safetensors"
        try:
            os.link(model_source, model_target)
        except OSError:
            try:
                os.symlink(model_source, model_target)
            except OSError:
                shutil.copy2(model_source, model_target)

        for name in ("encoder", "tokenizer"):
            source = checkpoint / name
            target = view / name
            try:
                os.symlink(source, target, target_is_directory=True)
            except OSError:
                shutil.copytree(source, target)
        yield view


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

    with laya_checkpoint_view(Path(verified["checkpoint"])) as laya_checkpoint:
        agent = laya.load(
            str(laya_checkpoint),
            device=args.device,
            expected_sha256={"model.safetensors": EXPECTED_MODEL_SHA256},
        )
        result = agent.predict(args.state, questions)
    result["model"] = "Scotty"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
