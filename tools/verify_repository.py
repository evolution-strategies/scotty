#!/usr/bin/env python3
"""Verify release-safe repository invariants using only the standard library."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_GIT_FILE_BYTES = 100 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".gguf"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    provenance = json.loads((ROOT / "provenance/artifacts.json").read_text())
    paper = provenance["paper"]
    expected = {
        ROOT / "paper/original/Beam-1_ESANN_paper.pdf": paper["original_beam_1_pdf_sha256"],
        ROOT / "paper/original/Beam-1_ESANN_LaTeX_source.zip": paper["original_beam_1_source_zip_sha256"],
        ROOT / "paper/Scotty-1_ESANN_paper.pdf": paper["scotty_1_pdf_sha256"],
        ROOT / "paper/Scotty-1_ESANN_LaTeX_source.zip": paper["scotty_1_source_zip_sha256"],
    }
    for path, wanted in expected.items():
        actual = sha256(path)
        assert actual == wanted, f"SHA mismatch for {path.relative_to(ROOT)}: {actual}"

    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "tmp" in path.parts or "_snapshot" in path.parts:
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.stat().st_size > MAX_GIT_FILE_BYTES:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"Excluded/oversized artifacts found: {offenders}"

    metrics = json.loads((ROOT / "results/metrics.json").read_text())
    splits = json.loads((ROOT / "results/splits.json").read_text())
    assert metrics["scotty_1"]["beam_protected_holdout"]["raw_accuracy"] == 0.830316047867444
    assert metrics["scotty_1"]["typed_official_test"]["hard_accuracy"] == 0.7545
    assert splits["combined"] == {
        "train_decisions": 18092,
        "validation_decisions": 4527,
        "separate_test_decisions": 5259,
    }
    assert not (ROOT / "LICENSE").exists(), "No license has been selected"
    print(f"Verified {len(expected)} artifact hashes, exclusions, metrics, and split counts.")


if __name__ == "__main__":
    main()
