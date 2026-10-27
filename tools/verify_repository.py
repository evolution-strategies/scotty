#!/usr/bin/env python3
"""Verify release-safe repository invariants using only the standard library."""

from __future__ import annotations

import ast
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
        ROOT / "paper/scotty-1-paper.pdf": paper["scotty_1_pdf_sha256"],
    }
    for path, wanted in expected.items():
        actual = sha256(path)
        assert actual == wanted, f"SHA mismatch for {path.relative_to(ROOT)}: {actual}"

    paper_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "paper").rglob("*")
        if path.is_file()
    )
    assert paper_files == ["paper/scotty-1-paper.pdf"], (
        f"Only the paper PDF may be tracked/present: {paper_files}"
    )
    assert paper["paper_source_in_git"] is False

    reproduction = provenance["reproduction"]
    for filename, hashes in reproduction["helpers"].items():
        path = ROOT / "reproduction/scripts" / filename
        actual = sha256(path)
        assert actual == hashes["repository_sha256"], (
            f"SHA mismatch for {path.relative_to(ROOT)}: {actual}"
        )

    laya = reproduction["laya"]
    vendor_expected = {
        "LICENSE": laya["license_sha256"],
        "pyproject.toml": laya["pyproject_sha256"],
        "laya/__init__.py": laya["init_sha256"],
        "laya/common.py": laya["common_sha256"],
        "laya/train.py": laya["train_sha256"],
    }
    vendor_root = ROOT / "reproduction/vendor/laya_checkout"
    for relative, wanted in vendor_expected.items():
        path = vendor_root / relative
        actual = sha256(path)
        assert actual == wanted, f"SHA mismatch for {path.relative_to(ROOT)}: {actual}"

    python_files = sorted((ROOT / "reproduction").rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

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
    licensing = provenance["licensing"]
    assert licensing["project_spdx"] == "Apache-2.0"
    assert sha256(ROOT / "LICENSE") == licensing["root_license_sha256"]
    assert (ROOT / licensing["scope_file"]).is_file()
    hosting = provenance["hosting"]
    assert hosting == {
        "huggingface_repo": "DarwinGuru/scotty-1",
        "repo_type": "model",
        "private": True,
        "checkpoint_sha256": provenance["model"]["selected_checkpoint_sha256"],
        "checkpoint_in_git": False,
        "verified_huggingface_commit": "b377f2ab1b92c3f114bea9fc70e4a619c04fe04b",
    }
    verification = provenance["verification"]
    assert verification["checkpoint_sha256"] == provenance["model"]["selected_checkpoint_sha256"]
    assert verification["selected_option"] == "true"
    assert verification["selected_probability"] == 0.8872
    assert verification["input_tokens"] == 31
    assert verification["truncated"] is False
    assert verification["training_rerun"] is False
    hf_card = (ROOT / "huggingface/README.md").read_text()
    for required in (
        "Build with Beam",
        "How Beam was used",
        "DarwinGuru/scotty-1",
        provenance["model"]["selected_checkpoint_sha256"],
        "Reflection Beam 1.0 501B-A23B",
    ):
        assert required in hf_card, f"Hugging Face card is missing: {required}"
    submission = (ROOT / "submission/README.md").read_text()
    for required in (
        "2 November 2026 at 11:59 PM Pacific Time",
        "DarwinGuru/scotty-1",
        "evolution-strategies/scotty-1",
        "Ready-to-post discussion entry",
    ):
        assert required in submission, f"Submission package is missing: {required}"
    provider_name_hits = []
    disallowed_provider_name = b"base" + b"ten"
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or "tmp" in path.parts
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        if disallowed_provider_name in path.read_bytes().lower():
            provider_name_hits.append(str(path.relative_to(ROOT)))
    assert not provider_name_hits, f"Named provider found in current tree: {provider_name_hits}"
    print(
        f"Verified {len(expected)} artifacts, {len(reproduction['helpers'])} helpers, "
        f"{len(vendor_expected)} vendored-source hashes, {len(python_files)} Python files, "
        "exclusions, metrics, split counts, and submission materials."
    )


if __name__ == "__main__":
    main()
