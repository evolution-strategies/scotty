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
        ROOT / "paper/scotty-paper.pdf": paper["scotty_pdf_sha256"],
    }
    for path, wanted in expected.items():
        actual = sha256(path)
        assert actual == wanted, f"SHA mismatch for {path.relative_to(ROOT)}: {actual}"

    paper_files = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "paper").rglob("*")
        if path.is_file()
    )
    assert paper_files == ["paper/scotty-paper.pdf"], (
        f"Only the paper PDF may be tracked/present: {paper_files}"
    )
    assert paper["paper_source_in_git"] is False
    assert paper["scotty_pages"] == 6

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
    final = metrics["scotty"]
    assert final["checkpoint_sha256"] == provenance["model"]["selected_checkpoint_sha256"]
    assert final["three_seed_validation"]["primary_objective_mean"] == 0.8595323507839705
    assert final["group_safe_source_audit"]["primary_objective"] == 0.841031
    assert final["group_safe_source_audit"]["external_or_fresh_distribution"] is False
    assert provenance["model"]["public_name"] == "Scotty"
    assert provenance["model"]["selected_checkpoint_sha256"] == (
        "e79b758aea590bcc4504e6b622c206080a7d1fedc49d572d3a185321b8e2eeda"
    )
    assert splits["training"] == {
        "unique_decisions": 18092,
        "optimizer_updates": 849,
        "effective_batch": 64,
        "total_presentations": 54336,
        "source_sampling": "50:50 Beam/Typed; uniform within source",
    }
    licensing = provenance["licensing"]
    assert licensing["project_spdx"] == "Apache-2.0"
    assert sha256(ROOT / "LICENSE") == licensing["root_license_sha256"]
    assert (ROOT / licensing["scope_file"]).is_file()
    hosting = provenance["hosting"]
    assert hosting["huggingface_repo"] == "DarwinGuru/scotty"
    assert hosting["repo_type"] == "model"
    assert hosting["private"] is False
    assert hosting["checkpoint_sha256"] == provenance["model"]["selected_checkpoint_sha256"]
    assert hosting["checkpoint_in_git"] is False
    assert len(hosting["verified_huggingface_commit"]) == 40
    verification = provenance["verification"]
    assert verification["download_repo"] == hosting["huggingface_repo"]
    assert verification["download_revision"] == hosting["verified_huggingface_commit"]
    assert verification["checkpoint_sha256"] == provenance["model"]["selected_checkpoint_sha256"]
    assert verification["status"] == "verified"
    assert verification["selected_option"] in {"false", "true"}
    assert 0.0 <= verification["selected_probability"] <= 1.0
    assert verification["input_tokens"] > 0
    assert verification["truncated"] is False
    assert verification["training_rerun"] is False
    hf_card = (ROOT / "huggingface/README.md").read_text()
    for required in (
        "Build with Beam",
        "How Beam was used",
        "DarwinGuru/scotty",
        provenance["model"]["selected_checkpoint_sha256"],
        "Reflection Beam 1.0 501B-A23B",
        "scotty_config.json",
    ):
        assert required in hf_card, f"Hugging Face card is missing: {required}"
    submission = (ROOT / "submission/README.md").read_text()
    for required in (
        "2 November 2026 at 11:59 PM Pacific Time",
        "DarwinGuru/scotty",
        "run_scotty.py",
        "paper/scotty-paper.pdf",
        "Ready-to-post discussion entry",
    ):
        assert required in submission, f"Submission package is missing: {required}"
    audience_docs = (
        "README.md",
        "MODEL_CARD.md",
        "REPRODUCIBILITY.md",
        "LICENSE_AUDIT.md",
        "DATA_CARD.md",
        "THIRD_PARTY.md",
        "RELEASE_CHECKLIST.md",
        "huggingface/README.md",
        "submission/README.md",
    )
    forbidden_availability_phrases = (
        "private repository",
        "private model",
        "hosted privately",
        "private checkpoint",
        "prepared privately",
        "submission preparation",
        "repository is private",
        "remain **private**",
    )
    availability_hits = []
    for relative in audience_docs:
        text = (ROOT / relative).read_text().lower()
        for phrase in forbidden_availability_phrases:
            if phrase in text:
                availability_hits.append(f"{relative}: {phrase}")
    assert not availability_hits, f"Stale availability wording found: {availability_hits}"
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
    stale_name_hits = []
    historical_name = b"scotty" + b"-1"
    historical_record = ROOT / "provenance/artifacts.json"
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or "tmp" in path.parts
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
            or path == historical_record
        ):
            continue
        if historical_name in path.read_bytes().lower():
            stale_name_hits.append(str(path.relative_to(ROOT)))
    assert not stale_name_hits, f"Stale public name found outside provenance: {stale_name_hits}"
    print(
        f"Verified {len(expected)} artifacts, {len(reproduction['helpers'])} helpers, "
        f"{len(vendor_expected)} vendored-source hashes, {len(python_files)} Python files, "
        "exclusions, metrics, split counts, and submission materials."
    )


if __name__ == "__main__":
    main()
