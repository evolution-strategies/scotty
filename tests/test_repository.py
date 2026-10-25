from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryFactsTest(unittest.TestCase):
    def test_split_arithmetic(self) -> None:
        splits = json.loads((ROOT / "results/splits.json").read_text())
        beam = splits["beam_generated"]
        typed = splits["typed_decisions"]
        self.assertEqual(beam["train"] + typed["train_decisions"], 18092)
        self.assertEqual(beam["validation"] + typed["validation_decisions"], 4527)
        self.assertEqual(beam["protected_test"] + typed["official_test_decisions"], 5259)
        self.assertEqual(sum(beam["protected_test_families"].values()), 3259)

    def test_public_name_and_checkpoint_hash(self) -> None:
        provenance = json.loads((ROOT / "provenance/artifacts.json").read_text())
        self.assertEqual(provenance["model"]["public_name"], "Scotty-1")
        self.assertEqual(len(provenance["model"]["selected_checkpoint_sha256"]), 64)
        self.assertFalse(provenance["model"]["checkpoint_in_git"])

    def test_paper_directory_contains_only_pdf(self) -> None:
        files = sorted(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "paper").rglob("*")
            if path.is_file()
        )
        self.assertEqual(files, ["paper/scotty-1-paper.pdf"])
        provenance = json.loads((ROOT / "provenance/artifacts.json").read_text())
        self.assertFalse(provenance["paper"]["paper_source_in_git"])

    def test_reproduction_dependency_closure_is_present(self) -> None:
        expected_helpers = {
            "beam_one_common.py",
            "smoke_beam_one.py",
            "train_laya_stage_compare.py",
            "evaluate_beam_three_typed_test.py",
        }
        scripts = ROOT / "reproduction/scripts"
        self.assertTrue(all((scripts / name).is_file() for name in expected_helpers))
        vendor = ROOT / "reproduction/vendor/laya_checkout"
        self.assertTrue((vendor / "LICENSE").is_file())
        self.assertTrue((vendor / "laya/train.py").is_file())

    def test_reproduction_provenance(self) -> None:
        provenance = json.loads((ROOT / "provenance/artifacts.json").read_text())
        reproduction = provenance["reproduction"]
        self.assertEqual(
            reproduction["laya"]["commit"],
            "eba3b0367643498e478ed02e870fa53737da5061",
        )
        self.assertEqual(reproduction["laya"]["license"], "Apache-2.0")
        self.assertTrue(reproduction["environment"]["full_freeze_is_post_run_evidence"])

    def test_portable_reproduction_roots(self) -> None:
        for path in (ROOT / "reproduction/scripts").glob("*.py"):
            source = path.read_text()
            self.assertNotIn('/home/oliver', source, path.name)


if __name__ == "__main__":
    unittest.main()
