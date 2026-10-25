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

    def test_paper_source_name(self) -> None:
        source = (ROOT / "paper/source/scotty1.tex").read_text()
        self.assertIn("pdftitle={Scotty-1:", source)
        self.assertIn("\\newcommand{\\model}{Scotty-1}", source)
        self.assertIn("Reflection Beam 1.0 501B-A23B", source)


if __name__ == "__main__":
    unittest.main()
