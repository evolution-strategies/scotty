from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from tools.run_local_checkpoint import choice_question, verify_checkpoint


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
        self.assertEqual(provenance["model"]["public_name"], "Scotty")
        self.assertEqual(provenance["model"]["historical_public_names"], ["Scotty" + "-1"])
        self.assertEqual(len(provenance["model"]["selected_checkpoint_sha256"]), 64)
        self.assertFalse(provenance["model"]["checkpoint_in_git"])

    def test_paper_directory_contains_only_pdf(self) -> None:
        files = sorted(
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "paper").rglob("*")
            if path.is_file()
        )
        self.assertEqual(files, ["paper/scotty-paper.pdf"])
        provenance = json.loads((ROOT / "provenance/artifacts.json").read_text())
        self.assertFalse(provenance["paper"]["paper_source_in_git"])
        self.assertEqual(provenance["paper"]["scotty_pages"], 6)

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

    def test_local_checkpoint_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "tokenizer").mkdir()
            (checkpoint / "encoder").mkdir()
            (checkpoint / "rl_agent_config.json").write_text("{}\n")
            payload = b"small test tensor placeholder"
            (checkpoint / "model.safetensors").write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            result = verify_checkpoint(checkpoint, expected_sha256=expected)
            self.assertEqual(result["status"], "verified")
            with self.assertRaises(ValueError):
                verify_checkpoint(checkpoint, expected_sha256="0" * 64)

    def test_local_choice_question(self) -> None:
        question = choice_question("Choose", ["alpha", "beta"])
        self.assertEqual(question["decision"]["criteria"], ["alpha", "beta"])
        with self.assertRaises(ValueError):
            choice_question("Choose", ["same", "same"])

    def test_license_scope(self) -> None:
        provenance = json.loads((ROOT / "provenance/artifacts.json").read_text())
        self.assertEqual(provenance["licensing"]["project_spdx"], "Apache-2.0")
        scope = (ROOT / "LICENSE_SCOPE.md").read_text()
        self.assertIn("paper/scotty-paper.pdf", scope)
        self.assertIn("does **not** apply", scope)

    def test_private_huggingface_hosting_record(self) -> None:
        provenance = json.loads((ROOT / "provenance/artifacts.json").read_text())
        hosting = provenance["hosting"]
        self.assertEqual(hosting["huggingface_repo"], "DarwinGuru/scotty")
        self.assertEqual(hosting["repo_type"], "model")
        self.assertTrue(hosting["private"])
        self.assertFalse(hosting["checkpoint_in_git"])
        self.assertEqual(
            hosting["checkpoint_sha256"],
            provenance["model"]["selected_checkpoint_sha256"],
        )

    def test_competition_submission_materials(self) -> None:
        hf_card = (ROOT / "huggingface/README.md").read_text()
        self.assertIn("Build with Beam", hf_card)
        self.assertIn("How Beam was used", hf_card)
        self.assertIn("Reflection Beam 1.0 501B-A23B", hf_card)
        submission = (ROOT / "submission/README.md").read_text()
        self.assertIn("Ready-to-post discussion entry", submission)
        self.assertIn("2 November 2026 at 11:59 PM Pacific Time", submission)
        self.assertIn("DarwinGuru/scotty", submission)

    def test_smoke_inference_record(self) -> None:
        provenance = json.loads((ROOT / "provenance/artifacts.json").read_text())
        verification = provenance["verification"]
        self.assertEqual(verification["download_repo"], "DarwinGuru/scotty")
        self.assertEqual(
            verification["download_revision"],
            provenance["hosting"]["verified_huggingface_commit"],
        )
        self.assertEqual(
            verification["checkpoint_sha256"],
            provenance["model"]["selected_checkpoint_sha256"],
        )
        self.assertEqual(verification["selected_option"], "true")
        self.assertEqual(verification["selected_probability"], 0.8872)
        self.assertEqual(verification["input_tokens"], 31)
        self.assertFalse(verification["truncated"])
        self.assertFalse(verification["training_rerun"])


if __name__ == "__main__":
    unittest.main()
