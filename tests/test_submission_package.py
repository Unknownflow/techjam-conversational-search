from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from submission.agent import Agent
from submission.validate import ROOT as SUBMISSION_ROOT, validate


ALLOWED_ATTRIBUTES = {
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
    None,
}


def write_catalog(path: Path, count: int = 12) -> set[str]:
    rows = [
        {
            "parent_asin": f"P{index:02d}",
            "title": f"Everyday walking shoe {index}",
            "features": ["cotton lining", "cushioned sole"],
            "details": {"department": "women", "style": "casual"},
            "description": ["comfortable footwear for daily walking"],
            "categories": ["Clothing", "Women", "Shoes"],
            "store": "Example",
            "average_rating": 4.0 + index / 100,
            "rating_number": 20 + index,
            "price": 49.0 + index,
        }
        for index in range(count)
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return {row["parent_asin"] for row in rows}


class SubmissionPackageTest(unittest.TestCase):
    def test_manifest_allowlist_and_frozen_hash_validate(self) -> None:
        manifest = validate()
        self.assertEqual(manifest["entry_point"], "agent:Agent")
        self.assertFalse(manifest["requires_network"])
        self.assertEqual(manifest["environment_variables"], [])

    def test_response_matches_strict_contract_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            valid_ids = write_catalog(catalog)
            with patch("socket.socket", side_effect=AssertionError("network used")):
                agent = Agent(catalog)
                agent.reset("contract", {"preference_tags": ["comfort"]})
                response = agent.respond(
                    "contract", "I'm looking for Women's Shoes.", 3, 10
                )

            json.dumps(response)
            self.assertEqual(
                set(response),
                {"message", "ask_attribute", "recommendations", "usage"},
            )
            self.assertIsInstance(response["message"], str)
            self.assertIn(response["ask_attribute"], ALLOWED_ATTRIBUTES)
            self.assertLessEqual(len(response["recommendations"]), 10)

            identifiers = [
                recommendation["parent_asin"]
                for recommendation in response["recommendations"]
            ]
            self.assertEqual(len(identifiers), len(set(identifiers)))
            self.assertTrue(set(identifiers).issubset(valid_ids))
            scores = [
                recommendation["score"]
                for recommendation in response["recommendations"]
                if "score" in recommendation
            ]
            self.assertEqual(scores, sorted(scores, reverse=True))
            for recommendation in response["recommendations"]:
                self.assertLessEqual(set(recommendation), {"parent_asin", "score"})

            self.assertEqual(
                set(response["usage"]), {"prompt_tokens", "completion_tokens"}
            )
            for value in response["usage"].values():
                self.assertIsInstance(value, int)
                self.assertGreaterEqual(value, 0)

    def test_extracted_bundle_imports_from_unrelated_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            bundle = temporary_root / "bundle"
            bundle.mkdir()
            manifest = json.loads(
                (SUBMISSION_ROOT / "MANIFEST.json").read_text(encoding="utf-8")
            )
            for relative in manifest["files"]:
                source = SUBMISSION_ROOT / relative
                destination = bundle / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

            catalog = temporary_root / "catalog.jsonl"
            write_catalog(catalog)
            script = """
import json
import socket
import sys

def blocked(*args, **kwargs):
    raise AssertionError("network used")

socket.socket = blocked
from agent import Agent

agent = Agent(sys.argv[1])
agent.reset("isolated", {"preference_tags": []})
print(json.dumps(agent.respond("isolated", "I need Shoes.", 1, 10)))
"""
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment["PYTHONNOUSERSITE"] = "1"
            completed = subprocess.run(
                [sys.executable, "-c", script, str(catalog)],
                cwd=bundle,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            response = json.loads(completed.stdout)
            self.assertEqual(
                set(response),
                {"message", "ask_attribute", "recommendations", "usage"},
            )


if __name__ == "__main__":
    unittest.main()
