from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.candidate_diagnostics import diagnose_sessions


class CandidateDiagnosticsTest(unittest.TestCase):
    def test_classifies_a_retrieval_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            dataset = root / "public_set.jsonl"
            catalog.write_text(
                json.dumps(
                    {
                        "parent_asin": "TARGET",
                        "title": "Hidden formal bracelet",
                        "categories": ["Clothing, Shoes & Jewelry", "Women", "Jewelry"],
                        "features": ["silver"],
                        "details": {},
                        "description": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            dataset.write_text(
                json.dumps(
                    {
                        "sample_id": "sample_1",
                        "scenario_type": "buying",
                        "user_profile": {"preference_tags": [], "summary": ""},
                        "ground_truth": {"parent_asin": "TARGET"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            class MissingAgent:
                def __init__(self, catalog_path: str | Path) -> None:
                    self.sessions: dict[str, dict] = {}

                def reset(self, session_id: str, user_profile: dict) -> None:
                    self.sessions[session_id] = {
                        "constraints": [],
                        "intent": "buying",
                        "workflow": {"allow_dense": False, "strategy": "test"},
                    }

                def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
                    return {
                        "message": "No match",
                        "ask_attribute": None,
                        "recommendations": [],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    }

                def _candidate_rows(self, constraints: list[str], intent: str, allow_dense: bool = True) -> dict:
                    return {}

                def _rank(self, state: dict, top_k: int) -> list[dict]:
                    return []

            with patch("tools.candidate_diagnostics.Agent", MissingAgent):
                result = diagnose_sessions(
                    catalog=catalog,
                    dataset=dataset,
                    include_session_traces=True,
                )

        self.assertEqual(result["aggregate_metrics"]["hit_rate_at_10"], 0.0)
        self.assertEqual(result["failure_summary"], {"retrieval_miss": 1})
        self.assertEqual(result["rank_distribution"], {"miss": 1})
        self.assertEqual(
            result["candidate_rank_distribution"],
            {"not_retrieved": 1},
        )
        self.assertEqual(
            result["improvement_focus"]["retrieval_routes"]["priority_sessions_to_inspect_first"][0]["sample_id"],
            "sample_1",
        )
        self.assertEqual(result["sessions"][0]["failure_mode"], "retrieval_miss")


if __name__ == "__main__":
    unittest.main()
