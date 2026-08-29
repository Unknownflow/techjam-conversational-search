from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starter.agent import Agent


class AgentDialogTest(unittest.TestCase):
    def _agent(self, root: Path, count: int = 4) -> Agent:
        catalog = root / "catalog.jsonl"
        rows = [{
            "parent_asin": f"A{index}",
            "title": f"Everyday Shoe {index}",
            "features": ["cotton lining"],
            "details": {"department": "adult"},
            "description": ["comfortable walking footwear"],
            "categories": ["Clothing", "Shoes"],
            "store": "Example",
            "average_rating": 4.2,
            "rating_number": 20 + index,
            "price": 49.0,
        } for index in range(count)]
        catalog.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return Agent(catalog)

    def test_intent_override_erases_soft_preference_but_keeps_category(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            agent.reset("s", {"preference_tags": []})
            agent.respond("s", "I'm looking for Shoes. I prefer leather.", 1, 10)
            response = agent.respond(
                "s",
                "Actually, ignore my earlier preference. What I need is: cotton.",
                2,
                10,
            )
            active = [value.lower() for value in agent.sessions["s"]["constraints"]]
            self.assertIn("shoes", active)
            self.assertIn("cotton", active)
            self.assertNotIn("leather", active)
            self.assertEqual(response["dialog_state"]["intent"], "buying")
            self.assertEqual(response["dialog_state"]["phase"], "intent_override")

    def test_over_general_pool_triggers_structured_clarification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            agent.reset("s", {"preference_tags": []})
            with patch("starter.agent.OVERGENERALITY_THRESHOLD", 3):
                response = agent.respond("s", "I'm looking for Shoes, but I'm still exploring.", 1, 10)
            self.assertEqual(response["ask_attribute"], "other")
            self.assertTrue(response["dialog_state"]["over_general"])
            self.assertEqual(response["dialog_state"]["phase"], "clarification")


if __name__ == "__main__":
    unittest.main()
