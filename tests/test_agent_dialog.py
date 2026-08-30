from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starter.agent import Agent, NextQuestionSelector


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
            first = agent.respond("s", "I'm looking for Shoes. I prefer leather.", 1, 10)
            self.assertEqual(
                agent.sessions["s"]["long_term_profile"]["learned_preferences"],
                {"material": ["leather"]},
            )
            self.assertEqual(first["dialog_state"]["context_version"], 1)
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
            self.assertEqual(
                agent.sessions["s"]["long_term_profile"]["learned_preferences"], {}
            )
            self.assertEqual(response["dialog_state"]["intent"], "buying")
            self.assertEqual(response["dialog_state"]["phase"], "intent_override")
            self.assertEqual(response["dialog_state"]["strategy"], "override_recovery")

    def test_over_general_pool_triggers_structured_clarification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            agent.reset("s", {"preference_tags": []})
            with patch("starter.agent.OVERGENERALITY_THRESHOLD", 3):
                response = agent.respond("s", "I'm looking for Shoes, but I'm still exploring.", 1, 10)
            self.assertEqual(response["ask_attribute"], "other")
            self.assertTrue(response["dialog_state"]["over_general"])
            self.assertEqual(response["dialog_state"]["phase"], "clarification")
            self.assertEqual(response["dialog_state"]["strategy"], "clarify_overload")

    def test_context_history_is_distilled_to_a_bounded_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            agent.reset("s", {"preference_tags": ["fit", "comfort"]})
            for turn in range(1, 9):
                agent.respond("s", "I prefer leather.", turn, 10)
            state = agent.sessions["s"]
            self.assertEqual(len(state["history"]), 6)
            self.assertEqual(state["context_version"], 8)
            self.assertEqual(state["long_term_profile"]["base_tags"], ["fit", "comfort"])
            self.assertIn("leather", state["profile_terms"])

    def test_next_question_maximizes_expected_candidate_reduction(self) -> None:
        candidates = [
            *("red cotton shoe" for _ in range(4)),
            *("blue cotton shoe" for _ in range(3)),
            "blue leather shoe",
        ]
        state = {
            "slots": [],
            "asked": {"other", "feature"},
            "long_term_profile": {"rejected_attributes": set()},
        }
        state["question_scores"] = NextQuestionSelector.score(candidates, state)
        attribute, _message, _utility = NextQuestionSelector.choose(state)
        self.assertEqual(attribute, "color")
        self.assertGreater(
            state["question_scores"]["color"],
            state["question_scores"]["material"],
        )

    def test_known_or_declined_facets_are_discounted(self) -> None:
        candidates = [
            *("red cotton shoe" for _ in range(2)),
            *("red leather shoe" for _ in range(2)),
            *("blue cotton shoe" for _ in range(2)),
            *("blue leather shoe" for _ in range(2)),
        ]
        state = {
            "slots": [{"kind": "color", "active": True}],
            "asked": {"other", "feature"},
            "long_term_profile": {"rejected_attributes": {"color"}},
        }
        state["question_scores"] = NextQuestionSelector.score(candidates, state)
        attribute, _message, _utility = NextQuestionSelector.choose(state)
        self.assertEqual(attribute, "material")

    def test_sparse_facet_does_not_overstate_information_gain(self) -> None:
        candidates = [
            *("cotton shoe" for _ in range(5)),
            *("leather shoe" for _ in range(3)),
            "red leather shoe",
            "blue leather shoe",
        ]
        state = {
            "slots": [],
            "asked": {"other", "feature"},
            "long_term_profile": {"rejected_attributes": set()},
        }
        state["question_scores"] = NextQuestionSelector.score(candidates, state)
        attribute, _message, _utility = NextQuestionSelector.choose(state)
        self.assertEqual(attribute, "material")

    def test_recommendations_widen_after_evidence_gathering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory), count=12)
            agent.reset("s", {"preference_tags": []})
            early = agent.respond("s", "I'm looking for Shoes.", 1, 10)
            late = agent.respond("s", "For that, what matters is: cotton lining.", 3, 10)
            self.assertEqual(len(early["recommendations"]), 1)
            self.assertEqual(early["dialog_state"]["recommendation_limit"], 1)
            self.assertEqual(len(late["recommendations"]), 10)
            self.assertEqual(late["dialog_state"]["recommendation_limit"], 10)

    def test_second_turn_collects_remaining_broad_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory))
            agent.reset("s", {"preference_tags": []})
            first = agent.respond(
                "s", "I'm looking for Shoes, but I'm still exploring.", 1, 10,
            )
            second = agent.respond(
                "s", "For that, what matters is: cotton; comfortable.", 2, 10,
            )
            self.assertEqual(first["ask_attribute"], "other")
            self.assertEqual(second["ask_attribute"], "other")
            self.assertEqual(agent.sessions["s"]["last_ask"], "other")

    def test_intent_override_gets_one_precision_recovery_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory), count=12)
            agent.reset("s", {"preference_tags": []})
            agent.respond("s", "I'm looking for Shoes. I prefer leather.", 1, 10)
            response = agent.respond(
                "s",
                "Actually, ignore my earlier preference. What I need is: cotton.",
                4,
                10,
            )
            self.assertEqual(len(response["recommendations"]), 1)
            self.assertEqual(response["dialog_state"]["recommendation_limit"], 1)

    def test_declined_initial_clarification_delays_widening_one_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent(Path(directory), count=12)
            agent.reset("s", {"preference_tags": []})
            agent.respond("s", "I'm looking for Shoes, but I'm still exploring.", 1, 10)
            agent.respond(
                "s", "I don't have a preference for other; please use your judgment.", 2, 10,
            )
            recovery = agent.respond(
                "s", "For that, what matters is: cotton lining.", 3, 10,
            )
            widened = agent.respond(
                "s", "I don't have an additional preference for feature.", 4, 10,
            )
            self.assertEqual(len(recovery["recommendations"]), 1)
            self.assertEqual(recovery["dialog_state"]["recommendation_limit"], 1)
            self.assertEqual(len(widened["recommendations"]), 10)

    def test_exact_category_tail_beats_ancestor_only_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            shared = {
                "features": ["generic fabric"],
                "details": {},
                "description": [],
                "rating_number": 0,
            }
            rows = [
                {
                    **shared,
                    "parent_asin": "TARGET",
                    "title": "Target garment",
                    "categories": ["Clothing", "Novelty", "Women"],
                    "average_rating": 3.0,
                },
                {
                    **shared,
                    "parent_asin": "ANCESTOR",
                    "title": "Popular garment",
                    "categories": [
                        "Clothing", "Novelty", "Women", "Tops", "T-Shirts",
                    ],
                    "average_rating": 5.0,
                    "rating_number": 100000,
                },
            ]
            catalog.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            agent = Agent(catalog)
            agent.reset("s", {"preference_tags": []})
            response = agent.respond(
                "s", "I'm looking for Novelty Women.", 4, 10,
            )
            self.assertEqual(
                response["recommendations"][0]["parent_asin"], "TARGET",
            )

    def test_exact_feature_evidence_beats_description_only_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            rows = [
                {
                    "parent_asin": "TARGET",
                    "title": "Target shoe",
                    "features": ["Unique stitched gusset"],
                    "details": {},
                    "description": [],
                    "categories": ["Clothing", "Shoes"],
                    "average_rating": 3.0,
                    "rating_number": 0,
                },
                {
                    "parent_asin": "DESCRIPTION",
                    "title": "Popular shoe",
                    "features": ["generic construction"],
                    "details": {},
                    "description": ["Unique stitched gusset"],
                    "categories": ["Clothing", "Shoes"],
                    "average_rating": 5.0,
                    "rating_number": 100000,
                },
            ]
            catalog.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            agent = Agent(catalog)
            agent.reset("s", {"preference_tags": []})
            response = agent.respond(
                "s",
                "I'm looking for Shoes. Key requirement is: Unique stitched gusset.",
                4,
                10,
            )
            self.assertEqual(
                response["recommendations"][0]["parent_asin"], "TARGET",
            )

    def test_color_field_label_is_not_required_catalog_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            rows = [
                {
                    "parent_asin": "TARGET",
                    "title": "Red walking shoe",
                    "features": ["lightweight"],
                    "details": {},
                    "description": [],
                    "categories": ["Clothing", "Shoes"],
                    "average_rating": 3.0,
                    "rating_number": 0,
                },
                {
                    "parent_asin": "FIELD_ONLY",
                    "title": "Walking shoe",
                    "features": ["color accent"],
                    "details": {},
                    "description": [],
                    "categories": ["Clothing", "Shoes"],
                    "average_rating": 5.0,
                    "rating_number": 100000,
                },
            ]
            catalog.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            agent = Agent(catalog)
            agent.reset("s", {"preference_tags": []})
            response = agent.respond(
                "s", "I'm looking for Shoes. Key requirement is: color: red.", 4, 10,
            )
            self.assertEqual(response["recommendations"][0]["parent_asin"], "TARGET")

    def test_profile_terms_do_not_override_current_session_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.jsonl"
            rows = [
                {
                    "parent_asin": "TARGET",
                    "title": "Walking shoe",
                    "features": ["cotton lining"],
                    "details": {},
                    "description": [],
                    "categories": ["Clothing", "Shoes"],
                    "average_rating": 4.0,
                    "rating_number": 21,
                },
                {
                    "parent_asin": "PROFILE_ONLY",
                    "title": "Durability walking shoe",
                    "features": ["cotton lining"],
                    "details": {},
                    "description": [],
                    "categories": ["Clothing", "Shoes"],
                    "average_rating": 4.0,
                    "rating_number": 20,
                },
            ]
            catalog.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            agent = Agent(catalog)
            agent.reset("s", {"preference_tags": ["durability"]})
            response = agent.respond("s", "I'm looking for Shoes.", 4, 10)
            self.assertEqual(response["recommendations"][0]["parent_asin"], "TARGET")

if __name__ == "__main__":
    unittest.main()
