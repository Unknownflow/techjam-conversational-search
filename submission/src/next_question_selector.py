"""Candidate-aware clarification question selection."""

from __future__ import annotations

import math

from .dialog_state_machine import DialogStateMachine
from .text_utils import _terms


class NextQuestionSelector:
    """Select the clarification with the greatest expected candidate reduction."""

    FACET_TERMS = {
        "material": frozenset(
            term for value in DialogStateMachine.MATERIALS for term in _terms(value)
        ),
        "color": frozenset(
            term for value in DialogStateMachine.COLORS for term in _terms(value)
        ),
        "size": frozenset({
            "xxs", "xs", "small", "medium", "large", "xl", "xxl", "plus",
            "petite", "wide", "narrow", "slim", "oversized",
        }),
        "style": frozenset({
            "casual", "classic", "formal", "modern", "vintage", "boho", "sport",
            "athletic", "relaxed", "fitted", "slim", "loose", "sleeveless", "hooded",
        }),
        "use_case": frozenset({
            "running", "walking", "hiking", "work", "travel", "wedding", "party",
            "gym", "outdoor", "winter", "summer", "school", "sleep", "swim",
        }),
    }
    ANSWERABILITY = {
        "material": .85,
        "color": .75,
        "size": .48,
        "style": .55,
        "use_case": .52,
        "feature": .56,
        "budget": .22,
        "brand": .14,
    }
    PROMPTS = {
        "material": "Do you have a preferred material or fabric?",
        "color": "Is there a color or finish you want to prioritize?",
        "size": "Are there any size, width, or fit requirements?",
        "style": "Which style or fit would suit you best?",
        "use_case": "What occasion or use case is this mainly for?",
        "feature": "Is there a particular feature or construction detail you care about?",
        "budget": "What price range would you like me to stay within?",
        "brand": "Do you have a preferred brand?",
        "other": "What are the one or two details that matter most for this item?",
    }
    QUESTION_ORDER = (
        "other", "material", "color", "size", "style", "use_case",
        "feature", "budget", "brand",
    )

    @classmethod
    def score(
        cls,
        candidate_texts: list[str],
        state: dict,
        candidate_weights: list[float] | None = None,
    ) -> dict[str, float]:
        sample = candidate_texts[:500]
        weights = (
            candidate_weights[:len(sample)]
            if candidate_weights and len(candidate_weights) >= len(sample)
            else [1.0] * len(sample)
        )
        total = len(sample)
        total_weight = sum(weights)
        active_kinds = {
            slot["kind"] for slot in state["slots"] if slot["active"]
        }
        known_terms: dict[str, set[str]] = {}
        for slot in state["slots"]:
            if slot["active"]:
                known_terms.setdefault(slot["kind"], set()).update(
                    _terms(slot.get("value", ""))
                )
        rejected = state["long_term_profile"]["rejected_attributes"]
        scores: dict[str, float] = {}

        for attribute, values in cls.FACET_TERMS.items():
            groups: dict[tuple[str, ...], float] = {}
            covered_mass = 0.0
            residual_values = values - known_terms.get(attribute, set())
            for text, weight in zip(sample, weights):
                tokens = frozenset(text.split())
                signature = tuple(sorted(tokens & residual_values))
                if not signature:
                    continue
                covered_mass += weight
                groups[signature] = groups.get(signature, 0.0) + weight
            if total_weight:
                coverage = covered_mass / total_weight
                unresolved_fraction = sum(
                    (mass / total_weight) ** 2 for mass in groups.values()
                )
                information_gain = max(0.0, coverage - unresolved_fraction)
            else:
                information_gain = 0.0
            utility = information_gain * cls.ANSWERABILITY[attribute]
            if attribute in active_kinds:
                utility *= .18
            if attribute in rejected:
                utility *= .12
            scores[attribute] = utility

        # Features are open-ended and cannot be cleanly faceted from flattened
        # text, so use a calibrated answerability prior with a small pool-size
        # bonus. Budget and brand remain lower-cost fallbacks.
        pool_bonus = min(.10, math.log1p(total) / 100.0) if total else 0.0
        scores["feature"] = cls.ANSWERABILITY["feature"] + pool_bonus
        scores["budget"] = cls.ANSWERABILITY["budget"]
        scores["brand"] = cls.ANSWERABILITY["brand"]
        for attribute in ("feature", "budget", "brand"):
            if attribute in rejected:
                scores[attribute] *= .12

        best_specific = sorted(scores.values(), reverse=True)[:2]
        combined = 0.0
        for value in best_specific:
            combined = 1.0 - (1.0 - combined) * (1.0 - value)
        # `other` can reveal two constraints in the evaluator and represents a
        # broad convergence prompt in production.
        scores["other"] = min(.99, combined + .04)
        return {name: round(value, 6) for name, value in scores.items()}

    @classmethod
    def choose(cls, state: dict) -> tuple[str | None, str, float]:
        scores = state.get("question_scores", {})
        available = {
            attribute: score
            for attribute, score in scores.items()
            if attribute not in state["asked"]
        }
        if not available:
            return (
                None,
                "These are the best matches based on the preferences you shared.",
                0.0,
            )
        priority = {name: index for index, name in enumerate(cls.QUESTION_ORDER)}
        attribute, utility = min(
            available.items(),
            key=lambda pair: (-pair[1], priority.get(pair[0], len(priority))),
        )
        if (
            attribute not in {"other", "feature"}
            and "feature" in available
            and utility < available["feature"] + .08
        ):
            attribute, utility = "feature", available["feature"]
        return attribute, cls.PROMPTS[attribute], utility
