"""Short- and long-term conversational context management."""

from __future__ import annotations

import re

from .text_utils import _terms


class ContextProgram:
    """Distill dialog memory and select the next runtime workflow."""

    HISTORY_LIMIT = 6

    @classmethod
    def distill(cls, state: dict, user_message: str, turn: int) -> None:
        state["history"].append(
            {"turn": turn, "role": "user", "text": user_message[:500]}
        )
        del state["history"][:-cls.HISTORY_LIMIT]

        declined = bool(
            re.search(
                r"\b(?:don't have|do not have|no additional)\b.*"
                r"\b(?:preference|requirement)\b",
                user_message,
                re.I,
            )
        )
        if declined and state.get("last_ask"):
            state["long_term_profile"]["rejected_attributes"].add(
                state["last_ask"]
            )

        active_by_kind: dict[str, list[str]] = {}
        learned: dict[str, list[str]] = {}
        for slot in state["slots"]:
            if not slot["active"]:
                continue
            active_by_kind.setdefault(slot["kind"], []).append(slot["value"])
            if slot["source"] == "preference":
                learned.setdefault(slot["kind"], []).append(slot["value"])
        # Recompute rather than append forever, so overridden preferences are
        # also removed from the distilled profile.
        state["long_term_profile"]["learned_preferences"] = learned
        learned_terms = _terms(
            " ".join(value for values in learned.values() for value in values)
        )
        state["profile_terms"] = list(
            dict.fromkeys(
                [*state["long_term_profile"]["base_tags"], *learned_terms]
            )
        )
        state["context_version"] += 1
        state["short_term_context"] = {
            "version": state["context_version"],
            "intent": state["intent"],
            "phase": state["phase"],
            "active_slots": active_by_kind,
            "recent_turns": list(state["history"]),
            "turns_remaining": max(0, 10 - turn),
            "declined_last_attribute": declined,
        }

    @staticmethod
    def orchestrate(state: dict) -> None:
        """Re-program the next retrieval and guidance workflow from context."""
        slot_count = len(state["constraints"])
        candidate_count = state["candidate_count"]
        if state["phase"] == "intent_override":
            strategy, allow_dense = "override_recovery", False
        elif state["over_general"]:
            strategy, allow_dense = "clarify_overload", False
        elif state["intent"] == "buying":
            strategy, allow_dense = "precision_filter", False
        elif slot_count >= 3 or (candidate_count and candidate_count <= 40):
            strategy, allow_dense = "focused_rerank", False
        else:
            strategy, allow_dense = "discovery_expand", True
        state["workflow"] = {
            "strategy": strategy,
            "allow_dense": allow_dense,
        }
