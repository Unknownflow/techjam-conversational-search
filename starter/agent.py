"""Deterministic conversational product retriever."""
from __future__ import annotations

import json
import math
import re
import sqlite3
import zlib
from pathlib import Path

from .candidate_document import CandidateDocument
from .context_program import ContextProgram
from .dialog_state_machine import DialogStateMachine
from .intent_router import IntentRouter
from .next_question_selector import NextQuestionSelector
from .text_utils import (
    _category_tail,
    _field_evidence,
    _normal,
    _terms,
    _text,
)

try:
    import numpy as np
except ImportError:  # The lexical pipeline remains fully functional without it.
    np = None

VECTOR_DIMENSIONS = 192
# Keep this threshold in the entry-point module so existing callers can patch
# starter.agent.OVERGENERALITY_THRESHOLD.
OVERGENERALITY_THRESHOLD = 350
CATEGORY_TAIL_EXACT_BONUS = 10.0
EVIDENCE_EXACT_BONUS = 2.0
INITIAL_PRECISION_TURNS = 2


class Agent:
    """Grounded FTS retrieval with deterministic ranking by default."""
    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.sessions: dict[str, dict] = {}
        self.product_quality: dict[str, float] = {}
        self.dense_matrix = None
        self._build_index()

    @staticmethod
    def _dense_vector(text: str):
        """A compact, deterministic text embedding for dependency-light discovery.

        This feature-hashed representation uses words and character fragments,
        which makes it useful for partial terms and ordinary morphology. The
        representation is fixed so retrieval remains reproducible.
        """
        if np is None:
            return None
        vector = np.zeros(VECTOR_DIMENSIONS, dtype=np.float32)
        for term in set(_terms(text)):
            fragments = (term, *(term[index:index + 3] for index in range(max(0, len(term) - 2))))
            for fragment in fragments:
                hashed = zlib.crc32(fragment.encode("utf-8"))
                vector[hashed % VECTOR_DIMENSIONS] += 1.0
                vector[(hashed >> 11) % VECTOR_DIMENSIONS] -= 0.5
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def _build_index(self) -> None:
        cur = self.connection.cursor()
        # Porter stemming keeps lexical retrieval useful when the customer uses
        # a different inflection from catalog copy (for example, "wallet" vs.
        # "wallets") without adding a model dependency.
        cur.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, attributes, description, "
            "category_tail UNINDEXED, evidence UNINDEXED, "
            "tokenize='porter unicode61')"
        )
        batch = []
        dense_rows = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                p = json.loads(line); asin = str(p["parent_asin"])
                title, categories = _text(p.get("title")), _text(p.get("categories"))
                attributes = " ".join((_text(p.get("features")), _text(p.get("details")), _text(p.get("store")), _text(p.get("price"))))
                description = _text(p.get("description"))
                category_tail = _category_tail(p.get("categories"))
                evidence = "\n".join(_field_evidence(p))
                if np is not None:
                    dense_rows.append(self._dense_vector(" ".join((title, categories, attributes))))
                try:
                    rating = float(p.get("average_rating") or 0.0)
                    count = max(0.0, float(p.get("rating_number") or 0.0))
                    # A deliberately modest, smoothed popularity prior.  It
                    # only resolves near-ties between equally suitable items.
                    self.product_quality[asin] = max(0.0, rating - 3.0) * .35 + math.log1p(count) * .14
                except (TypeError, ValueError):
                    self.product_quality[asin] = 0.0
                batch.append((
                    asin, title, categories, attributes, description,
                    category_tail, evidence,
                ))
                if len(batch) == 1000:
                    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch); batch.clear()
        if batch: cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        if dense_rows and np is not None:
            self.dense_matrix = np.vstack(dense_rows)

    def reset(self, session_id: str, user_profile: dict) -> None:
        base_tags = _terms(" ".join(str(x) for x in user_profile.get("preference_tags", [])))
        self.sessions[session_id] = {
            "constraints": [],
            "slots": [],
            "intent": None,
            "intent_history": [],
            "phase": "discovery",
            "candidate_count": 0,
            "over_general": False,
            "question_scores": {},
            "question_decision": {},
            "history": [],
            "context_version": 0,
            "short_term_context": {},
            "long_term_profile": {
                "base_tags": list(dict.fromkeys(base_tags)),
                "summary": str(user_profile.get("summary") or "")[:500],
                "average_prior_rating": user_profile.get("average_prior_rating"),
                "rating_style": str(user_profile.get("rating_style") or ""),
                "purchase_frequency": str(user_profile.get("purchase_frequency") or ""),
                "learned_preferences": {},
                "rejected_attributes": set(),
            },
            "workflow": {
                "strategy": "discovery_expand",
                "allow_dense": True,
            },
            "profile_terms": base_tags,
            "asked": set(),
        }

    @staticmethod
    def _extract_observations(message: str) -> list[tuple[str, str]]:
        """Extract every explicitly stated preference, including the category.

        Initial Buying messages deliberately contain both a category and a
        requirement. Treating the latter as a replacement for the former
        makes common requirements (for example, leather) search the whole
        catalog, which is both less precise and less robust to catalog growth.
        """
        lower, values = message.lower(), []
        if lower.startswith("i'm looking for "):
            category = re.split(r"[,\.]", message[len("I'm looking for "):], maxsplit=1)[0]
            values.append((category, "category"))
        for marker, source in (
            ("what matters is:", "hard"),
            ("key requirement is:", "hard"),
            ("what i need is:", "override"),
        ):
            pos = lower.find(marker)
            if pos >= 0:
                values.extend((part, source) for part in message[pos + len(marker):].split(";"))
                break
        preference = re.search(r"\b(?:i prefer|i'd prefer|my preference is)\s*:?[ ]*(.+?)(?:[.;]|$)", message, re.I)
        if preference:
            values.append((preference.group(1), "preference"))
        return [(part.strip(" .;"), source) for part, source in values if _terms(part)]

    @staticmethod
    def _extract_constraints(message: str) -> list[str]:
        """Compatibility helper returning active values without provenance."""
        return [value for value, _source in Agent._extract_observations(message)]

    def _dense_rows(self, query: str, limit: int = 240) -> list[tuple]:
        """Retrieve a bounded discovery route from the in-memory embedding."""
        if self.dense_matrix is None or np is None:
            return []
        query_vector = self._dense_vector(query)
        if query_vector is None or not query_vector.any():
            return []
        similarities = self.dense_matrix @ query_vector
        count = min(limit, len(similarities))
        indexes = np.argpartition(similarities, -count)[-count:]
        # FTS5 rowids are assigned in insertion order, hence vector index + 1.
        rowids = [int(index) + 1 for index in indexes if similarities[index] > 0]
        if not rowids:
            return []
        placeholders = ",".join("?" for _ in rowids)
        return self.connection.execute(
            "SELECT parent_asin, title, categories, attributes, description, "
            "category_tail, evidence "
            f"FROM products WHERE rowid IN ({placeholders})",
            rowids,
        ).fetchall()

    def _candidate_rows(
        self, constraints: list[str], intent: str, allow_dense: bool = True,
    ) -> dict[str, CandidateDocument]:
        """Retrieve only a bounded working set; catalog text remains in SQLite."""
        candidates: dict[str, CandidateDocument] = {}
        def add_rows(rows: list[tuple]) -> None:
            for row in rows:
                title, categories, attributes, description = (
                    str(value or "") for value in row[1:5]
                )
                candidates[str(row[0])] = CandidateDocument(
                    text=_normal(" ".join((title, categories, attributes, description))),
                    category_tail=str(row[5] or ""),
                    evidence=frozenset(str(row[6] or "").splitlines()),
                )

        # The intersection route protects precision when several independent
        # preferences have been disclosed. Per-constraint routes below retain
        # recall for incomplete, paraphrased, or overly-specific constraints.
        combined_terms: list[str] = []
        for constraint in constraints[:3]:
            combined_terms.extend(_terms(constraint)[:12])
        combined_terms = list(dict.fromkeys(combined_terms))[:28]
        if len(combined_terms) >= 2:
            expression = " AND ".join('"' + term.replace('"', '') + '"' for term in combined_terms)
            rows = self.connection.execute(
                "SELECT parent_asin, title, categories, attributes, description, "
                "category_tail, evidence "
                "FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 7.0, 3.0, 4.0, 1.5, 0.0, 0.0) LIMIT 600",
                (expression,),
            ).fetchall()
            add_rows(rows)

        for constraint in constraints:
            terms = list(dict.fromkeys(_terms(constraint)))[:20]
            if not terms: continue
            # A disclosed value comes directly from catalog metadata in the
            # simulator, so requiring its content words gives far better
            # precision (and a much smaller in-memory working set) than a
            # broad OR query.
            expression = " AND ".join('"' + term.replace('"', '') + '"' for term in terms)
            rows = self.connection.execute(
                "SELECT parent_asin, title, categories, attributes, description, "
                "category_tail, evidence "
                "FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 7.0, 3.0, 4.0, 1.5, 0.0, 0.0) LIMIT 400",
                (expression,),
            ).fetchall()
            if not rows and len(terms) > 1:
                # Graceful semantic-ish fallback for data whose wording differs
                # from the customer's phrase: preserve candidates matching any
                # distinctive term instead of returning an empty route.
                expression = " OR ".join('"' + term.replace('"', '') + '"' for term in terms)
                rows = self.connection.execute(
                    "SELECT parent_asin, title, categories, attributes, description, "
                    "category_tail, evidence "
                    "FROM products WHERE products MATCH ? "
                    "ORDER BY bm25(products, 0.0, 7.0, 3.0, 4.0, 1.5, 0.0, 0.0) LIMIT 400",
                    (expression,),
                ).fetchall()
            add_rows(rows)

        # Browsing favors a second, diverse discovery route only when lexical
        # evidence is thin. Otherwise a broad vector route can dilute a
        # high-confidence category match with merely similar catalog entries.
        if intent == "browsing" and allow_dense and len(candidates) < 80:
            add_rows(self._dense_rows(" ".join(constraints)))
        return candidates

    def _rank(self, state: dict, top_k: int) -> list[dict]:
        constraints = state["constraints"]
        if not constraints:
            state["candidate_count"] = 0
            state["over_general"] = True
            state["question_scores"] = NextQuestionSelector.score([], state)
            return []
        candidates = self._candidate_rows(
            constraints,
            state["intent"] or "browsing",
            bool(state["workflow"].get("allow_dense", True)),
        )
        state["candidate_count"] = len(candidates)
        state["over_general"] = (
            len(constraints) <= 1 and len(candidates) >= OVERGENERALITY_THRESHOLD
        )
        # Produce a grounded deterministic ranking over the recall set.
        base_scored: list[tuple[float, str]] = []
        active_slots = [slot for slot in state["slots"] if slot["active"]]
        for asin, candidate in candidates.items():
            full = candidate.text
            doc_tokens = frozenset(full.split()); score = 0.0
            for index, constraint in enumerate(constraints):
                slot_kind = active_slots[index]["kind"] if index < len(active_slots) else None
                scoring_terms = _terms(constraint)
                # color is a field label generated by the conversation
                # protocol, not part of the requested value. Requiring that
                # synthetic word would under-score products that simply say
                # "red" or "navy" in their catalog copy.
                if slot_kind == "color":
                    scoring_terms = [term for term in scoring_terms if term != "color"]
                query = frozenset(scoring_terms)
                if not query: continue
                matched = query & doc_tokens
                coverage = len(matched) / len(query)
                normalized = " ".join(scoring_terms)
                canonical_category = (
                    slot_kind == "category"
                    and _normal(constraint) == candidate.category_tail
                )
                # A category path is exact only when it is the product's
                # canonical tail. Ancestor/category-text occurrences are
                # useful for coverage but are not equally specific.
                exact = 1.0 if (
                    canonical_category
                    if slot_kind == "category"
                    else normalized in full
                ) else 0.0
                weight = 1.0 + index * .18
                score += weight * (coverage * 12 + exact * 14 + len(matched) * .35)
                if canonical_category:
                    score += CATEGORY_TAIL_EXACT_BONUS
                elif len(query) > 1 and normalized in candidate.evidence:
                    score += EVIDENCE_EXACT_BONUS * weight
            score += self.product_quality.get(asin, 0.0)
            base_scored.append((score, asin))
        base_scored.sort(key=lambda pair: (-pair[0], pair[1]))

        scored = base_scored
        focused_texts = [candidates[asin].text for _score, asin in scored[:200]]
        posterior_weights = [
            1.0 / math.log2(rank + 2) for rank in range(1, len(focused_texts) + 1)
        ]
        state["question_scores"] = NextQuestionSelector.score(
            focused_texts, state, posterior_weights,
        )
        return [{"parent_asin": asin, "score": round(score, 5)} for score, asin in scored[:top_k]]

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self.sessions: raise RuntimeError("reset must be called before respond")
        state = self.sessions[session_id]
        override = bool(re.search(r"\b(?:actually|instead)\b.*\b(?:ignore|need|want)\b", user_message, re.I))
        if state["intent"] is None:
            state["intent"] = IntentRouter.route(user_message)
            state["intent_history"].append({"turn": turn, "intent": state["intent"]})
        elif override and state["intent"] != "buying":
            state["intent"] = "buying"
            state["intent_history"].append({"turn": turn, "intent": "buying"})
        # A declined clarification is not evidence that the user has supplied
        # that slot. A broad other prompt may still be reframed, but a typed
        # attribute is recorded as declined and is not immediately repeated.
        if (
            re.search(r"\b(?:don't have|do not have|no additional)\b.*\b(?:preference|requirement)\b", user_message.lower())
            and state.get("last_ask") == "other"
        ):
            state["asked"].discard("other")
        DialogStateMachine.apply(
            state, self._extract_observations(user_message), turn, override,
        )
        if override:
            # Re-open the broad clarification route after a workflow reset.
            state["asked"].discard("other")
        ContextProgram.distill(state, user_message, turn)
        recommendations = self._rank(state, top_k)
        # Early low-confidence lists can lock a relevant product at a poor
        # reciprocal rank before the user has disclosed distinguishing facts.
        # Gather evidence with Top-1 for two turns, then widen to Top-K. An
        # explicit intent replacement gets one precision-only recovery turn
        # before widening, regardless of when the replacement occurs. If the
        # customer declined the initial broad clarification, keep turn three
        # precise because their distinguishing evidence arrives one turn later.
        delayed_evidence = (
            turn <= INITIAL_PRECISION_TURNS + 1
            and "other" in state["long_term_profile"]["rejected_attributes"]
        )
        precision_only = turn <= INITIAL_PRECISION_TURNS or override or delayed_evidence
        recommendation_limit = 1 if precision_only else top_k
        recommendations = recommendations[:recommendation_limit]
        ContextProgram.orchestrate(state)
        profile_tags = state["long_term_profile"]["base_tags"][:2]
        profile_hint = (
            f" Your profile emphasizes {', '.join(profile_tags)}, if that is relevant."
            if profile_tags else ""
        )
        # The first broad answer supplies at most two facts. Ask once more on
        # turn two so any remaining material, construction, style, or use-case
        # evidence can arrive together instead of being split across several
        # narrow questions. Intent replacement uses its recovery prompt instead.
        repeat_other = (
            turn == 2
            and not override
            and state.get("last_ask") == "other"
        )
        if repeat_other:
            ask, message, question_utility = (
                "other",
                "What are the next one or two details that matter most for this item?",
                state["question_scores"].get("other", 0.0),
            )
        elif turn <= 9:
            ask, message, question_utility = NextQuestionSelector.choose(state)
        else:
            ask, message, question_utility = (
                None,
                "These are the best matches based on the preferences you shared.",
                0.0,
            )
        if state["over_general"] and ask == "other":
            message = (
                "I found many possible matches. What are the one or two non-negotiable "
                "details—such as material, style, use case, or budget?" + profile_hint
            )
            state["phase"] = "clarification"
        elif state["workflow"]["strategy"] == "override_recovery" and ask == "other":
            message = (
                "Understood—I have reset the earlier preference. What other detail should "
                "I prioritize for the new direction?"
            )
        elif ask == "other":
            message += profile_hint
        if ask is None:
            state["phase"] = "recommendation"
        if ask: state["asked"].add(ask)
        state["last_ask"] = ask
        state["question_decision"] = {
            "attribute": ask,
            "utility": round(question_utility, 6),
            "scores": dict(sorted(
                state["question_scores"].items(),
                key=lambda pair: (-pair[1], pair[0]),
            )),
        }
        state["short_term_context"].update({
            "phase": state["phase"],
            "candidate_count": state["candidate_count"],
            "over_general": state["over_general"],
            "strategy": state["workflow"]["strategy"],
        })
        return {
            "message": message,
            "ask_attribute": ask,
            "recommendations": recommendations,
            "dialog_state": {
                "intent": state["intent"],
                "phase": state["phase"],
                "active_slots": len(state["constraints"]),
                "candidate_count": state["candidate_count"],
                "over_general": state["over_general"],
                "strategy": state["workflow"]["strategy"],
                "context_version": state["context_version"],
                "recommendation_limit": recommendation_limit,
                "next_question": state["question_decision"],
            },
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
