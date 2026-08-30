"""Deterministic conversational product retriever."""
from __future__ import annotations

import json
import math
import re
import sqlite3
import zlib
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # The lexical pipeline remains fully functional without it.
    np = None

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOPWORDS = {"a","an","and","are","as","at","be","but","by","for","from","i","in","is","it","me","my","of","on","or","please","some","that","the","this","to","want","with","would","you","looking","have","has","do","does","not","no","about","additional","what","matters","need","key","requirement","preference","prioritize"}
VECTOR_DIMENSIONS = 192
OVERGENERALITY_THRESHOLD = 350
CATEGORY_TAIL_EXACT_BONUS = 10.0
EVIDENCE_EXACT_BONUS = 2.0
INITIAL_PRECISION_TURNS = 2

def _text(value: object) -> str:
    if value is None: return ""
    if isinstance(value, dict): return " ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list): return " ".join(str(v) for v in value)
    return str(value)

def _terms(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 1 and t.lower() not in STOPWORDS]

def _normal(text: str) -> str:
    return " ".join(_terms(text))


def _category_tail(value: object) -> str:
    """Normalize the final two meaningful catalog category nodes."""
    excluded = {
        "clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry",
    }
    values = value if isinstance(value, list) else [value]
    cleaned: list[str] = []
    for item in values:
        for part in str(item or "").split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return _normal(" ".join(cleaned[-2:]))


def _field_evidence(product: dict) -> list[str]:
    """Keep exact feature/detail values without retaining the full product."""
    values: list[str] = []
    features = product.get("features")
    if isinstance(features, list):
        values.extend(_normal(str(item)) for item in features)
    elif features not in (None, ""):
        values.append(_normal(str(features)))
    details = product.get("details")
    if isinstance(details, dict):
        values.extend(_normal(f"{key} {item}") for key, item in details.items())
    elif isinstance(details, list):
        values.extend(_normal(str(item)) for item in details)
    elif details not in (None, ""):
        values.append(_normal(str(details)))
    return list(dict.fromkeys(value for value in values if value))


@dataclass(frozen=True, slots=True)
class CandidateDocument:
    text: str
    category_tail: str
    evidence: frozenset[str]


class IntentRouter:
    """Choose precision-first Buying or discovery-first Browsing retrieval."""

    BUYING_SIGNALS = ("key requirement", "what i need", "must have", "need ", "budget")
    BROWSING_SIGNALS = ("exploring", "browse", "ideas", "inspiration", "not sure")

    @classmethod
    def route(cls, message: str) -> str:
        lowered = message.lower()
        if any(signal in lowered for signal in cls.BROWSING_SIGNALS):
            return "browsing"
        if any(signal in lowered for signal in cls.BUYING_SIGNALS):
            return "buying"
        return "browsing"


class DialogStateMachine:
    """Accumulate slots, retire superseded preferences, and expose dialog phase."""

    MATERIALS = {
        "acetate", "acrylic", "aluminum", "aluminium", "bamboo", "brass",
        "canvas", "carbon", "cashmere", "ceramic", "chiffon", "chrome",
        "corduroy", "cork", "cotton", "crystal", "denim", "elastic", "fabric",
        "faux fur", "faux leather", "felt", "fiberglass", "flannel", "foam",
        "fur", "glass", "gold", "hemp", "iron", "jute", "lace", "latex", "leather",
        "linen", "lycra", "mesh", "metal", "microfiber", "modal", "neoprene", "nickel",
        "nylon", "paper", "pearl", "pewter", "plastic", "pleather", "polyamide", "polyester",
        "polypropylene", "polyurethane", "porcelain", "rayon", "resin", "rubber", "satin",
        "silicone", "silk", "silver", "spandex", "stainless steel", "steel", "suede",
        "synthetic", "tencel", "textile", "titanium", "tungsten", "velvet", "vinyl",
        "viscose", "wood", "zinc",
    }
    COLORS = {
        "aqua", "aquamarine", "apricot", "azure", "beige", "black", "blush", "blue",
        "bronze", "brown", "burgundy", "camel", "charcoal", "chocolate", "clear",
        "cobalt", "copper", "coral", "cream", "cyan", "ecru", "emerald", "fluorescent",
        "fuchsia", "gold", "gray", "green", "grey", "indigo", "ivory", "khaki", "lavender", "lilac", "lime",
        "magenta", "maroon", "mint", "multicolor", "mustard", "navy", "olive", "orange",
        "neon", "peach", "periwinkle", "pink", "plum", "purple", "raspberry", "red", "rose",
        "rust", "salmon", "sand", "scarlet", "silver", "slate", "tan", "taupe", "teal", "turquoise", "violet",
        "white", "wine", "yellow",
    }

    @classmethod
    def slot_kind(cls, value: str, source: str) -> str:
        terms = set(_terms(value))
        lowered = value.lower()
        if source == "category": return "category"
        if "budget" in lowered or "$" in value or "under " in lowered: return "budget"
        if terms & cls.MATERIALS: return "material"
        if "color" in terms or terms & cls.COLORS: return "color"
        if terms & {"size", "sizing", "width", "wide", "narrow"}: return "size"
        if terms & {"hiking", "running", "gym", "winter", "outdoor", "work"}: return "use_case"
        if terms & {"style", "fit", "sleeve", "neck", "department"}: return "style"
        return "feature"

    @classmethod
    def apply(cls, state: dict, observations: list[tuple[str, str]], turn: int, override: bool) -> None:
        if override:
            # An override erases prior soft preferences, while retaining the
            # category and independently confirmed hard constraints.
            for slot in state["slots"]:
                if slot["active"] and slot["source"] == "preference":
                    slot["active"] = False

        active_values = {_normal(slot["value"]) for slot in state["slots"] if slot["active"]}
        for value, source in observations:
            normalized = _normal(value)
            if not normalized or normalized in active_values:
                continue
            kind = cls.slot_kind(value, source)
            if source == "override":
                # A later override rewrites an earlier override of the same
                # slot type but does not erase unrelated hard requirements.
                for slot in state["slots"]:
                    if slot["active"] and slot["source"] == "override" and slot["kind"] == kind:
                        slot["active"] = False
            if kind == "category":
                for slot in state["slots"]:
                    if slot["active"] and slot["kind"] == "category":
                        slot["active"] = False
            state["slots"].append({
                "kind": kind, "value": value, "source": source,
                "turn": turn, "active": True,
            })
            active_values.add(normalized)

        state["constraints"] = [slot["value"] for slot in state["slots"] if slot["active"]]
        state["phase"] = "intent_override" if override else (
            "refinement" if len(state["constraints"]) > 1 else "discovery"
        )


class ContextProgram:
    """Distill dialog memory and select the next runtime workflow."""

    HISTORY_LIMIT = 6

    @classmethod
    def distill(cls, state: dict, user_message: str, turn: int) -> None:
        state["history"].append({"turn": turn, "role": "user", "text": user_message[:500]})
        del state["history"][:-cls.HISTORY_LIMIT]

        declined = bool(re.search(
            r"\b(?:don't have|do not have|no additional)\b.*\b(?:preference|requirement)\b",
            user_message,
            re.I,
        ))
        if declined and state.get("last_ask"):
            state["long_term_profile"]["rejected_attributes"].add(state["last_ask"])

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
        learned_terms = _terms(" ".join(
            value for values in learned.values() for value in values
        ))
        state["profile_terms"] = list(dict.fromkeys(
            [*state["long_term_profile"]["base_tags"], *learned_terms]
        ))
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
                known_terms.setdefault(slot["kind"], set()).update(_terms(slot.get("value", "")))
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
            attribute: score for attribute, score in scores.items()
            if attribute not in state["asked"]
        }
        if not available:
            return None, "These are the best matches based on the preferences you shared.", 0.0
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
        requirement.  Treating the latter as a replacement for the former
        makes common requirements (for example, ``leather``) search the whole
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
        # preferences have been disclosed.  Per-constraint routes below retain
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
        # evidence is thin.  Otherwise a broad vector route can dilute a
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
                # ``color`` is a field label generated by the conversation
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
        # that slot. A broad `other` prompt may still be reframed, but a typed
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
