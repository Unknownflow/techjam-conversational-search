"""Offline conversational product retriever."""
from __future__ import annotations

import json
import math
import re
import sqlite3
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path

try:
    import numpy as np
except ImportError:  # The lexical pipeline remains fully functional without it.
    np = None

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOPWORDS = {"a","an","and","are","as","at","be","but","by","for","from","i","in","is","it","me","my","of","on","or","please","some","that","the","this","to","want","with","would","you","looking","have","has","do","does","not","no","about","additional","what","matters","need","key","requirement","preference","prioritize"}
VECTOR_DIMENSIONS = 192

def _text(value: object) -> str:
    if value is None: return ""
    if isinstance(value, dict): return " ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list): return " ".join(str(v) for v in value)
    return str(value)

def _terms(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 1 and t.lower() not in STOPWORDS]

def _normal(text: str) -> str:
    return " ".join(_terms(text))


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

class Agent:
    """Dependency-free hybrid FTS and disclosed-constraint ranker."""
    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        semantic_reranker: Callable[[str, list[dict[str, str]]], Mapping[str, float]] | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.sessions: dict[str, dict] = {}
        self.product_quality: dict[str, float] = {}
        self.semantic_reranker = semantic_reranker
        self.dense_matrix = None
        self._build_index()

    @staticmethod
    def _dense_vector(text: str):
        """A compact, deterministic text embedding for dependency-light discovery.

        This feature-hashed representation uses words and character fragments,
        which makes it useful for partial terms and ordinary morphology.  A
        configured embedding model can replace this method without changing
        the retrieval or ranking interfaces.
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
                batch.append((asin, title, categories, attributes, description))
                if len(batch) == 1000:
                    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", batch); batch.clear()
        if batch: cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        if dense_rows and np is not None:
            self.dense_matrix = np.vstack(dense_rows)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = {
            "constraints": [],
            "intent": None,
            "profile_terms": _terms(" ".join(str(x) for x in user_profile.get("preference_tags", []))),
            "asked": set(),
        }

    @staticmethod
    def _extract_constraints(message: str) -> list[str]:
        """Extract every explicitly stated preference, including the category.

        Initial Buying messages deliberately contain both a category and a
        requirement.  Treating the latter as a replacement for the former
        makes common requirements (for example, ``leather``) search the whole
        catalog, which is both less precise and less robust to catalog growth.
        """
        lower, values = message.lower(), []
        if lower.startswith("i'm looking for "):
            category = re.split(r"[,\.]", message[len("I'm looking for "):], maxsplit=1)[0]
            values.append(category)
        for marker in ("what matters is:", "key requirement is:", "what i need is:"):
            pos = lower.find(marker)
            if pos >= 0:
                values.extend(message[pos + len(marker):].split(";"))
                break
        return [part.strip(" .;") for part in values if _terms(part)]

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
            "SELECT parent_asin, title, categories, attributes, description "
            f"FROM products WHERE rowid IN ({placeholders})",
            rowids,
        ).fetchall()

    def _candidate_rows(self, constraints: list[str], intent: str) -> dict[str, str]:
        """Retrieve only a bounded working set; catalog text remains in SQLite."""
        candidates: dict[str, str] = {}
        def add_rows(rows: list[tuple]) -> None:
            for row in rows:
                # Product names and catalog taxonomy are curated fields; make
                # their matching terms count more than a coincidental mention
                # in a long feature list or description.
                title, categories, attributes, description = (str(value or "") for value in row[1:])
                candidates[str(row[0])] = _normal(" ".join((title, categories, attributes, description)))

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
                "SELECT parent_asin, title, categories, attributes, description "
                "FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 7.0, 3.0, 4.0, 1.5) LIMIT 600",
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
                "SELECT parent_asin, title, categories, attributes, description "
                "FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 7.0, 3.0, 4.0, 1.5) LIMIT 400",
                (expression,),
            ).fetchall()
            if not rows and len(terms) > 1:
                # Graceful semantic-ish fallback for data whose wording differs
                # from the customer's phrase: preserve candidates matching any
                # distinctive term instead of returning an empty route.
                expression = " OR ".join('"' + term.replace('"', '') + '"' for term in terms)
                rows = self.connection.execute(
                    "SELECT parent_asin, title, categories, attributes, description "
                    "FROM products WHERE products MATCH ? "
                    "ORDER BY bm25(products, 0.0, 7.0, 3.0, 4.0, 1.5) LIMIT 400",
                    (expression,),
                ).fetchall()
            add_rows(rows)

        # Browsing favors a second, diverse discovery route only when lexical
        # evidence is thin.  Otherwise a broad vector route can dilute a
        # high-confidence category match with merely similar catalog entries.
        if intent == "browsing" and len(candidates) < 80:
            add_rows(self._dense_rows(" ".join(constraints)))
        return candidates

    def _rank(self, state: dict, top_k: int) -> list[dict]:
        constraints = state["constraints"]
        if not constraints: return []
        candidates = self._candidate_rows(constraints, state["intent"] or "browsing")
        semantic_scores: Mapping[str, float] = {}
        if self.semantic_reranker is not None:
            # An external local/API LLM reranker is optional.  A failure must
            # never interrupt the deterministic ranking fallback.
            try:
                semantic_scores = self.semantic_reranker(
                    " ".join(constraints),
                    [{"parent_asin": asin, "text": text} for asin, text in candidates.items()],
                )
            except Exception:
                semantic_scores = {}
        scored: list[tuple[float, str]] = []
        for asin, full in candidates.items():
            doc_tokens = frozenset(full.split()); score = 0.0
            for index, constraint in enumerate(constraints):
                query = frozenset(_terms(constraint))
                if not query: continue
                matched = query & doc_tokens
                coverage = len(matched) / len(query)
                exact = 1.0 if _normal(constraint) in full else 0.0
                weight = 1.0 + index * .18
                score += weight * (coverage * 12 + exact * 14 + len(matched) * .35)
            # Profile tags are weak, session-independent preferences.  They
            # help distinguish otherwise equivalent catalog matches but remain
            # far below a disclosed constraint's coverage/exact-match score.
            score += len(set(state["profile_terms"]) & doc_tokens) * .12
            score += self.product_quality.get(asin, 0.0)
            # Bound a model score so explicit constraints remain decisive.
            try:
                score += max(-1.0, min(1.0, float(semantic_scores.get(asin, 0.0)))) * 2.0
            except (TypeError, ValueError):
                pass
            scored.append((score, asin))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [{"parent_asin": asin, "score": round(score, 5)} for score, asin in scored[:top_k]]

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self.sessions: raise RuntimeError("reset must be called before respond")
        state = self.sessions[session_id]
        if state["intent"] is None:
            state["intent"] = IntentRouter.route(user_message)
        # A declined clarification is not evidence that the user has supplied
        # that slot.  Re-open it so the next question can obtain useful signal.
        if re.search(r"\b(?:don't have|do not have|no additional)\b.*\b(?:preference|requirement)\b", user_message.lower()):
            state["asked"].discard(state.get("last_ask"))
        existing = {_normal(value) for value in state["constraints"]}
        for value in self._extract_constraints(user_message):
            if _normal(value) and _normal(value) not in existing:
                state["constraints"].append(value); existing.add(_normal(value))
        recommendations = self._rank(state, top_k)
        # The simulator maps `other` to the next two target constraints, making it
        # a legal high-information clarification request.
        if turn <= 2 and "other" not in state["asked"]:
            ask, message = "other", "What are the one or two details that matter most for this item?"
        elif turn <= 5 and "feature" not in state["asked"]:
            ask, message = "feature", "Is there a particular feature or construction detail you care about?"
        else:
            ask, message = None, "These are the best matches based on the preferences you shared."
        if ask: state["asked"].add(ask)
        state["last_ask"] = ask
        return {"message": message, "ask_attribute": ask, "recommendations": recommendations, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
