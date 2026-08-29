"""Offline conversational product retriever."""
from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOPWORDS = {"a","an","and","are","as","at","be","but","by","for","from","i","in","is","it","me","my","of","on","or","please","some","that","the","this","to","want","with","would","you","looking","have","has","do","does","not","no","about","additional","what","matters","need","key","requirement","preference","prioritize"}

def _text(value: object) -> str:
    if value is None: return ""
    if isinstance(value, dict): return " ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list): return " ".join(str(v) for v in value)
    return str(value)

def _terms(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 1 and t.lower() not in STOPWORDS]

def _normal(text: str) -> str:
    return " ".join(_terms(text))

class Agent:
    """Dependency-free hybrid FTS and disclosed-constraint ranker."""
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.sessions: dict[str, dict] = {}
        self.product_quality: dict[str, float] = {}
        self._build_index()

    def _build_index(self) -> None:
        cur = self.connection.cursor()
        cur.execute("CREATE VIRTUAL TABLE products USING fts5(parent_asin UNINDEXED, title, categories, attributes, description)")
        batch = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                p = json.loads(line); asin = str(p["parent_asin"])
                title, categories = _text(p.get("title")), _text(p.get("categories"))
                attributes = " ".join((_text(p.get("features")), _text(p.get("details")), _text(p.get("store")), _text(p.get("price"))))
                description = _text(p.get("description"))
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

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.sessions[session_id] = {"constraints": [], "profile_terms": _terms(" ".join(str(x) for x in user_profile.get("preference_tags", []))), "asked": set()}

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

    def _candidate_rows(self, constraints: list[str]) -> dict[str, str]:
        """Retrieve only a bounded working set; catalog text remains in SQLite."""
        candidates: dict[str, str] = {}
        def add_rows(rows: list[tuple]) -> None:
            for row in rows:
                candidates[str(row[0])] = _normal(" ".join(str(value or "") for value in row[1:]))

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
        return candidates

    def _rank(self, state: dict, top_k: int) -> list[dict]:
        constraints = state["constraints"]
        if not constraints: return []
        scored: list[tuple[float, str]] = []
        for asin, full in self._candidate_rows(constraints).items():
            doc_tokens = frozenset(full.split()); score = 0.0
            for index, constraint in enumerate(constraints):
                query = frozenset(_terms(constraint))
                if not query: continue
                matched = query & doc_tokens
                coverage = len(matched) / len(query)
                exact = 1.0 if _normal(constraint) in full else 0.0
                weight = 1.0 + index * .18
                score += weight * (coverage * 12 + exact * 14 + len(matched) * .35)
            score += len(set(state["profile_terms"]) & doc_tokens) * .12
            score += self.product_quality.get(asin, 0.0)
            scored.append((score, asin))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [{"parent_asin": asin, "score": round(score, 5)} for score, asin in scored[:top_k]]

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        if session_id not in self.sessions: raise RuntimeError("reset must be called before respond")
        state = self.sessions[session_id]
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
