"""Shared text normalization helpers for conversational retrieval."""

from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "have", "has", "do", "does", "not", "no", "about", "additional", "what",
    "matters", "need", "key", "requirement", "preference", "prioritize",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        term.lower()
        for term in TOKEN_RE.findall(text)
        if len(term) > 1 and term.lower() not in STOPWORDS
    ]


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
