"""Candidate document representation used by the retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateDocument:
    text: str
    category_tail: str
    evidence: frozenset[str]
