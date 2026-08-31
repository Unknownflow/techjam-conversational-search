"""Candidate document model used while ranking catalog matches."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateDocument:
    text: str
    category_tail: str
    evidence: frozenset[str]
