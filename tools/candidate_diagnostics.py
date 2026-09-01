"""Explain public-set retrieval, ranking, and dialog failures.

This script is a development diagnostic. It intentionally uses released public
labels through the evaluator helpers, but it does not change the production
agent or the official evaluator contract.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    Agent,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)


def _histogram(values: list[int | None], miss_label: str = "miss") -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        counts[str(value) if value is not None else miss_label] += 1
    return dict(sorted(counts.items(), key=lambda pair: (pair[0] == miss_label, pair[0])))


def _metric_summary(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    if not sessions:
        return {
            "sample_count": 0,
            "hit_rate_at_10": 0.0,
            "mrr": 0.0,
            "mttc": None,
            "efficiency": 0.0,
            "technical_score": 0.0,
        }
    hit_rate = sum(1 for session in sessions if session["hit"]) / len(sessions)
    reciprocal_ranks = [
        0.0 if session["best_rank"] is None else 1.0 / session["best_rank"]
        for session in sessions
    ]
    turns = [
        session["first_hit_turn"]
        if session["first_hit_turn"] is not None
        else MAX_TURNS + 1
        for session in sessions
    ]
    mrr = statistics.fmean(reciprocal_ranks)
    mttc = statistics.fmean(turns)
    efficiency = max(0.0, min(1.0, (11.0 - float(mttc)) / 10.0))
    technical_score = 0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency
    return {
        "sample_count": len(sessions),
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(technical_score, 6),
    }


def _compact_session(session: dict[str, Any], metric_name: str, metric_value: Any) -> dict[str, Any]:
    turns = session.get("turns", [])
    last_turn = turns[-1] if turns else {}
    return {
        "sample_id": session["sample_id"],
        "scenario_type": session["scenario_type"],
        "metric": metric_name,
        "metric_value": metric_value,
        "failure_mode": session["failure_mode"],
        "first_hit_turn": session["first_hit_turn"],
        "best_rank": session["best_rank"],
        "best_candidate_rank": session["best_candidate_rank"],
        "target_ever_retrieved": session["target_ever_retrieved"],
        "last_ask_attribute": last_turn.get("ask_attribute"),
        "last_candidate_count": last_turn.get("candidate_count"),
        "last_strategy": last_turn.get("strategy"),
        "active_constraints": last_turn.get("active_constraints", []),
    }


def _miss_sort_value(value: int | None) -> int:
    return value if value is not None else MAX_TURNS + 1


def _candidate_sort_value(value: int | None) -> int:
    return value if value is not None else 1_000_000


def _improvement_focus(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Return compact, actionable diagnostics for the main improvement areas."""

    retrieval_problem_sessions = [
        session
        for session in sessions
        if not session["target_ever_retrieved"]
        or (
            session["best_candidate_rank"] is not None
            and session["best_candidate_rank"] > TOP_K
        )
    ]
    retrieval_ranked = sorted(
        retrieval_problem_sessions,
        key=lambda session: (
            not session["target_ever_retrieved"],
            _candidate_sort_value(session["best_candidate_rank"]),
        ),
        reverse=True,
    )
    ranking_problem_sessions = [
        session
        for session in sessions
        if session["best_rank"] is None
        or (
            session["best_rank"] is not None
            and session["best_rank"] > 1
        )
    ]
    ranking_ranked = sorted(
        ranking_problem_sessions,
        key=lambda session: (
            session["best_rank"] is None,
            _miss_sort_value(session["best_rank"]),
        ),
        reverse=True,
    )
    dialog_problem_sessions = [
        session
        for session in sessions
        if session["first_hit_turn"] is None
        or (
            session["first_hit_turn"] is not None
            and session["first_hit_turn"] > 2
        )
    ]
    dialog_ranked = sorted(
        dialog_problem_sessions,
        key=lambda session: _miss_sort_value(session["first_hit_turn"]),
        reverse=True,
    )
    override_sessions = [
        session
        for session in sessions
        if session["scenario_type"] == "intent_override"
        and (
            session["first_hit_turn"] is None
            or (
                session["first_hit_turn"] is not None
                and session["first_hit_turn"] > 3
            )
        )
    ]
    override_ranked = sorted(
        override_sessions,
        key=lambda session: (
            session["best_rank"] is None,
            _miss_sort_value(session["first_hit_turn"]),
            _miss_sort_value(session["best_rank"]),
        ),
        reverse=True,
    )

    return {
        "retrieval_routes": {
            "metric": "best_candidate_rank",
            "why": "High values mean the target entered the working set too weakly; not_retrieved means route recall failed.",
            "underperformance_rule": "target not retrieved, or best_candidate_rank > 10",
            "priority_sessions_to_inspect_first": [
                _compact_session(
                    session,
                    "best_candidate_rank",
                    session["best_candidate_rank"] if session["best_candidate_rank"] is not None else "not_retrieved",
                )
                for session in retrieval_ranked[:3]
            ],
        },
        "ranking_weights": {
            "metric": "best_rank",
            "why": "High values mean the target was returned low in Top-K; misses mean ranking/truncation did not surface it.",
            "underperformance_rule": "miss, or returned best_rank > 1",
            "priority_sessions_to_inspect_first": [
                _compact_session(
                    session,
                    "best_rank",
                    session["best_rank"] if session["best_rank"] is not None else "miss",
                )
                for session in ranking_ranked[:3]
            ],
        },
        "dialog_clarification_policy": {
            "metric": "first_hit_turn",
            "why": "High values mean the agent needed too many turns before collecting enough useful evidence.",
            "underperformance_rule": "miss, or first_hit_turn > 2",
            "priority_sessions_to_inspect_first": [
                _compact_session(
                    session,
                    "first_hit_turn",
                    session["first_hit_turn"] if session["first_hit_turn"] is not None else "miss",
                )
                for session in dialog_ranked[:3]
            ],
        },
        "intent_override_handling": {
            "metric": "first_hit_turn_on_intent_override",
            "why": "High values identify sessions where replacement intent still took too long to recover.",
            "underperformance_rule": "intent_override miss, or first_hit_turn > 3",
            "priority_sessions_to_inspect_first": [
                _compact_session(
                    session,
                    "first_hit_turn",
                    session["first_hit_turn"] if session["first_hit_turn"] is not None else "miss",
                )
                for session in override_ranked[:3]
            ],
        },
    }


def _rank_all_candidates(agent: Agent, state: dict[str, Any]) -> tuple[list[str], int]:
    """Return the full current candidate ranking and candidate-pool size."""

    candidates = agent._candidate_rows(
        state["constraints"],
        state["intent"] or "browsing",
        bool(state["workflow"].get("allow_dense", True)),
    )
    if not candidates:
        return [], 0
    ranked = agent._rank(state, max(len(candidates), TOP_K))
    return [str(item["parent_asin"]) for item in ranked], len(candidates)


def diagnose_sessions(
    *,
    catalog: str | Path,
    dataset: str | Path,
    sample_limit: int | None = None,
    late_turn_threshold: int = 6,
    include_session_traces: bool = False,
) -> dict[str, Any]:
    """Run the simulator and classify each session's retrieval/ranking outcome."""

    samples = load_jsonl(dataset)
    if sample_limit is not None:
        samples = samples[:sample_limit]
    catalog_ids, categories, products = catalog_index(catalog)
    agent = Agent(catalog)
    sessions: list[dict[str, Any]] = []

    for sample in samples:
        session_id = "diagnostic_" + str(sample["sample_id"])
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        intent_card, behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": intent_card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective_sample,
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        turns: list[dict[str, Any]] = []
        first_hit_turn: int | None = None
        best_rank: int | None = None
        target_ever_retrieved = False
        best_candidate_rank: int | None = None

        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception as exc:
                response = {
                    "message": "",
                    "ask_attribute": None,
                    "recommendations": [],
                    "diagnostic_error": str(exc),
                }
            if not isinstance(response, dict):
                response = {"message": "", "ask_attribute": None, "recommendations": []}

            ranked_top = normalize_recommendations(
                response.get("recommendations"),
                catalog_ids,
            )
            returned_rank = ranked_top.index(target) + 1 if target in ranked_top else None
            state = agent.sessions[session_id]
            full_ranked, candidate_count = _rank_all_candidates(agent, state)
            candidate_rank = (
                full_ranked.index(target) + 1 if target in full_ranked else None
            )
            if candidate_rank is not None:
                target_ever_retrieved = True
                best_candidate_rank = (
                    candidate_rank
                    if best_candidate_rank is None
                    else min(best_candidate_rank, candidate_rank)
                )

            turns.append(
                {
                    "turn": turn,
                    "user_message": user_message,
                    "ask_attribute": response.get("ask_attribute"),
                    "returned_count": len(ranked_top),
                    "returned_rank": returned_rank,
                    "candidate_count": candidate_count,
                    "candidate_rank": candidate_rank,
                    "intent": state.get("intent"),
                    "phase": state.get("phase"),
                    "strategy": state.get("workflow", {}).get("strategy"),
                    "over_general": state.get("over_general"),
                    "active_constraints": list(state.get("constraints", [])),
                    "question_decision": state.get("question_decision", {}),
                }
            )

            if override_applied and returned_rank is not None:
                first_hit_turn = turn
                best_rank = returned_rank
                break
            if turn == MAX_TURNS:
                break

            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(
                    override.get(
                        "message",
                        "Actually, please ignore my earlier preference.",
                    )
                )
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )

        hit = first_hit_turn is not None
        if hit and first_hit_turn >= late_turn_threshold:
            failure_mode = "late_conversion"
        elif hit:
            failure_mode = "hit"
        elif not target_ever_retrieved:
            failure_mode = "retrieval_miss"
        elif best_candidate_rank is not None and best_candidate_rank > TOP_K:
            failure_mode = "ranking_miss"
        else:
            failure_mode = "dialog_or_contract_miss"

        sessions.append(
            {
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "target_parent_asin": target,
                "hit": hit,
                "first_hit_turn": first_hit_turn,
                "best_rank": best_rank,
                "failure_mode": failure_mode,
                "target_ever_retrieved": target_ever_retrieved,
                "best_candidate_rank": best_candidate_rank,
                "turns": turns,
            }
        )

    failure_counts = Counter(session["failure_mode"] for session in sessions)
    scenario_failure_counts: dict[str, dict[str, int]] = defaultdict(dict)
    scenario_metrics: dict[str, dict[str, Any]] = {}
    for scenario in sorted({session["scenario_type"] for session in sessions}):
        scenario_sessions = [
            session for session in sessions if session["scenario_type"] == scenario
        ]
        counts = Counter(
            session["failure_mode"]
            for session in scenario_sessions
        )
        scenario_failure_counts[scenario] = dict(sorted(counts.items()))
        scenario_metrics[scenario] = _metric_summary(scenario_sessions)

    rank_values = [session["best_rank"] for session in sessions]
    turn_values = [session["first_hit_turn"] for session in sessions]
    candidate_rank_values = [session["best_candidate_rank"] for session in sessions]
    optimization_targets = {
        "late_conversions": [
            session["sample_id"]
            for session in sessions
            if session["failure_mode"] == "late_conversion"
        ],
        "low_mrr_hits": [
            session["sample_id"]
            for session in sessions
            if session["hit"] and session["best_rank"] is not None and session["best_rank"] >= 4
        ],
        "ranking_misses": [
            session["sample_id"]
            for session in sessions
            if session["failure_mode"] == "ranking_miss"
        ],
        "retrieval_misses": [
            session["sample_id"]
            for session in sessions
            if session["failure_mode"] == "retrieval_miss"
        ],
    }

    result = {
        "sample_count": len(sessions),
        "late_turn_threshold": late_turn_threshold,
        "aggregate_metrics": _metric_summary(sessions),
        "failure_summary": dict(sorted(failure_counts.items())),
        "scenario_failure_summary": dict(sorted(scenario_failure_counts.items())),
        "scenario_metrics": dict(sorted(scenario_metrics.items())),
        "rank_distribution": _histogram(rank_values),
        "turn_distribution": _histogram(turn_values),
        "candidate_rank_distribution": _histogram(candidate_rank_values, "not_retrieved"),
        "optimization_targets": optimization_targets,
        "improvement_focus": _improvement_focus(sessions),
    }
    if include_session_traces:
        result["sessions"] = sessions
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate diagnostics for public sessions")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results/candidate_diagnostics.json")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--late-turn-threshold", type=int, default=6)
    parser.add_argument(
        "--include-session-traces",
        action="store_true",
        help="Include every per-session turn trace in the output JSON.",
    )
    args = parser.parse_args()

    diagnostics = diagnose_sessions(
        catalog=args.catalog,
        dataset=args.dataset,
        sample_limit=args.sample_limit,
        late_turn_threshold=args.late_turn_threshold,
        include_session_traces=args.include_session_traces,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in diagnostics.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
