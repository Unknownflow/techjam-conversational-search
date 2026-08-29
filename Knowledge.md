# Conversational Product Retrieval: Knowledge Document

## Objective

Build a lightweight, offline shopping agent that recommends the hidden purchased product quickly and accurately. The agent must work with a read-only, 50,000-product Amazon clothing catalog, manage a ten-turn maximum conversation, and avoid reliance on external services or model training.

The evaluation rewards three properties:

| Metric | Meaning | Weight |
| --- | --- | ---: |
| Hit Rate@10 | Whether the purchased product appears in the top ten | 50% |
| MRR | How high the purchased product ranks | 30% |
| Efficiency | How few turns are needed before the first hit | 20% |

## Implemented Approach

### 1. Intent routing and hybrid pipeline

`IntentRouter` assigns the initial request to one of two execution paths:

| Route | Trigger | Retrieval behavior |
| --- | --- | --- |
| Buying | Explicit requirements such as “key requirement”, “need”, or a budget | Preserves category and hard constraints, then prioritizes conjunctive FTS retrieval for precision. |
| Browsing | Exploratory wording such as “exploring”, “browse”, or “ideas” | Uses the same grounded lexical routes first, then activates an in-memory vector discovery route only when fewer than 80 lexical candidates are available. |

The pipeline is therefore **intent routing → multi-route candidate retrieval → constraint/semantic ranking**. Keyword, category, conjunction, fallback-OR, and optional dense candidates are deduplicated into one bounded working set before ranking.

The dense discovery route is a 192-dimensional, deterministic feature-hashed text embedding over product title, category, and attributes. It is NumPy-backed when NumPy is present and fails safely to the lexical pipeline when it is not. It is deliberately gated: experiments showed that always adding broad vector candidates slightly reduced MRR by introducing weakly related, popular products into otherwise precise result sets.

`Agent` also accepts an optional `semantic_reranker` callable. This is the integration point for a local or API-backed LLM reranker when credentials, an approved model, and a latency budget are available. Its normalized contribution is capped, so current-session hard constraints remain decisive. The submitted default uses no external model and therefore has zero reported token cost.

### 2. In-memory hybrid retrieval

The catalog is loaded into an in-memory SQLite FTS5 index. It indexes product title, category, features, details, store, price, and description. The index uses Porter stemming, so lexical retrieval tolerates ordinary inflection changes such as `wallet` versus `wallets`. This keeps execution local and avoids the operational overhead of a vector database.

Retrieval uses two complementary routes:

1. **Combined-constraint route**: an AND query across the category and the earliest disclosed high-value preferences. This is the precision route; for example, it can retrieve products matching `wallets`, `leather`, and `red` together.
2. **Per-constraint routes**: each stated preference is searched independently. These routes protect recall when a combined query is too restrictive or catalog wording differs from the user wording.

If a restrictive individual query has no results, it falls back to an OR query over the same distinctive terms. Candidates from all routes are merged and deduplicated before reranking.

### 3. Structured conversational state

Each session stores:

- typed slot records containing value, source, turn, and active/retired status;
- the current intent and intent-change history;
- the dialog phase (`discovery`, `refinement`, `clarification`, `intent_override`, or `recommendation`);
- the latest candidate count and over-generality decision;
- profile preference tags;
- questions already asked; and
- the most recent requested attribute.

The extraction logic preserves both category and hard requirement in a Buying request. This is important because a common requirement such as `leather` alone is too broad; `wallets + leather` is a substantially better retrieval signal.

Normal turns accumulate category, material, color, size, budget, use-case, style, and feature slots. The state machine deduplicates repeated values while preserving multiple independently confirmed hard constraints.

An explicit phrase such as “Actually, ignore my earlier preference. What I need is…” causes an intent transition to Buying. Prior soft-preference slots are retired, a previous override of the same slot type can be rewritten, and the category plus unrelated confirmed hard constraints remain active. This selective erasure avoids both stale-preference contamination and destructive loss of useful evidence.

When a user declines a requested attribute (for example, “I don't have a preference”), that attribute is made eligible again rather than being treated as useful information. This lets the agent continue the dialogue without wasting the remainder of the ten-turn budget. Responses include a compact `dialog_state` object for observability; the evaluator ignores this optional metadata.

### 4. High-information clarification policy

The agent first requests `other`, which permits the simulator to disclose the next most useful one or two constraints. This is followed by a feature/construction question if more information is needed. A no-preference response reopens the question so a subsequent clarification can still collect signal.

Over-generality is detected when a request has at most one active constraint and lexical retrieval reaches the configured candidate-pool threshold. The dialog immediately enters `clarification` and asks for one or two non-negotiable details, with examples covering material, style, use case, and budget. The optional dense/LLM stage is cut off for that turn because broad semantic expansion would add noise. A grounded lexical shortlist is still returned, preserving the opportunity for an early conversion while the clarification guides the next turn.

This policy is deliberately compact: the system retrieves after every customer message and asks only when additional information is likely to reduce ambiguity. That directly supports Mean Turns to Conversion.

### 5. Constraint-aware reranking

Candidates are reranked using:

- token coverage for each disclosed constraint;
- exact normalized phrase matches;
- a small recency weight for later information; and
- a minor overlap boost from profile preference tags.

The primary score remains based on stated requirements. This keeps the ranking interpretable and makes the system responsive to intent changes instead of using profile data as a replacement for current needs.

### 6. Catalog-quality tie-breaker

Many catalog products are indistinguishable from the available conversational constraints: several products can be leather, red, and in the same category. For these near-ties, the agent applies a deliberately modest quality prior derived only from catalog-visible fields:

- average product rating; and
- log-smoothed review count.

This is a commercial retrieval consideration rather than a label-derived shortcut. It is applicable to the private split because it uses no public-session identifiers, targets, or hidden intent cards. The prior is small enough to resolve similarly matched products rather than override a material user requirement.

## Evaluation Findings and Changes

The initial implementation achieved the following full public-set result:

```json
{
  "sample_count": 200,
  "hit_rate_at_10": 0.765,
  "mrr": 0.59299,
  "mttc": 4.395,
  "efficiency": 0.6605,
  "recommended_technical_score": 0.692497
}
```

Analysis of the scenario results highlighted several issues:

| Observation | Root cause | Change |
| --- | --- | --- |
| Buying performance lagged | The parser kept the hard constraint but discarded the category when both appeared in the initial message. | Retain category and hard constraint as separate slots. |
| Broad constraints produced weak top-ten ranking | Independent retrieval over common words such as `leather` created a large, noisy candidate set. | Add a conjunction-based precision route alongside recall routes. |
| Boundary sessions used unnecessary turns | A declined clarification was treated as a completed slot. | Reopen the attribute after a no-preference response. |
| Exact-match products remained tied | Metadata alone can leave many valid products with equal semantic match scores. | Add a small rating/review-count quality tie-breaker. |

After the initial changes, the full 200-session public evaluation produced:

```json
{
  "sample_count": 200,
  "hit_rate_at_10": 0.99,
  "mrr": 0.666379,
  "mttc": 1.735,
  "efficiency": 0.9265,
  "recommended_technical_score": 0.880214
}
```

The latest robustness pass added Porter stemming to FTS5. It improved the complete public evaluation without using any session labels:

```json
{
  "sample_count": 200,
  "hit_rate_at_10": 0.995,
  "mrr": 0.670552,
  "mttc": 1.705,
  "efficiency": 0.9295,
  "recommended_technical_score": 0.884566
}
```

### Improvement summary

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.765 | 0.995 | +0.230 |
| MRR | 0.592990 | 0.670552 | +0.077562 |
| MTTC | 4.395 | 1.705 | -2.690 turns |
| Efficiency | 0.6605 | 0.9295 | +0.2690 |
| Technical score | 0.692497 | 0.884566 | +0.192069 |

## Generalization and Private-Test Considerations

The changes were selected to improve behavior on the task class, rather than exploit public examples:

- No public sample IDs, target ASINs, or public labels are read by the agent.
- The system uses only information available at inference: catalog metadata, the user profile, and the current conversation.
- The precision route has recall fallbacks, which is safer if private messages include paraphrasing or incomplete attributes.
- Product quality is catalog-derived and smoothed, so it is a tie-breaker rather than a target memorization mechanism.
- The agent remains dependency-free, deterministic, and fully in-memory, making it appropriate for constrained offline evaluation.

## Verification

The repository evaluator and dialog-state tests were executed successfully:

```text
Ran 5 tests ... OK
```

The improved result above was produced by running the complete public evaluator over all 200 sessions, not a small sample.

## Limitations and Future Work

- The built-in dense route is a deterministic feature-hashed embedding, not a pretrained semantic model. The optional LLM reranker hook requires a separately configured model, credentials, latency budget, and token reporting.
- Constraint extraction is rule-based and benefits from the controlled input format. Negation, comparative preferences, nested conditions, and long free-form dialogue would benefit from a schema-constrained model parser.
- The current quality prior does not account for price compatibility unless price is explicitly disclosed. A structured budget scorer would improve that case.
- Override erasure is intentionally conservative: it retires tracked soft preferences and rewritten override slots, but does not infer that unrelated hard constraints should be discarded unless the user explicitly identifies them.
- The over-generality threshold is catalog-size dependent. A production system should calibrate it from retrieval latency, candidate entropy, and observed conversion behavior rather than a fixed count alone.
