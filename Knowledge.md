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

### 1. In-memory hybrid retrieval

The catalog is loaded into an in-memory SQLite FTS5 index. It indexes product title, category, features, details, store, price, and description. This keeps execution local and avoids the operational overhead of a vector database.

Retrieval uses two complementary routes:

1. **Combined-constraint route**: an AND query across the category and the earliest disclosed high-value preferences. This is the precision route; for example, it can retrieve products matching `wallets`, `leather`, and `red` together.
2. **Per-constraint routes**: each stated preference is searched independently. These routes protect recall when a combined query is too restrictive or catalog wording differs from the user wording.

If a restrictive individual query has no results, it falls back to an OR query over the same distinctive terms. Candidates from all routes are merged and deduplicated before reranking.

### 2. Structured conversational state

Each session stores:

- disclosed category and preference constraints;
- profile preference tags;
- questions already asked; and
- the most recent requested attribute.

The extraction logic preserves both category and hard requirement in a Buying request. This is important because a common requirement such as `leather` alone is too broad; `wallets + leather` is a substantially better retrieval signal.

The state is additive for normal information accumulation. When a user declines a requested attribute (for example, “I don't have a preference”), that attribute is made eligible again rather than being treated as useful information. This lets the agent continue the dialogue without wasting the remainder of the ten-turn budget.

### 3. High-information clarification policy

The agent first requests `other`, which permits the simulator to disclose the next most useful one or two constraints. This is followed by a feature/construction question if more information is needed. A no-preference response reopens the question so a subsequent clarification can still collect signal.

This policy is deliberately compact: the system retrieves after every customer message and asks only when additional information is likely to reduce ambiguity. That directly supports Mean Turns to Conversion.

### 4. Constraint-aware reranking

Candidates are reranked using:

- token coverage for each disclosed constraint;
- exact normalized phrase matches;
- a small recency weight for later information; and
- a minor overlap boost from profile preference tags.

The primary score remains based on stated requirements. This keeps the ranking interpretable and makes the system responsive to intent changes instead of using profile data as a replacement for current needs.

### 5. Catalog-quality tie-breaker

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

After the changes, the full 200-session public evaluation produced:

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

### Improvement summary

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.765 | 0.990 | +0.225 |
| MRR | 0.592990 | 0.666379 | +0.073389 |
| MTTC | 4.395 | 1.735 | -2.660 turns |
| Efficiency | 0.6605 | 0.9265 | +0.2660 |
| Technical score | 0.692497 | 0.880214 | +0.187717 |

## Generalization and Private-Test Considerations

The changes were selected to improve behavior on the task class, rather than exploit public examples:

- No public sample IDs, target ASINs, or public labels are read by the agent.
- The system uses only information available at inference: catalog metadata, the user profile, and the current conversation.
- The precision route has recall fallbacks, which is safer if private messages include paraphrasing or incomplete attributes.
- Product quality is catalog-derived and smoothed, so it is a tie-breaker rather than a target memorization mechanism.
- The agent remains dependency-free, deterministic, and fully in-memory, making it appropriate for constrained offline evaluation.

## Verification

The repository evaluator tests were executed successfully:

```text
Ran 3 tests ... OK
```

The improved result above was produced by running the complete public evaluator over all 200 sessions, not a small sample.

## Limitations and Future Work

- Lexical FTS does not understand true semantic equivalence. If allowed by runtime constraints, a compact local embedding model could add a third dense-retrieval route.
- Constraint extraction is intentionally simple and benefits from the controlled input format. More natural production dialogue would benefit from a robust slot parser.
- The current quality prior does not account for price compatibility unless price is explicitly disclosed. A structured budget scorer would improve that case.
- Intent overrides are represented by fresh user constraints; a production version should additionally attach provenance to each slot and explicitly retire superseded values.
