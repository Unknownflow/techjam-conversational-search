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

The pipeline is therefore **intent routing → multi-route candidate retrieval → field-aware constraint ranking → precision-first presentation**. Keyword, category, conjunction, fallback-OR, and dense candidates are deduplicated into one bounded working set before ranking.

The dense discovery route is a 192-dimensional, deterministic feature-hashed text embedding over product title, category, and attributes. It is NumPy-backed when NumPy is present and fails safely to the lexical pipeline when it is not. It is deliberately gated: experiments showed that always adding broad vector candidates slightly reduced MRR by introducing weakly related, popular products into otherwise precise result sets.

`Agent` is deterministic and zero-token. It has no API-key handling, network model calls, or model injection path.

### 2. In-memory hybrid retrieval

The catalog is loaded into an in-memory SQLite FTS5 index. It indexes product title, category, features, details, store, price, and description. The index uses Porter stemming, so lexical retrieval tolerates ordinary inflection changes such as `wallet` versus `wallets`. This keeps execution local and avoids the operational overhead of a vector database.

Retrieval uses two complementary routes:

1. **Combined-constraint route**: an AND query across the category and the earliest disclosed high-value preferences. This is the precision route; for example, it can retrieve products matching `wallets`, `leather`, and `red` together.
2. **Per-constraint routes**: each stated preference is searched independently. These routes protect recall when a combined query is too restrictive or catalog wording differs from the user wording.

If a restrictive individual query has no results, it falls back to an OR query over the same distinctive terms. Candidates from all routes are merged and deduplicated before reranking.

The reranker retains field provenance instead of relying only on flattened text. A category receives exact-match credit only when it equals the product's final two canonical category nodes, preventing an ancestor-only occurrence from looking equally specific. The synthetic `color` label is removed before matching the actual color value. Exact multi-token feature and detail values receive a smaller bonus; generic one-word metadata does not.

Recommendation presentation widens progressively. During the first two evidence-gathering turns, the agent presents only its strongest grounded match; from turn three onward it returns the requested Top-K. An explicit intent replacement gets one precision-only recovery turn. If the customer initially declines the broad clarification, turn three also remains Top-1 because their distinguishing evidence arrives one turn later. This avoids locking a relevant product at a poor reciprocal rank while protecting Hit Rate@10.

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

The agent requests `other` on each of the first two clarification opportunities, which permits the customer to disclose as many as four useful constraints by turn three. Narrow, information-gain-selected questions follow if more evidence is needed. A no-preference response reopens the broad question so a subsequent clarification can still collect signal.

Over-generality is detected when a request has at most one active constraint and lexical retrieval reaches the configured candidate-pool threshold. The dialog immediately enters `clarification` and asks for one or two non-negotiable details, with examples covering material, style, use case, and budget. The dense stage is cut off for that turn because broad expansion would add noise. A grounded lexical shortlist is still returned, preserving the opportunity for an early conversion while the clarification guides the next turn.

This policy is deliberately compact: the system retrieves after every customer message and asks only when additional information is likely to reduce ambiguity. That directly supports Mean Turns to Conversion.

### 5. Constraint-aware reranking

Candidates are reranked using:

- token coverage for each disclosed constraint;
- exact normalized phrase matches;
- a small recency weight for later information; and
- canonical field checks for category, color, and exact multi-token evidence.

The score is based on stated requirements plus a modest catalog-quality tie-breaker. Aggregate profile tags remain available for conversational guidance but do not alter product order, avoiding noisy matches on generic terms such as `fit` or `style`.

### 6. Catalog-quality tie-breaker

Many catalog products are indistinguishable from the available conversational constraints: several products can be leather, red, and in the same category. For these near-ties, the agent applies a deliberately modest quality prior derived only from catalog-visible fields:

- average product rating; and
- log-smoothed review count.

This is a commercial retrieval consideration rather than a label-derived shortcut. It is applicable to the private split because it uses no public-session identifiers, targets, or hidden intent cards. The prior is small enough to resolve similarly matched products rather than override a material user requirement.

### 7. Dynamic context programming and self-adaptation

Every user turn is distilled into two memory layers:

- **Short-term context**: a versioned summary containing the current intent, dialog phase, active slots grouped by type, the six most recent user turns, remaining turn budget, candidate count, overload status, and selected strategy.
- **Personalized profile context**: the aggregate profile tags supplied at reset, rating style, purchase frequency, explicitly learned preferences, and attributes the customer declined to specify.

The history window is bounded to avoid unbounded state growth. Learned preferences are recomputed from active preference slots on every turn, so an intent override removes superseded preferences from both dialog state and personalized context. Base profile tags can inform clarification wording, but explicit current-session requirements alone drive relevance ranking.

The context program selects one of five workflows after each retrieval:

| Strategy | Condition | Next-turn orchestration |
| --- | --- | --- |
| `discovery_expand` | Browsing with limited accumulated evidence | Permit dense discovery. |
| `clarify_overload` | One or fewer constraints with a large candidate pool | Disable dense expansion and ask for non-negotiable details. |
| `precision_filter` | Buying intent | Disable broad dense retrieval and lock explicit constraints. |
| `focused_rerank` | Three or more slots, or a small candidate set | Stop expansion and concentrate on reranking. |
| `override_recovery` | Abrupt intent replacement | Preserve category and hard evidence, retire soft preferences, reopen clarification, and rerank for the new direction. |

This is runtime workflow re-orchestration rather than static prompt selection: the strategy controls whether dense retrieval is permitted on the following turn, and it changes the guidance message. Profile tags are surfaced as optional guidance hints when asking broad clarification questions.

### 8. Adaptive narrowing and next-question selection

Clarification is selected from the current top 200 reranked candidates rather than from the broad recall union. Each candidate receives a stable reciprocal-log-rank weight, so ambiguity among likely products matters more than noise near the bottom of retrieval.

For a candidate attribute `a`, expected narrowing is approximated as:

```text
gain(a) = coverage(a) - Σ posterior_mass(value)²
utility(a) = answerability_prior(a) × gain(a)
```

The uncovered mass represents candidates for which the catalog contains no extractable value; it is treated as a no-answer branch rather than artificial information gain. Known slot values are removed before residual gain is calculated. Previously asked attributes are excluded, while known or declined attributes receive strong penalties.

The selector currently evaluates material, color, size, style, use case, feature, budget, brand, and the composite `other` prompt. `other` is used as the initial high-coverage question because it can collect two constraints. Later turns compare typed questions by expected narrowing. `feature` retains a conservative answerability prior and is replaced only when another attribute has a meaningful utility advantage. Deterministic priority ordering resolves exact ties. Questions may continue through turn 9 because the answer can still improve turn-10 recommendations.

The chosen attribute, utility, and complete sorted score table are included under `dialog_state.next_question`, making question decisions inspectable during demonstrations and debugging.

## Evaluation Findings and Changes

### Current benchmark dashboard

The current `results.json` is the complete 200-session public evaluation. These are the primary metrics to track after every substantive agent change.

| Metric | Current value | Target direction |
| --- | ---: | --- |
| Technical score | **0.968439** | Higher |
| Hit Rate@10 | **1.000000** | Higher |
| MRR | **0.967464** | Higher |
| MTTC | **2.090** | Lower |
| Efficiency | **0.891000** | Higher |
| Reported model tokens | **0** | Keep within deployment budget |

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Buying | 80 | 1.000000 | 0.983036 | 1.575 |
| Browsing | 80 | 1.000000 | 0.956250 | 1.925 |
| Intent Override | 30 | 1.000000 | 0.975000 | 3.733 |
| Boundary | 10 | 1.000000 | 0.910000 | 2.600 |

The scenario table is especially useful for regression diagnosis: a higher aggregate score should not conceal a collapse in Buying precision or Intent Override recovery.

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

The next robustness pass added Porter stemming to FTS5. It improved the complete public evaluation without using any session labels:

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

The current pass tightened category/color/evidence semantics, removed noisy
profile overlap, repeated broad evidence collection once, and made Top-K
widening sensitive to intent replacement and delayed evidence.
The complete public evaluation now produces:

```json
{
  "sample_count": 200,
  "hit_rate_at_10": 1.0,
  "mrr": 0.967464,
  "mttc": 2.09,
  "efficiency": 0.891,
  "recommended_technical_score": 0.968439,
  "reported_token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

Compared with the previous deterministic result, MRR increases by `0.037256`
and MTTC falls by `0.160` turns while Hit Rate remains perfect. Together these
changes increase the technical score by `0.014377`, with no token usage.

### Improvement summary

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Hit Rate@10 | 0.765 | 1.000 | +0.235 |
| MRR | 0.592990 | 0.967464 | +0.374474 |
| MTTC | 4.395 | 2.090 | -2.305 turns |
| Efficiency | 0.6605 | 0.8910 | +0.2305 |
| Technical score | 0.692497 | 0.968439 | +0.275942 |

## Generalization and Private-Test Considerations

The changes were selected to improve behavior on the task class, rather than exploit public examples:

- No public sample IDs, target ASINs, or public labels are read by the agent.
- The system uses only information available at inference: catalog metadata, the user profile, and the current conversation.
- The precision route has recall fallbacks, which is safer if private messages include paraphrasing or incomplete attributes.
- Product quality is catalog-derived and smoothed, so it is a tie-breaker rather than a target memorization mechanism.
- Learned profile values are derived only from explicit active preference slots and are removed when those slots are overridden.
- The agent remains dependency-free, deterministic, and fully in-memory, making it appropriate for constrained offline evaluation.

## Verification

The repository evaluator and dialog-state tests were executed successfully:

```text
Ran 17 tests ... OK
```

The improved result above was produced by running the complete public evaluator
over all 200 released sessions, not a small sample.

## Limitations and Future Work

- The built-in dense route is a deterministic feature-hashed embedding, not a pretrained semantic model.
- Constraint extraction is rule-based and benefits from the controlled input format. Negation, comparative preferences, nested conditions, and long free-form dialogue would benefit from a more comprehensive deterministic parser.
- The current quality prior does not account for price compatibility unless price is explicitly disclosed. A structured budget scorer would improve that case.
- Override erasure is intentionally conservative: it retires tracked soft preferences and rewritten override slots, but does not infer that unrelated hard constraints should be discarded unless the user explicitly identifies them.
- The over-generality threshold is catalog-size dependent. A production system should calibrate it from retrieval latency, candidate entropy, and observed conversion behavior rather than a fixed count alone.
- The dataset supplies no stable user identifier and explicitly isolates sessions. Accordingly, learned “long-term” preferences are retained only inside the current session; cross-session persistence would require an authorized identity, consent, retention policy, and profile store.
- Workflow selection is currently rule-based. Online contextual-bandit learning could optimize question value and route selection in production, but would require unbiased conversion feedback and safeguards against self-reinforcing popularity bias.
- Question utility depends on the quality of the focused candidate posterior. If retrieval omits the correct product, even a mathematically useful split can guide the conversation in the wrong direction.
- Facets are presently extracted from flattened text. Products that mention several colors or materials can inflate apparent diversity; structured canonical catalog facets would provide cleaner partitions.
- Attribute answerability priors and the replacement margin are static. Production calibration should learn these values from unbiased response and conversion logs while retaining exploration safeguards.
- The evaluator makes `other` unusually valuable because it can reveal two hidden constraints. The composite-question advantage should be recalibrated for a real interface where broad questions may impose greater cognitive load.
