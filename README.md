# TechJam Conversational E-Commerce Search Challenge

Build a conversational shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

## Current Public Benchmark

The following values are taken from the current `results.json` run over all 200 public sessions. Re-run the evaluator after a change and update this table only when the complete run improves or validates the result.

| Metric | Current value | Why it matters |
| --- | ---: | --- |
| Technical score | **0.954062** | Weighted objective: Hit Rate@10, MRR, and efficiency. |
| Hit Rate@10 | **1.000000** | Target appears in the top ten recommendations. |
| MRR | **0.930208** | Target is placed near the top of the list. |
| MTTC | **2.250** | Fewer turns to first successful recommendation is better. |
| Efficiency | **0.875000** | Turn-efficiency component of the score. |
| Prompt / completion tokens | **0 / 0** | Default offline path uses no external model tokens. |

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
| --- | ---: | ---: | ---: | ---: |
| Buying | 80 | 1.000000 | 0.948854 | 1.750 |
| Browsing | 80 | 1.000000 | 0.907708 | 2.175 |
| Intent Override | 30 | 1.000000 | 0.945000 | 3.700 |
| Boundary | 10 | 1.000000 | 0.916667 | 2.500 |

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

The score increase comes from two deterministic changes. Ranking now rewards
an exact match against the final category path and the original feature/detail
field, instead of treating every occurrence in flattened catalog text as
equivalent. Presentation is precision-first: while the first three turns gather
distinguishing evidence, only the strongest grounded match is returned; from
turn four onward the list widens to the requested Top-K to protect recall. This
trades a modest MTTC increase for a substantially larger MRR gain under the
published weighting, and fixes the previous single Top-10 miss.

## Deterministic retrieval

The agent is deliberately offline. It does not inspect API credentials, make
network calls, or consume model tokens. The current field-aware deterministic
policy scores `0.954062` with zero tokens.


## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** zero for this fully local implementation.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
