# TechJam deterministic conversational search submission

This directory is the complete participant submission. Its official entry point is
`agent:Agent`. The catalog is organizer-provided and is deliberately not included in
the bundle.

## Reproducible environment

The reported result was produced with:

- CPython 3.12.13 (use the Python 3.12 series; 3.12.13 is the frozen reference);
- NumPy 2.3.5, pinned in `requirements.txt`;
- SQLite 3.53.1 with FTS5, supplied by the reference CPython build.

From the repository root on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r submission\requirements.txt
```

On POSIX systems, replace the last two interpreter paths with
`.venv/bin/python`. No environment variables, API keys, credentials, privileged
host access, or external services are required.

## Official interface and run command

The organizer should import the extracted bundle as follows and pass the path of
its frozen catalog explicitly:

```python
from agent import Agent

agent = Agent("/absolute/path/to/catalog.jsonl")
```

The constructor accepts a string or `pathlib.Path` to the organizer-provided JSONL
catalog. It does not download or discover a catalog and therefore does not depend
on the process working directory when an explicit path is supplied.

Using the released official-equivalent local harness, the single evaluation
command from the repository root is:

```powershell
.venv\Scripts\python -m submission.run_local --catalog data/catalog.jsonl --dataset data/public_set.jsonl --output results.json
```

The harness calls `reset(session_id, user_profile)` once per session and
`respond(session_id, user_message, turn, top_k)` once per turn. Responses contain
only the contract fields `message`, `ask_attribute`, `recommendations`, and
`usage`; recommendations are valid unique catalog identifiers ordered best to
worst.

## Method and model choice

This is a deterministic, CPU-only retrieval and dialog policy. It uses an
in-memory SQLite FTS5 index with Porter stemming, per-constraint recall queries,
canonical category-tail matching, exact structured-field evidence, and a modest
quality prior. A deterministic NumPy feature-hashed representation supplements
lexical retrieval for exploratory queries. It is not a trained model and does not
call an LLM.

The dialog state machine distinguishes buying, browsing, intent replacement, and
clarification states. It asks a broad evidence question for the first two turns,
uses Top-1 while evidence is incomplete, and then widens to the requested Top-K.
This policy improves reciprocal rank and time to conversion without model tokens.

On the released 200-session public set, the frozen implementation produces:

| Metric | Value |
| --- | ---: |
| Technical score | 0.968439 |
| Hit Rate@10 | 1.000000 |
| MRR | 0.967464 |
| MTTC | 2.090 |
| Efficiency | 0.891000 |

These public-set figures are development measurements, not a guarantee of private
evaluation performance.

## Network, token, cost, and performance disclosure

Official scoring requires no network access. The source imports no network client,
does not inspect credentials, and behaves the same when outbound access is
disabled. The complete official path is therefore its own offline implementation.
If NumPy is unavailable, a lexical-only fallback remains operational, but that is
not the frozen environment used for the score above; install the pinned dependency
for reproducibility.

| Disclosure | Measured value |
| --- | ---: |
| Prompt tokens per run | 0 |
| Completion tokens per run | 0 |
| Estimated model/API cost | $0.00 |
| Cold catalog initialization | 72.060 s |
| Mean turn latency | 490.056 ms |
| p50 turn latency | 466.975 ms |
| p95 turn latency | 660.960 ms |
| Maximum measured turn latency | 710.549 ms |
| Approximate process working set | 453 MiB |

Latency was measured over 50 independent first-turn requests after one cold index
build, using the complete 50,000-product catalog. The reference machine ran Windows
11, CPython 3.12.13, and an Intel64 Family 6 Model 186 CPU with 20 logical
processors. Working-set memory was observed during a full local evaluator run and
is approximate rather than a strict instrumented peak. Re-run the benchmark on the
target machine with:

```powershell
.venv\Scripts\python -m submission.benchmark --catalog data/catalog.jsonl --samples 50
```

## Limitations

- Building the in-memory lexical and dense indexes has material cold-start and
  memory cost; the agent is intended to be constructed once and reused.
- The exact frozen score requires the pinned NumPy path and a Python SQLite build
  with FTS5 enabled.
- Free-form facet extraction is rule-based, so unusual paraphrases, misspellings,
  multilingual requests, and attributes outside the catalog schema can be missed.
- Ranking heuristics are calibrated to the supplied catalog fields and may need
  retuning for a materially different catalog distribution.
- No live semantic model is available as a fallback, by design.

## Bundle validation

`MANIFEST.json` is an explicit allowlist. It excludes catalogs, evaluation data,
results, evaluator or organizer files, environment files, credentials, and Git
history. From the repository root, validate and build the upload archive with:

```powershell
.venv\Scripts\python -m submission.validate
.venv\Scripts\python -m submission.validate --build submission/dist/techjam-submission.zip
```

The validator also verifies the frozen implementation hash, required method
signatures, absence of network-capable imports, and absence of likely embedded
secrets. The resulting archive can be extracted directly and imported as
`agent:Agent`.
