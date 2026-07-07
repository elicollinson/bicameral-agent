# Harder benchmark integration (Issue #42)

Replaces the saturated `research_qa` evaluation pool (which a frontier answerer
one-shots, pinning the judge at ceiling — see the RCA in #41) with two harder,
externally-sourced task pools that map onto the existing `ResearchQATask` shape.

## What ships in the repo vs. what is fetched

**No external dataset is committed to this repository.** We map and point at
upstream sources; you download the data yourself. This keeps the repo free of
data whose redistribution terms are unclear (see Licensing below).

| In git (owned by us) | Fetched locally (git-ignored) |
|---|---|
| `src/bicameral_agent/hard_benchmark.py` — mappers + fetch/load | `data/external/hard_benchmark.json` — the pulled subset |
| `scripts/fetch_hard_benchmark.py` — fetch CLI | |
| `tests/fixtures/hard_benchmark_sample.json` — synthetic, author-owned | |

## How to fetch

```bash
python scripts/fetch_hard_benchmark.py            # defaults: 100 FRAMES + 60 CREPE
python scripts/fetch_hard_benchmark.py --frames 200 --crepe 100
```

Writes `data/external/hard_benchmark.json` (git-ignored). Stdlib-only — no extra
dependency, no API key. Load it in code:

```python
from bicameral_agent.hard_benchmark import load_hard_benchmark
dataset = load_hard_benchmark()            # -> ResearchQADataset
```

`load_hard_benchmark()` raises an actionable `FileNotFoundError` if you haven't
fetched yet. The returned `ResearchQADataset` is a drop-in for the existing
interface and works with `select_tasks` in `scripts/run_baseline_benchmark.py`.

## Datasets and mapping

### Hard tier — FRAMES (`google/frames-benchmark`)
Multi-hop factual QA requiring synthesis across multiple Wikipedia articles.
824 questions; we pull a subset. Frontier headroom: ~0.40 accuracy with no
retrieval, ~0.66 with a multi-step retrieval pipeline (arXiv:2409.12941).

| FRAMES field | `ResearchQATask` field |
|---|---|
| `Prompt` | `question` |
| `Answer` | `gold_answer` |
| *(synthesized, anchored on the gold answer)* | `scoring_rubric` |
| — | `difficulty = hard`, `split = eval` |

FRAMES ships no rubric, so we synthesize a 1–5 rubric anchored on the gold
answer and the multi-hop reasoning the question demands.

### Tricky tier — CREPE (false-presupposition QA, arXiv:2211.17257)
Open-domain questions built on a **false premise**, with the presupposition and
its correction annotated. Pulled via the `tasksource/CREPE` re-host; we keep
only rows labelled `false presupposition` that carry both a presupposition and a
correction. Frontier models still fail false-premise correction badly
(<30% on related false-presupposition probes).

| CREPE field | `ResearchQATask` field |
|---|---|
| `question` | `question` |
| `corrections` (joined) | `gold_answer` |
| `presuppositions` | `known_assumptions` |
| *(synthesized)* | `scoring_rubric` |
| — | `difficulty = tricky`, `split = eval` |

This is the cleanest possible mapping for the `tricky` tier: the dataset's own
presupposition annotation populates `known_assumptions` directly, satisfying the
`ResearchQATask` invariant that tricky tasks carry an assumption.

## Licensing / attribution

- **FRAMES** — Apache-2.0 (`google/frames-benchmark`). Cite: *"Fact, Fetch, and
  Reason: A Unified Evaluation of Retrieval-Augmented Generation"*, arXiv:2409.12941.
  Apache-2.0 would permit redistribution, but we fetch-at-build for uniformity.
- **CREPE** — the original false-presupposition CREPE
  (`github.com/velocityCavalry/CREPE`, arXiv:2211.17257) declares **no license**
  (GitHub: NOASSERTION). We therefore **do not redistribute it**; the fetch
  script pulls it locally for your own research use under the upstream terms.
  Cite: *"CREPE: Open-Domain Question Answering with False Presuppositions"*,
  Yu et al., ACL 2023.

## Fallback datasets — and the triggers to watch for

The pairing above is the primary choice. Keep these alternates in mind; each row
is a concrete condition under which to switch.

| Trigger to watch for | Fall back to |
|---|---|
| **Headroom collapses** — the 2026 answerer scores near-ceiling on FRAMES (the reported 0.40/0.66 figures are 2024-era Gemini-1.5-Pro; model progress may have eaten the gap). This is the #1 risk and must be checked by the #46 pilot. | **ResearchQA** (arXiv:2509.00496) or **ResearchRubrics** (arXiv:2511.07685): rubric-native long-form research QA, no system >70% rubric coverage. Better schema fit (per-question rubrics) but license/HF-loadability unconfirmed and scores were measured on agentic deep-research systems, not a one-shot answerer. |
| **CREPE proves too easy or too noisy**, or its no-license status becomes a problem even for local use. | **FalseQA** (arXiv:2307.02394, `thunlp/FalseQA`): 2,365 human-written false-premise questions. Also has no declared license, so it is fetch-at-build only too — it was the original pick, demoted because CREPE ships presupposition+correction annotations that map more cleanly. |
| **Single `scoring_rubric: str` can't capture rubric granularity** — ResearchQA/ResearchRubrics carry ~26 rubric items/task; collapsing to one string may lose the variance that makes them discriminating. | Extend the judge to multi-criterion scoring (overlaps with #45's discriminating-scorer work) before adopting the rubric-native sets. |

## Known limitation — the pilot (AC #3) is deferred to #46

Issue #42 asks for a pilot showing quality scores below ceiling with real
variance. That can't be validly run from the dataset change alone:

1. The judge currently saturates at ~5/5/5 **regardless of dataset difficulty**
   (the `max_output_tokens=100` cap + leniency); fixing that is #45's job. A
   pilot run now would show ceiling because of the *scorer*, not the data —
   misleading.
2. No Gemini API key is configured in the integration environment.

So below-ceiling-with-variance is validated in **#46** (the re-run), after #45's
discriminating-scorer fix lands. The right pilot is answerer-correctness against
gold (independent of the judge): if the answerer gets a meaningful fraction of
FRAMES/CREPE wrong, the headroom is real.
