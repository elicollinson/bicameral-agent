# Evaluation datasets (Issue #56)

Plug-and-play external benchmarks for the research QA harness. Every dataset
is one small adapter in `src/bicameral_agent/eval_datasets/` that maps
upstream rows onto the existing `ResearchQATask` shape, declares a default
verification metric, and shares one fetch/cache/load lifecycle.

Supersedes `docs/hard_benchmark.md` (Issue #42), which now points here.

## What ships in the repo vs. what is fetched

**No external dataset is committed to this repository.** We map and point at
upstream sources; you download the data yourself. This keeps the repo free of
data whose redistribution terms are unclear (see Licensing below).

| In git (owned by us) | Fetched locally (git-ignored) |
|---|---|
| `src/bicameral_agent/eval_datasets/` — adapters + registry | `data/external/<name>.json` — the pulled subsets |
| `scripts/fetch_dataset.py` — fetch CLI | |
| synthetic test fixtures / mocked pagers in `tests/` | |

## Usage

```bash
# Fetch (writes data/external/<name>.json; prints license + citation)
python scripts/fetch_dataset.py --dataset frames [--limit N]

# Run the benchmark against it
uv run python scripts/run_baseline_benchmark.py --dataset frames [--metric llm_judge]
```

In code:

```python
from bicameral_agent.eval_datasets import build_dataset, resolve_metric

ds = build_dataset("supergpqa")
ds.build(50)              # fetch from upstream, write the cache
dataset = ds.load()       # -> ResearchQADataset
metric = resolve_metric(ds)          # dataset default, CLI-overridable
```

`--dataset builtin` (the default) is the packaged 130-task pool — exactly the
pre-factory behavior, nothing fetched or cached.

## Dataset matrix

| name | source | tier | default metric | license / gating |
|---|---|---|---|---|
| `builtin` | packaged `research_qa.json` | mixed | `llm_judge` | ours |
| `frames` | HF `google/frames-benchmark` | hard | `llm_judge` | Apache-2.0 |
| `crepe` | HF `tasksource/CREPE` | tricky | `llm_judge` | none declared — fetch-only, never redistribute |
| `hard_benchmark` | FRAMES + CREPE composite | hard+tricky | `llm_judge` | see both |
| `simpleqa_verified` | HF `google/simpleqa-verified` | typical | `exact_match` | MIT |
| `supergpqa` | HF `m-a-p/SuperGPQA` | hard | `multiple_choice` | ODC-BY (attribution) |
| `bbeh` | GitHub `google-deepmind/bbeh` | hard | `exact_match` | CC-BY-4.0 |
| `healthbench_hard` | `openai/simple-evals` JSONL blob | hard | `rubric_coverage` | MIT |
| `researchqa` | HF `realliyifei/ResearchQA` | hard | `rubric_coverage` | MIT |
| `abstentionbench` | HF `facebook/AbstentionBench` | hard | `abstention` | **CC-BY-NC-4.0 — non-commercial only; fetch-only** |
| `hle` | HF `cais/hle` | hard | `llm_judge` | MIT, **gated** (accept terms + `HF_TOKEN`) |

Tier notes: adapters that cannot populate per-row `known_assumptions` map to
`hard`, never `tricky` (the `tricky` invariant requires an annotated
assumption). CREPE's presupposition annotations populate `known_assumptions`
directly, so it is the `tricky` source.

## Verifier matrix

Metric names are registry keys for `bicameral_agent.verifiers.build_verifier`;
each dataset declares a `default_metric` overridable via `--metric` within its
`supported_metrics`. All verifiers return the normalized `TaskScore` shape
(`overall` in [0, 1] feeds `EpisodeOutcome.quality_score`); the mode-specific
report goes to `TaskScore.detail` and lands in
`episode.metadata["verification"]`.

| metric | backend | behavior |
|---|---|---|
| `llm_judge` | LLM (judge client) | rubric + gold-answer 1–5 grading (`TaskScorer`) |
| `lexical` | deterministic | token F1 / ROUGE-L vs gold (`LexicalScorer`) |
| `exact_match` | deterministic | normalized / numeric equality, with trailing "answer is X" extraction |
| `multiple_choice` | deterministic | letter extraction (stated answer, bare letter, unique choice-text match) vs gold letter |
| `rubric_coverage` | LLM (judge client) | which `rubric_items` are satisfied; earned points over positive points, negative points penalize (HealthBench-style) |
| `abstention` | LLM (judge client) | did the answer abstain, compared to `abstention_expected` |
| `llm_autorater` | LLM (judge client) | official SimpleQA 3-way grading (verbatim openai/simple-evals template): correct→1.0, incorrect→0.0, not_attempted→0.0 with the verdict kept in `detail` |

## Task-schema extensions

`ResearchQATask` gained three optional fields (all `None` for existing data):

- `choices: list[str] | None` — multiple-choice options (SuperGPQA); options
  are also embedded in the question text, since the answerer sees only
  `question`.
- `rubric_items: list[RubricItem] | None` — weighted criteria
  (`RubricItem(criterion, points)`); HealthBench points may be negative.
- `abstention_expected: bool | None` — AbstentionBench label.

`gold_answer=""` is valid **only** when `rubric_items` is set (ResearchQA has
no single gold answer); a validator enforces the pairing.

## Per-dataset notes and caveats

- **FRAMES** (`google/frames-benchmark`, Apache-2.0). Multi-hop factual QA.
  Cite: *Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented
  Generation*, arXiv:2409.12941. Mapping: `Prompt`→question, `Answer`→gold;
  rubric synthesized around the gold answer.
- **CREPE** (via `tasksource/CREPE` re-host, arXiv:2211.17257). False-premise
  questions; `presuppositions`→`known_assumptions`, `corrections`→gold. The
  original repo declares **no license** (NOASSERTION): fetch-only, do not
  redistribute. Cite: Yu et al., ACL 2023.
- **SimpleQA Verified** (`google/simpleqa-verified`, MIT). 1,000 verified
  short-answer facts; `problem`/`answer` columns, config `simpleqa_verified`,
  split `eval`. `exact_match` is the deterministic default; `llm_autorater`
  runs the official SimpleQA 3-way grading prompt and `llm_judge` gives
  graded credit.
- **SuperGPQA** (`m-a-p/SuperGPQA`, ODC-BY). `options`→`choices` (and embedded
  in the question), `answer_letter`→gold. Cite: arXiv:2502.14739.
- **BBEH** (`google-deepmind/bbeh`, CC-BY-4.0). Raw `task.json` per subtask on
  GitHub; sampled round-robin across the 23 subtasks for coverage. Cite:
  arXiv:2502.19187.
- **HealthBench Hard** (openai/simple-evals blob, MIT). JSONL records:
  conversation `prompt` (flattened into the question), weighted `rubrics`
  → `rubric_items`, published ideal completion → gold answer. Cite:
  arXiv:2505.08775.
- **ResearchQA** (`realliyifei/ResearchQA`, MIT). Rubric-native: `query` +
  unweighted `rubric` items (mapped at 1.0 point each), `gold_answer=""`.
  Cite: arXiv:2509.00496.
- **AbstentionBench** (`facebook/AbstentionBench`, **CC-BY-NC-4.0**).
  Non-commercial license — fetched locally for research use only, never
  redistributed. `should_abstain`→`abstention_expected`; rows without
  reference answers get an explicit "abstain" gold statement. **Caveat:**
  upstream currently ships a `datasets` loading script the HF datasets-server
  cannot build, so a live fetch fails with an actionable error until upstream
  publishes data files; the adapter targets the script's declared feature
  schema. Cite: arXiv:2506.09038.
- **HLE** (`cais/hle`, MIT, gated). Accept the dataset terms on Hugging Face
  and export `HF_TOKEN`; the pager sends it (to the datasets-server only) as
  a bearer token. Multi-modal rows are filtered out — text-only harness.
  Field mapping is best-effort from the public dataset card (the gate blocks
  offline schema verification). Cite: arXiv:2501.14249.

## Output: EvalReport

`scripts/run_baseline_benchmark.py` writes `summary.json` via
`bicameral_agent.eval_report.EvalReport`: dataset/metric identity, answerer +
measurement-model provenance, per-condition `MetricSummary` aggregates
(unchanged shape — the ui/ Review screen keeps working), plus per-task
`results` with each episode's score and verification detail.
