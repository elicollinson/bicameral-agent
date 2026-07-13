# Search experiment, 2026-07 (#101)

Rerun of the hard-pool baseline benchmark with a **real search backend**
(Brave, #100) in place of the 20-snippet mock, testing the #101 hypothesis:
with genuine information asymmetry — the gap scanner can retrieve facts the
answerer does not have — the tool conditions (heuristic/random) should finally
separate from no_subconscious on quality, especially on the FRAMES hard tier.

Artifacts: `data/baseline_search/` (report, summary, per-condition parquets,
run log, `paired_analysis.json`). Comparison run: the committed no-search
baseline `data/baseline/` (#46) — same task pool, same pinned judge; the
search provider is the only intentionally changed variable. All numbers below
reproduce via:

    uv run python scripts/analyze_search_experiment.py

## 1. Run configuration

| Item | Value |
|---|---|
| Command | `scripts/run_baseline_benchmark.py --provider ollama --judge-provider ollama --dataset hard_benchmark --tasks-per-condition 50 --max-turns 6 --parallel-episodes 10 --search-provider brave` |
| Answerer | `ollama` / `gemma4:31b-cloud` |
| Judge + simulated user | pinned to `ollama` / `gemma4:31b-cloud` across all conditions (#53) |
| Dataset | `hard_benchmark` — FRAMES (hard, multi-hop) + CREPE (tricky, false-presupposition) (#42) |
| Tasks per condition | 50 planned (no_subconscious / random / heuristic) |
| Search backend | **Brave** (`BRAVE_API_KEY` set; run.log: `Gap scanner search backend: brave`) |
| Parallelism | 10 concurrent episodes |
| Wall clock | ~1.6 h (13:54–15:32 on 2026-07-13 per `run.log`) |

**Live Brave verified from episode content.** The heuristic arm produced 33
context injections; 6 contain literal source URLs (wmar2news.com,
encyclopedia.com, atlanticrecords.com, transfermarkt.com, …) with snippet
content specific to the FRAMES questions (e.g. Pope Paul II's 1466 support of
Scanderbeg; Tyson Fury's May 2024 first loss to Usyk). The mock provider is a
fixed set of 20 built-in research snippets with no URLs, so this content is
impossible under the mock — the run demonstrably hit the real web.

**Provenance caveat (#103, found during this analysis):** per-episode
`metadata.hyperparameters` records the *config-file defaults* (`provider:
gemini`, `search_provider: mock`, `parallel_episodes: 1`), not the resolved
CLI values. Do not filter episodes by that metadata. Provenance for this run
rests on `summary.json`'s answerer/measurement blocks, the run log, and the
URL-bearing injection content above.

## 2. Why the analysis is paired

Raw per-arm ns are skewed by transport-failure clustering (Ollama Cloud read
timeouts): no_subconscious lost 10 episodes vs heuristic's 3, and the failing
tasks are largely the same FRAMES tasks that failed in prior runs (the
no-search run's five no_subconscious failures are a subset of this run's ten).
Raw means are therefore survivor-biased — arms with more failures lose more
hard tasks. **All conclusions below come from the 37-task intersection
completed in all six arms** (3 conditions × 2 runs; 24 hard, 13 tricky),
pairing episodes by `task_id` so every mean and delta is computed over the
identical task set. Paired-delta 95% CIs use the repo's t-based
`compute_summary`.

Raw report-level numbers stay in `data/baseline_search/report.txt` for
reference (heuristic 0.599, no_subconscious 0.738, random 0.509 — the
no_subconscious raw mean is inflated by losing 10 mostly-hard tasks).

## 3. Paired results (n = 37 tasks)

Paired quality means on the intersection:

| Arm | no-search (`data/baseline`) | search (`data/baseline_search`) |
|---|---|---|
| no_subconscious | 0.709 | 0.770 |
| random | 0.676 | 0.588 |
| heuristic | 0.736 | 0.718 |

Paired deltas (mean [95% CI]):

| Contrast | Value |
|---|---|
| heuristic: search − no-search | **−0.018** [−0.158, +0.122] |
| random: search − no-search | −0.088 [−0.203, +0.028] |
| no_subconscious: search − no-search | +0.061 [−0.066, +0.187] |
| heuristic − no_subconscious, search run | **−0.052** [−0.163, +0.060] |
| heuristic − no_subconscious, no-search run | +0.027 [−0.091, +0.145] |

Tier splits of heuristic − no_subconscious:

| Tier | search run | no-search run |
|---|---|---|
| hard (n=24) | **−0.104** [−0.253, +0.045] | −0.007 [−0.177, +0.163] |
| tricky (n=13) | +0.045 [−0.130, +0.220] | +0.090 [−0.056, +0.235] |

Tool-activity and reliability stats (all episodes):

| Stat | no-search run | search run |
|---|---|---|
| heuristic injections (consumed) | 25 (25) | 33 (32) |
| heuristic URL-bearing injections | 0 | **6** |
| random injections (consumed) | 6 (4) | 2 (1) |
| transport failures (no_subc / random / heuristic) | 5 / 1 / 2 = 8 | 10 / 5 / 3 = **18** |

## 4. Verdict: hypothesis NOT supported

Real information asymmetry did **not** improve tool-condition quality. Every
directional prediction of #101 came out flat or reversed:

- The heuristic arm did not gain from real search (−0.018 paired vs its own
  no-search run; CI straddles zero).
- Within the search run, heuristic trails no_subconscious (−0.052), whereas
  in the no-search run it led (+0.027).
- The hard tier — where multi-hop retrieval was supposed to unlock the
  advantage — moved *against* the tools: heuristic − no_subconscious is
  −0.104 with search vs −0.007 without. The tricky tier stayed mildly
  positive either way (+0.045 vs +0.090).

**Power caveat:** none of these deltas is individually significant. At n = 37
pairs the delta CIs are roughly ±0.14 wide, so this run only *excludes*
paired effects larger than ~0.13–0.14 in magnitude. "No effect up to ~0.13"
is the defensible claim; it is still the opposite of the hypothesized
unlock, which predicted a visible positive separation.

### Candidate explanations (not established, listed for follow-up)

1. **SERP-snippet shallowness vs multi-hop needs.** Brave returns short
   result snippets; FRAMES hard tasks need chained facts. Injections carry a
   sentence or two per gap — full page content was never fetched, so the
   asymmetry delivered may be much thinner than the hypothesis assumed.
2. **Scanner re-ranking discounts relevant evidence.** The gap scanner's LLM
   relevance filter assigned 0.3 — the minimum score that survives the
   cutoff — to a directly relevant snippet (Encyclopedia.com on Pope Paul II
   backing Scanderbeg in 1466, exactly the fact the gap asked for,
   `frames_hard_010`). Real results may be systematically under-ranked
   relative to how the mock's canned snippets were scored.
3. **Search latency compounding transport flakiness.** The search run had 18
   transport failures vs 8 without search, concentrated in arms/tasks with
   long episodes. Added wall-clock (search-run episodes averaged ~213–233 s
   vs ~147–160 s) plausibly pushes more Ollama Cloud calls into timeout,
   degrading the very episodes search was meant to help.
4. **Answerer utilization unmeasured.** 32 of 33 heuristic injections were
   consumed (drained into context), but whether the answerer *used* the
   injected facts in its final answers was not instrumented. Consumption is
   not utilization.

## 5. Acceptance-criteria scorecard (#101)

| Criterion | Status | Notes |
|---|---|---|
| Run completes with per-condition n ≥ 45 | PARTIAL | random 45 ✓, heuristic 47 ✓; **no_subconscious 40 — MISSED** (10 clustered Ollama Cloud read timeouts, same FRAMES tasks that failed in prior runs). Documented rather than re-run; the paired analysis removes the resulting survivor bias from all conclusions. |
| Analysis doc comparing search vs no-search per tier | PASS | This doc + `scripts/analyze_search_experiment.py` + `data/baseline_search/paired_analysis.json`. Paired t-CIs on the 6-arm intersection supersede raw-n Welch tests (which are also in both `report.txt` files: all NO). |
| Honest verdict either way | PASS | §4: hypothesis not supported at the observable effect sizes; power bound stated. |

## 6. Downstream pointers

- #103 — fix per-episode hyperparameter provenance before any metadata-based
  corpus filtering.
- Follow-ups suggested by §4: fetch full page content behind top results;
  instrument answer-side utilization of injected facts; re-examine the
  relevance-filter calibration on real SERP snippets; #44-style timeout
  tuning before another N=10 cloud run.
