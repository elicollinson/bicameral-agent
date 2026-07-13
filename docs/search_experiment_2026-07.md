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

### Candidate explanations (ranked after the §5 utilization analysis)

1. **SERP-snippet shallowness vs multi-hop needs** — *strengthened by §5*.
   Brave returns short result snippets; FRAMES hard tasks need chained
   facts, and quality is conjunctive over the hops. The spot-checks show
   injections fixing exactly one hop while the un-searched hops stay
   hallucinated (`frames_hard_026`: the injected Usyk fact was used and the
   Fury half became correct, but the Mike Tyson half stayed wrong → 0.0).
   Full page content behind the snippets was never fetched.
2. **Evidence-induced retraction** — *new, observed in §5*. Snippets that
   fail to *confirm* a claim (rather than refute it) can push the answerer
   into retracting a correct answer: in `frames_hard_024` the injected
   findings supported Gandhi/Gram Swaraj generally but not the specific
   Chikhali link, and the answerer withdrew its (scoring) Gandhi/INC answer
   entirely → 0.0. On FRAMES tasks built from obscure chained facts, shallow
   confirmation-seeking snippets will often under-confirm.
3. **Search latency compounding transport flakiness.** The search run had 18
   transport failures vs 8 without search, concentrated in arms/tasks with
   long episodes. Added wall-clock (search-run episodes averaged ~213–233 s
   vs ~147–160 s) plausibly pushes more Ollama Cloud calls into timeout,
   degrading the very episodes search was meant to help.
4. **Scanner re-ranking discounts relevant evidence** — *weakened by §5*.
   The relevance filter did assign 0.3 — the minimum surviving score — to a
   directly relevant snippet (Encyclopedia.com on Pope Paul II backing
   Scanderbeg in 1466, `frames_hard_010`), but that snippet was still
   injected, engaged with, and the episode scored 1.0. The residual concern
   is only results silently dropped *below* the 0.3 cutoff, which the
   episode data cannot show.

A fifth candidate from the first draft of this analysis — *answerer
utilization unmeasured* — is now measured (§5) and eliminated as an
explanation: uptake of injected content is high, so the null is not
explained by the answerer ignoring the evidence.

## 5. Utilization: consumed ≠ cited, but the evidence is engaged

The #101 run plan asked whether injected search content is actually *cited*
in final answers. `utilization_stats` in the analysis script measures this
lexically per heuristic-arm episode with a consumed injection: an episode
counts as utilizing when an injected URL is quoted verbatim in a
post-injection assistant message, or ≥ 2 *distinctive novel* tokens from the
injection reappear there (distinctive = not scanner boilerplate by document
frequency across the arm's injections; novel = absent from the transcript
before the injection was consumed, since the scanner paraphrases the
assistant's own claims). This is a lenient lexical proxy — an upper bound on
genuine citation; the verbatim-URL count is the strict lower bound.

| Stat | no-search run | search run |
|---|---|---|
| episodes with consumed injections | 25 | 30 |
| … showing lexical utilization | 16 (64%) | 24 (80%) |
| verbatim URL citations | **0** | **0** |
| quality, utilized episodes | 0.505 (n=16) | 0.545 (n=24) |
| quality, non-utilized episodes | 0.667 (n=9) | 0.431 (n=6) |

Two results matter. First, **the answerer does engage with injected
evidence** — 80% lexical uptake in the search run, frequently explicit
("as your provided research indicates", "the research gap scanner
confirms") — so the null is not a delivery failure. Second, **engagement
does not buy quality**: the utilized-vs-not splits are small-n, inconsistent
in sign across the two runs, and confounded (the scanner injects most on
episodes already in trouble), and no answer ever cites an injected URL as a
source.

### Transcript spot-checks (search run, heuristic arm)

- `frames_hard_010` (utilized, 1.0): the relevance-0.3 Scanderbeg snippet is
  engaged directly — the answerer corrects its invented "Papal States vs
  Venice war" justification to "intensifying the struggle … supporting
  Scanderbeg, not ending the war" while keeping the correct Bayeux Tapestry
  answer. Evidence engaged, answer right, judge rewards it.
- `frames_hard_026` (utilized, 0.0): the injected Usyk/May-2024 fact (the
  run's highest-relevance snippet, 0.8) is adopted and Fury's half of the
  calculation becomes correct, but the un-searched Mike Tyson half stays
  hallucinated (first loss given as McNeeley 1996 instead of Douglas 1990),
  so the final figure is wrong. One-hop evidence, multi-hop task.
- `frames_hard_024` (utilized, 0.0): the answerer engages heavily with the
  injected findings, and *because* they confirm only the general
  Gandhi–Gram-Swaraj link and not the specific Chikhali connection, it
  retracts its original Gandhi/INC answer and ends refusing to name anyone.
  Engagement actively destroyed a scoring answer.
- `frames_hard_008` (utilized, 0.0): the injected Friends-fandom snippet lets
  the answerer correctly kill its own "Ross Gellar, Olympic diver"
  hallucination ("as indicated in the context updates you provided") — but
  it then invents a new athlete and never finds the real one. Refutation
  worked; the snippets could not supply the true chain.
- `frames_hard_012` (utilized, 1.0): the Primary Wave snippet is explicitly
  reasoned about ("refers to her publishing rights, not her record label"),
  sharpening a correct Warner Music Group answer.
- `frames_hard_028` (utilized, 1.0): both snippets engaged (the Shackleton
  quote is cited "as noted in the provided context") and folded into a
  confident, correct number-9 answer.
- `frames_hard_020` (not utilized, 1.0): the injection is a gaps-only list
  (no search findings); the already-correct Tom Ridge/Pennsylvania answer is
  re-verified without needing it. Non-utilization here is benign.
- `frames_hard_025` (not utilized, 0.0): the scanner identified exactly the
  right gaps (Rognoni's birth/death dates — both hallucinated), but this
  injection carried no findings, and the answerer re-asserted its invented
  dates across two more turns. Right diagnosis, no evidence delivered.
- `crepe_tricky_003` (utilized, 0.17): the gap list's critical-angle-of-
  attack/stall material is worked into the reply as a whole section — clear
  uptake, but the judge still scores the episode low. Engagement is not what
  the judge is paying for on the tricky tier.

Pattern across the spot-checks: injected evidence reliably *refutes or
confirms single claims* (010, 012, 028 — all 1.0), but on multi-hop FRAMES
tasks it covers one link at best (026, 008) and under-confirmation triggers
retraction of correct answers (024). That is the §4 explanation ranking.

## 6. Acceptance-criteria scorecard (#101)

| Criterion | Status | Notes |
|---|---|---|
| Run completes with per-condition n ≥ 45 | PARTIAL | random 45 ✓, heuristic 47 ✓; **no_subconscious 40 — MISSED** (10 clustered Ollama Cloud read timeouts, same FRAMES tasks that failed in prior runs). Documented rather than re-run; the paired analysis removes the resulting survivor bias from all conclusions. |
| Analysis doc comparing search vs no-search per tier | PASS | This doc + `scripts/analyze_search_experiment.py` + `data/baseline_search/paired_analysis.json`. Paired t-CIs on the 6-arm intersection supersede raw-n Welch tests (which are also in both `report.txt` files: all NO). The run plan's citation spot-check is §5 (measured utilization + 9 read transcripts). |
| Honest verdict either way | PASS | §4: hypothesis not supported at the observable effect sizes; power bound stated. |

## 7. Downstream pointers

- #103 — fix per-episode hyperparameter provenance before any metadata-based
  corpus filtering.
- Follow-ups suggested by §4–§5: fetch full page content behind top results
  (the one-hop-snippet ceiling is the strongest signal in the data); make
  the injection framing distinguish "not confirmed by these snippets" from
  "refuted" to prevent evidence-induced retraction; #44-style timeout tuning
  before another N=10 cloud run.
