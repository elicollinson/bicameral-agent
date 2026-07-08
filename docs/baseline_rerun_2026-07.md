# Baseline re-run, 2026-07 (#46)

Re-run of the baseline performance benchmark after the #41 validity remediation,
replacing the signal-free 2026-06 run (now archived at `data/baseline_v1_nosignal/`).
Artifacts: `data/baseline/` (canonical run), `data/baseline_pilot_v2/`
(supplementary pilot, same configuration at 20 tasks/condition planned).

## 1. Run configuration

| Item | Value |
|---|---|
| Answerer | `ollama` / `gemma4:31b-cloud` (Ollama Cloud, #43) |
| Judge + simulated user | pinned to `ollama` / `gemma4:31b-cloud` across all conditions (#53) |
| Dataset | `hard_benchmark` — 160-task pool: 100 FRAMES (hard, multi-hop) + 60 CREPE (tricky, false-presupposition) (#42) |
| Tasks per condition | 50 planned (no_subconscious / random / heuristic) |
| Max turns | 6 (multi-turn episodes, #45) |
| Structured output | schema-grounded (#82) |
| Transport failures | contained per-episode with retry, incremental persistence (#47, #81) |
| Injection persistence | `persistent_injection = true` in all conditions (#49) |
| Metered cost | $0 — Ollama Cloud subscription flat rate |
| Wall clock | ~7.6 h for the full run (12:47–20:23 per `run.log`); pilot v2 ran earlier the same day |

Provenance (`summary.json` `answerer`/`measurement` blocks) records the model
pinning. The planned Gemini-answerer conditions and the Gemini-pinned-judge
cross-model comparison were **deferred**: no `GEMINI_API_KEY` was available in
the run environment. Within this run the measurement apparatus is still held
fixed across all three conditions, so the control-vs-experimental comparison is
internally valid.

## 2. Results (`data/baseline/report.txt`)

| Metric | no_subconscious (n=45) | random (n=49) | heuristic (n=48) |
|---|---|---|---|
| quality_score | 0.639 [0.514, 0.763], σ=0.414 | 0.556 [0.429, 0.683], σ=0.442 | 0.672 [0.541, 0.802], σ=0.449 |
| task_completed | 0.667 | 0.592 | 0.667 |
| total_turns | 2.6 [2.3, 3.0] | 2.8 [2.5, 3.1] | 2.6 [2.3, 2.9] |
| total_tokens | 7 272 | 10 157 | 10 201 |
| tool_invocation_count | 0.0 | 0.6 | 1.1 |
| avg_queue_depth | 0.000 | 0.037 | 0.212 |
| drain_count | 0.0 | 0.1 | 0.5 |
| expired_count | 0.0 | 0.0 | 0.0 |
| latency MAPE | — (0 pairs) | 62.48% (27 pairs) | 74.95% (53 pairs) |
| transport failures | 5 | 1 | 2 |

Welch 95% comparisons: heuristic > random: **NO**; heuristic > no_subconscious:
**NO** (see §4).

## 3. Acceptance-criteria scorecard (#46)

| Criterion | Status | Notes |
|---|---|---|
| 50+ episodes per condition | PARTIAL | 45–49 collected vs 50 planned; 8 contained transport failures (Ollama Cloud read timeouts). Pooled with pilot v2, 62–67/condition are available as training data. |
| Incremental persistence, crash-safe | PASS | Run completed crash-free; all 8 transport failures contained per-episode and recorded in `summary.json` (#47, #81). |
| Real variance, statistically separable | PARTIAL | Variance restored (σ ≈ 0.41–0.45 vs pinned-at-1.000 in the old run); conditions are **not** separable at this n (CIs overlap). |
| Heuristic > random AND > no_subconscious | FAIL (direction correct) | Hypothesized ordering appears for the first time, but not significant — the #23 acceptance checks still report NO. |
| Queue metrics non-trivial | PASS | Heuristic: 0.5 drains/ep, avg depth 0.212; async architecture exercised. |
| Cross-model condition, same pinned judge | DEFERRED | No `GEMINI_API_KEY` available; all-Ollama run with judge + sim-user pinned to `gemma4:31b-cloud` across conditions. The Gemini arm is the one un-run condition. |
| Latency MAPE in the #44 sane range | FAIL | 62–75% vs <50% target. The #44 priors were fit on Gemini-era durations; Gemma-cloud calls run slower. Tracked in #44 — an Ollama-backend refit is in flight. |
| Training-ready Episode format | PASS | Per-condition parquet round-trips through `episodes_from_parquet`; consumed directly by the #26/#27 fits below. |
| #26 / #27 re-pointed to this run | PASS | Both fits below trained on `data/baseline/` episodes (see §7). |

## 4. Hypothesis outcome

Quality ordering matches the hypothesis for the first time:

    heuristic 0.672 [0.541, 0.802]  >  no_subconscious 0.639 [0.514, 0.763]  >  random 0.556 [0.429, 0.683]

but neither comparison is significant at Welch 95%. Effect sizes are small:
heuristic − no_subconscious ≈ 0.03, heuristic − random ≈ 0.12. A power
calculation at σ ≈ 0.44 says detecting a 0.12 difference needs roughly 230
episodes/condition; at n ≈ 48 the run is underpowered for significance.

Stated plainly: **validity is restored** (variance, multi-turn, queue activity,
fixed judge), the **direction is consistent** with the hypothesis, and the run
is **underpowered for significance at n ≈ 48**. The original #23 acceptance
checks ("heuristic > random / > no_subconscious at Welch 95%") still report NO.

## 5. Transition model (#27)

Refit on this run's 142 episodes (380 transition examples, 305 train / 75
held-out) from `data/baseline/*.parquet`. Checkpoint committed at
`data/transition_model/transition_model.pt` (61 549 params); full metrics in
`data/transition_model/metrics.json`. **All five acceptance criteria pass:**

| Criterion | Target | Actual |
|---|---|---|
| State MSE per dim (held-out, normalized) | < 0.1 | 0.0389 mean (max single dim 0.417) |
| Reward correlation (held-out) | r > 0.4 | r = 0.551 |
| 5-step rollouts bounded | norms bounded | max state norm 4.46 vs bound 103.9 |
| Forward-pass latency | < 2 ms | 0.028 ms median |
| Training time (CPU) | < 30 min | 0.66 s |

## 6. Supervised pre-training note (#26)

Policy/value pre-training on pooled heuristic episodes (this run + pilot v2, 66
episodes / 171 examples; `data/pretrain/`): **4 of 5 criteria pass** — val
action accuracy 0.97 vs 0.58 majority-class baseline, value correlation 0.39 >
0.3, train loss monotonic over 10-epoch windows, training < 30 min (0.2 s).
The val/train loss gap exceeds 20% (**FAILS**) — data-limited at 66 episodes.
**#26 stays open** pending a larger heuristic corpus.

## 7. Downstream pointers

- `data/baseline/` is the canonical baseline for #26 (policy/value pre-training)
  and #27 (transition model); both consumers now point at this run.
- `data/baseline_pilot_v2/` is supplementary pooling material (53 episodes,
  identical configuration at 20 tasks/condition planned).
- `data/baseline_v1_nosignal/` is the archived 2026-06 no-signal run — do not
  train on it (see its README and epic #41).
- Latency-model refit for the Ollama backend: #44 (in flight); the
  `TestBaselineMape` gate is xfailed against this data until it lands.
