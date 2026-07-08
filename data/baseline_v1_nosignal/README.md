# baseline_v1_nosignal — superseded 2026-06 baseline run

This is the original #23 baseline control run (2026-06). It is kept for
provenance only and must not be used for training or comparison: the RCA in
epic #41 found it produced no usable signal — quality scores pinned at ~1.0
(saturated judge), every episode collapsed to a single turn, and the
subconscious queue was never exercised (all queue metrics zero).

It was superseded by the 2026-07 re-run under `data/baseline/` (issue #46),
which restored variance, multi-turn episodes, and queue activity. See
`docs/baseline_rerun_2026-07.md`.
