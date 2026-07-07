# Harder benchmark integration (Issue #42)

**Moved:** this document was generalized into
[`docs/eval_datasets.md`](eval_datasets.md) (Issue #56), which covers the full
dataset factory — FRAMES and CREPE included — plus the licensing/citation
matrix and the verifier registry.

Quick pointers for the original #42 workflow:

```bash
python scripts/fetch_dataset.py --dataset hard_benchmark   # FRAMES + CREPE
uv run python scripts/run_baseline_benchmark.py --dataset hard_benchmark
```

`bicameral_agent.hard_benchmark` (`build_hard_benchmark` /
`load_hard_benchmark`) remains as a re-exporting shim over
`bicameral_agent.eval_datasets`, and the `data/external/hard_benchmark.json`
cache format is unchanged, so previously fetched caches stay valid.
