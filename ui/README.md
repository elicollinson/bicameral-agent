# Experiment console (`ui/`)

A React Ink terminal UI for driving `bicameral-agent` benchmark experiments
end-to-end: configure and launch a run, watch it live, and review/compare
finished runs. It is a thin operator console over the existing Python CLI —
it spawns `uv run python scripts/run_baseline_benchmark.py ...` as a child
process and reads its artifacts. No experiment logic lives in Node.

## Requirements

- Node.js ≥ 20
- `uv` on PATH (only needed to actually launch runs; the app starts and the
  Review screen works without any Python present)
- Provider credentials in the environment when launching, e.g.
  `GEMINI_API_KEY` for `--provider gemini`

## Install & run

```bash
cd ui
npm install
npm start          # or: npx tsx src/cli.tsx
```

Run it from anywhere inside the repo — the app walks up to the directory
containing `pyproject.toml` and uses that as the working directory for the
runner and for `data/` discovery.

## Screens

- **New experiment** — step-through form: provider (gemini/ollama), model,
  tasks-per-condition, max turns, parallel episodes, per-episode budget,
  output dir. Parallel episodes (default 1) maps to the runner's
  `--parallel-episodes` and should match the provider plan's
  concurrent-request allowance. With
  provider=ollama the model picker is populated live from
  `GET https://ollama.com/api/tags` (type to filter); if the request fails it
  falls back to free-text entry, and `tab` switches to free text at any time.
  Model names are shown and passed exactly as they appear in the catalog.
  Selecting a tag with no `MODEL_PRICING` entry shows a warning before launch
  (unregistered tags have crashed runs mid-flight, see #52). The exact `uv`
  command is shown before you confirm.
- **Running** — per-condition episode progress (row counts of the
  incrementally rewritten `<output-dir>/<condition>.parquet` files), elapsed
  time, streamed runner stdout/stderr, per-condition mean cost once the final
  report prints, and the child's exit code. `k` kills the run; leaving the
  screen does not.
- **Review** — lists completed runs (directories under `data/` containing a
  `summary.json`), renders the summary metrics table and `report.txt`
  (read-only), and compares the headline metrics of any two marked runs
  side-by-side (`space` to mark, `c` to compare).

## Development

```bash
npm test           # vitest unit tests
npm run typecheck  # tsc --noEmit
```
