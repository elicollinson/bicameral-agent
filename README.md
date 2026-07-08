# Bicameral Agent

A dual-process agent framework that uses MCTS-learned policy to coordinate "subconscious" tool primitives running alongside a "conscious" LLM reasoning loop.

## Concept

Traditional LLM agents run tools synchronously — the reasoning loop blocks while waiting for tool results. This project explores an alternative architecture inspired by dual-process theory:

- **Conscious loop**: A Gemini 3 Flash reasoning process that handles multi-turn conversation with the user.
- **Subconscious tools**: Lightweight tool primitives (Research Gap Scanner, Assumption Auditor, Context Refresher) that run asynchronously and deposit results into a priority queue.
- **Context injection queue**: A priority queue that sits between tools and the conscious loop, supporting breakpoint-drain and interrupt-and-retry consumption modes.
- **Learned controller**: An MCTS-trained policy network that decides *when* to invoke each tool, replacing hand-coded heuristics with learned timing strategies.

The hypothesis: a small neural network trained via Monte Carlo Tree Search can learn better tool invocation timing than hand-coded rules — discovering emergent patterns like preemptive invocation, latency-aware staggering, and queue-depth-sensitive inhibition.

## Architecture

```
                         ┌─────────────────────┐
                         │   User / Simulated   │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │   Signal Classifier   │
                         │  (5 behavioral dims)  │
                         └──────────┬───────────┘
                                    │
┌───────────────┐       ┌──────────▼───────────┐
│  State Encoder │◄──────│      Controller       │
│  (feature vec) │       │ (heuristic / learned) │
└───────────────┘       └──────────┬───────────┘
                                    │ decide()
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             ┌───────────┐  ┌─────────────┐  ┌───────────┐
             │  Research  │  │ Assumption  │  │  Context   │
             │Gap Scanner │  │  Auditor    │  │ Refresher  │
             └─────┬──────┘  └──────┬──────┘  └─────┬─────┘
                   │                │                │
                   └────────────────┼────────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  Context Injection   │
                         │       Queue          │
                         └──────────┬───────────┘
                                    │ drain / interrupt
                         ┌──────────▼───────────┐
                         │   Conscious Loop      │
                         │  (Gemini 3 Flash)     │
                         └──────────────────────┘
```

**Injection persistence**: by default, context drained from the queue is folded into the user message that enters conversation history, so injected findings stay visible to every later generation (and their tokens are accounted once, at injection). Setting `queue.persistent_injection = false` switches to transient "whisper" mode, where injected context is shown to the model for exactly one generation and then vanishes from history — kept as an experimental variable for A/B comparison.

## Project Phases

> **Status note:** the baseline control run (#23) produced no usable signal — see the RCA in epic [#41](https://github.com/elicollinson/bicameral-agent/issues/41). Remediation is in progress under that epic (harder benchmark, multi-provider backends, experiment-validity fixes); the original phase plan below is superseded by that epic. The #46 baseline re-run completed 2026-07 (validity restored) — see [docs/baseline_rerun_2026-07.md](docs/baseline_rerun_2026-07.md). Live MCTS training (#29), the 5-condition comparative evaluation (#30), and refreshed supervised pre-training (#26) completed 2026-07 — see [docs/live_training_and_comparative_2026-07.md](docs/live_training_and_comparative_2026-07.md).

The project is organized into 6 phases with 40 tracked issues:

| Phase | Name | Issues | Description |
|-------|------|--------|-------------|
| 0 | Foundation Infrastructure | #1–#12, #39, #40 | Episode schema, logging, replay, state encoding, queue, latency model, API client, evaluation dataset, scorer, config, cost tracking |
| 1 | Intelligent Tool Primitives | #13–#16, #35 | Tool interface contract, Research Gap Scanner, Assumption Auditor, Context Refresher, latency data collection |
| 2 | User Signal Processing | #17–#18 | Follow-up type classifier, behavioral signal aggregator |
| 3 | Heuristic Baseline Controller | #19–#23, #33, #34, #36, #37 | Conscious loop runner, heuristic/random/null controllers, state encoder extensions, simulated user, episode runner, A/B testing, baseline benchmarks |
| 4 | MCTS Training Infrastructure | #24–#29, #38 | Training data pipeline, policy/value network, supervised pre-training, transition model, MCTS engine, training loop, data storage |
| 5 | Evaluation and Iteration | #30–#32 | Comparative evaluation harness, MVP success criteria validation, emergent behavior analysis |

See the [project board](https://github.com/users/elicollinson/projects/4) for current status and dependencies.

## MVP Success Criteria

1. Learned policy outperforms random invocation by >= 15% on task quality
2. Learned policy outperforms heuristic baseline by >= 8% on task quality
3. At least one emergent timing pattern not present in the heuristic
4. Training converges within 500 episodes
5. Learned policy triggers fewer unintended interrupts than heuristic
6. Queue-based delivery produces fewer reasoning derailments than synchronous injection

## Setup

Requires Python >= 3.11.

```bash
# Clone and install
git clone https://github.com/elicollinson/bicameral-agent.git
cd bicameral-agent
uv pip install -e ".[dev]"

# Run tests
pytest
```

## Development

```bash
# Lint
ruff check src/ tests/

# Test with coverage
pytest --cov=bicameral_agent
```

A React Ink terminal console for launching, tracking, and reviewing benchmark runs lives in [`ui/`](ui/README.md).

## Model Providers

Two backends satisfy the same client contract (`src/bicameral_agent/model_client.py`), so episodes and benchmarks are provider-agnostic:

- **Gemini** (default) — requires `GEMINI_API_KEY`
- **Ollama Cloud** — open Gemma-class models; requires `OLLAMA_API_KEY` (see `docs/ollama_cloud.md`)

The research gap scanner can additionally use real web search via the Brave
Web Search API — requires `BRAVE_API_KEY`; enable with `--search-provider
brave` or `[tools] search_provider = "brave"` (default `mock` keeps runs
offline; see `docs/eval_datasets.md`).

Select a backend per run with the `--provider` flag on scripts that support it:

```bash
uv run python scripts/run_baseline_benchmark.py --provider ollama --model gemma4:31b-cloud
```

or via the `[model]` config section (`provider = "gemini"` or `"ollama"`), overridable with `BICAMERAL_` environment variables.

## State Encoder

The `StateEncoder` compresses conversation state into a fixed 64-dimensional feature vector (`FEATURE_DIM = 64` in `src/bicameral_agent/encoder.py`) for the MCTS controller. The layout:

| Index | Feature | Dims |
|-------|---------|------|
| 0 | turn_number (user messages so far) | 1 |
| 1 | total_tokens_so_far | 1 |
| 2–33 | topic_embedding | 32 |
| 34 | estimated_confidence | 1 |
| 35–38 | last_tool_invoked (one-hot) | 4 |
| 39 | turns_since_last_tool | 1 |
| 40 | user_stop_count | 1 |
| 41–45 | last_followup_type (one-hot) | 5 |
| 46–48 | response_latency_bucket (one-hot) | 3 |
| 49 | message_length_ratio | 1 |
| 50–52 | sentiment_shift (one-hot) | 3 |
| 53 | queue_depth | 1 |
| 54 | queue_token_total | 1 |
| 55 | queue_max_priority (ordinal, 0 = empty) | 1 |
| 56 | queue_time_since_last_drain | 1 |
| 57 | queue_pending_tool_count | 1 |
| 58 | queue_arrival_interval_ema | 1 |
| 59 | latency_research_gap_scanner | 1 |
| 60 | latency_assumption_auditor | 1 |
| 61 | latency_context_refresher | 1 |
| 62 | episode_turn_progress | 1 |
| 63 | episode_completion_fraction | 1 |

Scalars use cap-and-divide normalization (`min(val, cap) / cap`), so every value is deterministically bounded to [0, 1]. The module docstring in `encoder.py` is the authoritative reference for the layout.

By default, topic embeddings use a deterministic SHAKE-256 hash. For semantic embeddings, install the optional ML extra:

```bash
uv pip install -e ".[ml]"
```

This adds `fastembed` with the `all-MiniLM-L6-v2` ONNX model (~150MB).

## Project Structure

```
src/bicameral_agent/     # Main package
tests/                   # Test suite
scripts/                 # Runnable entry points (benchmarks, data collection)
docs/                    # Design notes and dataset documentation
pyproject.toml           # Project config (hatchling build, ruff, pytest)
```

## Documentation

- [`docs/hard_benchmark.md`](docs/hard_benchmark.md) — the harder external benchmark (FRAMES + CREPE): sources, licenses, attribution, and fetch instructions
- [`docs/ollama_cloud.md`](docs/ollama_cloud.md) — the Ollama Cloud model backend, a drop-in alternative to Gemini
- [`docs/benchmark_comparison_issue42.md`](docs/benchmark_comparison_issue42.md) — dataset comparison and recommendation behind the hard benchmark (#42)

## Tech Stack

- **LLM**: Gemini 3 Flash Preview (conscious loop + tool internals), with Ollama Cloud (Gemma-class) as an alternate provider
- **ML**: PyTorch (policy/value network, transition model, MCTS)
- **Data**: Pydantic v2 (schemas), PyArrow (Parquet serialization)
- **Dev**: pytest, ruff, uv
