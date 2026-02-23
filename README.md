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

## Project Phases

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

## Project Structure

```
src/bicameral_agent/     # Main package
tests/                   # Test suite
github_issues.md         # Full issue specifications
pyproject.toml           # Project config (hatchling build, ruff, pytest)
```

## Tech Stack

- **LLM**: Gemini 3 Flash Preview (conscious loop + tool internals)
- **ML**: PyTorch (policy/value network, transition model, MCTS)
- **Data**: Pydantic v2 (schemas), PyArrow (Parquet serialization)
- **Dev**: pytest, ruff, uv
