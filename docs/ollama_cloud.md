# Ollama Cloud backend (Issue #43)

Adds a second model backend alongside Gemini so episodes/benchmarks can run
against an open Gemma-class model. The new client is a **drop-in** for
`GeminiClient` at every call site (`EpisodeRunner`, `SimulatedUser`,
`TaskScorer`, tools) — it satisfies the same duck-typed contract: a `model`
property and a `generate(...) -> GeminiResponse` method with the identical
signature.

## Model tag

Default: **`gemma4:31b-cloud`** — verified against the Ollama Cloud catalog
(`ollama.com/library/gemma4:31b-cloud`): cloud-hosted, 256K context, and a
**reasoning model with configurable thinking modes**. This supersedes the
issue's tentative "gemma 4 31b". The tag is configurable, never hard-coded:

- CLI: `--model gemma4:31b-cloud`
- Config: `[model] name = "gemma4:31b-cloud"`
- Code: `OllamaCloudClient(model="…")` / `build_client("ollama", "…")`

`gemma3:27b-cloud` is also available if a non-reasoning fallback is wanted.

## Auth & endpoint

- `OLLAMA_API_KEY` — required (Bearer token). No secret is committed.
- `OLLAMA_HOST` — optional, defaults to `https://ollama.com`. Override for a
  self-hosted/local Ollama (e.g. `http://localhost:11434`).

Transport is stdlib `urllib` (no extra dependency): one non-streaming
`POST {host}/api/chat`, with the same retry/backoff and client-side latency
timing as `GeminiClient`.

## Selecting the backend

```bash
# Benchmark against Gemma on Ollama Cloud
OLLAMA_API_KEY=… uv run python scripts/run_baseline_benchmark.py \
    --provider ollama --model gemma4:31b-cloud \
    --tasks-per-condition 50
```

In config, set `[model] provider = "ollama"`; `HyperConfig.to_model_client()`
builds the right client. Both paths route through
`bicameral_agent.model_client.build_client(provider, model)`.

## Parameter mapping (Gemini → Ollama `/api/chat`)

| `generate(...)` arg | Ollama field |
|---|---|
| `messages` (role `model`) | `messages` (role `assistant`) |
| `system_prompt` | leading `{"role":"system"}` message |
| `thinking_level` | `think`: `minimal`→`false`, else the level string |
| `temperature` | `options.temperature` |
| `max_output_tokens` | `options.num_predict` |
| `response_schema` | `format` (JSON schema, native structured output) |
| `tools` | `tools[].function` (`parameters_json_schema`→`parameters`) |

Response: `message.content`→`content`, `prompt_eval_count`/`eval_count`→
input/output tokens, `done_reason`→`finish_reason`,
`message.tool_calls[].function.{name,arguments}`→`function_calls[].{name,args}`.

### Structured output

Native — the scorer and simulated user pass a JSON schema via `response_schema`
and parse `json.loads(response.content)` exactly as with Gemini. No shim.

### Thinking

`thinking_level` maps to Ollama's `think` parameter. This requires a
reasoning-capable model; `gemma4:31b-cloud` is one. `minimal` disables thinking.

## Cost

Gemma cloud is subscription-flat, so `gemma4:31b-cloud` is registered in
`MODEL_PRICING` at **$0/token**. Call and token counts still flow into
`CostTracker` (the call counter increments); only the dollar cost is zero.

## Testing

- `tests/test_ollama_cloud.py` — fully offline; mocks `urllib.request.urlopen`.
- `tests/test_model_client.py` — factory + config provider + flat-rate pricing.

```bash
uv run --extra dev python -m pytest tests/test_ollama_cloud.py tests/test_model_client.py -q
```

A live smoke run requires `OLLAMA_API_KEY` and is the only step that hits the
network (see the benchmark command above with small `--tasks-per-condition`).
