# pi-ollama-api-native

Native Ollama API provider extension for the [pi coding agent](https://github.com/badlogic/pi-mono). Talks directly to Ollama's `/api/chat` (and other native endpoints), bypassing the OpenAI compatibility shim.

## Why this extension exists

Pi ships with an `openai-completions` adapter that routes Ollama traffic through Ollama's `/v1/chat/completions` endpoint (the OpenAI compatibility shim). While convenient, this has known limitations that affect agentic workflows:

### Problems with the OpenAI compatibility shim

1. **Tool calls dropped during streaming.** Ollama's OpenAI-compatible endpoint has a known bug where `tool_calls` are silently dropped from streamed responses. Without those tool calls, pi's agent loop stalls on the first tool use — the model produces a tool call, the wire eats it, pi never sees it.

2. **Lossy capability mapping.** The compatibility layer normalises all models into an OpenAI-shaped box. Ollama-native capabilities like `thinking` (with model-specific levels), `vision`, and `embeddings` are either lost or incorrectly mapped.

3. **No native thinking control.** The Ollama `/api/chat` endpoint supports model-specific thinking levels (`"low"`, `"medium"`, `"high"`, `"max"` for DeepSeek V4), but the OpenAI compat shim's `reasoning_effort` mapping is coarse and doesn't handle per-model nuances.

### Advantages of native API

| | OpenAI compat (`/v1/`) | Native (`/api/`) |
|---|---|---|
| Tool calls in streaming | ❌ Bugged — dropped | ✅ Works |
| Model discovery | `/v1/models` (limited) | `/api/tags` + `/api/show` |
| Capability detection | Inferred from names | From `capabilities` array |
| Context window | Not exposed | Parsed from `model_info` |
| Thinking/reasoning | Coarse mapping | Model-specific levels |
| Vision | Unreliable detection | `capabilities` + family heuristics |

### DeepSeek compatibility

This extension specifically addresses two DeepSeek-specific issues:

1. **DSML tool call parsing.** DeepSeek models (V3, V4, R1) are trained to emit tool calls in DSML (DeepSeek Markup Language) format — `<｜tool_calls｜><｜invoke｜ name="X"｜><｜parameter｜ ... /></｜invoke｜></｜tool_calls｜>`. When Ollama's native API returns these in the `content` field instead of the structured `tool_calls` field, the extension parses them into proper toolCall blocks rather than silently discarding them.

2. **Thinking level mapping.** Pi's `xhigh` thinking level maps to `"max"` for DeepSeek V4 models (following the pattern used by the OpenRouter provider), while GPT-OSS models get `"low"`/`"medium"`/`"high"` string values.

## Install

```bash
pi install git:github.com/kuyavinny/pi-ollama-api-native
```

Requires Ollama running with at least one tool-capable model pulled.

Set `OLLAMA_API_KEY` for `https://ollama.com/api` (cloud). For local Ollama leave it unset — it defaults to `http://localhost:11434`.

## Commands

| Command | Description |
|---|---|
| `/ollama-models` | List all models with context + capabilities |
| `/ollama-refresh` | Re-discover models from `/api/tags` + `/api/show` |
| `/ollama-show <model>` | Dump `/api/show` response |

## How it works

On startup, the extension:
1. Calls `GET /api/tags` to list available models
2. For each model, calls `POST /api/show` to extract context window, capabilities, and family info
3. Registers models with pi via `registerProvider` with a custom `streamSimple` handler
4. The handler converts pi's internal message format to Ollama's `/api/chat` wire format, parses NDJSON streaming responses, and returns `AssistantMessageEventStream` events

No core pi changes required — `streamSimple` fully replaces the built-in handler for the registered API string.

## Related projects

- [pi-mono](https://github.com/badlogic/pi-mono) — the pi coding agent
- [ollama#12557](https://github.com/ollama/ollama/issues/12557) — upstream tool-calling streaming bug
- [pi-mono#3712](https://github.com/badlogic/pi-mono/issues/3712) — DeepSeek DSML tool call issue
- [CaptCanadaMan/pi-ollama](https://github.com/CaptCanadaMan/pi-ollama) — alternative native Ollama extension

## License

MIT
