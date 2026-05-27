# Using DFlash with OpenAI Codex CLI

This guide covers installing Codex CLI, configuring it to use the local
DFlash server, and common usage patterns.

## Prerequisites

- **DFlash server** built and model files downloaded (see [DEVELOPER.md](DEVELOPER.md))
- **Node.js ≥ 18**

## 1. Install Codex CLI

```bash
npm install -g @openai/codex
codex --version          # should print 0.1xx.x
```

> If you get a permission error, use a user-level prefix:
> ```bash
> mkdir -p ~/.npm-global && npm config set prefix ~/.npm-global
> export PATH=~/.npm-global/bin:$PATH   # add to ~/.bashrc
> npm install -g @openai/codex
> ```

## 2. Start the DFlash Server

```bash
cd dflash
python scripts/server.py \
  --draft models/draft/draft-Qwen3.6-27B.gguf \
  --port 8080
```

Wait until you see:

```
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### Quick smoke test

```bash
curl -s http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"luce-dflash","input":"Say hi","max_output_tokens":32}' \
  | python3 -m json.tool
```

## 3. Configure Codex

Create or edit `~/.codex/config.toml`:

```toml
model = "luce-dflash"
model_provider = "dflash"

[model_providers.dflash]
name = "DFlash"
base_url = "http://localhost:8080/v1"
wire_api = "responses"
supports_websockets = false
```

Key points:

| Field | Why |
|---|---|
| `name` | **Required** — Codex rejects providers with an empty name |
| `wire_api = "responses"` | Codex only supports the Responses API |
| `supports_websockets = false` | DFlash serves HTTP only |
| No `env_key` | Local server needs no auth token |

## 4. Using Codex

### Interactive mode

```bash
codex                               # opens the TUI
```

### Non-interactive (exec)

```bash
# Simple question
codex exec "What is 2+2?"

# Coding task in a repo
cd /path/to/your/project
codex exec "Add input validation to the login handler"

# Code review
codex exec review
```

### Approval policies

```bash
codex --approval-policy suggest ...   # suggest commands, ask before running
codex --approval-policy auto-edit ... # auto-approve file edits, ask for shell
codex --approval-policy full-auto ... # auto-approve everything
```

### Reasoning effort

The server maps reasoning effort to Qwen's thinking mode:

- `low` (default) — thinking disabled, fastest responses
- `medium` / `high` — thinking enabled (`<think>` tags)

```bash
codex -c 'model_reasoning_effort="medium"' exec "Refactor this module"
```

## 5. Troubleshooting

### "provider name must not be empty"

Add `name = "DFlash"` (or any non-empty string) to `[model_providers.dflash]`.

### "Reconnecting… 1/5" loop

The server is returning an error. Check:

1. **Server is running** — `curl http://localhost:8080/health`
2. **Port matches config** — `base_url` must match `--port`
3. **Server logs** — look at the terminal where `server.py` is running

Common causes:

- Server not started yet (daemon still loading model)
- Port mismatch between config and server
- Proxy intercepting localhost (set `no_proxy=localhost`)

### "Unknown model luce-dflash"

This warning is normal — Codex doesn't recognize custom model names but
uses fallback metadata. It does not affect functionality.

### Slow first response

The first request triggers model loading into GPU memory. Subsequent
requests reuse the loaded model and are much faster.

## 6. Server CLI Reference

```
python scripts/server.py [OPTIONS]

Options:
  --host HOST           Bind address (default: 0.0.0.0)
  --port PORT           Bind port (default: 8080)
  --target PATH         Target model GGUF (default: models/Qwen3.6-27B-Q4_K_M.gguf)
  --draft PATH          Draft model for speculative decoding
  --budget N            Speculation budget (default: 22)
  --max-ctx N           Max context length (default: model-dependent)
  --tokenizer NAME      HuggingFace tokenizer (auto-detected from model)
```
