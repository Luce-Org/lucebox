# Client Launchers

These scripts run real clients against Lucebox. They are useful when you want to
use Lucebox from a specific tool, and when you want to check that a server
change did not break that tool.

Run from the repo on the GPU machine:

```bash
cd /workspace/lucebox-hub-harness
harness/clients/run_codex.sh
```

Each launcher starts `dflash/scripts/server.py`, runs the client, writes logs
under `/workspace/lucebox-client-harness-runs`, then stops the server.

Set `LUCEBOX_SERVER_BACKEND=cpp` to run the native C++ HTTP server instead.
The launcher will start `dflash/build/dflash_server` by default, or the path in
`DFLASH_SERVER_BIN`.

```bash
LUCEBOX_SERVER_BACKEND=cpp \
DFLASH_SERVER_BIN=dflash/build/dflash_server \
MAX_CTX=32768 MAX_TOKENS=512 \
BUDGET=22 VERIFY_MODE=ddtree \
harness/clients/run_codex.sh
```

The C++ server is expected to handle the same client protocol shapes covered by
these launchers and probes: OpenAI Chat Completions, streaming chunks, tool
metadata, OpenAI Responses for Codex, Anthropic Messages for Claude Code, and
Open WebUI model metadata.

## Defaults

The defaults below are the current RTX 3090 starting points for
`Qwen3.6-27B-Q4_K_M` plus the Lucebox DFlash draft.

| Client | Launcher | Default profile |
| --- | --- | --- |
| Claude Code | `run_claude_code.sh` | `MAX_CTX=49152 BUDGET=22 VERIFY_MODE=ddtree EXTRA_SERVER_ARGS=--lazy-draft` |
| Codex | `run_codex.sh` | `MAX_CTX=32768 BUDGET=22 VERIFY_MODE=ddtree EXTRA_SERVER_ARGS=--lazy-draft` |
| OpenCode | `run_opencode.sh` | `MAX_CTX=86016 BUDGET=22 VERIFY_MODE=ddtree EXTRA_SERVER_ARGS=--lazy-draft` |
| Hermes Agent | `run_hermes.sh` | `MAX_CTX=98304 BUDGET=22 VERIFY_MODE=ddtree EXTRA_SERVER_ARGS=--lazy-draft` |
| Pi | `run_pi.sh` | `MAX_CTX=65536 BUDGET=22 VERIFY_MODE=ddtree EXTRA_SERVER_ARGS=--lazy-draft` |
| OpenClaw | `run_openclaw.sh` | `MAX_CTX=204800 BUDGET=22 VERIFY_MODE=ddtree EXTRA_SERVER_ARGS=--lazy-draft` |
| Open WebUI chat | `run_openwebui.sh` | `MAX_CTX=262144 BUDGET=22 VERIFY_MODE=ddtree EXTRA_SERVER_ARGS=--lazy-draft` |
| Open WebUI tools | `run_openwebui_tools.sh` | `MAX_CTX=65536 BUDGET=22 VERIFY_MODE=ddtree EXTRA_SERVER_ARGS=--lazy-draft` |

Override any setting inline:

```bash
MAX_CTX=32768 harness/clients/run_claude_code.sh
PROMPT='Explain the repo and end with lucebox-client-ok' harness/clients/run_opencode.sh
PROMPT_FILE=harness/clients/prompts/repo_inspection.txt harness/clients/run_hermes.sh
```

Claude Code uses the real Anthropic Messages client path. Lucebox trims
Claude-specific prompt boilerplate by default for local-model reliability. To
test the raw prompt, set:

```bash
DFLASH_ANTHROPIC_RAW_SYSTEM=1 DFLASH_ANTHROPIC_RAW_USER=1 \
  harness/clients/run_claude_code.sh
```

## Compare Backends

Use `run_backend_pair.sh` to run the same client once with llama.cpp and once
with Lucebox:

```bash
CLIENT=opencode PROMPT_FILE=harness/clients/prompts/repo_inspection.txt \
  harness/clients/run_backend_pair.sh
```

OpenAI Chat Completions clients can call llama.cpp directly. Claude Code and
Codex use `llamacpp_compat_proxy.py` so their real Anthropic Messages and
Responses requests can be compared too.

## Notes

- `common.sh` contains the shared server startup logic.
- `run_openwebui_tools.sh` supports `OPENWEBUI_FUNCTION_CALLING=default` and
  `OPENWEBUI_FUNCTION_CALLING=native`.
- Every launcher redirects stdin from `/dev/null`; this prevents SSH input from
  being accidentally treated as a user prompt by interactive clients.
