# Client Launchers

These scripts run real clients against Lucebox. They are useful when you want to
use Lucebox from a specific tool, and when you want to check that a server
change did not break that tool.

Run from the repo on the GPU machine:

```bash
cd lucebox-hub
harness/clients/run_codex.sh
```

Each launcher starts `server/build/dflash_server`, runs the client, writes logs
under `.harness-work/runs`, then stops the server. Override `REPO_DIR`,
`CLIENT_WORK_DIR`, or `RUN_DIR` for custom/shared locations.
If a client CLI is missing, the launcher installs it automatically. Set
`AUTO_INSTALL_CLIENTS=0` to require a preinstalled binary instead.

To preinstall real-client CLIs yourself:

```bash
python3 harness/client_test_runner.py install --clients codex,hermes,openwebui
```

The launcher will start `server/build/dflash_server` by default, or the path in
`DFLASH_SERVER_BIN`. The default model paths are
`server/models/Qwen3.6-27B-Q4_K_M.gguf` and
`server/models/draft/dflash-draft-3.6-q4_k_m.gguf`; override them with
`TARGET`/`DRAFT` or the standard `DFLASH_TARGET`/`DFLASH_DRAFT` env vars.
When you set a custom target without setting a draft, the launcher does not
attach the default Qwen draft. Use `DRAFT=none` explicitly for no-draft targets
such as Gemma, Laguna, or standalone Qwen3.

```bash
DFLASH_SERVER_BIN=server/build/dflash_server \
DFLASH_TARGET=/path/to/Qwen3.6-27B-Q4_K_M.gguf \
DFLASH_DRAFT=/path/to/dflash-draft-3.6-q4_k_m.gguf \
MAX_CTX=32768 MAX_TOKENS=512 \
BUDGET=22 VERIFY_MODE=ddtree \
harness/clients/run_codex.sh
```

Gemma example:

```bash
DFLASH_TARGET=/path/to/gemma.gguf \
DRAFT=none \
MAX_CTX=32768 MAX_TOKENS=512 \
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
| luce-bench | `run_lucebench.sh` | `MAX_CTX=32768 BUDGET=22 VERIFY_MODE=ddtree EXTRA_SERVER_ARGS=--lazy-draft` |

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

## luce-bench

`run_lucebench.sh` is the odd one out: the "client" is `luce-bench` (the
in-tree capability bench at `luce-bench/`), not a vendored binary. It hits
`/v1/chat/completions` with the standard ds4-eval / HumanEval / longctx /
agent / forge case sets and writes per-case PASS/FAIL + timings.

Useful as a regression gate: a server change that breaks tool-call parsing,
chat-template rendering, or sampling defaults will show up here the same way
it would break a real-client launcher above.

```bash
# Full sweep (default — runs all 4 stdlib areas)
harness/clients/run_lucebench.sh

# Single area
LUCEBENCH_AREA=code harness/clients/run_lucebench.sh
LUCEBENCH_AREA=ds4-eval LUCEBENCH_THINK=1 harness/clients/run_lucebench.sh

# Knobs (see top of run_lucebench.sh): LUCEBENCH_AREA, LUCEBENCH_THINK,
# LUCEBENCH_MAX_TOKENS, LUCEBENCH_TIMEOUT, LUCEBENCH_PARALLEL.
```

## Notes

- `common.sh` contains the shared server startup logic.
- `run_openwebui_tools.sh` supports `OPENWEBUI_FUNCTION_CALLING=default` and
  `OPENWEBUI_FUNCTION_CALLING=native`.
- Every launcher redirects stdin from `/dev/null`; this prevents SSH input from
  being accidentally treated as a user prompt by interactive clients.
