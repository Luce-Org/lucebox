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

## GPU selection

All launchers inherit `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES`. Native
Lucebox runs also accept `TARGET_DEVICE` and `DRAFT_DEVICE`; when the latter is
omitted it follows `TARGET_DEVICE`. Device numbers use the runtime-visible
namespace, so exposing one physical GPU makes it device zero inside the server:

```bash
# NVIDIA GPU 0 with a CUDA build.
CUDA_VISIBLE_DEVICES=0 \
DFLASH_SERVER_BIN=server/build-cuda/dflash_server \
TARGET_DEVICE=cuda:0 \
harness/clients/run_codex.sh

# Physical HIP GPU 1, exposed as hip:0 to a HIP build.
HIP_VISIBLE_DEVICES=1 \
DFLASH_SERVER_BIN=server/build-hip/dflash_server \
TARGET_DEVICE=hip:0 \
harness/clients/run_codex.sh
```

CUDA and HIP servers are separate build artifacts. A host with Strix Halo
(`gfx1151`) and an R9700 (`gfx1201`) can use one dual-architecture HIP build:

```bash
cmake -S server -B server/build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DDFLASH27B_GPU_BACKEND=hip \
  -DDFLASH27B_HIP_ARCHITECTURES='gfx1151;gfx1201' \
  -DDFLASH27B_HIP_SM80_EQUIV=ON
cmake --build server/build-hip --target dflash_server -j"$(nproc)"
```

Select a server binary and matching visibility variable together. The launcher
prints resolved placement before startup and uses `nvidia-smi` or `rocm-smi`
for the matching backend in its final report.

## Interactive terminal clients

The default launcher behavior remains a deterministic one-shot compatibility
test. Set `HARNESS_INTERACTIVE=1` to attach the real client TUI to the terminal
while the launcher manages the Lucebox server:

```bash
HARNESS_INTERACTIVE=1 harness/clients/run_claude_code.sh
HARNESS_INTERACTIVE=1 harness/clients/run_codex.sh
HARNESS_INTERACTIVE=1 harness/clients/run_opencode.sh
HARNESS_INTERACTIVE=1 harness/clients/run_hermes.sh
HARNESS_INTERACTIVE=1 harness/clients/run_pi.sh
HARNESS_INTERACTIVE=1 harness/clients/run_openclaw.sh
```

`INTERACTIVE_PROMPT` supplies an optional first message where the client
supports it. Interactive client state and sessions persist under
`.harness-work/interactive/<client>`. Exit the client or press Ctrl+C to stop
the launcher-managed server. One-shot client timeouts do not apply to a TUI.
Set `HARNESS_PROGRESS=0` to suppress launcher progress and heartbeat messages.

Open WebUI is already interactive through its browser UI; its harness scripts
remain deterministic HTTP probes.

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
| Pi | `run_pi.sh` | `MAX_CTX=65536 BUDGET=22 VERIFY_MODE=ddtree EXTRA_SERVER_ARGS=--lazy-draft PI_TIMEOUT=3600` |
| OpenClaw | `run_openclaw.sh` | `MAX_CTX=204800 BUDGET=22 VERIFY_MODE=ddtree EXTRA_SERVER_ARGS=--lazy-draft` |
| Open WebUI chat | `run_openwebui.sh` | `MAX_CTX=262144 BUDGET=22 VERIFY_MODE=ddtree EXTRA_SERVER_ARGS=--lazy-draft` |
| Open WebUI tools | `run_openwebui_tools.sh` | `MAX_CTX=65536 BUDGET=22 VERIFY_MODE=ddtree EXTRA_SERVER_ARGS=--lazy-draft` |

Override any setting inline:

```bash
MAX_CTX=32768 harness/clients/run_claude_code.sh
PROMPT='Explain the repo and end with lucebox-client-ok' harness/clients/run_opencode.sh
PROMPT_FILE=harness/clients/prompts/repo_inspection.txt harness/clients/run_hermes.sh
```

`PI_TIMEOUT` is Pi's total wall-clock limit in seconds. Its one-hour default
allows long-context prefill and long generations to finish; set
`PI_TIMEOUT=0` to run without a launcher deadline. The launcher also disables
Pi's separate five-minute HTTP idle timeout, which can otherwise terminate an
SSE connection during a long prefill. For a manually configured Pi install,
put the same setting in `~/.pi/agent/settings.json`:

```json
{"httpIdleTimeoutMs": 0}
```

The other real-client launchers also allow one hour by default. Override their
deadlines with `CLAUDE_TIMEOUT`, `CODEX_TIMEOUT`, `OPENCODE_TIMEOUT`,
`HERMES_TIMEOUT`, `OPENCLAW_TIMEOUT`, or (for Open WebUI's curl probe)
`CURL_MAX_TIME`. The CLI launcher timeouts accept `0` to disable the outer
deadline. OpenCode's provider-level request and chunk deadlines default to one
hour too and can be changed with `OPENCODE_REQUEST_TIMEOUT_MS` and
`OPENCODE_CHUNK_TIMEOUT_MS`. The server independently sends SSE heartbeat
comments during silent prefill and cancels backend work when a client
disconnects.

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
- One-shot launchers redirect stdin from `/dev/null`; this prevents SSH input
  from being accidentally treated as a user prompt. `HARNESS_INTERACTIVE=1`
  deliberately keeps the selected terminal client attached to the TTY.
