# Client Launchers

These scripts run real clients against Lucebox (C++ server by default).

## Headless bandit (5 clients, structured CSV)

```bash
cd /workspace/lucebox-hub-harness
python3 -m harness.client_test_runner --condition C_bandit \
  --clients claude_code,hermes,opencode,codex,pi
```

Each launcher starts `server/build/dflash_server`, runs the client, writes logs
under `/workspace/lucebox-client-harness-runs`, then stops the server.

The launcher will start `server/build/dflash_server` by default, or the path in
`DFLASH_SERVER_BIN`.

```bash
DFLASH_SERVER_BIN=server/build/dflash_server \
MAX_CTX=32768 MAX_TOKENS=512 \
BUDGET=22 VERIFY_MODE=ddtree \
harness/clients/run_codex.sh
```

Dry-run (preflight only, no server needed):

```bash
python3 -m harness.client_test_runner --condition C_bandit \
  --clients claude_code,hermes,opencode,codex,pi --dry-run
```

Output columns: `client, preflight_ok, session_id_captured, accept_rate, wall_s, exit_code`

When `codex` or `pi` binary is missing you will see:
```
PREFLIGHT ERROR: 'codex' not found on PATH.  Hint: run 'asdf reshim' or install it …
```

## Single-client bash launchers (kept for compatibility)

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
```

## Environment overrides (applies to all launchers)

| Variable | Default | Description |
| --- | --- | --- |
| `LUCEBOX_SERVER_BACKEND` | `cpp` | Native C++ server backend |
| `DFLASH_SERVER_BIN` | `$REPO_DIR/server/build/dflash_server` | C++ server binary |
| `MAX_CTX` | per-client | KV cache context size |
| `BUDGET` | 22 | Speculative decode budget |
| `PROMPT` | per-client | One-shot prompt |
| `PROMPT_FILE` | `` | Override prompt from file |
| `PFLASH_SESSION_ID` | `` | Session ID injected via proxy |

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

- `common.sh` contains the shared server lifecycle (`start_lucebox_server`, `preflight_require_bin`).
- C++ server default: `LUCEBOX_SERVER_BACKEND=cpp` is set before sourcing `common.sh` in every launcher.
- `run_openwebui_tools.sh` supports `OPENWEBUI_FUNCTION_CALLING=default` and `OPENWEBUI_FUNCTION_CALLING=native`.
- Every launcher redirects stdin from `/dev/null`.
