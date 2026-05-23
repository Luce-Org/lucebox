# Client Launchers

These scripts run real clients against Lucebox (C++ server by default).

## Headless bandit (5 clients, structured CSV)

```bash
cd /workspace/lucebox-hub-harness
python3 -m harness.client_test_runner --condition C_bandit \
  --clients claude_code,hermes,opencode,codex,pi
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

`run_claude_code.sh`, `run_openclaw.sh`, `run_openwebui.sh`, `run_openwebui_tools.sh`
are retained as bash launchers. GUI clients (openwebui, openclaw) require them.

```bash
MAX_CTX=32768 harness/clients/run_claude_code.sh
```

## Environment overrides (applies to all launchers)

| Variable | Default | Description |
| --- | --- | --- |
| `LUCEBOX_SERVER_BACKEND` | `cpp` | Use `python` to opt-in to the Python server fallback |
| `DFLASH_SERVER_BIN` | `$REPO_DIR/dflash/build/dflash_server` | C++ server binary |
| `MAX_CTX` | per-client | KV cache context size |
| `BUDGET` | 22 | Speculative decode budget |
| `PROMPT` | per-client | One-shot prompt |
| `PROMPT_FILE` | `` | Override prompt from file |
| `PFLASH_SESSION_ID` | `` | Session ID injected via proxy |

## Notes

- `common.sh` contains the shared server lifecycle (`start_lucebox_server`, `preflight_require_bin`).
- C++ server default: `LUCEBOX_SERVER_BACKEND=cpp` is set before sourcing `common.sh` in every launcher.
- `run_openwebui_tools.sh` supports `OPENWEBUI_FUNCTION_CALLING=default` and `OPENWEBUI_FUNCTION_CALLING=native`.
- Every launcher redirects stdin from `/dev/null`.
