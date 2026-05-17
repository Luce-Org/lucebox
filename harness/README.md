# Lucebox Harness

<p align="center">
  <img src="assets/hero.png" alt="Lucebox client harness experiments on RTX 3090" width="100%">
</p>

Use this folder for two things:

- launch Lucebox through a real client: Claude Code, Codex, OpenCode, Hermes,
  Pi, OpenClaw, or Open WebUI
- test a Lucebox change against the HTTP protocols those clients use

The defaults are tuned for an RTX 3090 / 24 GB setup. Paths and server settings
can be overridden with environment variables.

## Run a client

Each script starts Lucebox, runs one real client, saves logs, then stops the
server.

```bash
cd /workspace/lucebox-hub-harness

harness/clients/run_codex.sh
harness/clients/run_claude_code.sh
harness/clients/run_opencode.sh
```

Common overrides:

```bash
MAX_CTX=32768 BUDGET=22 VERIFY_MODE=ddtree harness/clients/run_codex.sh
PROMPT_FILE=harness/clients/prompts/repo_inspection.txt harness/clients/run_hermes.sh
CLIENT=opencode harness/clients/run_backend_pair.sh
```

The per-client defaults live in [`clients/README.md`](clients/README.md).
They are not all the same: a tool-heavy agent prompt and a chat/proxy prompt
need different context limits on a 24 GB card.

## Test a server change

If you already have `dflash/scripts/server.py` running, use `probe`:

```bash
python3 harness/client_test_runner.py probe \
  --url http://127.0.0.1:8000 \
  --clients claude_code,codex,opencode,openwebui,pi \
  --json-out /tmp/lucebox_harness_probe.json
```

Add `--install-packages` when you also want the runner to install/smoke the
client packages. Without it, the HTTP protocol probes still run.

For a GPU sweep, let the runner start Lucebox for each profile:

```bash
python3 harness/client_test_runner.py sweep \
  --target dflash/models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft dflash/models/draft \
  --bin dflash/build/test_dflash \
  --profiles rtx3090_dflash_safe,rtx3090_dflash_long \
  --clients all \
  --json-out /tmp/lucebox_harness_sweep.json
```

Use `--isolate-clients` when you want each client tested against a fresh server.

## Compare with llama.cpp

For client-level speed checks, run the same harness once against llama.cpp and
once against Lucebox:

```bash
CLIENT=opencode PROMPT_FILE=harness/clients/prompts/repo_inspection.txt \
  harness/clients/run_backend_pair.sh
```

This keeps the client prompt, tools, and request shape the same for both
backends.

## Files

- `client_test_runner.py`: package smoke tests, protocol probes, and server sweeps
- `clients/`: real client launchers
- `clients/prompts/`: short prompts used by the launchers
- `benchmarks/`: direct generation benchmark against OpenAI-compatible servers
