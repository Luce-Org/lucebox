<p align="left">
  <a href="../README.md">← lucebox-hub</a>
</p>

<p align="center">
  <img src="assets/hero.png" alt="Lucebox client harness experiments on RTX 3090" width="600" />
</p>

<h1 align="center">Lucebox Harness</h1>

<p align="center">
  <strong>Client launchers and regression tests for Lucebox server compatibility.</strong><br/>
  Run Lucebox from Claude Code, Codex, OpenCode, Hermes, Pi, OpenClaw, or Open WebUI.<br/>
  RTX 3090 / 24 GB defaults are included for each client.
</p>

---

Use this folder to launch Lucebox through a real client, or to test a server
change against the HTTP protocols those clients use. Paths and server settings
can be overridden with environment variables.

## Run a client

Each script starts Lucebox, runs one real client, saves logs, then stops the
server.

```bash
cd /workspace/lucebox-hub-harness

python3 -m harness.client_test_runner bandit --clients codex
harness/clients/run_claude_code.sh
python3 -m harness.client_test_runner bandit --clients opencode
```

Common overrides:

```bash
MAX_CTX=32768 BUDGET=22 VERIFY_MODE=ddtree python3 -m harness.client_test_runner bandit --clients codex
PROMPT_FILE=harness/clients/prompts/repo_inspection.txt python3 -m harness.client_test_runner bandit --clients hermes
CLIENT=opencode harness/clients/run_backend_pair.sh
```

Use the native C++ server instead of the Python server:

```bash
LUCEBOX_SERVER_BACKEND=cpp python3 -m harness.client_test_runner bandit --clients codex
```

The native server binary defaults to `dflash/build/dflash_server`. Override the
paths and profile the same way as the Python backend:

```bash
LUCEBOX_SERVER_BACKEND=cpp \
DFLASH_SERVER_BIN=dflash/build/dflash_server \
TARGET=dflash/models/Qwen3.6-27B-Q4_K_M.gguf \
DRAFT=dflash/models/draft/dflash-draft-3.6-q8_0.gguf \
MODEL_ID=luce-dflash \
MAX_CTX=32768 MAX_TOKENS=512 \
BUDGET=22 VERIFY_MODE=ddtree FA_WINDOW=2048 \
python3 -m harness.client_test_runner bandit --clients codex
```

To test an already-running native server:

```bash
dflash/build/dflash_server dflash/models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft dflash/models/draft/dflash-draft-3.6-q8_0.gguf \
  --host 127.0.0.1 --port 18080 \
  --max-ctx 32768 --max-tokens 512 \
  --fa-window 2048 \
  --ddtree --ddtree-budget 22 \
  --model-name luce-dflash

python3 harness/client_test_runner.py probe \
  --url http://127.0.0.1:18080 \
  --clients all
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
