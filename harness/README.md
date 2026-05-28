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

Use the native C++ server:

```bash
harness/clients/run_codex.sh
```

The native server binary defaults to `server/build/dflash_server`. Override the
paths and profile:

```bash
DFLASH_SERVER_BIN=server/build/dflash_server \
TARGET=server/models/Qwen3.6-27B-Q4_K_M.gguf \
DRAFT=server/models/draft/dflash-draft-3.6-q4_k_m.gguf \
MODEL_ID=luce-dflash \
MAX_CTX=32768 MAX_TOKENS=512 \
BUDGET=22 VERIFY_MODE=ddtree FA_WINDOW=2048 \
harness/clients/run_codex.sh
```

To test an already-running native server:

```bash
server/build/dflash_server server/models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft server/models/draft/dflash-draft-3.6-q4_k_m.gguf \
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

If you already have `dflash_server` running, use `probe`:

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
  --target server/models/Qwen3.6-27B-Q4_K_M.gguf \
  --draft server/models/draft \
  --bin server/build/test_dflash \
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
