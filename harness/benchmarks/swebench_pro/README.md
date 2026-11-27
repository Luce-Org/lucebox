# SWE-bench Pro via Harbor

SWE-bench Pro is natively available in [Harbor](https://github.com/laude-institute/harbor):

- **Harbor package registry (V1, 731 tasks):** `scale-ai/swe-bench-pro` (and anti-exploitation variant `cais/swebenchpro`).
- **Official V2 task tree (642 validated tasks + `HARD-51`):** `https://github.com/scaleapi/SWE-bench_Pro-os` (`v2/tasks/`, prebuilt `ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>` images, 642/642 oracle pass, 0/642 nop pass, `[agent] network_mode = "no-network"`).

## Curated 24-task long-horizon slice (`long_horizon_24.txt`)

Designed for benchmarking long-context multi-turn coding workloads (e.g., KVFlash + PFlash on `Qwen3.8-27B`):

- **24 tasks** (divisible by 1, 2, 3, 4, 6, 8 for concurrency sweeps).
- **Balanced across languages:** 8 Python (`ansible`, `openlibrary`, `qutebrowser`), 8 Go (`flipt`, `vuls`, `teleport`, `navidrome`), 8 JS/TS (`element-web`, `webclients`, `tutanota`).
- **14 multi-file standard V2 tasks + 10 HARD-51 tasks** (1 per repository) to preserve solve-rate signal on ~27B models while still stressing deep context accumulation.
- Every ID is in V2's 642-task release gate and matches exact-case in `scale-ai/swe-bench-pro`.

## Quick start

### 1. Default agent (`pi` 1.5.2, matching Terminal-Bench 4.0 `mix-pi.yaml`)

Defaults match Lucebox's Terminal-Bench 4.0 `pi` setup (`pi` `1.5.2`, `openai_chat_completions`, SSE `stream=true` + native `read`/`bash`/`edit`/`write` tool calls). Streaming avoids Harbor's `gost` egress-control sidecar 15s idle read timeout on SWE-bench Pro V2's `no-network` tasks (which breaks non-streaming `mini-swe-agent` turns longer than 15s). Start `luce_server` with `--host 0.0.0.0`:

```bash
# Terminal 1: start luce_server listening on the Docker bridge
server/build/luce_server /path/to/Qwen3.8-27B.gguf \
  --host 0.0.0.0 --port 8000 --model-name Qwen3.8-27B

# Terminal 2: dry-run check, then run the 24-task slice
DRY_RUN=1 harness/benchmarks/swebench_pro/run_swebench_pro.sh
SERVER_HOST=172.17.0.1 CONCURRENCY=1 harness/benchmarks/swebench_pro/run_swebench_pro.sh
```

### 2. Other agents (`mini-swe-agent` 2.2.4, `opencode`, `claude-code`, `codex`, `terminus-2`)

`AGENT=mini-swe-agent` pins `2.2.4` with `mini-swe-agent.yaml` (`cost_limit: 0`, `step_limit: 0`, matching TB4's `mix-mini.yaml`; recommended only on V1 `public`-network tasks since it sends `stream=false`). Host-side `terminus-2` reaches `127.0.0.1` directly without container networking:

```bash
AGENT=mini-swe-agent CONCURRENCY=1 harness/benchmarks/swebench_pro/run_swebench_pro.sh
AGENT=opencode       CONCURRENCY=4 harness/benchmarks/swebench_pro/run_swebench_pro.sh
AGENT=claude-code    CONCURRENCY=4 harness/benchmarks/swebench_pro/run_swebench_pro.sh
AGENT=terminus-2     CONCURRENCY=4 harness/benchmarks/swebench_pro/run_swebench_pro.sh
```

### 3. Using SWE-bench Pro V2 (prebuilt images, no Docker build step)

```bash
git clone --depth 1 https://github.com/scaleapi/SWE-bench_Pro-os /tmp/SWE-bench_Pro-os
TASKS_PATH=/tmp/SWE-bench_Pro-os/v2/tasks \
  CONCURRENCY=4 \
  harness/benchmarks/swebench_pro/run_swebench_pro.sh
```

V2 tasks enforce `[agent] network_mode = "no-network"`. With the default host-side `terminus-2` agent no container network is needed; for in-container agents against V2 on Linux Docker, pass the Docker host bridge IP explicitly (e.g. `SERVER_HOST=172.17.0.1`) so Harbor's egress sidecar allowlists it directly.

### 4. Verifier smoke test (`oracle` / `nop`)

```bash
AGENT=oracle LIMIT=1 harness/benchmarks/swebench_pro/run_swebench_pro.sh
```

## Overrides

| Variable | Default | Notes |
| --- | --- | --- |
| `DATASET` | `scale-ai/swe-bench-pro` | Harbor registry dataset (ignored when `TASKS_PATH` is set) |
| `TASKS_PATH` | *(unset)* | Local Harbor task directory (e.g. `SWE-bench_Pro-os/v2/tasks`) |
| `SLICE_FILE` | `long_horizon_24.txt` | Task ID list, or `all` for the full dataset |
| `AGENT` | `pi` | `pi`, `mini-swe-agent`, `terminus-2`, `opencode`, `claude-code`, `codex`, `hermes`, `oracle`, `nop` |
| `AGENT_VERSION` | `1.5.2` (`pi`) / `2.2.4` (`mini-swe-agent`) | Pinned TB4 agent version |
| `MINI_SWE_CONFIG` | `mini-swe-agent.yaml` | `cost_limit: 0`, `step_limit: 0` config passed to `mini-swe-agent` |
| `MODEL` | `Qwen3.8-27B` | Exposed model name on `luce_server` (`--model-name`) |
| `SERVER_HOST` | `127.0.0.1` | Auto-mapped to `host.docker.internal` for in-container agents |
| `SERVER_PORT` | `8000` | `luce_server` port |
| `CONCURRENCY` | `1` | Passed to `harbor run -n` |
| `LIMIT` | *(unset)* | Optional cap passed to `harbor run -l` |
| `DRY_RUN` | `0` | Set to `1` to validate config and task resolution without running |
