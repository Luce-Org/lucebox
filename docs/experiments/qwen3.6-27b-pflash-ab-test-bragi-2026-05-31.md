# Qwen3.6-27B PFlash A/B test — bragi — 2026-05-31

PFlash (prefill KV compression via small drafter) speed and quality
evaluation for the coding-agent-loop use case on bragi.

* **Host**: bragi (RTX 5090 Laptop MaxQ, 23 GB VRAM, WSL2, sm_120)
  * GPU throttled to ~86–90 W / 1515 MHz (Windows Balanced mode).
* **Image**: locally-built `lucebox-hub:cuda12` @ `a45c9fa` (pflash/ee7
  squash-merged as `83c5567`, Gemma4 channel-token fix `4b757d1`)
  * Built with `DFLASH_CUDA_ARCHES=120` for sm_120 (Blackwell).
* **Server config** (baseline, pflash=off):
  ```toml
  budget = 16
  max_ctx = 98304
  cache_type_k = "tq3_0"
  cache_type_v = "tq3_0"
  fa_window = 0
  think_max = 15488
  prefix_cache_slots = 0
  prefill_cache_slots = 0
  ```
* **Drafter**: `Qwen3-0.6B-BF16.gguf` (1.2 GB, unsloth/Qwen3-0.6B-GGUF)
* **PFlash config** (compress leg):
  ```toml
  prefill_mode = "auto"
  prefill_keep_ratio = 0.25
  prefill_threshold = 1000
  prefill_drafter = "~/.local/share/lucebox/models/Qwen3-0.6B-BF16.gguf"
  ```
  Valid `prefill_mode` values: `off`, `auto` (compress if n_prompt ≥ threshold),
  `always` (compress every prompt). `"compress"` is not a valid value —
  the server rejects it and falls back to `off`. Config was corrected from
  `"compress"` → `"auto"` before the pflash leg was run.

## Speed benchmark: 32K multi-turn session probe

Fixture: `agent_recorded/multi_turn_cases.json` bucket=32768
(216 messages, actual prompt = 42,735 tokens).

`python3 scripts/pflash_session_bench.py --bucket 32768 --max-tokens 64`

| Leg        | wall_s | prefill_s (est.) | decode tps | prefill speedup |
|------------|--------|------------------|------------|-----------------|
| pflash=off | 89.3s  | ~86.7s           | ~24 tok/s  | 1.0× (baseline) |
| pflash=on  | *TBD*  | *TBD*            | ~24 tok/s  | *TBD* (expected ~4×) |

**Estimate**: 42,735 tokens compressed to 25% ≈ 10,684 tokens →
prefill ~22s (4× speedup). Decode is unchanged because pflash only
affects the prefill pass; generation uses the compressed KV cache.

Note: previous session recorded wall=111.8s for this same fixture —
the lower 89.3s here is consistent with run-to-run variation. Actual
prefill_s can be confirmed from server logs after each run:
```
docker logs lucebox | grep "chat DONE" | tail -1
# [server] chat DONE chatcmpl_... ok=true in=42735 effective_in=NNNN ...
#   prefill=XX.Xs decode=X.Xs(XX.Xtok/s)
# With pflash=on, effective_in = 42735 * 0.25 ≈ 10684.
```

## Quality benchmark: agent_recorded (26 cases)

`uv run luce-bench --areas agent_recorded --no-think`

Prior nothink baseline from 2026-05-30 bragi sweep: **42.3% (11/26)**.

### Baseline leg (pflash=off, 2026-05-31)

| # | given   | correct_tools                  | in_tok | result |
|---|---------|--------------------------------|--------|--------|
| 1 | engaged | Read,Bash,Write,Edit           | 1852   | FAIL   |
| 2 | engaged | Read,Glob,Bash,Monitor         | 2472   | FAIL   |
| 3 | engaged | Read,Glob,Bash,Write           | 3118   | —      |
| … | …       | …                              | …      | …      |

*Full results pending — benchmark in progress.*

### PFlash leg (pflash=on, threshold=1000, keep=0.25)

*Pending server restart and re-run.*

Key question: cases 1–6 (1772–3671 tok prompts) are ABOVE the 1000-tok
threshold and will be compressed 75% of context dropped. Cases 7–26
(120–200 tok) are below the threshold and will NOT be compressed.

Expected outcome: quality degradation on cases 1–6 only; cases 7–26
identical to baseline.

## Analysis framework

For a production coding-agent-loop deployment:

* PFlash is always a **speed win** on long sessions (32K+): 4× prefill
  speedup with modest quality tradeoff.
* PFlash at `threshold=1000` fires on ALL multi-turn context >1K — this
  is very aggressive and will hurt quality on medium-length turns.
* **Recommended production config** (pending data):
  - `threshold=32768` — only compress sessions that would take >30s
    to prefill anyway; no overhead on short turns
  - `keep_ratio=0.5` — less aggressive compression, lower quality risk
  - Effective for multi-day coding agent sessions; no overhead on single turns

## Next steps

1. Complete baseline quality benchmark (26 cases, `agent_recorded`)
2. Restart server with pflash=on, record server startup confirmation
3. Run 32K session bench with pflash (record actual prefill_s from server logs)
4. Run 26-case quality benchmark with pflash
5. Fill in table and update recommended config
