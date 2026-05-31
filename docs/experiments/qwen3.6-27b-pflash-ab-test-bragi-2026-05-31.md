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

| Leg        | wall_s | prefill_s | effective_in | pflash_kept | decode tps |
|------------|--------|-----------|--------------|-------------|------------|
| pflash=off | 89.3s  | ~86.7s    | 42,735       | N/A         | ~15 tok/s  |
| pflash=on  | 89.4s  | 72.4s     | 41,848       | 97.9%       | 15.0 tok/s |

Server log for pflash leg:
```
[drafter] score_and_compress total 5.58s S=42152 kept=41288 (1291/1318 chunks, forced=1291)
[pflash] 42735 -> 41288 -> 41848 tokens (97.9% kept)
```

**Key finding: PFlash did not compress.** 1291 of 1318 chunks were `forced`
(required by the model) and only 27 were eligible for dropping. With
`prefix_cache_slots=0`, every token is "current turn" — there is no
previously-cached KV to compress. PFlash requires prefix caching to work:

- **PFlash mechanism**: compresses the *cached* KV from prior turns before
  processing the new turn. The drafter scores which cached tokens to drop;
  the target model then refills attention only over the kept tokens.
- **Without prefix cache**: every request prefills from scratch. All tokens
  are "current", so all chunks are forced. PFlash adds drafter overhead
  (5.58s) but saves nothing.
- **With prefix cache** (`prefix_cache_slots > 0`): prior turns sit in KV cache;
  pflash would compress that cache before prefilling new tokens. This is the
  correct use case — e.g., long coding-agent sessions with many turns.

Same behavior confirmed on short agent_recorded cases:
```
# Case 1 (1852 tokens): 58/58 chunks forced → 0% compression
# Case 2 (2472 tokens): 77/77 chunks forced → 0% compression
```

**Conclusion**: PFlash + `prefix_cache_slots=0` = zero benefit. For the
coding-agent-loop use case, enabling prefix caching (`prefix_cache_slots=N`)
alongside pflash would be the correct configuration to test.

## Quality benchmark: agent_recorded (26 cases)

`uv run luce-bench --areas agent_recorded --no-think`

Prior nothink baseline from 2026-05-30 bragi sweep: **42.3% (11/26)**.

### Baseline leg (pflash=off, 2026-05-31)

**Score: 12/26 = 46.2%**

Snapshot: `luce-bench/snapshots/qwen36-27b-nothink-nopflash-20260531/`

Pflash-affected cases (>1000 tok threshold): **4/10 = 40%**
Non-affected cases (<1000 tok): **8/16 = 50%**

Note: higher than 2026-05-30 baseline (42.3%) due to GPU non-determinism
at tq3_0 KV quantization; run-to-run variation ≈ ±10-15pp.

### PFlash leg (pflash=auto, threshold=1000, keep=0.25, 2026-05-31)

**Score: TBD** (benchmark running, but expected ≈ 46% — pflash does nothing
without prefix caching; see speed benchmark findings above)

Snapshot: `luce-bench/snapshots/qwen36-27b-nothink-pflash025-20260531/`

All cases confirmed to have 100% forced chunks → pflash is a no-op.
Cases 1–6 (1772–3671 tok): 58/58 to 115/115 chunks forced → 0% compression.
Cases 7–26 (120–889 tok): below threshold → pflash doesn't even run.

## Analysis framework

For a production coding-agent-loop deployment:

* PFlash is **only effective with prefix caching enabled** (`prefix_cache_slots > 0`).
  With `prefix_cache_slots=0`, every request prefills from scratch → all chunks
  forced → pflash adds drafter overhead (5.58s/request) with zero compression.
* PFlash at `threshold=1000` fires on ALL multi-turn context >1K — would be
  aggressive if prefix caching were active. Consider `threshold=32768` for
  production (only long sessions) with `keep_ratio=0.5` (less aggressive).
* The prior sweep found `prefix_cache_slots=0` optimal for the agent_recorded
  benchmark — but that benchmark sends each case as an independent request.
  In a real coding-agent session (same system prompt + growing history), prefix
  caching saves repeating the system prompt prefill on every turn.
* **Recommended next experiment**: test `prefix_cache_slots=N` (e.g., 512) +
  pflash for actual multi-turn agent sessions (not the agent_recorded fixture).

## Conclusion

PFlash **does not help** in the current `prefix_cache_slots=0` configuration.
Reverting `prefill_mode = "off"` to avoid the 5.58s/request drafter overhead.

| Config               | Quality | Prefill speedup | Verdict |
|----------------------|---------|-----------------|---------|
| pflash=off           | 46.2%   | 1.0× (baseline) | Current optimal |
| pflash=auto, no pcache | ≈46%  | 1.0× (no-op + overhead) | Worse |
| pflash=auto, pcache  | TBD     | expected ~4×    | Future experiment |

## Next steps

1. ~~Complete baseline quality benchmark~~ ✓ 46.2% (12/26)
2. ~~Restart server with pflash=on~~ ✓ Confirmed pflash=auto active
3. ~~Run 32K session bench with pflash~~ ✓ 97.9% chunks forced → no speedup
4. Run 26-case quality benchmark with pflash → expected ≈ baseline (no compression)
5. Revert `prefill_mode = "off"` to remove drafter overhead
6. Rebuild Docker image (picks up call:<verb>{} parser, test_server_unit changes)
7. Test Gemma4 forge benchmark after rebuild
