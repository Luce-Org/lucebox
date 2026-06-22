# Best-Config Real-Agentic Benchmark: Dense 27B vs MoE 35B-A3B

Harness: `bench/abc_cache_harness/replay_harness.py` — replay of a real goldgate
multi-turn agentic session (`traces/goldgate_fix.jsonl`), single-server in-RAM prefix
cache, temp=0, smoke mode = first 3 turns. Turn 1 is cold (prefix_len=0); turns 2-3
are cache hits (restore=true). Prompt sizes: turn1 34,135 tok / turn2 34,965 tok /
turn3 37,699 tok. All numbers below are MEASURED; provenance is `file:line` or `jq path`.

Build: git `42278139ce85`, binary SHA `717778c8…7160e047`, branch
`pr/kvflash-moe-prefill-snapshot`.

NOTE on a metadata bug: `BENCH_27B_…_smoke_raw.json` provenance claims `tq3_0` KV, but
the server log it came from (`srv_BENCH_27B_20260622_100403.log:48-49`) shows the binary
actually loaded `cache_type_k = q4_0 / cache_type_v = q4_0`, matching the harness arm def
(`replay_harness.py:435`). Real KV type for the dense run below = **q4_0**.

---

## 1. Best DENSE 27B agentic run

CONFIG (from `replay_harness.py:430-446`, confirmed `srv_BENCH_27B_20260622_100403.log:48-49`):
- Target: Qwen3.6-27B-Q4_K_M (dense)
- Drafter: `dflash-draft-3.6-bf16-reconverted.gguf`, `--draft-swa 2048`
- KV cache: **q4_0 / q4_0**
- max_ctx: 40960
- Env: `DFLASH_FEAT_RING_CAP=4096`, `DFLASH_SPEC_GATE=1`, `DFLASH_DRAFT_CTX_MAX=2048`, temp=0
- Source: `bench/abc_cache_harness/results/BENCH_27B_20260622_100403_smoke_raw.json`
  (mirror: `ctxsweep/agentic_27b_q4kv.log:16-18`)

| turn | restore | prefix_len | fresh_prefill | prefill_s | prefill_tps | decode tok/s | mode | accept% |
|------|---------|-----------|---------------|-----------|-------------|--------------|------|---------|
| 1 (cold) | false | 0 | 34135 | 53.0 | 644.1 | 15.8 | ar | — |
| 2 (warm) | true | 34077 | 888 | 3.6 | 9712.5 | 14.8 | spec | 38.6 |
| 3 (warm) | true | 34077 | 3622 | 12.5 | 3015.9 | 11.5 | ar | — |

Aggregate (`jq .arm_aggregate`): total_wall_s 109.6, mean_prefill_tps 4457.5,
mean_decode_tps 14.03, spec_engagement 0.333, mean_accept 38.6.

### Naive (crippled) dense numbers for contrast
- Cold, no DRAFT_CTX_MAX cap, ring=16384, max_ctx=65536
  (`BENCH_27B_20260622_005722_smoke_raw.json`): turn-1 prefill **384.1s** / **88.9 tps**,
  decode **7.9 tok/s**, spec NEVER engages (0%), total_wall **542.4s**.
- f16 KV variant turn-1 prefill **194s** wall 204.6s (`agentic_f16.log:16`).
- tq3_0 KV variant turn-1 prefill **199.9s** wall 215.5s (`agentic_cap2048.log:16`).
Gap: best-config cold turn-1 (53s / 644 tps) is **~7× faster prefill** than the
uncapped naive arm (384s / 89 tps); warm turn-2 prefill 3.6s vs 18.9s (~5×).

---

## 2. Best MoE 35B-A3B agentic run

CONFIG (from `replay_harness.py:409-423`, log `ctxsweep/agentic_kvflash.log:4-10`):
- Target: Qwen3.6-35B-A3B-UD-Q3_K_XL (MoE, all-hot)
- Drafter: `qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf`, `--draft-swa 2048`
- KV cache: **f16 / f16**, `--kvflash auto`
- max_ctx: 40960
- Env: `DFLASH_FEAT_RING_CAP=65536`, `DFLASH_SPEC_GATE=1`, `DFLASH_DRAFT_CTX_MAX=2048`, temp=0
- Source: `bench/abc_cache_harness/results/BENCH_35B_20260622_101002_smoke_raw.json`
  (mirror: `ctxsweep/agentic_kvflash.log:16-18`, `agentic_f16_40k.log` is the same recipe)

| turn | restore | prefix_len | fresh_prefill | prefill_s | prefill_tps | decode tok/s | mode | accept% |
|------|---------|-----------|---------------|-----------|-------------|--------------|------|---------|
| 1 (cold) | false | 0 | 34135 | 23.4 | 1458.8 | 64.7 | ar | — |
| 2 (warm) | true | 34077 | 888 | 0.8 | 43706.2 | 80.7 | ar | — |
| 3 (warm) | true | 34077 | 3622 | 2.9 | 12999.7 | 55.3 | ar | — |

Aggregate (`jq .arm_aggregate`): total_wall_s 34.8, mean_prefill_tps 19388.2,
mean_decode_tps 66.9, spec_engagement 0.0, mean_accept null.

Spec-decode variant of same recipe (`agentic_f16_40k.log:17`): turn-2 decode 66.4 tok/s
[spec] **25.0% accept**; spec engaged 0.333, total_wall 29.6s. Spec is roughly decode-neutral
on MoE at this ctx (the f16 KV decode is already ~65-80 tok/s autoregressive).

### MoE long-context note
At max_ctx=131072 (`agentic_kvflash_131k.log`): turn-1 prefill 49.7s / 686.8 tps,
decode 20.6 tok/s [spec] 22.3% accept; turn-2 went cache-MISS (prefix_len=0) and the
run FAILED its gate (server closed connection on turn 3). Not a clean warm-cache result.

---

## 3. Side-by-side (each at its BEST measured config)

| metric | DENSE 27B (best) | MoE 35B-A3B (best) |
|--------|------------------|---------------------|
| target / quant | Qwen3.6-27B Q4_K_M | Qwen3.6-35B-A3B Q3_K_XL |
| drafter | dflash-3.6 bf16 reconv | 35b-a3b-dflash-new bf16 reconv |
| KV cache | q4_0 / q4_0 | f16 / f16 (+kvflash auto) |
| max_ctx | 40960 | 40960 |
| cold turn-1 prefill_s / tps | 53.0s / 644.1 | 23.4s / 1458.8 |
| cold turn-1 decode tok/s | 15.8 (ar) | 64.7 (ar) |
| warm turn-2 prefill_s / tps | 3.6s / 9712.5 | 0.8s / 43706.2 |
| warm turn-2 decode tok/s | 14.8 (spec) | 80.7 (ar) |
| warm turn-3 prefill_s / tps | 12.5s / 3015.9 | 2.9s / 12999.7 |
| warm turn-3 decode tok/s | 11.5 (ar) | 55.3 (ar) |
| mean decode tok/s | 14.03 | 66.9 |
| accept% (when spec) | 38.6 | 25.0 (f16_40k spec arm); none in kvflash arm |
| total_wall (3 turns) | 109.6s | 34.8s |
| harness / ctx | replay goldgate, 34-38K prompt | replay goldgate, 34-38K prompt |

---

## 4. Config honesty flags

- **Different KV precision**: dense=q4_0, MoE=f16. The dense arm is NOT measured at f16
  KV; the dense f16/tq3_0 logs (`agentic_27b_f16.log`, no per-turn data — truncated) do
  not give a clean warm-cache dense f16 table. So the dense vs MoE decode gap (14 vs 67
  tok/s) is partly architecture (A3B active-param MoE) and partly KV format. NOT
  like-for-like on KV.
- **Spec engagement differs**: dense's best arm engages spec on turn 2 (38.6%); MoE's best
  (kvflash) is pure AR. The fastest MoE numbers do not even need spec — f16 AR decode
  already ~65-80 tok/s.
- **Both warm turns are real cache hits** (restore=true, 97.5% / 90.4% hit ratio) on both
  models — the comparison is apples-to-apples on the prefix-cache axis.

## 5. What's MISSING for a fully fair dense number

- **No dense 27B agentic run at f16 KV with per-turn data.** `agentic_27b_f16.log` is
  truncated (config header only, no turn rows). To compare dense vs MoE like-for-like on
  KV, re-run the BENCH_27B arm with `kv_cache_types=("f16","f16")` (the only change) on
  build 42278139 and capture the 3-turn replay table.
- **No multi-turn agentic MTP dense run exists.** `wt-mtp-spike/` is an unbenchmarked
  llama.cpp fork; the only MTP data on record is a turn-1-only 0.85-accept artifact that
  crashed on turn 2+ under compression (memory: `project_mtp_pflash_reinit_bug.md`). The
  dflash-drafter arm above is therefore dense's best PROVEN agentic config. A clean
  MTP-vs-dflash dense agentic A/B is missing and would need the MTP re-init bug fixed first.
