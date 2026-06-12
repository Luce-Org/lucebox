# KVFlash — measured results

All numbers: single RTX 3090 (24 GB), Qwen3.6-27B Q4_K_M target, Q8_0 KV,
Qwen3-0.6B pflash drafter as the scorer. June 2026, `test_kvflash` +
`dflash_server` + `harness/benchmarks`.

## End-to-end long-prompt A/B (`--longab`; needle depth 0.25, 240-token timed free run)

| context | mode | prefill | decode tok/s | needle /16 | KV in VRAM |
|---|---|---|---|---|---|
| 32K  | full    | 47.2 s  | 32.8 | 16 | 576 MiB |
| 32K  | KVFlash 4K | 41.8 s | 29.0 | 15 | 72 MiB |
| 64K  | full    | 130.6 s | 27.8 | 16 | 1152 MiB |
| 64K  | KVFlash 4K | 87.5 s | **38.6** | 14 | **72 MiB** |
| 128K | full    | 335.9 s | 19.6 | 16 | 2304 MiB |
| 128K | KVFlash 4K | 177.8 s | **38.6** | 14 | **72 MiB** |
| 256K | full    | 999.0 s | 13.1 | 16 | 4608 MiB |
| 256K | KVFlash 4K | **354.9 s** | **38.6** | 15 | **72 MiB** |

Decode is flat at 38.6 tok/s from 64K to native-max 256K (speedups 1.4x /
2.0x / 2.9x); prefill speedups 1.5x / 1.9x / 2.8x. One drafter rescore per
query: 9-70 s scaling with context (bisected above the drafter's ~65K
single-pass ceiling).

## Retrieval quality vs residency (synthetic NIAH, teacher-forced /16)

| context | residency | LRU (d=10/50/90%) | drafter (d=10/50/90%) | full control |
|---|---|---|---|---|
| 8K   | 25%   | 0 / 0 / 16 | 15 / 15 / 16 | 16/16 |
| 8K   | 9%    | 0 / 0 / 0  | 15 / 15 / 15 | 16/16 |
| 32K  | 25%   | 0 / 0 / 16 | 15 / 15 / 16 | 16/16 |
| 32K  | 9%    | 0 / 0 / 0  | 15 / 15 / 15 | 15-16/16 |
| 256K | 6.25% | 0 (d=0.5); 16/16 in-window | 14 / 15 / 15 | (in-window LRU = control) |

Drafter-scored residency retains 88-100% of perfect recall at every depth
down to 6-9% residency; recency-only LRU retains zero outside its tail
window (mirrors FlashMemory's Recency-Only ablation).

## Harness ground truth (pool sized per the heuristic, vs full cache)

| suite | baseline pass | KVFlash pass | exact text match |
|---|---|---|---|
| HumanEval | 10/10 | **10/10** | 10/10 |
| GSM       | 10/10 | **10/10** | 8/10 |
| MATH      | 10/10 | **10/10** | 4/10 |
| agent (to 24K prompts) | 6/6 | **6/6** | 2/6 |

Base-vs-base control: 16/16 byte-identical — the stack is deterministic.
Text drift under KVFlash is the masked decode kernel's different (equally
deterministic) rounding lineage, not noise and not a correctness effect.

## Spec decode (chain mode, slot-mapped verify, daemon)

| config | accept rate | avg_commit | output |
|---|---|---|---|
| full cache, 2400 tok | 15.3% | 3.45 | coherent |
| KVFlash 2K, 1800 tok | 15.4% | 3.47 | coherent |
| KVFlash 2K, 2400 tok (live eviction mid-spec) | 15.6% | 3.49 | coherent |

## Microbenchmarks

- Memory at 128K-logical: attn-KV 2304 -> 18 MiB (99.2%) with a 1K pool;
  whole cache buffer 2654 -> 218 MiB, confirmed via VRAM deltas.
- Exact slot mask is free: 25.10 ms/step masked vs 25.52 maskless.
- Paging: page_out p50 1.27 ms / page_in 0.64 ms per 64-token chunk
  (~2.2 MiB, synchronous); ~0.01 ms/token amortized at observed rates.
- reselect() repaging 20 chunks: 21.3 ms.
- Relocation equivalence (shuffled physical placement, teacher-forced
  1200 tokens): ~99% argmax agreement; page_out/page_in roundtrip
  bit-exact.

## Multi-architecture smokes (pool 1024, --max-ctx 8192, ~1235 logical tokens, live LRU eviction mid-request, RTX 3090)

| arch | model | mode | decode tok/s | output |
|---|---|---|---|---|
| qwen35 | Qwen3.6-27B Q4_K_M | all-GPU, masked pool | 37.4 | coherent |
| qwen35moe | Qwen3.6-35B-A3B UD-Q4_K_M | Spark hybrid (9403 hot / 837 cold experts), pipelined decode | 101.6 | coherent |
| laguna | Laguna-XS.2 Q4_K_M | Spark hybrid, single-graph decode, slot-space full+SWA masks | 137.1 | coherent |
| gemma4 | Gemma4 26B-A4B UD-Q4_K_M | all-GPU, slot-space full mask, SWA rings untouched | 119.0 | coherent |

Gemma4 control on the same build without the flag: 120.2 tok/s, no
kvflash code engaged — the default path is unchanged.

## Known limits

- DDTree tree-verify is not pool-aware (falls back to AR with KVFlash).
- Post-generation snapshots are skipped once cur_pos exceeds the pool
  (pooled snapshots need page-table serialization).
- Paging is synchronous (copy-stream overlap is a follow-up).
- Memory-dense tasks needing the entire context at once (MRCR-style) are
  a paradigm limit shared with FlashMemory; size the pool up for those.
- 512K+ requires RoPE scaling (model native max is 256K) — memory-side
  KVFlash already scales (host backing is the only growth).
