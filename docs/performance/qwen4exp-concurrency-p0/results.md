# Qwen4Exp concurrency v1 P0 feasibility

**Latest status (2026-09-25): N=4 @ ctx32768 passes the combined GTT headroom and overflow-isolation gates.** The initial no-go below is retained as history and is superseded by the clean-box follow-up at the end of this report.

Date: 2026-09-25. No production sources were changed and nothing was committed. All GPU probes were serialized behind `/tmp/qwen-perf/gpu.lock`; one IQ4_NL model was resident. The box reported `platform_profile=balanced`, gfx1151 performance level `auto`. `rocm-smi --showclocks --showpower --showtemp --showperflevel` was sampled around the numerical and chunk arms, but the numeric sclk/fclk/power/temp trace files stayed on the box and could not be retrieved after SSH stopped responding. Therefore the required per-arm clock record is incomplete; no clock values are inferred here.

## Decision

**No-go for claiming the full P0 gate passed.** Four per-slot caches and a 16,366-token prefill each fit in separate measured phases, but the combined peak with both the prefill workspace and an initialized decode workspace was not collected. The first phase had 12.39 GiB GTT headroom; the later phase reached 94.12 GB GTT used (8.35 GiB remaining) after cache and decode-workspace allocation but before prefill. That is too close to the 8 GiB requirement to infer the combined case. Width-4 numerics are not bit-identical for all dense operators, so this design needs a documented numeric margin criterion and cannot promise token-for-token parity from the arithmetic probe. The FIFO server does not preempt a prefill: all chunk sizes leave the following decode request waiting for the full prefill. The context-overflow isolation gate is unverified because the probe was interrupted by loss of box connectivity.

## Memory budget

The allocation probe called the production `load_qwen4exp_gguf` and `create_qwen4exp_cache` on the IQ4_NL weights, then allocated four independent F16 caches at 32768. Model load reported 1,222 tensors, 66.24 GiB, three shards. The loaded model plus four caches plus a 16,366-token prefill reached 89,778,561,024 B GTT used out of 103,079,215,104 B (96 GiB): **13,300,654,080 B / 12.39 GiB free**. System `MemAvailable` remained at least 40,479,088,640 B (~37.7 GiB) in that phase. Each cache was allocated and its tensor byte sizes printed by the probe.

| Component per cache | Formula at 32768 | Bytes | MiB |
|---|---:|---:|---:|
| F16 K+V | 12 full-attention layers × 2 KV heads × 256 K + 256 V × 2 B × 32768 | 805,306,368 | 768.000 |
| Indexer K | 12 full-attention layers × ceil(32768/4) × 128 × 4 B | 50,331,648 | 48.000 |
| SSM state | 36 linear layers × 128 × 128 × 48 × 4 B | 113,246,208 | 108.000 |
| Conv state | production cache tensor dimensions | 4,423,680 | 4.219 |
| PLE conv state | 1 PLE layer × (4−1) × 3 × 10240 × 4 B | 368,640 | 0.352 |
| **Total** | | **973,676,544** | **928.570** |
| **Four caches** | | **3,894,706,176** | **3714.281** |

`QWEN4EXP_FA_PAD256` is safe for 32768: it is exactly divisible by 256, so the padded KV capacity remains 32768. Indexer K storage is separate and uses its compression ratio of four; it is included above.

The allocation probe then completed a 16,366-token prefill successfully in 23.635 s with all four caches resident. That run included prefill graph/kernel allocations, but did not initialize a decode workspace. A second probe initialized a decode workspace on another slot; at that checkpoint GTT used was 94,121,095,168 B, leaving 8.35 GiB, before the prefill began. Its subsequent combined peak and overflow result were not recovered. Thus four-cache allocation is proven; the requested >=8 GiB headroom for caches + decode workspace + prefill workspace is not yet proven.

For a rough slot envelope, the first successful run's non-cache resident footprint (weights plus prefill/workspace and other allocations) was about 79.99 GiB. Holding that footprint fixed and reserving 8 GiB gives an estimated cache budget of about 8.01 GiB. This implies upper bounds of 8 slots at 32768, 7 at 40000 (KV padded to 40192), 3 at 96000, and 2 at 131072. These are arithmetic extrapolations, not demonstrated safe slot counts: per-slot decode graph workspaces and allocator fragmentation may reduce them. The second checkpoint shows why the 32768 eight-slot estimate is especially not a v1 capacity promise.

| Context | Padded KV ctx | KV/cache | Indexer/cache | recurrent+PLE/cache | Total/cache | Estimated slots with fixed 79.99 GiB base + 8 GiB reserve |
|---:|---:|---:|---:|---:|---:|---:|
| 32768 | 32768 | 768 MiB | 48 MiB | 112.570 MiB | 928.570 MiB | 8 |
| 40000 | 40192 | 941.97 MiB | 58.875 MiB | 112.570 MiB | 1113.42 MiB | 7 |
| 96000 | 96000 | 2250 MiB | 140.625 MiB | 112.570 MiB | 2503.195 MiB | 3 |
| 131072 | 131072 | 3072 MiB | 192 MiB | 112.570 MiB | 3376.570 MiB | 2 |

## Width-4 numeric falsifier

The probe loaded the real IQ4_NL model, formed a deterministic input row, and compared row 0 computed alone (M=1) with row 0 inside M=4 alongside three decoys. It used the production CUDA/HIP matmul and MoE operators. The model's sampled quantized types were IQ4_NL, Q5_K, Q6_K, Q8_0, BF16, and F32.

| Operator / real tensor | Weight type and shape | max absolute delta, M1 vs row0(M4) | Bitwise? | Decision/selection preserved? |
|---|---|---:|---|---|
| Router `blk.0.ffn_gate_inp.weight` | F32 K=2560, M=512 | 0 | yes | top-10 expert list identical (`56,265,26,312,508,27,378,37,153,322`) |
| Shared-expert gate `blk.0.ffn_gate_shexp.weight` | Q8_0 K=2560, M=640 | 1.9127503e−05 | no | argmax 481 preserved |
| Attention output `blk.3.attn_output.weight` | Q5_K K=6144, M=2560 | 0.00125523657 | no | argmax 2459 preserved |
| Attention Q `blk.3.attn_q.weight` | Q6_K K=2560, M=12288 | 2.03836244e−05 | no | argmax 5633 preserved |
| HC attention down `blk.0.hc_attn_down.weight` | IQ4_NL K=10240, M=320 | 7.59214163e−05 | no | argmax 213 preserved |
| HC inject `blk.0.hc_attn_inject.weight` | BF16 K=10240, M=4 | 9.57995653e−05 | no | argmax 0 preserved |
| Routed experts `blk.0.ffn_gate_exps.weight` (`MUL_MAT_ID`) | IQ4_NL K=2560, M=640, E=512 | 0 | yes | selected top-10 experts identical; output argmax preserved |
| HC normalize/state op | F32 gamma, width 2560, H=4 | 0 | yes | argmax preserved |

This falsifies “width-4 is universally bit-exact.” It does not show an expert-selection change in these samples, but a production batch implementation must track logit margins at every top-k boundary (router and output) and fail/retry or accept only when the measured perturbation is below that margin. The tested rows are representative, not an exhaustive token/context proof; exact greedy parity remains an end-to-end gate.

## Prefill/decode queueing by chunk

Each arm used one current single-sequence server with `--max-ctx 32768`; a 29,922-token prompt (`prompt_tokens=29922`) was submitted, followed 2 s later by a one-token decode request. The HTTP server has one worker that runs each `process_job` synchronously. It does not yield between prefill chunks to service another request.

| Chunk | Prefill time (s) | Enqueued decode → first output (s) | Stream gap median (ms) | p99/max observed gap (ms) |
|---:|---:|---:|---:|---:|
| 512 | 50.1150 | 49.1672 | 36.090 | 37.004 |
| 1024 | 38.3323 | 37.3808 | 35.913 | 36.765 |
| 2048 | 36.4743 | 35.5357 | 35.867 | 36.613 |
| 4096 | 37.2245 | 36.3198 | 35.835 | 36.625 |

Only one arm per chunk was collected, so these are observations, not min-of-N estimates. The measured decode stream itself has p99 gaps around 37 ms, well below 2 s. However, **no chunk meets a requirement that a decode request receive service during an active prefill**: it waits ~35–49 s for the whole prefill. Chunk 2048 had the shortest prefill in these samples and is the provisional throughput choice; meeting an interleaving/service-latency target requires yielding or scheduling between chunks, which is outside this P0 probe and would be a production change.

## Context overflow isolation

The test-only probe snapshots boundary bytes from all K/V, indexer, SSM, conv, and PLE state tensors, plus `cur_pos`, `indexer_blocks`, `ple_prev`, and decode-workspace context. It then calls `qwen4exp_forward` at `pos0=32768,n_tokens=1`, checks rejection and state equality, and finally attempts a valid token at position zero on the same cache. The guard emitted `context overflow: 32768 + 1 >32768`, but the box became unreachable before the probe's final `[overflow] ... isolated=1` result was collected. **Gate: unverified**, not passed. The test source is retained at `probes/memory_probe.cpp` for a rerun.

## Evidence and repeat instructions

Probe sources are under `probes/`. Remote raw logs are `/tmp/q4exp-p0-memory.log`, `/tmp/q4exp-p0-telemetry.log`, `/tmp/q4exp-p0-width*`, and `/tmp/q4exp-p0/chunks/`; copy them into this directory when the box is reachable. Memory/width commands were held under `/tmp/qwen-perf/gpu.lock`; chunk arms likewise used `probes/run_chunk_arm.sh`. The launched server environment was asserted from `/proc/<pid>/environ` by the chunk runner, including `LUCE_HIP_NO_AUTO_UMA=1`. No GPU or source changes were left running intentionally; the final remote probe's process status could not be checked after SSH timed out. Numeric sclk/fclk/power/temp traces and the complete final memory log remain remote and unavailable, which is a reporting limitation rather than a successful telemetry gate.

## Clean-box follow-up — 2026-09-25

This section supersedes the incomplete-memory and overflow entries above. No production code was changed. The test-only probe was rebuilt as a temporary target against the box's older dflash-namespaced checkout; only its temporary CMake target was removed afterward, preserving the box's unrelated pre-existing CMake edits. The memory run and every chunk arm held `/tmp/qwen-perf/gpu.lock`, with one IQ4_NL model resident at a time. Both `LUCE_HIP_NO_AUTO_UMA=1` and the old checkout's `DFLASH_HIP_NO_AUTO_UMA=1` were set. `HIP_VISIBLE_DEVICES=1`, delivered dense/HC flags, `QWEN4EXP_FA_PAD256=1`, and stable T=1 decode reuse were used. Chunk-arm process environments were checked from `/proc/<pid>/environ`.

The box stayed at `platform_profile=balanced`, gfx1151 perf level `auto`, and selected `mclk=1000 MHz`. For the memory run, sclk was 600 MHz / package 18.01 W / edge 40 °C before load and 961 MHz / 14.07 W / 45 °C after teardown; fclk current telemetry was unavailable. The raw per-arm snapshots are copied under `raw/box-2026-09-25/`.

### Combined memory peak and exact component accounting

The probe loaded 66.24 GiB of IQ4_NL weights (1,222 tensors, three shards), allocated all four production F16 caches at ctx32768, initialized a real stable T=1 workspace on slot 1, then ran a 16,366-token prefill on slot 0. The active-run trace sampled the peak while the prefill graph/allocator was live. `QWEN4EXP_FA_PAD256=1` adds no capacity at ctx32768 because 32768 is an exact multiple of 256.

| Resident component at peak | Bytes | GiB | Measurement |
|---|---:|---:|---|
| Idle GTT baseline | 18,636,800 | 0.017 | before model load |
| IQ4_NL model weights, allocated above idle | 71,271,817,216 | 66.363 | GTT after load 71,290,454,016 B; loader reported 66.24 GiB |
| Four full cache tensors | 3,894,706,176 | 3.628 | 4 × 973,676,544 B |
| Four pinned input rings | 5,996,544 | 0.006 | measured per-cache GTT increment was 975,175,680 B, 1,499,136 B above its tensor bytes |
| One persistent stable T=1 decode workspace | 30,117,888 | 0.028 | GTT delta from after four caches to `decode_workspace_ready` |
| Additional live GTT during 16,366-token prefill | 15,074,312,192 | 14.043 | peak trace minus GTT at `decode_workspace_ready`; graph/kernel buffers plus any prefill first-touch/pinning |
| **Combined observed GTT peak** | **90,295,586,816** | **84.094** | sampled during prefill execution |
| GTT capacity | 103,079,215,104 | 96.000 | gfx1151 GTT total |
| **Headroom at combined peak** | **12,783,628,288** | **11.906** | **passes 8 GiB floor by 3.906 GiB** |

The minimum sampled system `MemAvailable` during the combined run was 39,025,790,976 B (36.346 GiB). Prefill completed successfully in 37.328 s. The workspace-ready GTT was 75,221,274,624 B; after prefill teardown it was 83,521,785,856 B, confirming the active peak was transient. The 15,074,312,192 B prefill delta is an exact additional-live-GTT measurement, not an isolated allocator-buffer-size query; it includes graph/kernel temporary buffers and any pages first-touched or pinned during prefill. The prior measured allocation phases are reconciled: each cache's tensor payload is 973,676,544 B (KV 805,306,368; indexer 50,331,648; SSM 113,246,208; conv 4,423,680; PLE 368,640). The extra 1,499,136 B per cache is its pinned two-slot graph-input ring.

The earlier checkpoint was **94,121,095,168 bytes = 94.121 decimal GB = 87.657 GiB**, not 94.12 GiB. Against the matching clean workspace-ready checkpoint of 75,221,274,624 B, it had an extra 18,899,820,544 B (17.602 GiB). Its apparent ~24 GiB unexplained delta came from mixing decimal GB and GiB and from the known second-model co-load. The clean run accounts for the entire measured working set above; the old trace did not record the foreign model's process-level GTT allocation, so its exact historical per-process split cannot be recovered. Crucially, the decode workspace is only 30,117,888 B, not tens of GiB.

### Shared versus per-slot decode workspace and capacity envelope

For an exact-width batched graph, one graph arena/allocator workspace is shared by the operation; the four caches still own independent K/V, indexer, SSM, conv, and PLE state. Per-slot scratch is not a correctness requirement. The existing single-stream implementation stores a T=1 workspace in each cache; its measured cost is 30,117,888 B per slot. Four independent T=1 workspaces would total 120,471,552 B, only 90,353,664 B more than one shared workspace, and the projected four-slot headroom would still be 11.822 GiB. A true M=4 graph workspace has not been implemented or measured yet; this is a small-workspace extrapolation, and the engine must recheck its allocator peak.

With the measured non-slot footprint fixed at 86,394,884,096 B (weights, idle baseline, one decode workspace, and the prefill transient), one shared workspace, the 8 GiB reserve, and 975,175,680 B per slot (cache tensors plus input ring), the arithmetic max is **N=8 at ctx32768**; N=9 would breach the reserve. Four slots at ctx32768 pass with the observed 11.906 GiB headroom. At N=4, the fixed-workspace, linear-cache extrapolation reaches a 256-aligned ctx of **72,704** before headroom falls below 8 GiB. This is a memory-budget ceiling, not a tested compute/QSA limit; ctx32768 is the proven point. With four separate T=1 workspaces the slot maximum remains N=8 and the ctx ceiling is slightly lower, at 71,936.

| Workspace layout | Estimated max N @ ctx32768 | Estimated ctx ceiling @ N=4 | Headroom at N=4, ctx32768 |
|---|---:|---:|---:|
| One shared graph workspace | **8** | **72,704** | **11.906 GiB measured** |
| One 30,117,888 B T=1 workspace per slot | **8** | **71,936** | **11.822 GiB extrapolated** |

### Overflow isolation

The real cache boundary snapshot hashed the first and last 64 bytes of every K/V, indexer-K, SSM, conv, and PLE state tensor and compared `cur_pos`, `indexer_blocks`, `ple_prev`, and decode-workspace context. At `pos0=32768,n_tokens=1`, `qwen4exp_forward` rejected with `32768 + 1 >32768`; before/after state hash was `7dbb4b8cf05c1183`, metadata was unchanged, and the subsequent valid token at `pos0=0` succeeded with 248,320 logits. **isolated=yes**.

### Fresh chunk-size rerun

One fresh server run per chunk size was completed; min-of-3 was not cheap because each sample reloads ~66 GiB of weights and runs a ~30k-token prefill. Each prompt was 29,922 tokens; a 64-token decode request was enqueued two seconds after the prefill. The single-worker server still queues it until prefill finishes, so first-output wait is not the interleaved-engine latency. Decode stream p99 is measured after service begins.

| Chunk | Prefill wall (s) | Queued decode → first output (s) | Decode stream gap median (ms) | p99/max (ms) | Mean chunk time (prefill/chunks) |
|---:|---:|---:|---:|---:|---:|
| 512 | 58.155 | 57.185 | 35.210 | 41.025 | ~0.99 s (59 chunks) |
| 1024 | 51.634 | 50.649 | 34.436 | 34.994 | ~1.72 s (30 chunks) |
| 2048 | 51.916 | 50.946 | 34.349 | 34.856 | ~3.46 s (15 chunks) |
| 4096 | 53.522 | 52.544 | 34.126 | 35.497 | ~6.69 s (8 chunks) |

Recommend **chunk 512 provisionally** for the interleaving engine: it is the only tested size with an average prefill quantum near 1 s, leaving margin against the requested 2 s p99 service-gap target. It costs 6.52 s (12.6%) versus the fastest single sample at chunk1024. The decode-stream gap numbers above do not test service at chunk boundaries; measure chunk-latency p99 once the new engine yields between chunks before treating the 2 s SLA as passed. Power state stayed `balanced/auto`; sclk before→after was 600→2695 MHz (512), 754→2723 (1024), 856→2724 (2048), and 835→2624 (4096); mclk was 1000 MHz; fclk current telemetry was unavailable. Samples were sequential and temperatures rose from 41→61 °C through 68→76 °C, so subsecond differences between 1024/2048 are not significant.

All raw `/tmp/q4exp-p0*` artifacts were copied to `raw/box-2026-09-25/`. The memory GTT trace is in `raw/box-2026-09-25/q4exp-p0/memory-trace.csv`; phase output, overflow check, and power snapshots are alongside it. Chunk JSON, server logs, `/proc` environment checks, and before/during/after power traces are in the `chunk-rerun-20260925/c{512,1024,2048,4096}/` directories.

**Follow-up verdict:** memory fit and per-request overflow isolation pass for N=4, ctx32768, with >8 GiB GTT headroom; the prior high checkpoint was a co-load artifact plus a GB/GiB labeling error. Use one shared workspace for the exact-width batch graph. The earlier N=4 versus N=1 numeric probe still found nonzero dense deltas, so this is a memory/overflow **go**; exact bitwise batched parity is still not established. Chunk512 is a provisional service-latency choice pending the batched engine's chunk-boundary p99 measurement.
