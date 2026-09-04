# DS4V-1 baseline for Strix Halo plus RX 7900 XT

This file records the measured asymmetric serving baseline for DeepSeek V4 Flash.
It follows `docs/ds4v-uncensored-vision-plan.md` section DS4V-1.
Every number here is cited from PR 604 or the Lucebox report.
None were measured on the authoring machine. This machine has no GPU.

## Sources

- PR 604 `perf(ds4): accelerate ROCmFPx dual-GPU serving`. Merged to `main` as `491e568`. PR head SHA `027a5f2221509ce497e1f8ba3155f1783a768d93`.
- Lucebox report at `www.lucebox.com/blog/deepseek-v4-asymmetric-parallelism`.

## Hardware

- CPU. Ryzen AI MAX+ 395. This is the Strix Halo package with 128 GiB shared LPDDR5X.
- Discrete GPU. Radeon RX 7900 XT with 20 GiB VRAM.
- Both GPUs sit in one machine. One server process drives both.

## Software

- ROCm 7.2.4.
- One process. Two GPUs.
- gfx1100 is the 7900 XT. gfx1151 is Strix Halo.
- Build flags from PR 604. `DFLASH27B_GPU_BACKEND=hip`. `DFLASH27B_HIP_ARCHITECTURES=gfx1100;gfx1151`. `DFLASH27B_ROCMFP2_AFFINE=ON`. `GGML_HIP_GRAPHS=ON`. `CMAKE_BUILD_TYPE=Release`.
- Build output. `server/build-hip-dual/dflash_server`.

## Models

- Target. `DeepSeek-V4-Flash-ROCMFP2-STRIX.gguf`. ROCmFP2 fixed-codebook target weights.
- Draft. `DeepSeek-V4-Flash-DSpark-draft-Q4RMFP4-denseF16.gguf`. 11.3 GiB. DSpark drafter at Q4RMFP4.
- The GGUF files exist only on the operator machine.
- `scripts/ds4v-baseline.sh launch` records the SHA-256 of both files at start time. The values land in `artifacts/ds4v-1/$RUN_ID/model-sha256.txt`.

## Placement

- The 7900 XT is `hip:0`. Dense work and calibrated hot experts run there.
- Strix Halo is `hip:1`. Remaining experts and the DSpark drafter run there.
- The discrete-GPU expert budget is 10,200 MiB.
- Placement is workload dependent. Capture a routing CSV with `DFLASH_DS4_ROUTING_STATS_OUT`. Serve with it through `DFLASH_DS4_HOTNESS_CSV`. Without a CSV the server still runs, but placement is uncalibrated and the qualified numbers do not apply.

## Qualified numbers

Controlled 512-token decode. Temperature zero. True top-k-6. Fixed DSpark q=4. One warm-up. Three fresh requests. No prefix cache and no prefill cache.

| Run | Decode time | Throughput |
|---|---|---|
| 1 | 11.3772 s | 45.0 tok/s |
| 2 | 10.9023 s | 47.0 tok/s |
| 3 | 10.7302 s | 47.7 tok/s |

- Speculative acceptance was 100 percent on every run.
- All three measured outputs were byte-identical.
- True top-k-6 sparse prefill reached 111.2 tok/s at 132,981 tokens.

Context only. The Lucebox report measured 51.1 tok/s median serving decode on R9700 plus Strix Halo at top-k-4. That is a different discrete GPU and a different top-k. Do not compare it directly with the top-k-6 rows above.

## Launch

`scripts/ds4v-baseline.sh launch` starts `dflash_server` with the exact qualified flags from PR 604.
The script exports this environment, verbatim from the PR 604 profile `server/scripts/serve_ds4_dual_rocm_128k.sh`.

```
DFLASH_DS4_MOE_TP=1
DFLASH_DS4_MOE_TP_INPROC=1
DFLASH_DS4_MOE_TP_GPU=1
DFLASH_EXPERT_BUDGET_MB=10200
DFLASH_DS4_TP_MAIN_TO_PEER_RATE=100
DFLASH_DS4_TP_BALANCE_MIN_HOT=31
DFLASH_DS4_TP_BALANCE_MAX_HOT=50
DFLASH_DS4_TP_CRITICAL_PATH_PLACEMENT=1
DFLASH_DS4_LONG_CONTEXT_CHUNK=2048
DFLASH_DS4_DISABLE_LONG_CONTEXT_ARENA_HANDOFF=1
DFLASH_CUDA_MMQ_FP2_AFFINE_PREFILL_ONLY=1
DFLASH_CUDA_MMQ_FP2_AFFINE_CAPTURE=1
DFLASH_MOE_PREFILL_MASKED_COLD=0
DFLASH_DS4_HYBRID_PREFILL_GPU_HC=1
DFLASH_DS4_HYBRID_PREFILL_EAGER=1
DFLASH_MOE_EXPERT_MAJOR_PINNED_OUTPUT=1
LUCE_MMVQ_MAX_NCOLS=4
DFLASH_HIP_NO_AUTO_UMA=1
DFLASH_DS4_TP_GROUPED_MMVQ=1
DFLASH_MMID_GROUPED=1
DFLASH_MMID_GROUPED_TYPES=15
DFLASH_CUDA_MMVQ_MOE_FP2_PACKED32=1
DFLASH_CUDA_MMVQ_MOE_FP3_PACKED24=1
DFLASH_CUDA_MMVQ_MOE_FP3_PACKED24_DECODE_ONLY=1
DFLASH_CUDA_MMVQ_FP4_X4=1
DFLASH_DS4_TP_MASKED_ROUTES=1
DFLASH_DS4_TP_DEVICE_JOIN=1
DFLASH_DS4_TP_NATIVE_ROUTE_WIDTH=1
DFLASH_DS4_TP_SPLIT_COUNT=1
DFLASH_DS4_TP_ROUTE_PREFORK=1
DFLASH_DS4_TP_DEVICE_JOIN_SPLIT=1
DFLASH_DS4_TP_FUSED_HC_JOIN=1
DFLASH_DS4_TP_MAIN_ROUTE_WEIGHTS=1
DFLASH_DS4_TP_COARSE_OWNER=1
DFLASH_DS4_TP_COARSE_OWNER_SPLIT=0
GGML_BATCH_PEER_COPIES=1
DFLASH_CUDA_MMVQ_MOE_ROWS_PER_BLOCK=2
DFLASH_DS4_TP_CAPTURE_CACHE_SLOTS=4
DFLASH_DS4_TP_FUSED_CACHE_SLOTS=9
DFLASH_DS4_VERIFY_FORCE_GRAPH_REPLAY=1
DFLASH_DS4_GPU_ARGMAX_VERIFY=1
DFLASH_DS4_SPEC=1
DFLASH_DS4_DRAFT=$DRAFT
DFLASH_DS4_DRAFT_GPU=1
DFLASH_DS4_SPEC_Q=4
DFLASH_DS4_PINNED_ROLLBACK=1
DFLASH_DS4_FUSED_VERIFY=1
DFLASH_DS4_TOPK=6
```

The server arguments, verbatim from PR 604, follow.

```
dflash_server "$TARGET" \
    --target-device hip:0 \
    --peer-access \
    --ds4-expert-top-k 6 \
    --ds4-prefill sparse \
    --chunk 2048 \
    --max-ctx 135168 \
    --prefix-cache-slots 0 \
    --prefill-cache-slots 0 \
    --host 127.0.0.1 \
    --port 8216
```

Key settings.

- True top-k-6. `--ds4-expert-top-k 6` with `DFLASH_DS4_TOPK=6`.
- Fixed DSpark q=4. `DFLASH_DS4_SPEC_Q=4` with `DFLASH_DS4_SPEC=1`.
- 135168 context slots. `--max-ctx 135168`. That is 128 KiB prompt plus 4 KiB generation headroom.
- No prefix cache and no prefill cache. `--prefix-cache-slots 0` and `--prefill-cache-slots 0`.

## Proof steps

- `scripts/ds4v-baseline.sh doctor` curls `/props.build` and pretty prints the JSON with jq. Pass when it answers HTTP 200 with the expected image tag.
- `scripts/ds4v-baseline.sh chat-smoke` posts to `/v1/chat/completions`. Pass when it gets HTTP 200 and non empty content.
- `scripts/ds4v-baseline.sh spec-flag` reports whether speculative decode ran. It reads `usage.spec_decode_ran` from the served response. PR 604 added that field.
- `scripts/ds4v-baseline.sh cleanup` removes only its own recorded PID or container. It never kills by process name. Evidence stays under `artifacts/ds4v-1/$RUN_ID`.

Set `LUCEBOX_VERIFY_RUN_ID` before any subcommand so every proof file lands in one named evidence directory.

## Pending on the operator machine

- The ten GPU live lanes and the perf lanes need the Strix Halo plus 7900 XT pair. They stay unchecked until the operator runs them.
- The unit lane `ctest --output-on-failure -R deepseek4_unit` also runs there. The `server/` sources are absent from this sparse checkout.
