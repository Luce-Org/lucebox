# IQ4_NL prefill results, 2026-09-20

## Final result

**1002.20 t/s**, minimum TTFT **16.33 s** of eight measured 16,366-token
requests. Median **993.99 t/s**; slowest **989.48 t/s**. The existing benchmark's
“min-of-8” means minimum latency / best throughput. This does **not** establish
a minimum-throughput floor above 1000 t/s. Timings are rounded to 0.01 s.

Final TTFTs: `16.49, 16.50, 16.48, 16.45, 16.54, 16.37, 16.33, 16.34` seconds.
One warmup was excluded after the full quality suite in the same server process.
No measured requests were discarded. Raw results: [timings](qwen4exp-prefill-delivery-summary.json),
[quality](qwen4exp-prefill-delivery-quality.json).

Quality: **HE 10/10, GSM 10/10, Math 9/10, recall 2/2**. Every previously passing
case still passes; `math_10` remains the same 2048-token-cap failure.
Reference profiles: **GSQ and IQ4_NL pass**, at sequence length 16, with all
**7,530/7,530 expert IDs** matching per model. Raw bytes match at `L00.gnorm`,
`L03.faout`, `L03.att`, and final `logits` (393216, 393216, 163840, and 993280
bytes respectively). Both harnesses exit 0; no scalar-divergence onset. This
proves exactness at these checkpoints, not an exhaustive comparison of every
intermediate tensor or context length. Logs: [GSQ](qwen4exp-reference-gsq.log),
[IQ4_NL](qwen4exp-reference-iq4.log).

Retained changes: bit-preserving RMS+SCALE fusion, default enabled on eligible
RDNA3.5 F32 width-128 rows, and opt-in `QWEN4EXP_LAST_TOKEN_FFN=1` to select the
final output row before the last layer's HC/FFN. Attention/cache updates precede
the selection. It remains opt-in because matmul dispatch/rounding change.
Only IQ4_NL's fast profile received the quality suite; GSQ's fast profile is
not certified. Nine RMS bitwise unit cases pass after the final rebuild.

The retained configuration uses the original QSA/CUBLAS5/shadow1/HC16=2 settings,
`GGML_CUDA_MMB=1`, `QWEN4EXP_RMS_SCALE_FUSED=1`,
`QWEN4EXP_LAST_TOKEN_FFN=1`, `GGML_DS4_INDEXER_M32_CACHE_B=0`, and
`--max-ctx 40000 --chunk 16384`. Other parity/experimental flags remain unset.
Set `QWEN4EXP_RMS_SCALE_FUSED=0` for the original two-kernel normalization.

No retained changes to MMB, concat, indexer, FA dispatch, or KV padding. The
default already used RMS+SCALE before this task; fusion removes one launch.
Reference-only padding, F32 RoPE, unfused HC/MoE and vector-dot dispatch remain
opt-in. No new reference-profile work is added to the fast path. Other GPU
architectures retain their existing RMS dispatch.

## Measurement method and experimental notebook

Each A/B table compares arms in its own run/binary; do not compare results
across sessions. Thermal/clock variation remains larger than the narrow final
margin above 1000 t/s. The notebook below retains failed and rejected trials
as well as successful ones; its chronological candidate descriptions are not
the final source configuration.

Target: 16,366 prompt tokens, >1000 t/s, preserve quality and reference parity. No commits.

Decisions / scratchpad:
- All builds and GPU runs on lucebox4 via SSH, builds -j4, GPU runs serialized.
- Remote HEAD 5812e8b0 has preexisting uncommitted parity work. Local HEAD 7afd47a1. Do not reset either tree.
- HEAD source files match remotely except fattn.cu has extra opt-in debug overrides; preserve those.
- Installed ROCm reports 7.2.2, not 10.
- GDN default is already RMS_NORM + SCALE; legacy L2 is opt-in. Fusing must preserve intermediate F32 rounding, not merely real-number algebra.
- Baseline/legacy diagnostic uses A/B/B/A, four samples after one warmup per load, eight measured samples per variant. Report minimum, median, best throughput, plus minimum latency. Legacy is diagnostic only, never a parity-safe candidate.
- Experiment scripts/logs: remote /tmp/qwen-perf-ab.py and /tmp/qwen-perf/. Source edits mirrored individually, existing remote files backed up before editing.

Initial diagnostic (A/B/B/A, 8 samples each, unchanged build):

| Variant | Minimum t/s | Median t/s | Best t/s | Minimum TTFT s |
|---|---:|---:|---:|---:|
| Shipped baseline | 915.84 | 979.42 | 988.29 | 16.56 |
| Legacy GDN L2 (diagnostic only) | 958.20 | 979.12 | 989.48 | 16.54 |

The min-of-8 convention in `/tmp/bench_model.sh` means minimum latency, hence **best** throughput. Minimum throughput is separately retained above. Warmup samples excluded; first measured samples still show post-load ramp effects.

Initial candidate changes (at the time, all opt-in):
- `QWEN4EXP_RMS_SCALE_FUSED=1`: fuse 128-wide F32 RMS_NORM + SCALE on RDNA3.5. Preserve the reduction tree and intermediate F32 product. Unit differential passes contiguous/strided cases at 1, 17, 1024, 16366 tokens.
- `QWEN4EXP_LAST_TOKEN_FFN=1`: reuse upstream's final-row selection before final HC/FFN; all attention/cache writes remain before selection. Changes GEMM dispatch, so requires quality gate.
- `QWEN4EXP_CONCAT_ROWS16=1`: 16-row transpose tile versus existing 8-row tile on RDNA3.5.

Code inspection:
- No separate ubatch knob in Qwen4ExpBackend: `cfg_.chunk` directly bounds each call to qwen4exp_forward. At 16,366 tokens, chunk 24,576 and 16,384 submit the identical graph.
- MMQ tile selection already uses `args.ncols_max`, and best-config routed IQ4_NL uses MMB; do not blindly port the same change twice.
- rocWMMA 2 is installed. DS4 indexer already enables WMMA and M32 prefill on this device; upstream lightning-indexer is a different op. Its claimed default-off status does not describe this qwen4exp graph.
- GDN/PLE direct-convolution fusion can bypass transpose-concat, limiting the 16-row tile's impact.

Progress:
- The kernel ladder reached minimum TTFT 16.35 s (1000.979 t/s) for both fused RMS + last-token FFN, and that combination + concat16. Concat16 has no demonstrated gain and should not be retained absent contrary final evidence.
- Additional test candidate: RMS scale mode `2`, four independent rows per workgroup, each wave reproduces the original four segment sums and lane-zero second-stage tree. Intended to remove synchronization without changing bits. Must pass unit test and reference binary harness before use.
- Additional dense-FA candidate: `QWEN4EXP_DENSE_FA256=1`, restrict selection to non-sparse prefill with >=128 queries. Pad dense masks and physical KV capacity; retain QSA and single-token decode behavior. Pending measurement.

Completed kernel ladder (A/B/C/D/D/C/B/A; 8 samples per variant):

| Variant | Minimum t/s | Median t/s | Best t/s | Minimum TTFT s |
|---|---:|---:|---:|---:|
| Baseline | 929.36 | 967.08 | 984.13 | 16.63 |
| RMS fusion | 930.94 | 983.25 | 991.28 | 16.51 |
| RMS fusion + last FFN | 966.12 | 987.99 | 1000.98 | 16.35 |
| Above + concat16 | 934.13 | 988.88 | 1000.98 | 16.35 |

Rejected/removed:
- Concat16: no improvement over final-row candidate, removed from source.
- Warp-per-row RMS prototype: bitwise failures at longer inputs. Two targeted compiler-expression checks did not resolve them. Removed; no performance claim for this prototype. Logs `/tmp/qwen-perf/rms-warp-test*.log`.
- Retained block-based RMS fusion re-passed all nine tests after cleanup, including a nonstandard 3-head input. `/tmp/qwen-perf/rms-final-test.log`.

Completed sweep setup (remote, single GPU lock): `/tmp/qwen-perf-ab.py mode-sweep`; baseline is RMS fusion + last FFN, with single-variable candidates chunk8192, cublas3, cublas1, shadow2, hc0, GDN grouped, GDN scalar, QSA unset (dense), dense + restricted MMA, QSA + restricted MMA gate, indexer CACHE_B=1. Four measured requests after one warmup per load, all variants followed by reverse order. Results `/tmp/qwen-perf/mode-sweep.json`.

Dense MMA rejection / sweep interruption:
- With QSA unset and `QWEN4EXP_DENSE_FA256=1`, the first warmup completed at 28.93 s, then the second request failed with `ROCm error: unspecified launch failure` at `hipStreamSynchronize`. No valid measured MMA sample was recorded.
- Runner stopped and terminated its server. No stray inference processes remained. Fresh RMS unit run passed all nine cases; no host/GPU restart was performed. Kernel logs were unavailable from the unprivileged session.
- Removed the new dense-only gate from local/remote source; preserved preexisting FA256 opt-in selectors and remote debug overrides. The source cleanup was built after the mode sweep, so all mode comparisons used the same binary.
- Resumed the existing mode-sweep records (no samples discarded from safe arms): indexer cache twice, then dense/GDN/HC/shadow/GEMM/chunk/control in reverse. The failed MMA arm and its QSA companion are omitted. This interruption is disclosed; the uninterrupted kernel ladder remains the primary paired performance evidence.

Completed mode sweep (8 measured requests per safe arm; interruption disclosed above):

| Variant | Min t/s | Median t/s | Best t/s | Min TTFT s |
|---|---:|---:|---:|---:|
| best | 952.1 | 969.0 | 1009.6 | 16.21 |
| chunk8192 | 966.7 | 979.4 | 983.5 | 16.64 |
| cublas3 | 887.5 | 908.7 | 931.5 | 17.57 |
| cublas1 | 845.4 | 868.9 | 890.9 | 18.37 |
| shadow2 | 672.9 | 690.3 | 698.2 | 23.44 |
| hc0 | 880.8 | 922.3 | 930.4 | 17.59 |
| gdn_grouped | 925.2 | 956.2 | 973.0 | 16.82 |
| gdn_scalar | 606.6 | 830.6 | 838.9 | 19.51 |
| dense | 577.7 | 623.7 | 630.4 | 25.96 |
| index_cache | 1008.4 | 1009.9 | 1015.9 | 16.11 |

The final control block was slower than its opening block. Do not interpret the small index-cache delta causally from this long sweep; a fresh cache-off/on A/B/B/A is required. Index-cache alone recorded all eight samples above 1000 t/s.

Fresh indexer-cache A/B/B/A (cleaned binary, two warmups/load, eight samples/arm):

| Variant | Min t/s | Median t/s | Best t/s |
|---|---:|---:|---:|
| cache off | 981.8 | 990.4 | 996.7 |
| cache on | 961.6 | 969.3 | 984.1 |

The earlier cache gain did **not** reproduce. Reject CACHE_B=1 for the final configuration; do not quote the earlier 1015.9 result as the deliverable.

New bounded candidate: `QWEN4EXP_MMB_TILE64=1` selects 64-column routed MMB/GLU tiles on RDNA3.5 when average routes/expert >=128. Small-expert tiles are unchanged. Shared memory drops from 36 KiB to 27 KiB for the two big kernels. Six bitwise tests (balanced/skewed IDs, ragged token/row counts) pass. ROCprof confirms six old and six new big-kernel dispatches for each of routed and fused GLU. Runtime flag is off by default; reference GGML_CUDA_MMB=0 bypasses it. Fresh matched A/B `/tmp/qwen-perf/mmb-final-ab.json` is in progress, with indexer cache explicitly off in both arms.


Completed routed-MMB A/B/B/A (two warmups/load, eight samples/arm):

| Variant | Min t/s | Median t/s | Best t/s |
|---|---:|---:|---:|
| Existing 128-column | 979.41 | 987.39 | 1005.90 |
| Candidate 64-column | 917.38 | 955.12 | 962.71 |

Rejected the 64-column change; removed its source and test. Lower VGPR/shared-memory use did not improve this workload.

Final source selection:
- RMS+SCALE fusion is enabled by default only on RDNA3.5, F32 width128, with fusion liveness/layout checks. `QWEN4EXP_RMS_SCALE_FUSED=0` restores the old two-kernel sequence. Its new unit test passes nine cases after the final rebuild.
- Last-token FFN remains opt-in via `QWEN4EXP_LAST_TOKEN_FFN=1`; it changes dispatch/rounding and is not silently enabled for other models/configurations.
- No retained changes to MMB, indexer, concat, FA dispatch, or KV padding. Existing parity padding/FP32 RoPE/HC/MoE switches remain opt-in. The default GDN RMS+SCALE was already present before this task; fusion removes a launch without selecting legacy L2 math.
- Best-known QSA/CUBLAS5/shadow1/HC16=2/chunk16384 settings retained. Indexer cache explicitly off in final measurements.
- Final quality and raw-tensor parity gates completed successfully; see the result at the top.


## Reproduction and artifacts

On the SSH host, from `/home/duster/lucebox-qwen4exp`, launch:

```sh
HIP_VISIBLE_DEVICES=1 DFLASH_HIP_NO_AUTO_UMA=1 GGML_CUDA_MMB=1 QWEN4EXP_QSA=1 QWEN4EXP_MMB_CUBLAS=5 DFLASH_MMB_SHADOW=1 LLAMA_MMB_HC16=2 QWEN4EXP_RMS_SCALE_FUSED=1 QWEN4EXP_LAST_TOKEN_FFN=1 GGML_DS4_INDEXER_M32_CACHE_B=0 server/build-hip/dflash_server /home/duster/models/qwen4exp-iq4nl/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf --host 127.0.0.1 --port 8877 --target-device hip:0 --max-ctx 40000 --chunk 16384
```

Use a clean environment with upstream/FA-padding/RoPE and other experimentation
flags unset. QSA is presence-based: `=0` still enables it; unset to disable.
Run `python3 /tmp/pf.py 8877 8875 1` once for warmup, then eight times.

Remote drivers: `/tmp/qwen-perf-ab.py`, `/tmp/qwen-perf-quality.py`,
`/tmp/qwen-perf-reference.py`. Logs and raw tensor dumps: `/tmp/qwen-perf/`.
Quality used the unchanged `harness/client_test_runner.py bench --url
http://127.0.0.1:8877 --suite he,gsm,math,recall --model dflash`.
Case-level comparison baseline: `/tmp/qwen-fa-iq4-final-quality.json`.
Reference wrapper invokes `server/scripts/qwen4exp_upstream_diff.py --model
MODEL --seq 16 --reference --output-dir DIR`, with RMS fusion and final-row
selection enabled, fast QSA/shadow flags unset, and the harness selecting
`GGML_CUDA_MMB=0` / `QWEN4EXP_UPSTREAM=1` for ours. It deletes stale binary
dumps before each run, then compares the four tensor pairs as raw bytes.
The smoke executable was explicitly relinked against the final source.

All measured timing samples are retained in adjacent JSON files. No comparison
uses measurements from different sessions as causal evidence. The long mode
sweep interruption and failed indexer-cache replication are documented above.
The final clean binary's full quality + eight-pass run is the deliverable,
not the higher, unreproduced indexer result.
