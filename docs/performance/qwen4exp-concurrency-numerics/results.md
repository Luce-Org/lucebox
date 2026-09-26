# Qwen4Exp batched decode numerics and dispatch

Date: 2026-09-26. Hardware: gfx1151 on `lucebox4`, HIP, `GGML_HIP_GRAPHS=OFF`, IQ4_NL, ctx 32768, F16 cache, 16-token synthetic prefill, probe binary `/tmp/qwen4exp-p1-build/smoke_qwen4exp_batched`. Both GPU arms held `/tmp/qwen-perf/gpu.lock`, had no server resident (idle GTT 18,636,800 B), set `HIP_VISIBLE_DEVICES=1 LUCE_HIP_NO_AUTO_UMA=1`, and used the delivered QSA/MMB/HC settings. Power was `platform_profile=balanced`; rocm-smi reported GPU1 edge 34–40 C and GPU0 fclk 582 MHz/mclk 96 MHz, then aborted while parsing clocks. GPU1's clocks were therefore not captured. The default timed control did not have an arm-specific rocm-smi sample; the shared probe clocks are environmental context, not a matched GPU1 clock record. Available RAM was 59–60 GiB. No production code changed.

## Dispatch result

The observed dispatch is not identical between T=1 and T=4 for ordinary quantized dense `MUL_MAT`:

| Operation | T=1 | T=4 default | Why |
|---|---|---|---|
| Quantized dense projections | MMVQ | MMQ | `LUCE_MMVQ_MAX_NCOLS` defaults to 3; `use_mul_mat_vec_q` is selected before MMB/MMQ, and T=4 exceeds that ceiling. MMB is ineligible below T=512. `LUCE_MMB_TELEMETRY=1` observed 1,980 `path=mmq T=4` dispatch records in the probe. |
| Routed IQ4_NL `MUL_MAT_ID` | MMVQ | MMVQ | Separate type/architecture ceiling: gfx1151 IQ4_NL reports `mmvq_max=6`; telemetry saw widths 1, 2, and 4 on MMVQ. Width 16 switched to MMQ. |
| Dense dispatch table | not applicable | not applicable | Its rule is exact `tokens==16366`; it cannot fire in T=1/T=4 decode. |
| CUBLAS BF16 shadow | not applicable | not applicable | Its minimum token threshold is 512; it cannot fire in decode. |

MMVQ and MMQ both consume F32 activations through Q8_1 quantization. The mismatch is their tile/reduction path and scheduling, not “F32 accumulate versus Q8_1” in the simple sense. The `QWEN4EXP_DENSE_TABLE` prefill route is not implicated.

## Expert routing and layer trace

The first-decode layer-trace probe used four identical slot states/inputs; row 0 was also evaluated solo from the same state. With default dispatch, top-10 expert *sets* first diverged at L12 (solo-only expert 389, batch-only expert 21) and differed in 26/48 layers. With `LUCE_MMVQ_MAX_NCOLS=4`, the first set divergence remained at L12 with the same set delta, but changed sets fell to 22/48 layers. Router output itself is F32; this behavior is consistent with small earlier arithmetic differences crossing router boundaries and then compounding through MoE, rather than a different slot/cache layout. Layer traces also show non-MoE differences before L12: default first `ffnxn` >1e-3 at L0 (0.03134), attention at L0 was 0.001152; the first `hcmix` >1e-3 was L1 (0.07605). Thus MoE amplification is real, but it is not the only source of hidden-state drift.

Historical P0 width-four samples (same real row alone vs row 0 in width four) found small operator-level deltas: Q5_K attention output max 0.00125524, Q6_K attention Q 2.03836e-5, IQ4_NL HC attention-down 7.59214e-5, Q8_0 shared gate 1.91275e-5, IQ4_NL routed MoE output bitwise equal, and F32 HC/router sampled ops bitwise equal. Those are width-parity deltas, **not** errors against a dequantized F32 dot-product oracle. This investigation did not compute per-op MMVQ/MMQ errors against that oracle, so it cannot attribute the observed logit epsilon quantitatively to each kernel’s standalone F32 error.

## Dispatch tightening falsifier

Setting `LUCE_MMVQ_MAX_NCOLS=4` forces ordinary T=4 quantized dense operations onto MMVQ, matching the solo kernel family. It does not alter the separate `MUL_MAT_ID` dispatch. The same smoke probe gave:

| Probe result | Default (ceiling 3) | Forced MMVQ (ceiling 4) |
|---|---:|---:|
| Identical-row step-0 max logit delta | 1.77092 | 1.62100 |
| Identical-row step-1 max logit delta | 1.11634 | 0.80799 |
| Identical-row step-2 max logit delta | 1.72925 | 0.70543 |
| Distinct-row step-0 max logit delta | 1.96272 | 1.85115 |
| Distinct-row maximum observed over steps | 1.96272 (trajectory flips at step0) | 3.81961 (trajectory remains matched through step1; flips at step2) |
| Expert-set-different layers | 26/48 | 22/48 |
| Smoke-probe total wall time, one run | 35.67 s | 32.24 s |

Both arms passed the probe’s relative margin check, but both still showed greedy flips on near ties: identical rows at step 2 were solo token 3710 vs batch token 13 under default, and solo 13 vs batch 3710 with forced MMVQ; solo margins were 0.1463 and 0.0252 respectively. Distinct rows flipped at step 0 in the default arm (slot 2: 487 vs 4876; margin 0.1632). Forced MMVQ preserved all four tokens through step 1 but reached ε=3.81961 and flipped slots 0 and 3 at step 2. The route change modifies both the perturbation and the point at which greedy streams diverge; it does not cure batch-dependent output.

The one-run smoke wall times include model load, prefill, state snapshots, solo forwards and isolation/reset checks; the 9.6% lower candidate wall time is not a decode throughput measurement and is too noisy to claim a performance win. The timed default and forced arms were separate single runs; GPU1 clock matching was not established. No tok/s cost was established. A clean repeated decode benchmark is needed before changing the setting globally.

## Verdict

This is a **mixed case**. Batch-size-dependent reduction/tiling and subsequent MoE routing amplification are expected numerics; however, the current width-3 ceiling creates an avoidable MMQ-vs-MMVQ split for T=4 dense GEMMs. Matching the dense kernel family improves the observed epsilon and reduces downstream expert-set churn modestly, but residual ε remains O(1), expert routes still diverge, and greedy output still flips. Therefore the large end-to-end delta is not explained solely by ordinary tiny floating-point reassociation, and the dispatch choice contributes, but it is not the whole cause. Keep batched decode opt-in/default-off. Do not promise exact solo-greedy equivalence. Evaluate `LUCE_MMVQ_MAX_NCOLS=4` only as a gfx1151-scoped candidate after a min-of-N decode throughput run and a direct per-op dequantized-F32 reference probe.

## Raw evidence

- `default.log`, `default-timed.log`, `default.time`: default dispatch telemetry and smoke outputs.
- `force-mmvq.log`, `force-mmvq.time`: matched dense MMVQ arm.
- `trace-default.log`, `trace-forced-mmvq.log`, `traces.tgz`: captured first-decode traces. Expanded traces and layer-delta analyzer inputs are in `traces/`.
- Analyzer: `python3 docs/performance/qwen4exp-concurrency-p1/probes/analyze_layer_trace.py docs/performance/qwen4exp-concurrency-numerics/traces numdef` and likewise `nummmvq`.
