# Qwen4Exp concurrency P1 batched-decode prototype

Date: 2026-09-25. This is an experimental forward API only; the serving scheduler does not call it. No commit was created.

## Design

`qwen4exp_forward_batched()` is gated by `QWEN4EXP_BATCHED_DECODE=1` and returns immediately when `QWEN4EXP_UPSTREAM=1`. N=1 delegates to `qwen4exp_forward()`; the smoke probe now verifies bit-identical logits and state after both entry points. For N>1, one graph has T=N independent rows. Embedding and dense attention Q/K/V/O projections, linear-attention projections, PLE key/value projections, HC, routed MoE, output HC and output projection operate on the row batch. Full-attention KV writes/read and masks, GDN convolution/recurrent updates, and PLE convolution/history are branched per cache. PLE n-gram indices come from each cache's own `ple_prev`; masks are per slot. There is no shared causal time axis or last-row FFN trim.

One caller-owned `Qwen4ExpBatchedDecodeWorkspace` retains the graph metadata context and gallocr across calls. Graph topology is rebuilt per call because active row-to-cache edges can change. Each slot continues to own its full F16 cache. `reset_qwen4exp_state()` clears that slot's T=1 decode workspace / stable graph bucket; the batched arena contains no captured slot state and is shared by the caller.

Two fixes were needed while bringing up the graph: the per-slot PLE convolution tap view must be `[hc_dim, 1]` (the initial `[1, hc_dim]` shape tripped the CUDA fused-conv matcher); and the next-layer PLE boundary must preserve the unfused residual combine, matching the existing single-sequence graph. Full-attention positions and masks are separate input tensors per slot, including each slot's fixed PAD256 bucket.

## Validation

Built `smoke_qwen4exp_batched` in `/tmp/qwen4exp-p1/build` against HIP on gfx1151, `GGML_HIP_GRAPHS=OFF`. Every GPU run held `/tmp/qwen-perf/gpu.lock`; one IQ4_NL model was resident. The final process environment was captured in `raw/box-2026-09-25/final15-environ.txt`: `HIP_VISIBLE_DEVICES=1`, `LUCE_HIP_NO_AUTO_UMA=1`, `QWEN4EXP_BATCHED_DECODE=1`, plus the delivered dense/MMB/HC flags and `QWEN4EXP_FA_PAD256=1`. Power was `platform_profile=balanced`; device pp_dpm clocks and temperatures are in the before/during/after snapshots (gfx1151 is DRM card2). The full rocm-smi clock/power selector was avoided because that local ROCm command crashes; `--showtemp` and sysfs DPM state worked.

The probe used real IQ4_NL weights, ctx=32768, 16-token deterministic synthetic prompts, and three decode steps. `final12`–`final15` are separate fresh processes; the corrected final binary is `final15` (exit 0). `final12` was captured during exec startup and its environ file is empty; use final13–15 for environment evidence. `final13` and `final14` were compiled before the explicit N=1 assertion was added; core graph code was unchanged. The final source, build output and raw logs are retained here.

| Gate | Result |
|---|---|
| N=1 API vs existing path | PASS: bit-identical logits and all saved slot-state bytes (`N=1-single-path=PASS`). |
| Four identical slots | PASS: every slot's logits compared bitwise equal and the three generated greedy IDs agreed across all four rows at each step. |
| Distinct slots vs solo | Conditional PASS under the requested `margin < 2*epsilon` rule. At the first decode, slot0: epsilon 2.488699, same token 13868, solo margin 0.785589; slot1: epsilon 1.626825, same token 271, margin 0.230045; slot2: epsilon 8.323452, token 279 vs 13, margin 0.299193; slot3: epsilon 6.621463, token 1249 vs 837, margin 0.295493. Both changed-token margins are below their per-slot `2*epsilon` bounds (16.646904 and 13.242926). |
| Identical-state third-step vs solo | Conditional: batch slots remain mutually identical, but their shared token is 248044 while solo is 248045; max logit delta 6.482243, solo top-two margin 0.790478 (< 2*epsilon 12.964485). |
| Untouched slot bytes | PASS for active set `{3,1}`; snapshots of slots 0 and 2 unchanged. |
| Row permutation | PASS: reverse `{3,1}` order and outputs map bitwise back to their original slots. |
| Reset/reuse | PASS: reused slot equals newly allocated cache within each run; logits hash `25f8876a91fa045c`. The same hash repeated in final13–15 fresh processes. |
| N=4 per-slot full cache | Exercised at ctx 32768, F16, with distinct per-cache attention/recurrent/PLE state. |

## Interpretation / status

The implementation meets the specified structural and state-isolation tests. It does **not** preserve exact N=1 greedy output for all rows: distinct slots 2 and 3 diverge on the first generated token, and identical batch rows diverge from solo on step 3. Those divergences satisfy the requested margin rule only because observed end-to-end logit deltas are large (up to 8.323452). This is not evidence of tight numerical parity; batched dense GEMM/recurrent arithmetic shifts later routed-expert choices. Keep the feature opt-in and do not connect it to serving until the human accepts this loose margin policy or a stricter numerical tolerance is established. The P1 probe is not a general quality-suite run.
