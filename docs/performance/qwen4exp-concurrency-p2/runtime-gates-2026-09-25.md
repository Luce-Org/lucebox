# P2 real-engine runtime gates — 2026-09-25

No commit. Rebuilt `/tmp/qwen4exp-p1-src/server/build-hip/test_qwen4exp_seq_engine`
with `GGML_HIP_GRAPHS=OFF` and `-j4`. The pre-build guard passed: no
`luce_server`/`dflash_server`, GTT 18,636,800 bytes, and 64.6 GB available RAM.
The run used IQ4_NL, gfx1151, `HIP_VISIBLE_DEVICES=1`,
`LUCE_HIP_NO_AUTO_UMA=1`, `QWEN4EXP_BATCHED_DECODE=1`, ctx 32768, and the GPU
lock `/tmp/qwen-perf/gpu.lock`.

## Contract root cause and fix

The adapter compared the scheduler's logical `cur_pos` against
`Qwen4ExpCache::cur_pos`, but `qwen4exp_forward()` does not update that cache
metadata. After successful prefill/decode, the engine now advances the matching
cache metadata (`pos + count` / `position + 1`) alongside the scheduler state.
This makes valid subsequent plans pass validation and lets the FIFO prefill
owner consume its bounded slices until completion. No model state or kernel
behavior was changed. The test contract diagnostic now includes the engine
error string on a failed valid plan.

## Results

| Gate | Result |
|---|---|
| Fresh HIP build, graphs OFF, `-j4` | PASS |
| Real `seq_engine_contract.h`, N=2 | PASS |
| Real `seq_engine_contract.h`, N=4 | PASS |
| Randomized admit/retire soak, two rounds at each width | PASS; mixed prompt lengths, partial prefill/cancel, and over-context admissions exercised |
| Distinct concurrent request answers | PASS: `4`, `Paris`, `Jupiter`, `pink` all appeared in their corresponding slot outputs |
| Batched-vs-solo greedy comparison | Slot 0 first diverged at decode step 7; epsilon 0.50860035, solo top-two margin 0.021116257, so margin < 2ε (1.0172007). Slots 1–3 had no token divergence through solo EOS. |
| Process exit | 0 |

The maximum sampled GTT in the confirmed run was 75,416,875,008 bytes
(~70.2 GiB); after teardown it returned to 18,640,896 bytes, versus
18,636,800 bytes before the run. This shows no retained GTT growth after the
soak/test process. The available RAM before load was 64,611,262,464 bytes.

Power/thermal snapshots: platform profile `balanced`; pre-run SCLK was at the
600 MHz idle level (2.9 GHz maximum available), MCLK selected 1.0 GHz; post-run
idle SCLK was 828 MHz and MCLK remained 1.0 GHz. FCLK levels were available up
to 2.0 GHz; the sysfs snapshot did not mark an active FCLK level. GPU1 edge
temperature was 41°C before and 51°C after. The monitor sampled GTT every two
seconds; clocks and temperature are before/after snapshots, not a continuous
clock trace.

Raw command output and GTT samples are in `raw/remote-2026-09-25/`.

The sequence-engine and batched-decode paths remain opt-in/default-off and
excluded under `QWEN4EXP_UPSTREAM=1`. The slot-0 greedy mismatch is numerically
consistent with the P1 margin allowance, rather than exact-token parity.
