# Qwen4Exp SeqEngine P2 status

Date: 2026-09-25. No commit.

## Implementation

- Added `Qwen4ExpSeqEngine`, backed by `SeqSlotManager` and an internal
  bookkeeping-only `PagedKvPool`. Admission reserves each live slot's entire
  `max_ctx`; the model still uses that slot's full F16 Qwen4ExpCache.
- Each admission/retirement resets only its cache. One
  `Qwen4ExpBatchedDecodeWorkspace` is shared across decode cohorts.
- Prefill planning is one sequence at a time, capped at 512 tokens. The
  selected scheduler FIFO owner is held until completion or retirement.
  Completed prefill returns its first sampled token as pending.
- Step plans are checked before state mutation; device-forward failures return
  an empty fatal cohort result. Compute is synchronized before cache reuse.
- Serving integration is gated by both `LUCE_QWEN4EXP_SEQ_ENGINE=1` and
  `QWEN4EXP_BATCHED_DECODE=1`, and constrained to 2–4 slots, `max_ctx=32768`,
  one local device, F16 caches, and no paged-attention option. Both flags are
  default-off; UPSTREAM excludes the engine.
- Added `test_qwen4exp_seq_engine`: it runs `seq_engine_contract.h` against the
  real adapter at N=2 and N=4, then performs two randomized mixed-length
  admission / partial-prefill / cancellation / retirement rounds at each width.

## Validation status

The required pre-run guard could not be completed. Both
`duster@lucebox4.tail97592a.ts.net` and `duster@100.115.193.112` timed out on
SSH. Therefore no box build or GPU action was started. Server residency, GTT,
and available RAM are **unverified** for this run.

| Gate | Result |
|---|---|
| Fresh gfx1151 HIP build, `GGML_HIP_GRAPHS=OFF`, `-j4` | **Pending**: box unreachable; no build attempted |
| Real-engine SeqEngine contract | **Pending**: adapter test added, not executed |
| N=2/N=4 randomized soak and flat GTT | **Pending**: probe added, not executed |
| Four distinct coherent answers and greedy-vs-solo margin comparison | **Pending**: requires live model server/GPU |

No runtime or correctness claim is made until these gates run with the GPU lock
held and `LUCE_HIP_NO_AUTO_UMA=1`.

## Runtime-gate retry

On 2026-09-25, retried the pre-run guard via MagicDNS and direct IP. MagicDNS
timed out; direct IP also timed out. The local client reports that its
tailscaled service is not running. It was not restarted. The guard remains
unverified and no build or GPU run was started.

Raw command output: raw/pre-run-guard-2026-09-25.txt.

## Corrected adapter and completed runtime gates — 2026-09-25

The first reachable contract run reproduced the three reported failures at
N=2/N=4 while the soak passed. The cause was an adapter metadata mismatch:
`qwen4exp_forward()` updates device/model recurrence state but does not update
`Qwen4ExpCache::cur_pos`; the engine's plan checks compared it against the
scheduler position. On successful prefill/decode, the engine now advances the
cache metadata with the same consumed token count. The FIFO owner consequently
consumes each selected bounded prefill slice and reaches decode.

Fresh `-j4` HIP build succeeded with `GGML_HIP_GRAPHS=OFF`. On the guarded,
GPU-locked gfx1151 run, the real engine contract and two-round randomized
admit/retire soak passed at both N=2 and N=4. The 48-step distinct-request run
returned coherent answers for all four requests. Batched-vs-solo comparison
found one token divergence at slot 0 / step 7 with epsilon 0.50860035 and solo
top-two margin 0.021116257 (`margin < 2ε`); slots 1–3 matched through solo EOS.
The completed wrapper exited 0. Maximum sampled GTT was ~70.2 GiB and returned
to the 18.6 MB idle baseline after teardown. Exact logs, GTT trace, and hardware
snapshots are recorded in `runtime-gates-2026-09-25.md` and
`raw/remote-2026-09-25/`.

The sequence-engine and batched-decode flags remain default-off pending the
separate quality-under-batch gate.
