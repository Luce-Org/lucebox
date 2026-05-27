# HIP perf — diagnosis + kernel-side optimization plan

_Drafted 2026-05-11 against `Luce-Org/lucebox-hub` post-#122 with HIP/ROCm
support landing in the upcoming PR. Numbers below are from the canonical
DFlash bench (Qwen3.6-27B-Q4_K_M + z-lab DFlash drafter,
`--fast-rollback --ddtree --ddtree-budget=22`, HE-style 128-tok prompt
md5 `4280413edc0b45c2b09e1a45f4f5ee60`, n_gen=256, warmup + 2 measurement
runs)._

## tl;dr

Lucebox HIP decode on gfx1100 (7900 XTX) runs **3.22× over AR** today —
within 6% of the README's CUDA RTX 3090 3.43× anchor. That's the headline
DFlash speedup. But the **absolute throughput is 50 tok/s on gfx1100 vs
~200 tok/s on hipfire's RDNA-native engine** on the same physical card
(same prompt, same target, same context). The 4× gap is **not** in
attention — it's in `mul_mat_q` for `q4_K` / `q4_0` / `q5_0`. The fix is
upstream in `ggml-cuda/mmq.cuh` + `mmvq.cuh`.

Tier 1 of this plan is **already empirically verified**: setting
`--ddtree-budget=8` instead of the default 22 on gfx1100 lifts decode
from 49.81 tok/s to 76.02 tok/s — a **53% speedup from a single config
flag**, no kernel work. Same flag is a -9% regression on gfx1201, so the
ship is arch-aware. Details in the Tier 1 section below.

**Tier 2 has been tested and FALSIFIED (2026-05-11).** The plan's original
hypothesis — that extending MMVQ template instantiations to cover
`ncols_dst ∈ {16, 23}` would route the DFlash verify path through the
BW-amortised GEMV kernel and recover most of the 4× gap — does not
hold. Three-arch A/B (gfx1100 / gfx1151 / gfx1201, 10-prompt HE bench,
default budget=22): MMVQ regresses **−42 to −69 % across all RDNA3+
silicon**. MMQ + WMMA on modern AMD wins decisively even at the
supposedly "wasteful" 32-wide tile occupancy of ne[1]=23 → 28% empty
columns. Full table + analysis in the Tier 2 section below. Pivot
recommendation: **Tier 3 (hipfire-style multi-row q4_K decode GEMV)
becomes the only kernel-side lever worth pursuing**.

This doc traces the rocprofv3 evidence, identifies the dispatch decisions
that route lucebox onto the slow path, and proposes a four-tier
optimization plan for the lucebox `llama.cpp-dflash-ggml` fork. Tier 2's
falsification is documented in full so future contributors don't repeat
the experiment.

## rocprofv3 top-10 hot kernels on gfx1100

Captured via `scripts/lucebox_kernel_atlas.py` (kernel-trace + summary +
ISA manifest) on the canonical DFlash bench above. Total profiled wall
~112s; per-kernel totals below cover all 256 generated tokens.

| Time   | Calls | Kernel | Notes |
|-------:|------:|---|---|
| 2076 ms | 1820 | `mul_mat_q<q4_K, 32, false>` | **target q4_K matmul, DDTree batch-tile 32** |
| 1247 ms | 8064 | `mul_mat_q<q4_0, 32, false>` | KV cache q4_0 matmul |
|  741 ms | 3456 | `Cijk_Alik_Bljk_SB_MT64x64x8_SN_1LDSB0_...` | rocBLAS strided batched GEMM (no WMMA) |
|  211 ms | 2304 | `mul_mat_q<q4_0, 16, false>` | smaller-tile MMQ |
|  205 ms | 1344 | `mul_mat_q<q5_0, 32, false>` | MMQ q5_0 |
|  130 ms |  420 | `Cijk_Alik_Bljk_HB_MT64x64x32_MI16x16x16x1_...` | rocBLAS GEMM **with** WMMA |
|  125 ms |  540 | `mul_mat_q<q4_K, 16, false>` | smaller-tile MMQ for q4_K |
|   72 ms | 1344 | `gated_delta_net_cuda<128, false, true, __half>` | DeltaNet hybrid path |
|   47 ms |  149 | `Cijk_..._MI16x16x16x1_...` | rocBLAS WMMA, second shape |
|   27 ms |  448 | `flash_attn_tile<256, 256, 32, 1, false>` | **FA tile — 0.5% of total** |

**~76% of GPU time is `mul_mat_q` variants. FlashAttention is 0.5%.**
The "huge tax" frame is correct, but it doesn't live in the missing
`flashprefill_kernels.hip.cu` (that path is short-prompt-cold here) — it
lives in `mmq.cuh`'s `q4_K` MMA path on RDNA3+.

## The dispatch trace — why DDTree always lands on MMQ

`ggml-cuda.cu:2294` decides MMVQ vs MMQ:

```cpp
bool use_mul_mat_vec_q = ggml_is_quantized(src0->type)
                        && ... && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
```

`MMVQ_MAX_BATCH_SIZE = 8` (`mmvq.cuh:3`).

DDTree budget=22 → speculation batch = 22 → `src1->ne[1] = 22 > 8` →
**always falls to MMQ.** MMQ uses WMMA on RDNA3+ (via
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` in `mma.cuh`) but the tile
shape is `32 × mmq_y × K` — designed for big-batch prefill. On a 22-wide
spec-verify batch it does 32 columns of work and discards 10 → **31%
wasted GPU compute on every speculation step**.

For the non-spec AR baseline (batch=1 decode), MMVQ kicks in correctly,
which is why the AR baseline at 28 tok/s is closer to hipfire's per-call
throughput than DFlash's 50 tok/s.

## Where MMVQ stops scaling — the second hard wall

`mmvq.cu:calc_nwarps()` and `calc_rows_per_block()` are explicitly tuned
for `ncols_dst ∈ {1..8}`. The fall-through for `ncols_dst ≥ 9` returns
`nwarps=1, rows_per_block=1` — i.e. no parallelism, one batch per warp,
launch-overhead-bound. Even if you bumped `MMVQ_MAX_BATCH_SIZE` to 32,
the kernel would behave badly because the per-arch
`MMVQ_PARAMETERS_RDNA3_0` / `RDNA4` cases gate on `ncols_dst == 1`
specifically (`mmvq.cu:326-345`).

So MMVQ as-shipped is not a drop-in fix for the 22-batch spec-verify
shape. It needs new instantiations.

## Four-tier optimization plan

### Tier 1 — Config-only (15 min, zero risk, **arch-specific**)

Try `--ddtree-budget=8` on the HIP backend. Routes spec-verify through
MMVQ instead of MMQ.

**Empirically validated 2026-05-11, n_gen=256 on the canonical HE bench
above, warmup + 2 measurement runs each:**

| Arch | Card | budget=22 (MMQ) | budget=8 (MMVQ) | Delta |
|---|---|---:|---:|---:|
| gfx1100 | 7900 XTX | 49.81 tok/s | **76.02 tok/s** | **+53%** |
| gfx1151 | Strix Halo iGPU | **34.78 tok/s** | 30.71 tok/s | -13% |
| gfx1201 | R9700 | **84.70 tok/s** | 77.23 tok/s | -9% |

The win is **gfx110x-only** (vanilla RDNA3 desktop dGPUs: 7900 XTX/XT,
7800 XT, 7700 XT/S, 7600). RDNA3.5 (Strix Halo gfx1151) and RDNA4
(gfx1201) both prefer MMQ at budget=22 — likely a combination of:

- RDNA4 has well-tuned MMQ tile shapes that make wasted columns of a
  batch-32 tile cheap proportionally.
- RDNA3.5 Strix Halo's LPDDR5X UMA (~270 GB/s vs 7900 XTX's 960 GB/s
  GDDR6) makes one-big-MMQ-launch's launch-amortization more valuable
  than tile-utilization. MMVQ's per-batch separate launches hurt UMA.

The dispatch analysis is correct on all three archs; the threshold is
just on the wrong side specifically for the desktop RDNA3 SKUs.

**Suggested ship**: arch-aware default in the daemon's CLI parsing or
`server.py` — set `--ddtree-budget=8` when running on gfx1100, gfx1101,
gfx1102 (desktop RDNA3 only). Keep 22 on gfx115x (RDNA3.5), gfx120x
(RDNA4), and CUDA. Single-PR change, zero kernel work, recovers most of
the gfx110x-specific gap.

```bash
./test_dflash $T $D prompt.bin 256 out.bin --fast-rollback --ddtree --ddtree-budget=22  # current default
./test_dflash $T $D prompt.bin 256 out.bin --fast-rollback --ddtree --ddtree-budget=8   # MMVQ-routed
```

### Tier 2 — Extend MMVQ template instantiations (TESTED 2026-05-11, FALSIFIED)

**The original hypothesis**: extend the MMVQ switch from `ncols_dst ∈
{1..8}` to also include the real DFlash verify shapes (`ne[1] = 16` for
chain mode = `DFLASH27B_DRAFT_BLOCK_SIZE`, `ne[1] = 23` for default
DDTree `--ddtree-budget=22` = `1 + budget`). Add a per-(type, cc, batch)
gate `mmvq_is_supported_batch()` at the dispatcher so RDNA3+ q4_K routes
to the BW-amortised vec_dot kernel instead of MMQ's 32-wide tile that
wastes ~28 % of ALU at ne[1]=23. Projected: **+50-100 % decode tok/s
on gfx1100**, cutting the dominant `mul_mat_q<q4_K, 32>` cost in half.

**Implementation** (research artifact at
`Kaden-Schutt/llama.cpp-dflash-ggml@feat/mmvq-rdna3-batch16`, commit
`002db52`, default-off behind `GGML_MMVQ_NO_EXTENDED=1` opt-out):

- New `mmvq_is_supported_batch(ggml_type, int cc, int batch)` returning
  true for `batch ∈ [1, 8]` unconditionally plus
  `batch ∈ {16, 23}` for q4_K on RDNA3 / RDNA4 (`GGML_CUDA_CC_IS_RDNA3`
  / `_IS_RDNA4`).
- New template instantiations at `case 16:` and `case 23:` in
  `mul_mat_vec_q_switch_ncols_dst<type>()`, with the existing
  `nwarps=1, rows_per_block=1` shape for non-`MMVQ_PARAMETERS_GENERIC`
  tables. Each new value adds one compile unit per quant type; bench
  binary size unchanged within rounding.
- Bumped assert ceiling `MMVQ_MAX_BATCH_SIZE_EXTENDED = 23`.
- Replaced the two host-side `src1->ne[1] <= MMVQ_MAX_BATCH_SIZE`
  gates at `ggml-cuda.cu:2294,2337` with the new
  `mmvq_is_supported_batch(...)` call.

**Bench A/B** — single binary per arch, `GGML_MMVQ_NO_EXTENDED=1` for
baseline cell vs unset for tier2 cell, byte-identical 10-prompt
HumanEval set (`bench_he.py`, `--n-gen 256 --skip-tokenize`),
Qwen3.6-27B-Q4_K_M target + matched z-lab/Qwen3.6-27B-DFlash drafter,
ROCm 7.2.2:

| GPU | Arch | Budget | baseline (MMQ) AL / tok/s | tier2 (MMVQ) AL / tok/s | Δ |
|---|---|---:|---:|---:|---:|
| 7900 XTX (k9lin)        | gfx1100 (RDNA3)   | **22** | 7.57 / **40.91** | 7.00 / 23.46 | **−42.7 %** |
| 7900 XTX (k9lin)        | gfx1100 (RDNA3)   | 8      | 5.25 / 62.36     | 5.55 / 65.72 | +5.4 % (noise, paths ≡) |
| R9700 (hiptrx)          | gfx1201 (RDNA4)   | **22** | 8.26 / **77.53** | 7.69 / 24.13 | **−68.9 %** |
| R9700 (hiptrx)          | gfx1201 (RDNA4)   | 8      | 5.51 / 64.88     | 6.05 / 70.77 | +9.1 % (noise) |
| Strix Halo iGPU (hipx)  | gfx1151 (RDNA3.5) | **22** | 6.66 / **26.36** | 8.24 / 14.79 | **−43.9 %** |
| Strix Halo iGPU (hipx)  | gfx1151 (RDNA3.5) | 8      | 5.56 / 28.73     | 5.17 / 26.70 | −7.1 % (noise; AL drift) |

**Tier 2 falsified on all three RDNA3+ archs at the default workload**.
The budget=8 cells are bench-noise null: at ne[1]=9 the new gate also
routes to MMQ (9 ∉ {1..8, 16, 23}), so the cells exercise identical
kernels modulo FP-reduction-order drift.

**Why the hypothesis was wrong**:

1. **WMMA beats BW amortisation on modern AMD even with wasted lanes.**
   MMQ at ne[1]=23 uses a 32-wide tile (28 % of columns sit idle), but
   the WMMA matrix cores process 16×16 tiles at ~four bf16-FMA-per-cycle
   per lane. MMVQ's `vec_dot_q4_K_q8_1` is scalar `v_dot4` (RDNA3) /
   scalar fp16 FMA (RDNA1/2) — no matrix-core throughput. The
   ALU-density gap dwarfs the tile-occupancy loss.
2. **MMVQ re-reads activations per `ncols_dst`.** Inside the K-block
   loop the kernel does
   `tmp[j][i] += vec_dot(vx, &y[j*stride_col_y + kby], ...)` for
   `j ∈ [0, ncols_dst)`. At ncols_dst=23 that's 23 redundant activation
   loads per K-block, all hitting GDDR6 (RDNA3 dGPU) / LPDDR5X (Strix
   Halo). MMQ stages activations into LDS once per tile and amortises
   across `tile_M × tile_N = 16 × 32 = 512` output positions.
3. **Per-thread accumulator pressure.** `float tmp[23][1]` on the stack
   is 23 live VGPRs per thread, plus the vec_dot working set. The
   compiler doesn't spill on gfx1100, but the larger live-set reduces
   wave-occupancy on RDNA3.5 and RDNA4 enough to compound with the BW
   issue.
4. **FP non-associativity.** MMQ tile-summed accumulation and MMVQ
   K-block + warp_reduce_sum produce different bit-patterns for the
   same logits. That shifts argmax by enough on a fraction of tokens
   that AL drifts ±7 % per arch, on top of the kernel slowdown.

**Negative-result class**. This is the fifth synth-win → prod-falsify
cycle on RDNA3+ for kernel-level decode optimisations under ROCm 7.2.x
(hipfire's prior catalogue: FP8 dot4 GEMV on gfx1201, gfx12 FP8 WMMA on
HFP4G32, MFP4 MoE all-FP4, gemv graph cache PR3). The pattern: per-shape
microbenches show 1.5-2× wins; the same change loses 5-70 % in
end-to-end production because cross-kernel L2 state, scheduler
ordering, and WMMA-vs-scalar ALU density only become legible at the
full forward pass. **Only launch-reduction levers (β + graph capture,
fusion) cross zero in production on this codebase + ROCm version**.

**Disposition**:

- Keep the `feat/mmvq-rdna3-batch16` branch as a research artifact;
  default-off behind `GGML_MMVQ_NO_EXTENDED` so a non-set env still
  triggers it for reproduction. **Do not open as a perf PR**.
- Pivot the kernel-side roadmap to Tier 3 (next section): a multi-row
  q4_K decode GEMV that processes R=4-8 rows per warp with shared
  activation register state, the hipfire pattern that consistently
  ships on RDNA3+ without the wasted-tile / per-row-reread tradeoff
  that killed Tier 2.

### Tier 3 — Multi-row decode GEMV à la hipfire (1-2 weeks)

Hipfire's `kernels/src/gemv_hfq4g256_multirow.gfx1100.hip` processes
R=2/4/8 output rows per warp, sharing the X (activation) register state
across rows. For a wide decode batch (DDTree budget=22-32) this is
exactly the right shape:

- One warp processes 4 output cols × 22 batches → 88 dot products with
  one X-load
- Register pressure: ~38 VGPRs for R=4 multirow (hipfire-measured on
  gfx1100), still 16 waves/CU occupancy
- vs MMQ's tile-based approach which does 32 cols × mmq_y rows but
  burns more shared memory and launches more thread blocks

To port to ggml's q4_K shape, the kernel needs:
- The q4_K block layout reader (super-blocks of 256 with 6-bit scales +
  6-bit mins per sub-block of 32)
- A wave32 fast path using `v_dot4_i32_i8` for non-WMMA inner loops on
  gfx1010/1030, and `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` for
  gfx1100+
- Register-packed batch dimension to avoid LDS-staging for activations

Reference patterns in hipfire's tree (not directly portable, but the
tile/loop structure transfers):
- `crates/rdna-compute/src/dispatch.rs:601-613` — the per-arch ROCm-min +
  WMMA-builtin routing table
- `crates/rdna-compute/src/kernels.rs:multirow_*` — the R-selection logic
- `kernels/src/gemv_hfq4g256_multirow.gfx1100.hip` — the actual kernel

Expected impact: **2-3× on DFlash spec-verify**, bringing q4_K decode
within ~20% of hipfire on the same hardware.

### Tier 4 — gfx1010 / gfx1030 scalar-fallback score kernel (3-5 days)

Orthogonal to the q4_K decode work. Required to unblock PFlash on
RDNA1/RDNA2 cards where today the score-blocks kernel hangs (gfx1010 —
missing `v_dot4`) or runs ~7× slower than Strix Halo (gfx1030 — uses
SDWA fallback but no WMMA available).

Pattern: hipfire's `kernels/src/gemv_hfq4g256_multirow.gfx1010.hip`.
Wave32 RDNA1, scalar fp16 accumulation, no WMMA dependency.

## Ranked priority (revised post-Tier-2 falsification)

1. **Tier 1 today** — needs only an arch-aware default change in
   `bench_he.py` / `run.py` / `server.py`, gives a free 53 % bump on
   gfx1100 (`--ddtree-budget=8`). Zero kernel work, lowest risk, ships
   first.
2. ~~Tier 2~~ **— removed**. Bench data above shows MMVQ at extended
   `ncols_dst` regresses 42-69 % across all RDNA3+ silicon at the
   default workload. The branch survives as a documented research
   artifact (default-off env-gated) so future investigators can
   reproduce the negative result without re-implementing the path.
3. **Tier 3 — primary kernel-side lever**. The real engineering work
   that consistently ships on RDNA3+ in hipfire: multi-row q4_K decode
   GEMV with shared activation register state. Sidesteps both failure
   modes of Tier 2 — no wasted-tile waste (R=4-8 rows is the actual
   live output budget) and no per-`ncols_dst` activation re-read (the
   batch dimension is register-packed, not loop-extended). Lands as
   PR-C below. **~1-2 weeks of focused work; projected +1.5-3× on
   DFlash spec-verify on gfx1100/1201**.
4. **Tier 4 in parallel** — unlocks RDNA1/RDNA2 PFlash entirely. No
   overlap with Tier 3; can ship in any order.

Path B (rocWMMA port of `flashprefill_kernels.cu`) addresses the PFlash
*prefill* tax (compress + target_prefill on long ctx). It is **orthogonal
to this plan** — it helps long-context TTFT, not decode tok/s. Both
should ship.

## Sequence of PRs against `Luce-Org/llama.cpp-dflash-ggml`

1. **PR-A**: arch-aware `--ddtree-budget` default (Tier 1, daemon-side
   only, no submodule change).
2. **PR-C**: multi-row decode GEMV for q4_K on RDNA3+ (Tier 3, biggest
   remaining payoff). Lands a new
   `mul_mat_vec_q_multirow_rdna_<arch>.cu` template alongside the
   existing MMVQ kernel; dispatched from `ggml_cuda_mul_mat` at
   `ne[1] ≤ 8` (the regime where the existing MMVQ already wins) to
   capture R-row sharing without touching the > 8 dispatcher (which
   Tier 2 proved is MMQ's territory).
3. **PR-D**: gfx1010/1030 scalar-fallback score kernel (Tier 4).
4. **PR-E (separate, against `lucebox-hub`)**: rocWMMA port of
   `flashprefill_kernels.cu` → `flashprefill_kernels.hip.cu` (Path B,
   prefill tax).

Each PR is independent, separately bench-able, separately revertible.
Tier 2's original PR-A (extend MMVQ to ncols_dst ≤ 32) and PR-B
(generalise to all common quants) have been **dropped** from this
sequence; rationale in the Tier 2 section above.

## Validation per PR

- Bit-identical token output vs CUDA baseline (gfx1100 vs RTX 30/40-series
  on the same prompt). Lucebox already has `test_vs_oracle` for this.
- DFlash 3-tier coherence smoke (Path-A attractor / 3gram density /
  EOS-immediate) — port from hipfire `crates/hipfire-detect/` as a small
  Python script.
- NIAH retrieval at 8K / 32K / 64K / 128K — already verified end-to-end
  on the HIP support PR.
- Prompt-md5 disciplined bench: warmup + ≥2 measurement runs, fresh
  binary, prompt md5 logged. See lucebox's existing `bench_he.py` setup.

---

# 2026-05-11 evening update — cycles 9–16 empirical follow-on

After the Tier 1 validation + Tier 2 falsification above, this is the
log of eight additional cycles testing every remaining lever within the
HIP path's ceiling. The headline is mixed: **one new kernel-side
config win + one drafter-swap win, four falsified levers, one
sanity-check that quantifies the structural ceiling.** Combined gains
land at **+6.4% to +10.95% over the post-Tier-1 baseline**, but the
sanity-check vs hipfire's native implementation on the same silicon
shows lucebox-hub HIP runs at ~58% of the hardware's practical ceiling.

The strategic recommendation at the end (cycle 16) is to pivot the AMD
direction toward **Vulkan**, not because the HIP work is wasted (it
shouldn't be — ship it), but because the kernel-tuning ceiling on HIP
is structurally lower than what Vulkan offers across all RDNA archs.

## Correction to Tier 1 attribution

The original Tier 1 write-up framed the +53% gfx1100 win as
"MMQ→MMVQ dispatch via `MMVQ_MAX_BATCH_SIZE = 8`". The mechanism is
correct (budget=8 → `src1->ne[1]=8 ≤ 8` → MMVQ path for the target
spec-verify q4_K matmul). What's **wrong** is the implied severity:
cycle 11's `rocprof --stats` decode-time attribution on the same
configuration shows MMVQ accounts for **only 1.10% of decode GPU
time** (8 calls × 1128 µs avg on the 27B HE-0 workload). The +53% comes
from the routing decision for one specific tensor — the target q4_K
spec-verify matmul — which dominates _budget=22_'s wasted-tile cost
but is only one of several q4_K/q5_K/q6_K mat-muls in the forward
pass. The wider MMQ surface (52.66% of decode time across
`mul_mat_q<q4_K, 16>` + `<q5_K>` + `<q6_K>`) is unchanged by the
budget switch and remains the bottleneck after Tier 1 lands.

This doesn't change the recommendation to ship Tier 1 — it just
clarifies what's actually moving.

## Cycle 9 — Issue ggml-org/llama.cpp#21284 MMQ tile override on RDNA3_0

The closed PR #21344 (pedapudi, for gfx1151) hypothesized that the
upstream `mmq_x_max=128, mmq_y=128, nwarps=8` defaults spill VGPRs on
RDNA3-class WMMA architectures and proposed `48, 64, 4` for RDNA3_5.
IMbackK (ROCm maintainer) explicitly suggested testing the same values
on discrete RDNA3 (gfx1100/gfx1101) but never followed up. We tested it.

**Implementation**: 6 edits to `ggml-cuda/mmq.cuh`'s
`get_mmq_x_max_host/device`, `get_mmq_y_host/device`,
`mmq_get_nwarps_host/device` — all gated behind an opt-in macro
`LUCEBOX_GFX1100_TILE_OVERRIDE` (default OFF). Branch
`Kaden-Schutt/lucebox-hub@feat/mmq-q4_K-rdna3-custom` (local, unpushed).

**Bench** (HE 10-prompt, n_gen=256, budget=8, n=2 runs each):

| Tile override | Mean tok/s | AL  | Δ vs stock |
|---|---|---|---|
| OFF (= upstream defaults 128/128/8) | 62.56 | 5.25 | baseline |
| ON  (= 48/64/4 on RDNA3_0)           | **63.86** | 5.25 | **+1.93%** |

Per-prompt deltas: +1.5% to +3.0% across all 10 prompts, every prompt
positive → structural, not DPM drift. AL identical (tokens
bit-identical). The WMMA static-assert `nwarps × tile_C::I == mmq_y`
collapses the sweep space to 3 valid points; cycle 12 confirmed (48,
64, 4) is the empirical peak.

**Status**: small but real and reproducible. The first non-negative
result in 9 cycles of kernel-level work. Default OFF keeps upstream
behaviour intact; flip the macro to opt-in.

## Cycle 10 — ggml's built-in HIP graph capture: FALSIFIED at -0.73%

ggml ships `GGML_HIP_GRAPHS=ON` (default OFF, ggml's own description:
"experimental, slow"). We tested the full 2×2 matrix (tile-override ×
ggml-graphs):

| Tile | ggml graphs | Mean tok/s | Δ |
|---|---|---|---|
| OFF | OFF | 62.65 | baseline |
| ON  | OFF | 63.86 | +1.93% |
| OFF | ON  | 62.19 | **-0.73%** |
| ON  | ON  | 63.51 | +1.37% (graphs erase ~30% of tile gain) |

ggml's "experimental, slow" disclaimer holds. Per cycle 12's hipGraph
port subagent analysis, the actionable fix would be to bypass ggml's
cgraph-wide capture and do hipfire-style surgical per-forward capture
— but this is blocked by `build_target_step_tree` baking `kv_start`
into `ggml_view_3d` offsets (a single captured graph would write to
the wrong KV slot on every cycle past the first). Estimated 3-5 days
of integration + a ggml refactor for +1-2% best case. Not worth doing
without ggml-side work.

## Cycle 11 — rocprofv2 decode-time attribution on gfx1100

Ran `rocprof --stats` on test_dflash for HE-0, n_gen=32, budget=8.
27,766 kernel events, 48 unique kernel signatures.

**The decode-time breakdown that should have been in the original
plan**:

| Category | % of GPU time | Notes |
|---|---|---|
| **q4_K MMQ** | **34.98%** | target weights; cycle 9 tile override acts here |
| **rocBLAS FP32 GEMM** `Cijk_..._SB_MT64x64x8_..._ISA1100` | **29.75%** | **drafter forward** (FP32-class Tensile fallback for ncols=16 — see cycle 14) |
| **q6_K MMQ** | **13.96%** | token_embd + output projection in Q4_K_M's mixed-quant |
| **q5_K MMQ** | 3.72% | minor mix-in |
| rocBLAS FP16 WMMA GEMM (HB tiles) | ~4% | FA / attention paths |
| Qwen3 SSM (`gated_delta_net` + `ssm_conv`) | ~2.8% | DeltaNet path |
| Activation quant (`quantize_mmq_q8_1`) | ~1.2% | |
| **MMVQ** (`mul_mat_vec_q`) | **1.10%** | **The MMVQ rabbit-hole was attacking 1% of decode** |
| Norms + RoPE + k_bin_bcast + copies | ~5% | thousands of tiny launches |
| Launch overhead (estimated) | ~10-15% | ~10k tiny kernels at 3-5 µs each |

**Implication for Tier 3**: the original "multi-row decode GEMV à la
hipfire" hypothesis was attacking q4_K MMQ time (34.98% of decode).
That's the right surface area — but cycles 1-7 testing the multi-row
GEMV prototype against the lucebox stack landed at −20.4% to −23.5%
per-cycle (documented in the
`Kaden-Schutt/llama.cpp-dflash-ggml@feat/mmvq-rdna3-batch16` research
branch). The multi-row hipfire pattern beats Tensile on hipfire's
runtime; on lucebox-hub's ggml backend it regresses because
hipfire's wins depend on a Rust-native dispatch layer + per-arch
custom kernels + per-forward hipGraph capture that ggml's
infrastructure doesn't have. Tier 3 should be considered superseded
by cycle 9 (tile tuning) + cycle 14 (drafter swap) at the
HIP-friendly cost/benefit ratio.

## Cycle 12 — tile sweep + 3 parallel subagent investigations

**Tile sweep**: WMMA static-assert `nwarps × tile_C::I == mmq_y`
forces only three valid `(mmq_y, nwarps)` triples — (32, 2), (64, 4),
(128, 8). Cycle 9's (64, 4) is the empirical peak (63.86); (32, 2) is
within noise (63.70); (128, 8) is stock-equivalent (62.41).

**q6_K subagent**: cycle 9's override is type-agnostic, so q6_K and
q5_K already inherit the (48, 64, 4) values. The +1.93% global is the
combined effect across all MMQ types. Per-type tile tuning yields ±0.5%
expected — not worth pursuing.

**hipGraph port subagent**: hipfire's surgical capture is 30 LOC of
bridge code, but porting to lucebox requires bypassing ggml's
backend-graph layer AND refactoring `build_target_step_tree`'s
`kv_start` offsets. 3-5 days first-pass + 1-2 weeks if the ggml
refactor is harder than estimated. Best-case yield: +1-2% per
realistic gfx1100/ROCm-7.2 perf data.

**Drafter FP16 subagent (the cycle 14 setup)**: cycle 11's "drafter
runs FP32 = 29.75%" attribution was *literally* correct but the
*reason* was wrong. Drafter weights are already F16 on HIP
(`build_prefers_bf16_projection() == false`); the slowness comes from
F32 **activations** + small-N=16 GEMM shapes falling outside Tensile's
WMMA solution coverage on ISA1100. Dispatch trace:
`ggml_cuda_mul_mat` → MMF rejected by `mmf.cu:169` (RDNA3_0 +
ncols>8) → batched-cublas-f16 rejected (single-batch) →
`ggml_cuda_op_mul_mat_cublas` → Tensile picks
`_SB_MT64x64x8_SN_..._ISA1100` SGEMM-class tile.

## Cycle 13 — MMF cutoff lift on RDNA3_0: FALSIFIED at -2.78%

Tried the subagent's "free 1-line experiment": patch `mmf.cu:169`
from `ncols > 8` to `ncols > 16` on RDNA3_0 to route the drafter
q_len=16 case into MMF's FP16-WMMA path instead of Tensile.

| Tile | MMF lift | tok/s | AL  | Δ |
|---|---|---|---|---|
| OFF | OFF | 62.65 | 5.25 | baseline |
| ON  | OFF | 63.86 | 5.25 | +1.93% |
| OFF | ON  | **60.91** | 5.29 | **-2.78%** |
| ON  | ON  | 62.49 | 5.29 | -0.26% (MMF erases tile gain) |

The conservative `>8` cutoff at `mmf.cu:169` is empirically tuned and
correct. MMF FP16-WMMA at ncols=16 is slower per-call than Tensile's
SGEMM-class tile on RDNA3_0. AL went *up* slightly (5.25→5.29 —
drafter numerics under MMF FP16 align slightly better with target's
argmax) but per-cycle time worsened more than AL helped. **Patch
reverted to default OFF.**

## Cycle 14 — Drafter swap to spiritbuun Q8_0 GGUF: **+4.40% to +8.85%** on top of cycle 9

The biggest single lever found across all 14 cycles, and it required
**zero code changes**. test_dflash already auto-detects `.gguf`
extension and routes through `load_draft_gguf` (`test_dflash.cpp:2438-2440`).

Drafter swap from `z-lab/Qwen3.6-27B-DFlash/model.safetensors`
(F16 dense via Tensile FP32 fallback) →
`spiritbuun/Qwen3.6-27B-DFlash-GGUF/dflash-draft-3.6-q8_0.gguf`
(Q8_0 GGUF via ggml's MMQ path):

| Run | Tile | Drafter | tok/s | AL | Δ vs stock |
|---|---|---|---|---|---|
| Stock        | OFF | F16 safetensors | 62.65 | 5.25 | — |
| Cycle 9      | ON  | F16 safetensors | 63.86 | 5.25 | +1.93% |
| GGUF run 1   | ON  | Q8_0 GGUF       | **66.67** | 5.54 | **+6.42%** |
| GGUF run 2   | ON  | Q8_0 GGUF       | **69.51** | 5.54 | **+10.95%** |
| **GGUF mean** | ON | Q8_0 GGUF | **68.09** | 5.54 | **+8.69%** |

**Why it works**: replaces 29.75% Tensile FP32-class fallback kernel
with ggml's `mul_mat_q<Q8_0, 16, false>` MMQ kernel — which ALSO
benefits from cycle 9's tile override (compounding effect). AL rose
5.25 → 5.54 (Q8_0 drafter is slightly better aligned with target than
F16 safetensors). Per-prompt heterogeneous (separate_paren_groups
+50%, below_zero +56%, sum_product +22%; intersperse -26%,
truncate_number -20%); all outputs remain coherent.

**Suggested ship**: README recommendation + a tested-known-good URL
for the GGUF. No `test_dflash.cpp` change needed.

⚠️ **Watch out for Q4_K_M GGUF drafter**: spiritbuun's model card
warns that 3.6 drafters have causal sliding-window attention which is
Q4-fragile. Q4_K_M drops acceptance from ~43% to ~28%. Use Q8_0 only.

## Cycle 15 — Sanity check vs hipfire's native DFlash on the same silicon

Ran hipfire's `dflash_spec_demo` (vanilla DFlash, **no DDTree**) on
the same 7900 XTX with `~/.hipfire/models/qwen3.6-27b.mq4` target +
`~/.hipfire/models/qwen36-27b-dflash-mq4.hf4` drafter.

| Prompt | hipfire mean (3 runs) | lucebox-hub mean (cycle 9+14) | Hipfire advantage |
|---|---|---|---|
| HE 0 has_close_elements | 152.85 (AL 8.00) | 78.53 (AL 6.24) | **+95%** |
| HE 3 below_zero         | 111.79 (AL 4.90) | 73.93 (AL 5.95) | **+51%** |
| **Mean over 2 HE prompts** | **132.32** | **76.23** | **+73.6%** |

Important caveats:
- **DDTree HURTS hipfire's DFlash impl**. At `--ddtree-budget=8` on
  hipfire, perf drops to 27.46 tok/s on HE 0. Hipfire's win is
  vanilla linear block-diffusion drafts at AL τ=8. Lucebox-hub
  *requires* DDTree because that's how the fork's DFlash is wired up
  in test_dflash.
- Different quant formats (hipfire `.mq4` vs lucebox `.gguf Q4_K_M`),
  different tokenizers (built-in vs HF Qwen3.5-27B), different prompt
  normalization (hipfire default-on vs lucebox raw HF tokens).
- Per cycle 12 subagent B, the structural ceiling for lucebox-hub on
  HIP without ggml refactors is ~70-80 tok/s. We're at 68. The
  ~+74% gap to hipfire is real and architectural.

The four structural reasons hipfire wins (per the subagent analysis):
(a) surgical per-forward hipGraph capture works on hipfire's code but
is blocked by ggml's `kv_start` design on lucebox; (b) hand-tuned
`.mq4` kernels per arch (lucebox's ggml MMQ is shape-generic);
(c) Rust-native pipeline avoids the ggml backend abstraction tax +
Tensile FP32 fallback (29.75% recovered by cycle 14, the rest is
launch + abstraction overhead); (d) vanilla linear DFlash drafts
> DDTree on this codebase.

## Cycle 16 — Strategic recommendation: pivot AMD direction toward Vulkan

After 14 cycles measuring every available HIP-side lever, the
ceiling-to-hipfire gap is **structural to the ggml backend** on
RDNA3 + ROCm 7.2.x, not a lack of kernel-level effort. Three options
for what comes next:

### Option A: Ship the HIP path as-is

Ship the HIP support PR + the two cycle-validated improvements
(cycle 9 tile override default-off macro, cycle 14 README drafter
recommendation). Stop further HIP optimization. Realistic ceiling
on RDNA3 ROCm 7.2.x without ggml refactors: **~70-80 tok/s** on
canonical 27B Q4_K_M HE10 budget=8.

Pros: low risk, gets gfx101x/gfx102x/gfx110x/gfx115x/gfx120x users
working today. The cycle 9 / cycle 14 wins compose to +6.4-10.95%
over post-Tier-1 baseline.

Cons: leaves ~64% of the silicon's practical ceiling on the table
(132 vs 76 tok/s on 7900 XTX). All future maintenance is ROCm
version-sensitive; the Tensile FP32 fallback path will be hit again
whenever AMD ships a new quant variant.

### Option B: Pursue hipfire-pattern hipGraph port (3-5 days + ggml refactor)

Bypass ggml's graph layer, capture surgically at the test_dflash
decode-step level, refactor `kv_start` to scatter-style writes.
Expected yield: +1-3% best case per the cycle 12 subagent analysis.
Realistic: -1% to +0.5% if the kv_start refactor isn't surfaced
upstream first (would race the wrong KV slot on cycles > 1).

### Option C: Pivot AMD direction toward Vulkan (RECOMMENDED)

Vulkan covers RDNA1 → RDNA4 + Intel discrete + Apple integrated in
**one backend**. Eliminates the entire ROCm version-compatibility
matrix that PR #156's predecessor work documented. Community data
on gfx1100 cites **Vulkan +25-30% over ROCm** for Q4_0 decode
(ggml-org discussion #20934, #21526).

**Measured 2026-05-11 (post-cycles): vanilla upstream llama.cpp
Vulkan AR on the same 7900 XTX + Qwen3.6-27B Q4_K_M:**

| Backend | HE 0 tok/s | HE 3 tok/s | Mean tok/s | VRAM used |
|---|---|---|---|---|
| HIP (ROCm 7.2.2 + cycle 9 tile) | 30.8 | 31.0 | 30.9 | 23.0 GiB |
| **Vulkan (RADV NAVI31)** | **37.5** | **39.5** | **38.5** | **16.2 GiB** |
| **Delta** | **+21.8%** | **+27.4%** | **+24.6%** | **-29.7%** |

Two surprising findings: (a) Vulkan beats HIP **at Q4_K_M too**
(community had only reported on Q4_0); (b) Vulkan uses **7 GiB less
VRAM** for the same model (16.2 vs 23.0 GiB) — context and compute
buffers are dramatically smaller on the Vulkan path, giving real
headroom for larger context or MTP setups that HIP can't fit.

**Trade-off the user named**: Vulkan loses hipfire-style per-arch
kernel tuning. Cycle 9's tile override (mmq.cuh edits) is
HIP/CUDA-specific; doesn't transfer. Vulkan uses GLSL/SPIRV compute
shaders compiled by the driver, not custom HIP intrinsics. Future
kernel-level hand-tuning on lucebox becomes lower-priority on the
Vulkan path. But cycle 9 was a +1.93% win out of a +6.42-10.95%
total — kernel tuning is **NOT load-bearing** for lucebox in the
way it is for hipfire (which gets its 132 tok/s by stacking many
small kernel optimizations on top of the surgical hipGraph capture).

**Practical sequencing**:
1. Ship the HIP support PR + cycle 9 + cycle 14 as the **fallback
   AMD path** (Option A). Users on pre-ROCm-7 stacks, gfx1010, or
   CUDA-portable contexts need this anyway.
2. Open a parallel Vulkan port of the DFlash kernels as the
   **strategic AMD direction**. Most of the spec-decode
   infrastructure (KV state mgmt, draft tree, accept logic) is
   vendor-agnostic; only the matmul + flash-attn kernels need
   Vulkan compute shader implementations. ggml-vulkan already has
   q4_K MMQ; the integration effort is bridging ggml-vulkan to the
   DFlash forward graph.

The recommendation is C. Updates the doc's "Sequence of PRs" section
above (see PR-D / PR-E renumbering below).

## Revised PR sequence (post-cycles)

Replaces the earlier "PR-A → PR-C → PR-D → PR-E" sequence:

1. **PR-A**: arch-aware `--ddtree-budget` default (Tier 1, daemon-side
   only, no submodule change). **Unchanged from original plan**.
2. **PR-B (NEW)**: cycle 9 `mmq.cuh` tile override for RDNA3_0
   (gfx1100/gfx1101). 6-line patch behind opt-in macro
   `LUCEBOX_GFX1100_TILE_OVERRIDE`, default OFF to preserve
   upstream behavior. +1.93% on canonical bench, AL-preserving.
   Effectively a port of the spirit of closed PR
   `ggml-org/llama.cpp#21344` to discrete RDNA3 (IMbackK's
   never-acted-on suggestion).
3. **PR-C (renumbered, was original PR-C)**: dropped pending
   evidence — multi-row decode GEMV cascades 1-7 falsified on
   lucebox-hub's ggml backend; hipfire's pattern doesn't transfer
   without surgical hipGraph capture which is itself blocked.
4. **PR-D (NEW)**: README + bench script recommendation for
   `spiritbuun/Qwen3.6-27B-DFlash-GGUF/dflash-draft-3.6-q8_0.gguf`
   as the canonical drafter (replaces `z-lab/Qwen3.6-27B-DFlash`
   safetensors). +4.40-8.85% over PR-A baseline at cost of zero
   code changes. ⚠️ explicitly warn against Q4_K_M variant of same
   model (Q4-fragile per upstream model card).
5. **PR-E (renamed, was original PR-D)**: gfx1010/1030
   scalar-fallback score kernel (Tier 4). Unchanged.
6. **PR-F (renamed, was original PR-E)**: rocWMMA port of
   `flashprefill_kernels.cu`. Unchanged.
7. **PR-G (NEW, strategic)**: Vulkan port of DFlash kernels as the
   long-term AMD direction. Separate workstream from PR-A through
   PR-F (which keep the HIP path as fallback). Not blocked on any
   HIP-side work landing.

Validation rubric per PR unchanged from the original plan.

## Combined gain after PR-A + PR-B + PR-D

Stacking the three actionable wins (Tier 1 arch-aware budget + cycle 9
tile override + cycle 14 drafter swap):

| Lever | Δ |
|---|---|
| Stock (pre-PR-A, default budget=22) | baseline 49.81 tok/s |
| PR-A (budget=8 on gfx110x) | +53% → 76.02 tok/s |
| + PR-B (tile override on RDNA3_0) | +1.93% on top → 77.49 tok/s |
| + PR-D (Q8_0 GGUF drafter) | +6.6% on top → **82.61 tok/s mean** |

Run-to-run variance: ±2-5% (gfx1100 thermal/DPM drift) → realistic
range **~80-86 tok/s mean** after all three PRs land. Up from 49.81
at the original baseline. **+65% combined gain** vs the pre-Tier-1
state, still ~38% below hipfire's 132 tok/s on the same silicon.

## Findings local artifact

All 21 research markdown docs (cycle 9 through cycle 16) are kept at
`Kaden-Schutt/lucebox-hub@feat/mmq-q4_K-rdna3-custom:tier3-research/`
(LOCAL only, not pushed to any remote). The doc set is the complete
audit trail for anyone wanting to reproduce a falsified cycle without
re-running the experiment.
