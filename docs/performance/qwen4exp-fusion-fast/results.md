# Qwen4Exp lifetime-safe fast PLE/GDN convolution fusion

Date: 2026-09-22. Base: `b4bdf73a`. Candidate is uncommitted. All GPU work
held `/tmp/qwen-perf/gpu.lock`; one model was resident at a time.

## Decision

Accept the candidate. It retains `b4bdf73a`'s exact fresh-process state while
restoring the prefill throughput of parent `4ae60f1f`. The final path skips the
37 full CONCAT materializations at T=16,366, snapshots only the mutable
3/9-row recurrent history, and gives the allocator explicit lifetime edges for
the large coalesced inputs.

## Implementation

- The CONCAT dispatch again intercepts 36 GDN concats and one PLE concat. It
  writes the small checkpoint tail and saves the old recurrent state into
  otherwise-unused columns of the CONCAT allocation. It never writes the full
  `[T+H,C]` tensor.
- The late direct kernels read saved state from that graph-owned allocation and
  read the large input directly in coalesced `[T,C]` order.
- GDN adds the actual transpose input as unused `src[3]` of ordinary SSM_CONV;
  PLE adds it as unused `src[1]` of the first CONT. Those inputs are ignored by
  the ordinary operators but are real graph-node dependencies, so `ggml-alloc`
  retains them until the custom fused dispatch.
- No new environment flag was added. The path has the same gfx1151 and matcher
  guards as the original fusion.

An initial attempt used a `ggml_view_tensor` with `GGML_OP_NONE` as a keepalive.
`ggml_build_forward_expand()` classifies that tensor as a leaf, so its source
does not enter node lifetime accounting. It recovered throughput but failed
fresh-process equality. This attempt is rejected; the final compute-node edges
are the correction.

## Long-prefill performance

Delivered environment, `--max-ctx 40000 --chunk 16384`, effective prompt
16,366, one warmup plus N=8 measured requests. Rates are derived from each
reported TTFT. Parent values are E63/E64.

| Model | Parent `4ae60f1f` | Candidate median [min-max] | Candidate tok/s [min-max] | Delta vs parent |
|---|---:|---:|---:|---:|
| IQ4_NL | 14.490 s, 1129.48 tok/s | **14.340 [14.31-15.56] s** | **1141.28 [1051.80-1143.68]** | +1.04% median tok/s |
| GSQ IQ3_XXS | 20.950 s, 781.19 tok/s | **20.715 [20.69-20.75] s** | **790.06 [788.72-791.01]** | +1.14% median tok/s |

The IQ4 first measured sample was cold/slow at 15.56 s; the median and complete
spread are reported rather than discarding it. Dispatch telemetry recorded
36 GDN tail/direct pairs and one PLE tail/direct pair.

## Correctness and regression gates

| Gate | Result |
|---|---|
| Fresh-process C=1024 snapshots | **PASS**, N=3 byte-identical 144,784,108-byte files; shared SHA-256 `1f34fa0b724ae8ca93a1172077df1a02e8dc676b5d7b7c063a18b70a16bd926c` |
| C=1024 compute timing | **2827.769 [2712.529-3011.568] ms**, N=3 fresh processes; E61 pre-fix 2831.5 ms and materialized-CONCAT fix 2787.4 ms, within this test's spread |
| Default quality | **PASS**: HE 10/10, GSM 10/10, Math 9/10, recall 2/2 |
| `QWEN4EXP_UPSTREAM=1` differential | **PASS**, exit 0, no onset, all aligned ratios 1.0000, zero expert-ID mismatches in 48/48 layers |
| Build | **PASS**, Release HIP/gfx1151 candidate server, snapshot probe, and smoke differential target |

## Platform controls

The launched server environment was read from `/proc/PID/environ` and contains
all ten delivered settings. During IQ4, sclk was 2842/2878/2900 MHz
(min/median/max), fclk 2000 MHz, package power 17.1/131.1/136.1 W, temperature
31/68/76 C, and median GPU busy 100%. During GSQ, sclk was
2861/2889/2900 MHz, fclk 2000 MHz, power 18.0/121.1/133.0 W, temperature
37/68/77 C, and median GPU busy 100%. `rocm-smi` independently reported the
gfx1151 device at performance level high, 2900 MHz sclk, 2000 MHz fclk, and
1000 MHz mclk after each arm.

Raw compact evidence is in this directory. Large binary snapshots remain on
lucebox4 under `/tmp/qwen4exp-fusion-fast-evidence/`; they were excluded from
the repository evidence copy. `candidate.patch` is the exact tested product
patch (SHA-256 `c997e7849349d7a1f85ee0080d0cf934548265f09ce65fd33f5eb598dcc1a21d`).

