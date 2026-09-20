# T13: GSQ IQ3_XXS prefill dispatch table

Date: 2026-09-20. GPU runs were serialized with
`flock /tmp/qwen-perf/gpu.lock`; each timing set used one warm-up followed by
N=8 uncached requests. The live server environment was read from
`/proc/<pid>/environ` for every run and matched the delivered ten-variable
configuration. No source change is active under `QWEN4EXP_UPSTREAM=1`.

## Result

The accepted change adds 22 measured GSQ quantized `(type,M,K,shadow)` tuples
to the existing gfx1151, exactly-16,366-token `QWEN4EXP_DENSE_TABLE`. Each
tuple selects MMQ instead of MMB. The guard remains deliberately narrow:
other token counts, architectures, split buffers, probe runs, and the upstream
reference profile retain their previous dispatch.

| IQ3_XXS @ 16,366 tokens | Prefill ms, median [min-max] | Tokens/s, median [min-max] | Delta |
|---|---:|---:|---:|
| Table off, same binary | 28,235.1 [28,025.6-28,249.7] | 579.63 [579.33-583.97] | control |
| Accepted quant table | 21,591.2 [21,412.2-21,696.0] | **757.99 [754.33-764.33]** | **+30.77%** |
| BF16 rocBLAS extension | 19,076.1 [19,065.5-19,107.8] | 857.93 [856.51-858.41] | **rejected: corrupt long-context answer** |

Against the prior E18 headline of 589 t/s, the accepted result is +28.7%.
It does not reach the requested 1,000 t/s target. The post-change native trace
is 21.616 s wall / 21.158 s GPU-busy and attributes 6.189 s to routed MoE,
5.538 s to remaining dense MMB, 2.560 s to MMQ, and 1.387 s to rocBLAS.
The old 14.46 s “dense MMB” bucket therefore contained substantial native
BF16 HC/PLE work in addition to the GSQ quant matrices. Direct rocBLAS won the
BF16 event probes and produced 858 t/s end to end, but it failed the exact
16,366-token planted recall with corrupted text. It is not in the production
path. Routed MoE remains a separate bounded follow-up; this task does not
change it.

## Quantized dispatch probe

Routes are `1=shadow rocBLAS`, `3=MMQ`, `4=MMB-128`, and `5=MMB-256`.
Values are per-operation HIP-event medians in milliseconds over six repeats.
All 22 newly added rows are MMQ winners. Existing IQ4_NL and Q5_K rows are
shown for completeness; the larger shadow-backed Q5_K/Q6_K shapes remain on
their existing rocBLAS route.

| Type | M | K | Shadow | rocBLAS | MMQ | MMB-128 | MMB-256 | Winner / action |
|---|---:|---:|:---:|---:|---:|---:|---:|---|
| Q4_K | 512 | 2560 | no | - | 2.604 | 5.501 | 6.849 | add MMQ |
| Q4_K | 640 | 2560 | no | - | 3.052 | 6.848 | 8.091 | add MMQ |
| Q4_K | 2560 | 6144 | no | - | 23.915 | 63.988 | 83.604 | add MMQ |
| Q4_K | 6144 | 2560 | no | - | 22.185 | 60.229 | 75.417 | add MMQ |
| Q4_K | 10240 | 2560 | no | - | 36.342 | 99.588 | 126.563 | add MMQ |
| Q4_K | 12288 | 2560 | no | - | 43.186 | 115.661 | 155.442 | add MMQ |
| Q5_K | 512 | 2560 | no | - | 2.717 | 6.255 | 6.211 | add MMQ |
| Q5_K | 640 | 2560 | no | - | 3.227 | 7.672 | 7.544 | add MMQ |
| Q5_K | 2560 | 6144 | yes | 29.236 | 25.352 | 101.577 | 111.879 | existing MMQ |
| Q5_K | 6144 | 2560 | yes | 17.906 | 22.720 | 70.001 | 87.538 | keep rocBLAS |
| Q5_K | 10240 | 2560 | yes | 29.287 | 37.287 | 124.054 | 146.243 | keep rocBLAS |
| Q6_K | 512 | 2560 | yes | - | 3.516 | 5.528 | 7.592 | add MMQ |
| Q6_K | 640 | 2560 | yes | - | 4.110 | 6.370 | 8.819 | add MMQ |
| Q6_K | 2560 | 6144 | yes | 29.537 | 35.362 | 103.069 | 110.835 | keep rocBLAS |
| Q6_K | 6144 | 2560 | yes | 18.087 | 32.048 | 71.556 | 89.107 | keep rocBLAS |
| Q6_K | 10240 | 2560 | yes | 29.370 | 53.166 | 123.828 | 146.637 | keep rocBLAS |
| Q6_K | 12288 | 2560 | yes | 34.611 | 62.374 | 154.083 | 184.140 | keep rocBLAS |
| IQ4_NL | 2560 | 640 | no | - | 2.848 | 4.586 | 6.908 | existing MMQ |
| IQ3_S | 512 | 2560 | no | - | 2.676 | 8.491 | 7.625 | add MMQ |
| IQ3_S | 640 | 2560 | no | - | 3.118 | 10.231 | 9.013 | add MMQ |
| IQ3_S | 2560 | 6144 | no | - | 24.930 | 100.368 | 86.835 | add MMQ |
| IQ3_S | 6144 | 2560 | no | - | 21.898 | 89.679 | 80.216 | add MMQ |
| IQ3_S | 12288 | 2560 | no | - | 45.550 | 188.596 | 168.115 | add MMQ |
| IQ4_XS | 512 | 2560 | no | - | 2.494 | 9.876 | 7.955 | add MMQ |
| IQ4_XS | 640 | 2560 | no | - | 2.841 | 12.191 | 9.319 | add MMQ |
| IQ4_XS | 2560 | 6144 | no | - | 22.297 | 114.089 | 86.774 | add MMQ |
| IQ4_XS | 6144 | 2560 | no | - | 19.328 | 107.553 | 79.803 | add MMQ |
| IQ4_XS | 10240 | 2560 | no | - | 32.534 | 183.944 | 134.745 | add MMQ |
| IQ4_XS | 12288 | 2560 | no | - | 37.675 | 214.401 | 173.379 | add MMQ |
| Q2_0 | 2560 | 640 | no | - | 2.841 | 6.997 | 7.455 | add MMQ |

The diagnostic BF16 probe also compared native rocBLAS with both MMB tiles.
Those raw results are in `bf16-probe-summary.json`. Event-time wins alone were
not accepted because the combined candidate failed the planted-recall gate.

## Regression gates

| Gate | Result |
|---|---|
| GSQ short greedy sanity | PASS: `The capital of France is Paris` |
| GSQ exact 16,366-token planted recall | PASS: exact `QUINCE-AMBER-7731`; candidate branch exercised |
| GSQ default suite | PASS: HE 10/10, GSM 8/10, Math 10/10, recall 2/2. Suite prompts do not hit the exact-token table branch, so their compute path is unchanged. |
| GSQ upstream reference differential | PASS: no onset; all expert IDs match |
| IQ4_NL upstream reference differential | PASS: no onset; all expert IDs match |
| IQ4_NL prefill @16,366 | 1,073.69 [1,005.40-1,078.68] t/s; median 15,242.8 ms, consistent with the ~1,070 t/s baseline |
| IQ4_NL decode, ~2k | table on 29.8 [29.7-29.9] vs thermally matched table-off 29.85 [29.7-29.9] t/s |
| IQ4_NL decode, ~16k | table on 28.1 [27.9-28.1] vs thermally matched table-off 28.1 [27.9-28.1] t/s |

The first, cooler table-off decode server measured 30.0 and 28.5 t/s. A
reverse-order table-off replicate measured 29.85 and 28.1 t/s, matching the
candidate. This order effect is expected because the new branch requires
exactly 16,366 rows, while the decode A/B uses 1,996/16,348-row prefill and
T=1 decode; neither can execute a new row.

## Evidence map

- `measurement-summary.json`: computed timing medians and spreads.
- `probe-summary.json`, `bf16-probe-summary.json`: per-shape event results.
- `raw/t13-gsq-{control,quant-candidate}-n8.json`: accepted A/B requests.
- `raw/t13-gsq-candidate-n8.json`: rejected BF16 candidate timing.
- `raw/t13-trace-candidate-gsq.summary.json`: accepted native trace attribution.
- `raw/t13-gsq-quality.json`, `raw/t13-gsq-sanity.json`: quality gates.
- `raw/reference-{gsq,iq4}-summary.log`: upstream differential summaries.
- `raw/*.environ.json`: live delivered-environment assertions.

T12, the all-architecture regression required because HIP graphs are globally
enabled, remains a separate shipping gate.
