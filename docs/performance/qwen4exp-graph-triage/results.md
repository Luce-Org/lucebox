# T14: qwen4exp HIP-graph quality triage

Date: 2026-09-20. Hardware: Ryzen AI Max+ / gfx1151, ROCm 7.2.2 runtime.
Model: Qwen3.8-Flash-Next IQ4_NL. Every GPU run held
`/tmp/qwen-perf/gpu.lock`; only one model was resident. The delivered qwen4exp
environment was asserted from `/proc/<pid>/environ`. The graph candidate used
`server/build-hip-graphs-t14` (`GGML_HIP_GRAPHS=ON`) with
`QWEN4EXP_DECODE_STABLEGRAPH=1`; the control used `server/build-hip`
(`GGML_HIP_GRAPHS=OFF`). Both final binaries were rebuilt after removing the
temporary logit telemetry.

## Verdict

The E30 response variance does **not originate in the stable T=1 graph's KV
bucketing, capture/replay state, or device-scalar/input handoff**. A divergent
response can choose a different first generated token, and that token is
sampled from the prefill logits before the first T=1 graph call. Full-vocabulary
logit hashes also change between identical requests with HIP graphs disabled.

No graph-path source change is justified by these results. The exact low-level
prefill kernel responsible was not isolated: the variability survives disabling
QSA, then MMB, selecting `QWEN4EXP_UPSTREAM=1`, and disabling tiled GDN. The
smallest correctness containment remains the already-reverted defaults:
`GGML_HIP_GRAPHS=OFF` and `QWEN4EXP_DECODE_STABLEGRAPH` opt-in. This containment
meets the release policy after E30, although the graph-off exact path is itself
not strictly byte deterministic.

Semantic quality is healthy: the graph-on candidate passed 16/16 standard
recall cases and HE 10/10, GSM 10/10, Math 9/10. Strict byte stability still
fails across fresh graph-on processes (3 identical streams, 1 wording variant),
so T14 does not recommend re-enabling the defaults.

## Root-cause discriminators

| Experiment | Graph build | Decode path | Repeats | Result | Interpretation |
|---|---:|---|---:|---|---|
| E32 exact | OFF | exact-length, stable path OFF | 8 in one process | 7 dominant streams, 1 wording variant; facts 8/8 | Variance exists without bucketing and without HIP graphs. |
| E32 bucket | OFF | stable bucket | 8 in one process | 8/8 byte-identical | Bucketing alone does not reproduce the failure. |
| E33 capture | ON | stable bucket | 8 in one process | 8/8 byte-identical; capture 1, eager 1, replay 222 by the last reported interval | Warmup/capture/replay did not reproduce stale state. |
| E34 E30 order | OFF / ON | four small requests, then 8 recall requests | 8 each | OFF 8/8 and ON 8/8 byte-identical | Reusing a graph across the E30 request sequence did not reproduce stale workspace. |
| E40 fresh process | OFF | stable bucket | 4 | 4 dominant streams | Control sample. |
| E40 fresh process | ON | stable bucket | 4 | 3 dominant streams, 1 different first-token wording; facts 4/4 | Strict fresh-process byte gate fails. The divergent first token precedes T=1 execution. |

The dominant 13k answer SHA-256 prefix is `303a91466ac4`. The exact graph-off
variant is `98035ba78fdc`; the fresh graph-on variant is `944d4d7f7831` and
begins with lowercase `the` instead of `The`, proving first-token divergence.

## Logit evidence

Temporary `QWEN4EXP_LOGIT_TRACE=1` telemetry hashed the complete vocabulary
logit vector and recorded the top two logits. Step 0 is the prefill result; the
backend samples and emits it before calling T=1 `qwen4exp_forward`.

| Configuration, graphs OFF unless stated | Equal selected tokens | Equal top-2 pairs | Equal full hashes | Median / max absolute best-logit drift |
|---|---:|---:|---:|---:|
| Delivered, graph OFF | 32/32 | 27/32 | 0/32 | 0.172 / 0.809 |
| Delivered, graph ON | 32/32 | 27/32 | 0/32 | 0.485 / 1.541 |
| QSA OFF | 32/32 | 24/32 | 0/32 | 0.340 / 2.591 |
| QSA OFF, MMB OFF | 32/32 | 27/32 | 0/32 | 0.350 / 1.107 |
| Upstream profile, QSA/MMB OFF | 32/32 | 26/32 | 0/32 | 0.354 / 1.343 |
| QSA/MMB OFF, tiled GDN OFF | 32/32 | 29/32 | 0/32 | 0.334 / 1.119 |

Every row already differs at step 0. This excludes the stable decode KV scalar,
fixed input buffer, and replayed decode workspace as the cause of the first
token drift. It does not prove that graph support can never change later-token
rounding; it shows that the E30 symptom is part of a pre-existing prefill
numerics variance and cannot be repaired in the T=1 graph path alone.

## Required gates

| Gate | Result | Evidence |
|---|---|---|
| Standard recall, graph ON | 8 full suite runs, **16/16** cases correct | `raw/e41-graphs-on-recall8.*` |
| HE / GSM / Math, graph ON | **10/10 / 10/10 / 9/10** | `raw/e42-full-quality-graphs-on.*` |
| Same-process recall byte stream | **8/8 identical** in both matched OFF and ON sequences | `raw/e34-sequence-graphs-*.json` |
| Fresh-process byte stream | **failed**: ON has 2 streams across 4 processes | `raw/e40-fresh-*.json` |
| Upstream reference differential | All aligned activation counts, finiteness, absmax, and means match exactly. Current upstream run omitted all I32 expert-ID callbacks, so the strict wrapper is **inconclusive/nonzero**. Candidate IDs match the last valid upstream capture on all 48 layers. | `raw/e46-reference-diff-clean.log`, `raw/e46-reference-diff-clean/`, `measurement-summary.json` |
| Prefill at 16,366, graph ON | **1039.45 [1020.28–1043.94] t/s**, N=8 after one warm-up | `raw/e44-prefill-graphs-on.*` |

The upstream omission occurred in two fresh runs, before and after removing the
diagnostic telemetry. Both runs still matched every comparable activation
statistic exactly. The historical expert-ID comparison is supporting evidence,
not a substitute for a presently green strict wrapper; this report therefore
does not claim that gate as freshly passed.

## Decode A/B

One warm-up plus N=8 per depth, greedy 128-token generation. Values are median
`[min–max]` from server timings.

| Effective prompt | Graphs OFF, stable path | Graphs ON, stable path | Delta |
|---:|---:|---:|---:|
| 1,996 | **30.00 [29.80–30.10] t/s** | **29.95 [29.80–30.00] t/s** | -0.17% |
| 16,348 | **28.15 [28.00–28.20] t/s** | **28.25 [26.30–28.40] t/s** | +0.36% |

The graph-on deep run contains two thermal outliers (26.6 and 26.3 t/s); its
upper six samples are 28.2–28.4. Telemetry proves replay engaged: after the two
depth buckets, the final reported aggregate was 2 captures, 2 eager executions,
and 2,044 replays. In this matched stable-path comparison, capture/replay has no
material median throughput effect.

## Artifacts

- `measurement-summary.json`: machine-readable min/median/max and logit analysis.
- `analyze.py`: deterministic summary generator.
- `t14_probe.py`: request-order, fresh-process, quality, and logit probe.
- `raw/`: live environments, responses, telemetry, quality JSON, and reference logs.
- `final-build-state.txt`: final OFF/ON CMake cache values and clean diagnostic diff.

