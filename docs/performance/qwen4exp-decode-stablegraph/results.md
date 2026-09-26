# Qwen4Exp stable T=1 decode graph (E23/E24)

## Result

`QWEN4EXP_DECODE_STABLEGRAPH=1` now keeps one allocated T=1 graph per active
KV-span bucket and lets ggml's existing HIP graph machinery capture and replay
it. The flag is default-off and is ignored by `QWEN4EXP_UPSTREAM=1`. No
ggml-cuda source change or CUDA-graph override was needed.

The lock-held IQ4_NL A/B used the delivered environment, greedy decoding, 128
output tokens, one warm-up, and N=8 measured requests at each depth. Control and
candidate used the same `GGML_HIP_GRAPHS=ON` binary; only the stable-graph flag
changed.

| Effective prompt | Control median [min-max] | Stable graph median [min-max] | Delta |
|---:|---:|---:|---:|
| 1,996 | 28.578 [28.239-28.758] t/s | **29.887 [29.707-30.017] t/s** | **+4.58%** |
| 16,348 | 25.317 [25.089-25.511] t/s | **28.064 [27.955-28.171] t/s** | **+10.85%** |

The larger deep-context gain is still launch/rebuild removal: decode telemetry
remained dense flash attention (`qsa=0,dense=12`) for both stable decode graphs.
The `qsa=12,dense=0` lines at 16k belong to the preceding prefill graph.

## Implementation

- Stable decode uses fixed device allocations for embeddings, positions, mask,
  PLE input, KV-row index, and logits. Values are uploaded asynchronously before
  each replay, so the graph sees invariant pointers.
- KV append uses `ggml_set_rows` into the full fixed cache with a device-side
  row-index tensor. This replaces the advancing destination views that made E20
  incompatible with HIP graph replay.
- Dense attention uses a fixed 256-token bucket plus one 256-token generation
  window. An uploaded F16 mask hides the bucket tail. A graph is rebuilt only
  when live KV length exceeds the active bucket.
- The graph object, metadata context, and gallocr plan remain alive in
  `Qwen4ExpDecodeWorkspace`. The ordinary rotating UMA input ring is bypassed
  only in stable mode.
- The existing `ggml_backend_cuda_graph_compute` path recognizes the stable
  graph. No `ScopedCudaGraphOverrides` integration was necessary.

At the last sampled point in the N=8 candidate, HIP graph telemetry reported
`total=2240 replay=2236 capture=2 eager=2 enabled=1` for the 5,147-node stable
graph. Those two captures correspond to the ~2k and ~16k buckets.

## Construction and dispatch evidence

The phase profile contains two requests per depth (warm-up plus measured), or
254 T=1 forwards per depth because each request's first output comes from the
prefill logits.

| Depth | Control build+alloc mean, median [min-max] | Stable mean, median [min-max] | Calls with nonzero build |
|---:|---:|---:|---:|
| ~2k | 0.946, 0.9 [0.9-2.3] ms/token | **0.011, 0.0 [0.0-2.7]** | 254/254 -> **1/254** |
| ~16k | 1.211, 1.2 [0.9-2.4] ms/token | **0.010, 0.0 [0.0-2.5]** | 254/254 -> **1/254** |

A separate rocprof runtime trace covered two ~2k requests: 254 T=1 forwards.
It recorded 253 `hipGraphLaunch` calls. Every graph-launch correlation expanded
to exactly 2,800 device kernel dispatches (253/253 correlations). Thus replay
does not remove device work; after capture it replaces roughly 2,800 eager host
submissions with one graph launch per token. The profiler slowed decode to 6.7
t/s, so it is attribution evidence only and is excluded from the A/B timing.

## Correctness and regression gates

| Gate | Result |
|---|---|
| Greedy stream identity | **PASS**: exact emitted byte/token sequence and 24 completion tokens for control vs candidate at 1,834 and 14,864 prompt tokens |
| Default quality | **PASS**: HE 10/10, GSM 10/10, Math 9/10, recall 2/2 |
| Upstream differential | **PASS**: all aligned activation ratios 1.0000, zero expert-ID mismatches, no onset under `QWEN4EXP_UPSTREAM=1` |
| Prefill 16,366, N=8 | **PASS**: best 15.30 s / 1,069.67 t/s; median 15.345 s / 1,066.54 t/s; range 15.30-16.02 s / 1,021.60-1,069.67 t/s |
| Build | `dflash_server` target built successfully with HIP graphs enabled |

The prefill best matches the prior default-on E22 result (15.30 s, about 1,070
t/s), so no prefill regression is measurable. The first prefill sample was the
cold 16.02 s tail; it is retained in the spread.

## Evidence

- Exact numeric summary: `measurement-summary.json`
- A/B: `stablegraph-{control,candidate}.json` and corresponding server logs
- Identity: `identity.json` and `stablegraph-identity-*.server.log`
- Quality and prefill: `quality.json`, `quality-harness.log`, `prefill.txt`
- Reference: `reference-diff.log`, `reference/ours.log`, `reference/upstream.log`
- Phase attribution: `stablegraph-phase-*.server.log`
- Runtime trace summary: `measurement-summary.json`; the 15 MB HIP API and 244
  MB kernel CSVs remain on the box under
  `/tmp/q4exp-followup/stablegraph-runtime-trace-trace/` to avoid adding 259 MB
  of generated trace data to the working tree.

All GPU work was serialized by `/tmp/qwen-perf/gpu.lock`. No commit was made.
