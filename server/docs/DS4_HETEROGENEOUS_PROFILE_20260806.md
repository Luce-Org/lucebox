# DS4 heterogeneous decode profile — 2026-08-06

This note records the kernel-level profile of the qualified R9700 + Strix Halo
q=5 path. Its main conclusion is that attention head splitting does create real
concurrent work, but attention is not the throughput bottleneck. The routed
expert kernels are the largest optimization target.

## Measurement method

`rocprofv3` collected kernel and memory-copy traces from matched control and
25% attention-split runs. The publication client records Linux monotonic-clock
timestamps for request start, first token, and request completion. The analyzer
selects only the first-token-to-completion intervals of measured requests, so
model loading, prefill, warmups, and time between requests are excluded.

Three measured 2K-context, 128-output-token requests were present in each
short trace. All six responses matched SHA-256
`0f785a7ffa406498aafb14553966eaed0f52220fed0f7cc016b66921d104d194`.
Profiler throughput is lower than unprofiled throughput and is used only for
matched attribution.

Agent 1 is the R9700 (`gfx1201`); Agent 2 is the Strix Halo (`gfx1151`).

## Exact decode-window result

| Measurement | Control | 25% attention split |
| --- | ---: | ---: |
| Selected decode span, three requests | 5.658 s | 5.780 s |
| R9700 busy | 3.313 s | 3.225 s |
| Strix busy | 0.897 s | 1.335 s |
| Both devices busy | 0.703 s | 0.989 s |
| R9700 work overlapped | 21.20% | 30.67% |
| Strix work overlapped | 78.34% | 74.07% |
| Summed R9700 dispatch work per request | 1104.782 ms | 1075.195 ms |
| Summed Strix dispatch work per request | 298.961 ms | 445.068 ms |

The split increases simultaneous device work by about **95.3 ms per request**.
It removes only **29.6 ms per request** from the R9700 while adding
**146.1 ms per request** to Strix. The peer branch therefore extends past the
work it was intended to hide. This is why increased overlap does not produce a
throughput gain: useful overlap increased, but total work increased more.

The packed direct-KV path also performs 117.055 ms of device-to-device copies
over the three split requests, or about **39.0 ms per request**. Removing that
copy alone is insufficient: the added Strix attention kernels still cost much
more than the R9700 work they replace.

The main added Strix work per request was approximately:

- attention GEMMs: 41.6 ms;
- ROCmFP4-fast matrix-vector work: 27.4 ms;
- Q projection: 16.3 ms;
- RoPE: 8.5 ms;
- remaining copies, softmax, normalization, and joins: the balance.

The R9700's largest removed individual contribution was 37.9 ms per request,
followed by smaller matrix-vector, softmax, projection, and RoPE reductions.
Smaller 48-head R9700 shapes also select less efficient kernels, so removed
logical work does not translate linearly into saved dispatch time.

## Telemetry warning

Do not interpret the coarse `attention_us` counter as the complete fused
attention cost. In q=5 fused verification, the large scheduler graph contains
attention, expert work, and joins together. The small counter is populated by
fallback/replay bookkeeping and may also be divided across speculative steps.
It cannot support a claim that attention costs 0.7–0.9 ms. Use timestamped
kernel traces and matched request windows for component attribution.

## Rejected experiments

| Experiment | Exact 2K median | Decision |
| --- | ---: | --- |
| Persistent peer KV cache, ordinary replay policy | 60.32 tok/s | reject |
| Persistent peer KV cache, forced property-scan bypass | 60.246 tok/s | reject |
| Five-token same-expert weight-reuse kernel | 79.380 tok/s | reject |
| Two-token task-parallel weight-reuse kernel | 78.068 tok/s | reject |

The peer-cache design removed the full-history transfer but changed scheduler
topology and lost native HIP-graph replay for the main verification graph. The
property-scan bypass could not recover it. The two weight-reuse kernels were
correct, but serializing multiple activation dot products per wave reduced
occupancy and parallelism more than shared weight decoding reduced memory
traffic. All rejected paths remain disabled; the two-token source experiment
was removed after qualification.

## Real bottleneck and next target

The dominant active-device work is the full-width routed-expert ROCmFP2 and
ROCmFP3 matrix-vector computation. In the control trace, those kernels consume
roughly 0.69 s on the R9700 and 0.80 s on Strix across three requests, before
their launch-preparation and combine work. The peer is already almost fully
hidden when it runs, while the R9700 has the long tail.

The next useful optimization must make the existing full-width expert kernels
faster without reducing occupancy, or move a coarse independent stage whose
complete peer branch finishes before the R9700 tail. Small attention/shared
matrix shards and per-expert serialization are measured dead ends on this
hardware pair.

## Reproduce the analysis

Run the overlap analyzer on each trace:

```bash
server/scripts/analyze_rocprof_overlap.py \
  rocprof/trace_kernel_trace.csv \
  --requests-json decode-client.json \
  --memory-copy-trace rocprof/trace_memory_copy_trace.csv \
  --top 30
```

Compare matched kernel work:

```bash
server/scripts/compare_rocprof_decode.py \
  CONTROL/rocprof/trace_kernel_trace.csv CONTROL/decode-client.json \
  CANDIDATE/rocprof/trace_kernel_trace.csv CANDIDATE/decode-client.json \
  --top 30
```

Raw artifacts are retained on lucebox5 under:

```text
/home/lucebox5/ds4-attention-full-tp-proven-20260806/results/attention_tp/
  profile-control-kernel-short-20260806/
  profile-attention25-kernel-short-20260806/
  attention25-incremental-coherent-v2-2k-20260806/
  attention25-incremental-force-replay-2k-20260806/
/home/lucebox5/ds4-attention-full-tp-proven-20260806/results/expert_kernel/
  control-mmid-group-reuse-2k-20260806/
  control-mmid-pair-reuse-r2-2k-20260806/
```
