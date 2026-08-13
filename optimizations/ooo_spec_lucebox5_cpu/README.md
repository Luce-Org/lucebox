# Lucebox5 disjoint-CPU tool speculation

This experiment runs a predicted read-only/idempotent tool concurrently with
DeepSeek generation. The model stays on the R9700 + Strix GPU path with DS4
enabled, while the tool child process is pinned to CPU cores that the model
process cannot use. The result is released only when the generated canonical
tool call exactly matches the prediction.

## Measured result

The native engine benchmark on Lucebox5 passed its production gate:

| Metric | Result |
| --- | ---: |
| Sequential model + tool, p50 | 5511.64 ms |
| Speculative full task, p50 | 2910.40 ms |
| Exact-hit speedup | **1.8938x** |
| Bootstrap 95% CI | 1.8859x - 1.8964x |
| Task-latency reduction | 47.20% |
| Model-compute slowdown | 0.46% |
| DS4 median acceptance rate | 0.4167 |
| Correct predictions | 20 / 20 |

This workload's zero-interference ceiling is 1.9011x because model and tool
latencies are not perfectly equal. The implementation reaches 99.6% of that
ceiling; claiming 2x for this measured workload would be inaccurate.

The control arm runs the model and then the identical CPU-pinned sparse tool.
The speculative arm starts that tool before generation. Arm order is randomized
inside each of 20 warm pairs, with two warmups. A 20,000-resample paired
bootstrap supplies the confidence interval. Model outputs, canonical calls,
and tool checksums are identical across arms. A deliberately wrong prediction
produces an `invocation_mismatch` and exposes no private tool result.

Raw artifacts:

- `results/lucebox5-cpu-native-20pairs.json` (`sha256:4a7f224f44cc7f2385e476f8227c6d51d4f4b90052e6a7b090bafcfc1f3b68a7`)
- `results/lucebox5-cpu-lane-qualification.json` (`sha256:6fc3f6d95c12db817b687b8c9509517d24230a508f520989483c1c6ed96c67df`)
- `profiles/lucebox5-cpu-lane-qualified.json` (`sha256:5cdf2550bb5a95c835daddac9f8d0126f470a5ac359e14008207e82c5dafb718`)

## Isolation and compatibility

Lucebox5 reserves logical CPUs `14-15,30-31` for the two-thread sparse tool and
launches the model with `0-13,16-29`. Startup fails closed if either mask
overlaps, if a listed CPU does not exist, or if an in-process executor is used.
Each child is pinned and its mask is read back before the request payload is
sent, so tool work cannot begin on model CPUs.

The engine feature is Linux- and backend-neutral: a single-GPU system can use
the same child-process path when it has CPU cores to reserve. It does not
replace or disable autoregressive decoding or DS4 token speculation. The
benchmark requires a positive DS4 acceptance rate on every measured request;
the median was 0.4167.

This improves the full latency of a correctly predicted tool-using request; it
does not double token generation throughput. On a miss, generation remains
authoritative, the speculative result stays private, and the caller executes
the generated tool call normally.

## Reproduce

Build the deterministic sparse-compute executor:

```bash
JSON_INCLUDE=/path/to/server/deps/json/include \
  ./build_cpu_sparse_executor.sh ./cpu_sparse_tool_executor
```

Launch the qualified native server on an otherwise idle Lucebox5:

```bash
./run_native_cpu_server_lucebox5.sh
```

Then run the measured gate:

```bash
python3 benchmark_cpu_tool_speculation.py native \
  --url http://127.0.0.1:18145/v1/chat/completions \
  --binary ./cpu_sparse_tool_executor \
  --tool-cpus 14-15,30-31 \
  --iterations 172452 \
  --max-tokens 32 \
  --pairs 20 \
  --warmups 2 \
  --bootstrap-resamples 20000 \
  --min-speedup 1.8 \
  --min-speedup-ci-low 1.7 \
  --max-model-slowdown-percent 5 \
  --output results/lucebox5-cpu-native-20pairs.json
```

The harness exits nonzero if correctness, isolation, DS4 activity, slowdown,
or either speed threshold fails.
