# CPU-isolated tool speculation on Lucebox5

The engine asks a small predictor for one concrete tool call before the target
model runs. On Lucebox5, Qwen3-0.6B Q8_0 predicts on Strix, then exits its GPU
compute window; the predicted read-only tool runs on reserved CPU cores while
DeepSeek-V4-0731 decodes with DS4/DSpark on R9700 + Strix. The result stays
private unless DeepSeek emits the exact same canonical function and arguments.

This path does not inject tokens, replace DSpark, or retry speculative decoding
with autoregressive decoding. A wrong prediction is discarded and the caller
executes the target model's authoritative call normally.

## Production result

The 2026-08-17 paired run compiled a recurring, side-effect-free five-step
trace into one typed workflow tool. Independent branches ran concurrently on
the isolated CPU lane. The six measured tasks covered 10, 15, and 20 leaf calls
twice each, with randomized arm order and one warmup task.

| Metric | Result |
| --- | ---: |
| Normal stage-batched workflow, p50 | 81.030 s |
| Trace-compiled + speculative workflow, p50 | **14.597 s** |
| End-to-end speedup, paired p50 | **5.5961x** |
| End-to-end bootstrap 95% CI | **5.4577x–5.6599x** |
| Trace compilation alone | **3.2806x** |
| Early launch on top of compilation | **1.6954x** |
| Early-launch bootstrap 95% CI | **1.6760x–1.7210x** |
| Exposed tool wait, compiled / speculative p50 | 10.143 s / **0.027 ms** |
| Qwen prediction latency, p50 | 203.5 ms |
| Target model-compute slowdown, p50 / p95 | -0.458% / -0.332% |
| Target decode slowdown, p50 / p95 | -0.101% / 0.219% |
| Exact predictor-to-target hits | 6 / 6 |

All 20 production gates passed: identical leaf calls, tool-result hashes,
macro calls, and final outputs; positive DS4 acceptance on every call turn;
correct CPU isolation; and no measurable target slowdown. The 5.60x result
combines two independent gains: four fewer model/tool synchronization barriers
from trace compilation, plus the 1.70x gained by starting the compiled graph
before target authorization completes.

Artifact:

- `results/trace-compiled-engine-qwen-production-6pairs-compact.json`
  (`sha256:0807cca1d22453728b069a0150800fcfa9a513db6a9f25815663fc03b99285b9`)

The compact training fixture below reproduces the artifact's compiled pattern
fingerprint (`06d95882…0645`); the artifact retains the original full-report
hash for provenance.

## Safety and portability

- Only explicitly allowlisted, read-only/idempotent tools are eligible.
- The external result is committed only on an exact canonical call match.
- The executor is launched directly without a shell and has a hard timeout.
- Lucebox5 reserves CPUs `14-15,30-31`; the model uses `0-13,16-29`.
- Startup fails closed if CPU masks overlap or the measured lane profile fails.
- `before-model` is the native predictor default, so shared-GPU prediction
  cannot reduce target prefill/decode throughput.
- The same API works on a single GPU: run the predictor before the target and
  overlap only the CPU tool. An HTTP predictor can use the same verification
  and executor path on other model families.

The speedup applies to tool-using request latency, not token throughput. Its
real-world value depends on exact predictor hit rate and on how much tool work
can overlap target generation.

## Reproduce

Build the deterministic sparse tool used by the single-call qualification:

```bash
JSON_INCLUDE=/path/to/server/deps/json/include \
  ./build_cpu_sparse_executor.sh ./cpu_sparse_tool_executor
```

Launch the qualified single-call configuration on an otherwise idle Lucebox5:

```bash
./run_native_cpu_server_lucebox5.sh
```

The launcher defaults to Qwen3-0.6B Q8_0 on predictor GPU 1. Override placement
with `PREDICTOR_MODEL`, `PREDICTOR_GPU`, `PREDICTOR_MAX_CTX`, and
`PREDICTOR_MAX_TOKENS`. The adjacent `candidate-build` symlink in the wrapper
selects a build even though the qualified launcher clears ambient variables.

Run the single-call paired gate:

```bash
python3 benchmark_cpu_tool_speculation.py native-qwen \
  --url http://127.0.0.1:18145/v1/chat/completions \
  --binary ./cpu_sparse_tool_executor \
  --tool-cpus 14-15,30-31 \
  --iterations 172452 \
  --max-tokens 32 \
  --pairs 20 \
  --warmups 5 \
  --bootstrap-resamples 20000 \
  --min-speedup 1.6 \
  --min-speedup-ci-low 1.5 \
  --min-speedup-p05 1.5 \
  --min-prediction-hit-rate 1.0 \
  --max-model-slowdown-percent 5 \
  --output results/qwen-auto-production-20pairs.json
```

For the 10–20-call workflow gate, launch with the trace executor and macro
allowlist:

```bash
TOOL_SPEC_EXECUTOR=./trace_compiled_tool_executor.py \
TOOL_SPEC_ALLOW=resolve_customer,list_open_orders,get_order_details,calculate_shipping,prepare_customer_summary,execute_customer_workflows \
  ./run_native_cpu_server_lucebox5.sh
```

Then run:

```bash
python3 benchmark_trace_compiled_workflows.py \
  --binary ./bfcl_replay_tool_executor.py \
  --training-report results/trace-compiled-training-traces.json \
  --pairs 6 \
  --warmup-tasks 1 \
  --min-branches 2 \
  --max-branches 4 \
  --seed 814 \
  --bootstrap-resamples 20000 \
  --output results/trace-compiled-engine-qwen-production-6pairs-compact.json
```

The harness exits nonzero on any correctness, isolation, DS4-activity,
slowdown, hit-rate, or speed threshold failure.
