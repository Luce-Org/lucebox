# CPU-isolated tool speculation on Lucebox5

The engine can ask a small model for one concrete tool call before the target
model runs. If the call is allowlisted and its measured execution lane is
qualified, the engine starts the read-only tool privately. The target model
remains authoritative: the result is released only when the emitted function
name and canonical arguments match exactly.

On Lucebox5, Qwen3-0.6B Q8_0 predicts on Strix and then leaves its compute
window. The CPU tool runs on logical CPUs `14-15,30-31` while
DeepSeek-V4-0731 + DS4/DSpark runs on R9700 + Strix and CPUs
`0-13,16-29`. This path neither injects tokens nor replaces DSpark.

## Measured result

The 2026-08-18 production run used six paired tasks: two each with 10, 15,
and 20 leaf calls. Each branch contained five serial, deterministic read-only
calls; independent branches ran concurrently. Arm order was randomized.
Every arm included model turns, tool time, and a final answer produced from
the actual assistant/tool conversation.

| Metric | Result |
| --- | ---: |
| Stage-batched workflow, p50 | 87.998 s |
| Trace-compiled workflow, p50 | 32.877 s |
| Trace-compiled + speculative workflow, p50 | **20.474 s** |
| End-to-end paired speedup, p50 | **4.3597x** |
| End-to-end bootstrap 95% CI | **3.7696x–4.8252x** |
| End-to-end paired speedup, p05 | **3.6680x** |
| Trace compilation alone, paired p50 | **2.6876x** |
| Early launch on top of compilation, paired p50 | **1.7676x** |
| Early-launch bootstrap 95% CI | **1.2590x–1.8002x** |
| Exposed tool wait, compiled / speculative p50 | 10.149 s / **0.030 ms** |
| Qwen prediction latency, p50 | 201.0 ms |
| Target model-compute change, p50 / p95 | -0.027% / +0.090% |
| Target decode change, p50 / p95 | -0.658% / +0.374% |
| Exact Qwen predictor hits | 6 / 6 |

The slowdown figures come from a separate controlled A/B probe. Each task had
three alternating repetitions per arm, and every observation was preceded by
the same warm request. The gate compares the per-task median ratios and
requires matching cache state, completion tokens, call digest, and active
DS4 decoding.

All 22 gates passed, including stable leaf-call and result digests, exact
final answers, exact macro calls, CPU isolation, controlled p50/p95 model
slowdown, and a wrong-call probe in which no private result crossed the
exact-match gate.

These numbers apply to recognized, side-effect-free recurring workflows. The
4.36x combines fewer model/tool synchronization barriers from trace
compilation with the 1.77x gain from early tool launch. It is not a claim that
arbitrary single tool calls become 4.36x faster. The broader Qwen smoke suite
currently protects a 9/12 exact-argument baseline; a predictor miss falls back
to the authoritative call without changing the target output.

Evidence:

- `results/trace-compiled-engine-qwen-production-6pairs-compact.json`
  (`sha256:b1194bd1447f772dc9c90e6e801e99646bdce121e1b300398c45b54540b87d20`)
- `results/multiturn-cached-wordref-production-6tasks.json`
  (`sha256:2475697d418bffed0e9668da26ce6c88a85a952ce97d99749f440f97f9ac5bf9`)
- `results/trace-workflow-registry.json`
  (the benchmark artifact records its exact hash)

The report and registry paths inside the result are repository-relative, and
their recorded hashes match the committed files.

The full source report is retained deliberately. Its original per-call
speculation performance gate did not pass; the compiler reads only the two
executions whose calls, results, dataflow, and side-effect flags are valid.
That failed baseline is the reason the workflow was compiled. No model was
trained on this report, and the passing 4.36x result is recorded separately.

## Runtime contract

- Tools must be explicitly allowlisted and read-only or idempotent.
- A result is committed only on one exact canonical call match. Multiple,
  malformed, failed, timed-out, cancelled, or different calls are discarded.
- The child receives a minimal environment and only standard streams. Tool
  speculation fails closed unless Linux glibc 2.34+ descriptor isolation is
  available.
- CPU affinity is applied before the executor starts and is re-read before
  the request payload is released. The model and tool masks must be disjoint.
- Executors run in their own process group with a launch-based deadline and a
  bounded output size. Cancellation removes descendants.
- Native predictor IPC has a bounded startup, a private `0700` work directory,
  an inherited-descriptor allowlist, serialized requests, and deadline-aware
  prompt construction, tokenization, and generation.
- A production profile must declare a qualified non-accelerator or separate
  physical-GPU lane and match the configured executor contract.

Automatic prediction means clients do not need to send a prediction hint.
Clients do need to consume `dflash_tool_speculation.result` on a `hit` and use
it as the tool result; ignoring the extension remains correct but forfeits the
latency gain.

The schedule also supports a single GPU because predictor compute finishes
before target compute begins; only the CPU tool overlaps target generation.
Both models still have to fit in memory. The production numbers above were
measured on Lucebox5's dual-GPU placement, not on a single-GPU machine.

## Reproduce on Lucebox5

Build the deterministic sparse adapter used to qualify the CPU lane:

```bash
JSON_INCLUDE=/path/to/server/deps/json/include \
  ./build_cpu_sparse_executor.sh ./cpu_sparse_tool_executor
```

Launch the qualified server:

```bash
./run_native_cpu_server_lucebox5.sh
```

The launcher uses Qwen3-0.6B Q8_0 on predictor GPU 1 and the build selected by
the wrapper's adjacent `candidate-build` symlink. It clears ambient variables.
`PREDICTOR_MODEL`, `PREDICTOR_GPU`, `PREDICTOR_MAX_CTX`,
`PREDICTOR_MAX_TOKENS`, and `PREDICTOR_TIMEOUT_MS` apply only when invoking
the wrapper directly.

Run the single-call gate:

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

For the workflow gate, launch with the trace executor and macro allowlist:

```bash
TOOL_SPEC_EXECUTOR=./trace_compiled_tool_executor.py \
TOOL_SPEC_ALLOW=resolve_customer,list_open_orders,get_order_details,calculate_shipping,prepare_customer_summary,execute_customer_workflows \
  ./run_native_cpu_server_lucebox5.sh
```

Then run:

```bash
python3 benchmark_trace_compiled_workflows.py \
  --binary ./bfcl_replay_tool_executor.py \
  --training-report results/multiturn-cached-wordref-production-6tasks.json \
  --workflow-registry results/trace-workflow-registry.json \
  --pairs 6 \
  --warmup-tasks 1 \
  --min-branches 2 \
  --max-branches 4 \
  --interference-repetitions 3 \
  --seed 814 \
  --bootstrap-resamples 20000 \
  --output results/trace-compiled-engine-qwen-production-6pairs-compact.json
```

The harness exits nonzero on any correctness, privacy, isolation, DS4,
slowdown, hit-rate, or speed failure.
