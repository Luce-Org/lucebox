# DeepSeek V4 expert implementation

This document summarizes the current DeepSeek V4 Flash expert execution path in DFlash. It focuses on the complex split-expert implementation used for the CUDA parent plus HIP expert worker configuration.

## Scope

DeepSeek V4 Flash uses 43 layers, 256 routed experts per layer, 6 routed experts per token, 1 shared expert, and 3 hash-routed early layers. The implementation supports:

- full GPU load when memory allows,
- hybrid expert load when full load does not fit,
- split execution where non-expert/core weights run on CUDA, hot experts run in a HIP worker, and cold experts run locally on CPU,
- route statistics and placement experiments,
- timing telemetry for FFN, IPC, graph-cache behavior, and worker residency.

Main code paths:

| Area | Files |
| --- | --- |
| DeepSeek4 backend setup and lifecycle | `server/src/deepseek4/deepseek4_backend.cpp`, `.h` |
| Forward graph and routing | `server/src/deepseek4/deepseek4_graph.cpp` |
| Weights and loader metadata | `server/src/deepseek4/deepseek4_internal.h`, `deepseek4_loader.cpp` |
| Hybrid expert storage | `server/src/common/moe_hybrid_storage.*` |
| Hybrid FFN evaluation | `server/src/common/moe_hybrid_ffn_eval.*` |
| Expert IPC client/worker | `server/src/common/deepseek4_expert_ipc.*`, `deepseek4_expert_ipc_daemon.cpp` |
| Backend IPC CLI entry | `server/src/ipc/backend_ipc_main.cpp` |
| Routing stats and placement JSON | `server/src/common/moe_hybrid_routing_stats.*`, `moe_hybrid_placement.*` |

## Execution modes

### Full CUDA load

The backend first attempts a full DeepSeek4 load on the target CUDA GPU. On a 24 GiB RTX 3090 this currently fails for V4 Flash because expert tensors require far more VRAM than available, so the backend falls back to hybrid mode.

### Hybrid parent-only mode

Hybrid mode partially loads non-expert/core tensors on the CUDA backend and stores only a configured subset of expert tensors in GPU memory. Remaining experts are backed by CPU memory. This mode is controlled through the generic MoE hybrid storage/eval helpers.

### CUDA parent + HIP worker split

The main tested mode is:

```bash
DFLASH_DS4_EXPERT_SPLIT=hip_hot_cpu_cold
DFLASH_DS4_EXPERT_IPC=1
DFLASH_DS4_EXPERT_IPC_BIN=<HIP build>/backend_ipc_daemon
DFLASH_DS4_EXPERT_IPC_MODEL=<DeepSeek4 GGUF>
DFLASH_DS4_EXPERT_IPC_SUDO=1
DFLASH_DS4_EXPERT_WORKER_BUDGET_MB=61000
```

In this mode:

1. The CUDA parent loads non-expert weights and metadata.
2. The parent computes a hot-expert placement.
3. The parent materializes local CPU cold-tail storage using the complement of hot experts.
4. A HIP `backend_ipc_daemon` worker is launched for hot expert evaluation.
5. For each routed FFN call, selected experts are partitioned:
   - hot IDs owned by the HIP worker are sent over IPC,
   - cold IDs are evaluated locally by the CUDA parent process on CPU,
   - results are summed on the parent.

The current uniform split on the remote Halo worker uses `hot/layer=210`, `total_hot=9030`, leaving `1978` cold experts in CPU memory. The parent core load is about `7383.7 MB`; the worker loads expert metadata plus resident hot experts according to its placement.

## Routing and FFN flow

DeepSeek4 forward follows:

1. HC pre for attention.
2. MLA attention.
3. HC post for attention.
4. HC pre for FFN.
5. FFN RMSNorm and router probabilities.
6. Expert selection:
   - first hash-routed layers use token ID to expert ID table,
   - later layers use top-k over router probability plus optional bias.
7. Hybrid/worker FFN evaluation.
8. HC post for FFN.
9. Final logits.

`deepseek4_step_hybrid()` drives the hybrid forward path. For each layer it creates `selected_host` and `weights_host`, then calls `eval_ds4_hybrid_or_worker()`.

`eval_ds4_hybrid_or_worker()` has three important cases:

- no IPC worker: evaluate hot/cold using local `eval_moe_hybrid_ffn_*`;
- IPC worker owns cold IDs: local hot path plus remote cold path;
- split worker-hot mode: local CPU cold path plus remote HIP hot path.

For single-token decode, `eval_moe_hybrid_ffn_single()` caches ggml graphs per layer and selected expert count. The graph contains:

- input activation,
- selected local expert IDs,
- selected weights,
- routed expert matmuls,
- shared expert when present,
- weighted sum.

## Expert storage and placement

`MoeHybridPlacement` records:

- `n_layer`,
- `n_expert`,
- `n_expert_used`,
- `total_hot`,
- `hot_counts[layer]`,
- `hot_expert_ids[layer]`.

The storage builder materializes:

- hot experts in backend buffers,
- cold experts either in CPU memory or caller-selected backend memory,
- local-by-global lookup tables for hot and cold IDs.

For DS4 split mode, the same placement must be used by the parent and HIP worker. `DFLASH_DS4_PLACEMENT_IN=<json>` loads a placement in the parent and is forwarded to the sudo worker through `--placement-in`, which sets `DFLASH_DS4_PLACEMENT_IN` inside `backend_ipc_daemon`.

Placement JSON can also be exported with:

```bash
DFLASH_DS4_PLACEMENT_OUT=/tmp/ds4-placement.json
```

## Expert IPC protocol

The parent writes each request payload to a temporary file and sends an `eval <path>` command to the worker process. The worker reads:

- request header,
- activation tensor `[n_tokens, n_embd]`,
- selected expert IDs `[n_tokens, n_selected]`,
- selected expert weights `[n_tokens, n_selected]`.

The worker writes a binary response over the backend IPC stream:

1. `DeepSeek4ExpertIpcResponseHeader` version 2 base header.
2. Optional version 3 graph-timing extension when requested.
3. Output activation payload.

The graph-timing extension is requested by setting `DS4_EXPERT_IPC_FLAG_GRAPH_TIMING` in the request header. The client drains the optional extension before applying status and dimension validation so error responses cannot desynchronize the stream.

`DeepSeek4ExpertIpcClient::eval()` is now a compatibility wrapper around:

- `eval_begin()`: writes the request file and sends the command,
- `eval_end()`: reads the response, removes the temp file, and fills output/timing.

This split enables async overlap without changing the worker protocol.

## Async worker overlap

`DFLASH_DS4_ASYNC_WORKER=1` enables opt-in overlap in `eval_ds4_hybrid_or_worker()`:

1. Partition selected IDs into local and remote sets.
2. Launch the HIP worker request with `eval_begin()`.
3. Evaluate local CPU cold/shared FFN work.
4. Join the worker with `eval_end()`.
5. Sum local and remote outputs.

Default behavior remains synchronous. Remote HE n=1 showed a small improvement on uniform placement:

| Mode | TTFT | Decode tok/s | Wall tok/s | Total |
| --- | ---: | ---: | ---: | ---: |
| Uniform sync | `12.613s` | `12.39` | `7.69` | `33.04s` |
| Uniform async | `11.654s` | `12.51` | `7.97` | `31.88s` |

Async timing showed visible worker wait falling to about `3.59s`, while actual worker resident compute remained about `6.11s`. This means CPU cold work hid part of the HIP worker latency, but did not reduce the HIP compute itself.

## Telemetry

`DFLASH_DS4_TIMING=1` enables aggregate timing logs:

```text
[deepseek4-timing] prefill ...
[deepseek4-timing] decode ...
```

Important fields:

| Field | Meaning |
| --- | --- |
| `ffn`, `hot`, `cold`, `combine`, `partition` | Parent-side hybrid FFN wall and sub-phases |
| `worker` | Parent-observed worker cost; in async mode this is visible write/wait/read, not total worker compute |
| `worker_write`, `worker_wait`, `worker_read` | Parent IPC phases |
| `worker_req_read`, `worker_part`, `worker_resident`, `worker_miss_build`, `worker_miss` | Worker-side phases |
| `worker_req_kib`, `worker_resp_kib` | Payload volume |
| `ffn_hot_graph_build`, `ffn_hot_graph_hit` | Parent hot graph cache events |
| `ffn_cold_graph_build`, `ffn_cold_graph_hit` | Parent cold graph cache events |
| `worker_hot_graph_build`, `worker_hot_graph_hit` | Worker resident graph cache events |
| `hot_sel`, `cold_sel` | Routed expert counts by placement ownership |

The detailed timing showed temp-file IPC file read/write cost is not the primary bottleneck. Worker wait largely tracks HIP resident FFN compute.

## Fixed-slot graph reuse experiment

`DFLASH_MOE_FIXED_SLOT_GRAPHS=1` enables opt-in fixed-slot cached graphs for single-token FFN:

- graph shape uses at least `cfg.n_expert_used` slots,
- unused slots are padded with a valid dummy local expert ID and zero weight,
- output is preserved because dummy expert weights are zero,
- the flag is forwarded to the HIP worker via `--fixed-slot-graphs`.

Remote results:

| Mode | Effect |
| --- | --- |
| Short decode | worker graph builds dropped from `546` to `0` |
| HE n=1 | regressed from `10.71` to `8.32` decode tok/s |

Conclusion: fixed-slot reuse is useful diagnostic infrastructure but should stay opt-in. It removes graph rebuild churn, but forcing six slots increases expert work enough to regress performance.

## Route statistics and placement experiment

`DFLASH_DS4_ROUTE_STATS_OUT=<csv>` enables route statistics collection. Both hash-routed and top-k routed layers observe selected expert IDs. Stats are saved after requests and at backend shutdown.

A top-210-per-layer placement generated from HE n=1 stats predicted much better coverage:

| Placement | Hot coverage | Cold coverage |
| --- | ---: | ---: |
| Uniform first 210 | `82.7%` | `17.3%` |
| Route-stat top 210 | `99.8%` | `0.2%` |

Actual HE n=1 result regressed:

| Placement | TTFT | Decode tok/s | Wall tok/s | Total | Decode cold selections | Worker resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform/current | `12.613s` | `12.39` | `7.69` | `33.04s` | `11403` | `5.66s` |
| Top-210 route-stat | `19.889s` | `6.04` | `4.11` | `61.77s` | `815` | `18.21s` |

Conclusion: maximizing HIP hot coverage is not the right objective on Halo. The CPU cold tail is often cheaper than pushing those experts to the HIP worker. Future placement should be cost-aware: it needs per-layer/per-expert HIP-vs-CPU cost, not only hit counts.

## Current performance interpretation

The recent measurements changed the optimization priorities:

1. Temp-file IPC is not the main bottleneck.
2. Fixed-slot graph reuse removes graph builds but increases resident work.
3. Naive route-stat placement removes CPU cold hits but overloads the HIP worker.
4. Async overlap is safe and mildly beneficial.
5. HIP resident FFN compute is still the dominant remaining worker-side cost.

The next high-leverage step is profiling and optimizing the HIP resident FFN path rather than shared-memory IPC or naive hot placement.

## Useful environment variables

| Variable | Purpose |
| --- | --- |
| `DFLASH_DS4_EXPERT_SPLIT=hip_hot_cpu_cold` | Enable CUDA parent + HIP hot + CPU cold split |
| `DFLASH_DS4_EXPERT_IPC=1` | Use backend IPC worker |
| `DFLASH_DS4_EXPERT_IPC_BIN` | Path to worker `backend_ipc_daemon` |
| `DFLASH_DS4_EXPERT_IPC_MODEL` | Worker GGUF path |
| `DFLASH_DS4_EXPERT_IPC_SUDO=1` | Launch worker via sudo for HIP access |
| `DFLASH_DS4_EXPERT_WORKER_BUDGET_MB` | Worker expert memory budget |
| `DFLASH_DS4_HIP_HOT_PER_LAYER` | Override hot experts per layer |
| `DFLASH_DS4_EXPERT_WORKER_OFFSET` | Uniform placement first expert offset |
| `DFLASH_DS4_TIMING=1` | Emit `[deepseek4-timing]` logs |
| `DFLASH_MOE_FIXED_SLOT_GRAPHS=1` | Opt-in fixed-slot FFN graphs |
| `DFLASH_DS4_ROUTE_STATS_OUT` | Write routing stats CSV |
| `DFLASH_DS4_PLACEMENT_IN` | Load placement JSON in parent and worker |
| `DFLASH_DS4_PLACEMENT_OUT` | Write computed placement JSON |
| `DFLASH_DS4_ASYNC_WORKER=1` | Opt-in async worker overlap |

## Example remote command

```bash
export DFLASH_DS4_EXPERT_SPLIT=hip_hot_cpu_cold
export DFLASH_DS4_EXPERT_IPC=1
export DFLASH_DS4_EXPERT_IPC_SUDO=1
export DFLASH_DS4_EXPERT_IPC_BIN=$PWD/server/build-hip/backend_ipc_daemon
export DFLASH_DS4_EXPERT_IPC_MODEL=/opt/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
export DFLASH_DS4_EXPERT_WORKER_BUDGET_MB=61000
export DFLASH_DS4_TIMING=1
export DFLASH_DS4_ASYNC_WORKER=1

./server/build/dflash_server \
  /opt/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --port 8213
```

## Next work

Recommended next steps:

1. Profile HIP resident FFN eval per layer and selected-count shape.
2. Identify whether worker cost is dominated by graph rebuild, matmul shape, memory movement, or HIP backend scheduling.
3. Build a cost-aware placement policy using hit counts plus measured HIP/CPU cost.
4. Keep `DFLASH_DS4_ASYNC_WORKER=1` as an opt-in candidate for more multi-sample validation.
5. Defer shared-memory IPC until worker resident FFN cost is lower or payload overhead becomes visible in timing.
