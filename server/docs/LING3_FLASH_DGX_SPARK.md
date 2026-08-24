# Ling 3.0 Flash on DGX Spark

LuceBox has an experimental native backend for AntLing's 124B-parameter
`BailingMoE3` model, Ling 3.0 Flash. This is a LuceBox graph implementation,
not a proxy to vLLM or llama.cpp.

The first tested configuration serves a Q4_K_M GGUF on one NVIDIA DGX Spark
at 34.6 generated tokens/s. It also renders the official chat template,
emits OpenAI-compatible tool calls, consumes tool results, and returns a final
answer.

## Tested configuration

| Item | Value |
|---|---|
| Machine | NVIDIA DGX Spark |
| GPU | NVIDIA GB10, `sm_121` |
| Host architecture | `aarch64` |
| Unified memory | 121 GiB visible to the OS |
| Driver / CUDA | 580.173.02 / CUDA 13.0 |
| Model | `inclusionAI/Ling-3.0-flash` |
| GGUF | `Ling-3.0-flash-Q4_K_M.gguf`, 72.9 GiB |
| Weight quantization | Q4_K_M |
| KV cache | Q4_0 keys and values |
| Server context | 32,768 tokens |
| Request concurrency | One active decode slot |
| Prefix cache | Disabled for measurement |
| Benchmark date | 2026-08-24 |

The GGUF used for this bring-up came from
[`bloomer010/Ling-3.0-flash-GGUF`](https://huggingface.co/bloomer010/Ling-3.0-flash-GGUF).
The official model and tokenizer are at
[`inclusionAI/Ling-3.0-flash`](https://huggingface.co/inclusionAI/Ling-3.0-flash).

## Build and serve

Download the tested quantization:

```bash
hf download bloomer010/Ling-3.0-flash-GGUF \
  Ling-3.0-flash-Q4_K_M.gguf \
  --local-dir models/ling-3.0-flash-gguf
```

Build the CUDA server for GB10:

```bash
cmake -S server -B server/build-ling3-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DDFLASH27B_GPU_BACKEND=cuda \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=121

cmake --build server/build-ling3-cuda \
  --target dflash_server -j"$(nproc)"
```

Start the measured configuration:

```bash
server/build-ling3-cuda/dflash_server \
  models/ling-3.0-flash-gguf/Ling-3.0-flash-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 18081 \
  --max-ctx 32768 \
  --default-max-tokens 256 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --prefix-cache-slots 0 \
  --disk-prefix-cache off \
  --model-name ling-3.0-flash-lucebox
```

The server exposes the OpenAI-compatible `/v1/chat/completions` route. To use
a Spark reached over SSH without exposing the API on the network, create a
local tunnel:

```bash
ssh -N -L 18081:127.0.0.1:18081 <spark-user>@<spark-host>
```

Then send requests to `http://127.0.0.1:18081` on the client machine.

## Benchmark method

The benchmark uses [`bench_ling3_flash.py`](../scripts/bench_ling3_flash.py),
which depends only on the Python standard library. It records the timings
reported by the LuceBox server and checks output determinism. Prefix caching is
disabled, temperature is zero, and concurrency is reported separately from
single-request decode.

```bash
python3 server/scripts/bench_ling3_flash.py \
  --url http://127.0.0.1:18081 \
  --model ling-3.0-flash-lucebox \
  --weights-quant Q4_K_M \
  --kv-quant Q4_0 \
  --server-max-context 32768 \
  --decode-tokens 128 \
  --decode-runs 3 \
  --warmups 1 \
  --context-tokens 256,1024,4096,8192 \
  --context-runs 2 \
  --concurrency-levels 1,2 \
  --concurrency-tokens 64 \
  --output ling3-flash-lucebox-bench.json
```

### Decode

The decode prompt requests a deterministic repeated-token sequence and each
measured run generates 128 tokens. All three measured responses have the same
SHA-256 digest.

| Run | Prompt tokens | Output tokens | Prefill | Decode |
|---:|---:|---:|---:|---:|
| 1 | 57 | 128 | 157.8 ms | 34.6 tok/s |
| 2 | 57 | 128 | 157.4 ms | 34.6 tok/s |
| 3 | 57 | 128 | 157.2 ms | 34.5 tok/s |
| **Median** | **57** | **128** | **157.4 ms** | **34.6 tok/s** |

### Context prefill

Each row is the median of two uncached requests. The response is restricted to
one token so the measurement isolates prompt evaluation.

| Actual prompt tokens | Median prefill | Median prefill rate |
|---:|---:|---:|
| 267 | 277.55 ms | 962.0 tok/s |
| 1,035 | 810.80 ms | 1,277.0 tok/s |
| 4,107 | 2,642.15 ms | 1,554.4 tok/s |
| 8,203 | 5,238.60 ms | 1,565.9 tok/s |

The model advertises a 262,144-token context. This run capped the server at
32,768 and exercised prompt depth only through 8,203 tokens, so it is not a
claim of validated 262K behavior.

### Concurrency

| Submitted requests | Total output tokens | Wall time | Aggregate rate |
|---:|---:|---:|---:|
| 1 | 64 | 2.041 s | 31.35 tok/s |
| 2 | 128 | 4.033 s | 31.74 tok/s |

Two simultaneous HTTP requests complete safely, but the current LuceBox
backend queues them through one active decode slot. The second request took
4.032 seconds, versus 2.016 seconds for the first. This is serialization, not
continuous batching, and throughput does not scale with request concurrency.

### Tool-call round trip

The test gives the model a `get_weather(location)` function and asks for the
current weather in Rome. The model emits exactly one structured call:

```json
{"name":"get_weather","arguments":{"location":"Rome"}}
```

The harness supplies a deterministic fixture:

```json
{"temperature_c":29,"condition":"sunny","source":"test fixture"}
```

Ling then returns:

> The current weather in Rome is sunny with a temperature of 29°C.

The call completed with `finish_reason: "tool_calls"`; the final answer used
both fixture values. Decode measured 35.0 tok/s for the tool call and 34.9
tok/s for the final answer.

## Correctness checks

The backend implements Ling's mixed 35-layer KDA / 7-layer MLA attention,
512-expert MoE, top-8 routing, and grouped expert selection. The embedded MTP
layer is recognized and excluded from the autoregressive target graph.

For an oracle check, the same GGUF and exact chat prompt were run against the
Ling support branch proposed to llama.cpp. LuceBox and the oracle match the
opening text through `Running an open-weight language model locally offers`.
Their first different greedy token was a near tie: oracle `greater` at
-0.89073 log probability and LuceBox `three`, with oracle `three` at -0.96373.
That 0.073-natural-log-unit margin is consistent with GPU kernel numerical
drift rather than a prompt-template or graph-structure mismatch. A short exact
answer test produced `PARIS` in both engines.

The full LuceBox CTest suite passes: 477 tests, zero failures. Four tests that
require unavailable model assets or NCCL are skipped by configuration.

## Current verdict

Use this backend today for single-user, local Ling 3.0 Flash evaluation on a
DGX Spark, especially chat, tool-use, and agent experiments that need an
OpenAI-compatible endpoint. The 73 GiB Q4_K_M artifact fits comfortably in the
Spark's unified memory, and 34.6 tok/s is interactive.

Do not use this baseline yet for a multi-user serving claim, a validated 262K
context claim, or a headline comparison against another runtime. Continuous
batching, full-depth context validation, the model's MTP speculative path, and
matched-condition cross-runtime benchmarks remain follow-up work.
