# Qwen3.8-Flash-Next

Qwen3.8-Flash-Next (`general.architecture = qwen4exp`) runs on its own
backend, tuned for the Strix Halo (gfx1151, 128 GB unified memory). This page
covers the model files, how to run it, what it measures, what is supported,
and how the port is laid out.

The model has 48 layers: 36 Gated DeltaNet and 12 full attention with Qwen
Sparse Attention (QSA: each query attends to the top 512 blocks of 4 tokens),
4-stream hyper-connections, per-layer n-gram embeddings and a 512-expert MoE
with 10 active experts.

## Model files

| Quant | Where | GTT |
| --- | --- | --- |
| UD-Q4_K_XL | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), `UD-Q4_K_XL/` | 77 GB |
| IQ4_NL | [bartowski/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/bartowski/Qwen3.8-Flash-Next-GGUF) | 73 GB |
| GSQ-RCO IQ3_XXS | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), `IQ3_XXS/` | 47 GB |

Point the server at the first shard; the others are found next to it. The
model cards (`share/model_cards/qwen3.8-flash-next.json`, and
`qwen3.8-flash-next-gsq-rco.json` for GSQ) are picked up from the GGUF's
`general.name`.

## Running on the Strix Halo

```bash
./server/build-hip/luce_server \
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --max-ctx 65536
```

No environment variables are needed: on gfx1151 the backend sets up its
measured kernel profile itself before loading. Prefill runs in 2048-token
chunks; long prompts on UD prefill faster with `--chunk 8192` (a 16K prompt
reaches its first token in 13.2-13.8 s). `--chunk 16384` does not fit UD in
memory with a 40K context.

Build:

```bash
cmake -S server -B server/build-hip -DCMAKE_BUILD_TYPE=Release \
  -DLUCE_GPU_BACKEND=hip -DLUCE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build server/build-hip -j --target luce_server
```

### Thinking

The model thinks by default, rendered with the GGUF's official chat template
at its default effort (xhigh). `reasoning_effort` low / medium / xhigh pass
through to the template. Use the card's sampling (temperature 1.0, top_p
0.95, top_k 20) for thinking requests, not greedy decoding; requests with
thinking off get the card's instruct set (0.7 / 0.80 / 20, presence 1.5).

The card keeps requests within lucebox context budgets (32,768 tokens). Hard
problems need more room to think:

```bash
--max-ctx 151552 --think-max-tokens 131072 \
  --hard-limit-reply-budget 16384 --default-max-tokens 147456
```

On hardcode-bench E4 (UD) this takes the score from 0/13 to 12/13; the run
thinks for about 110K tokens. At medium effort with the same budget E4 scores
7/13 in 53K tokens, at low effort 0/13.

Multi-turn requests replay each assistant turn's `reasoning_content` into the
template, and `chat_template_kwargs.preserve_thinking` is honoured.

### Measured

Strix Halo, default flags, fresh server per run:

| | prefill | decode |
| --- | --- | --- |
| UD-Q4_K_XL, 16K prompt | 990-1,100 tok/s | 22.4 tok/s |
| UD-Q4_K_XL, 64K prompt | ~1,000 tok/s | 20.9 tok/s |
| UD-Q4_K_XL, short prompt | - | 24.1 tok/s |
| IQ4_NL, 16K prompt | 950-1,010 tok/s | 28.5 tok/s (30.6 short) |
| GSQ-RCO IQ3_XXS, 16K prompt | ~475 tok/s | 25-26 tok/s |

Quality (lucebox gates, prompts above 512 tokens with a 1,500-token
preamble): HE / GSM / Math 10 / 10 / 10 and 16K planted-fact recall 3/3 on
all three quants.

## Support

| Piece | State |
| --- | --- |
| Loader, graph and cache: Gated DeltaNet, full attention, hyper-connections, n-gram embeddings, MoE | done, byte-exact against upstream llama.cpp's reference path (0 expert-ID mismatches across 48 layers on IQ4_NL and UD) |
| QSA at prefill and decode, keys pooled per 4-token block | done (exactly dense up to 2,051 tokens) |
| gfx1151 kernels: MMB bf16 and Q8_0 -> F16 WMMA GEMMs, fused HC / GDN / PLE, M-RoPE into the flash-attention layout | done |
| Chat template, reasoning effort, thinking budget, `preserve_thinking`, `sampling_no_thinking` | done |
| Concurrent serving (`--max-concurrency > 1`) | refused; exact 4-slot serving is a follow-up PR |
| MTP speculative decoding | follow-up PR |
| Layer split | refused |
| Other GPUs | generic paths; kernels, defaults and quality gates are tuned and measured on gfx1151 only |

## Layout

| File | What |
| --- | --- |
| `src/qwen4exp/qwen4exp_loader.cpp` | GGUF keys and tensors |
| `src/qwen4exp/qwen4exp_graph.cpp` | prefill and decode graphs, QSA selection |
| `src/qwen4exp/qwen4exp_cache.cpp` | KV, recurrent state and QSA indexer cache |
| `src/qwen4exp/qwen4exp_backend.cpp` | `ModelBackend`: generation loop, gfx1151 profile |
| `deps/llama.cpp/ggml/src/ggml-cuda/` | QSA, MMB, HC / GDN / PLE and RoPE kernels |

Tests: `test_qwen4exp_qsa_ids` (QSA block selection, GPU and CPU),
`test_qwen4exp_indexer_score`, `test_rope_tail` (including the M-RoPE ->
CONT fusion alias), `test_backend_plan` (default prefill chunk),
`test_server_unit`, and `smoke_qwen4exp_forward` (split-prefill KLs and
cancel/reset/reuse on a real GGUF).
