# GLM-5.3-Flash — Lucebox Integration

This document describes the GLM-5.3-Flash (`glm5-next`) backend. It loads the
GGUFs produced by upstream llama.cpp (for example `ggml-org/GLM-5.3-Flash-GGUF`)
and runs autoregressive decode of one sequence, on one GPU or with the layers
split across several (`--target-devices`).

Not implemented yet: speculative decode (MTP, DFlash), PFlash, paged or
concurrent serving, and the vision tower.

## Model Architecture

GLM-5.3-Flash is a 45-layer MoE model (321B total, about 18B active) with:

| Parameter | Value |
|-----------|-------|
| Hidden dim (`n_embd`) | 4096 |
| Layers | 34 KDA + 11 MLA (every 4th layer, `head_count_kv[i] == 1`) |
| KDA | 64 heads x 128, conv kernel 4, low-rank decay and output gates |
| MLA | 64 heads, NoPE, q_lora 1536, kv_lora 512, qk/v head 256; cache holds the 512-wide latent |
| DSA indexer | 32 heads x 128, top-k 2048 tokens as 512 pools of 4, plus the query's unfinished pool |
| mHC | 4 residual streams, Sinkhorn-normalized combine (20 iterations) |
| FFN | 3 dense layers (12288), then 288 routed experts (top-8) + 1 shared (2048) |
| Routing | Sigmoid router with selection bias, weights normalized and scaled by 2.5 |
| SwiGLU | Clamped at 10 (routed and shared) |
| Vocab | 154880, `glm4` pre-tokenizer |

While a sequence has at most 512 complete pools (2048 tokens), DSA keeps every
visible pool and attention is exactly causal; the backend skips indexer scoring.

## Reasoning effort

GLM-5.3-Flash always reasons: its template opens `<think>` for every reply.
The chat template embedded in the GGUF renders the prompt unless
`--chat-template-file` is passed. `reasoning_effort` maps to the template's
three levels: `minimal` and `low` to `low`, `medium` and `high` to `high`, and
anything higher to the template default `max`.

## Code Layout

| Area | Files |
|------|-------|
| Weights, cache, GGUF loader (mmap, split GGUFs, per-device placement) | `src/glm5/glm5_model.h`, `src/glm5/glm5_loader.cpp` |
| Step graph (mHC, KDA, MLA + DSA, MoE) | `src/glm5/glm5_graph.h`, `src/glm5/glm5_graph.cpp` |
| `ModelBackend` (prefill, decode, snapshots, multi-GPU) | `src/glm5/glm5_backend.h`, `src/glm5/glm5_backend.cpp` |
| Factory and capabilities | `src/common/backend_factory.cpp`, `src/common/model_capabilities.h` |
| Chat template, effort, tokenizer | `src/server/server_main.cpp`, `src/server/chat_template.cpp`, `src/server/tokenizer.cpp` |
| Tests | `test/unit/test_glm5_unit.cpp`, `test/unit/test_server_unit.cpp` (`glm5`) |

Reused building blocks:

- KDA follows the Ling 3 block (`build_bailingmoe3_kda_block`) on the same
  `ggml_gated_delta_net` path; GLM's gates are low-rank and its `ssm_a` holds
  `-exp(A_log)`.
- MLA follows the Ling 3 absorbed-latent attention, without RoPE or the output gate.
- mHC follows the DeepSeek V4 computation, as one batched graph.
- The clamped SwiGLU is `ggml_swiglu_ds4_split`.

## Environment Variables

The backend reads none. `test_glm5_unit` reads:

| Variable | Purpose |
|----------|---------|
| `GLM5_REF_GGUF` | A `glm5-next` GGUF to compare against upstream llama.cpp reference logits. |
| `GLM5_REF_PREFIX` | Reference files: `<prefix>.tokens.txt`, `<prefix>.prefill.f32`, `<prefix>.decode.f32`. |
| `GLM5_REF_DEVICE` | `cpu` (default), a GPU index, or a list such as `0,1,2,3` to split the layers. |
| `GLM5_CPU_THREADS` | CPU backend threads for the reference run (default: all cores). |
| `GLM5_DUMP_PREFIX` | Write our own prefill/decode logits for offline comparison. |

## Example: Single GPU and Layer Split

Pass the first shard of a split GGUF; the others are found next to it.

```bash
./server/build/luce_server /opt/models/GLM-5.3-Flash-Q4_K-00001-of-00002.gguf \
  --target-device cuda:0 \
  --max-ctx 65536 \
  --port 8213
```

Four GPUs; the layers are split in proportion to their weight bytes:

```bash
./server/build/luce_server /opt/models/GLM-5.3-Flash-Q4_K-00001-of-00002.gguf \
  --target-devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --max-ctx 131072 \
  --port 8213
```

On 4x RTX PRO 6000 (96 GB) the Q4_K release loads in about 90 s, split as
layers 0-12, 13-23, 24-33, 34-44.

## Performance Notes

- **Memory**: the Q4_K release is about 180 GB of weights. State is about 22 KB
  per token (11 MLA layers x (1 KB latent + 1 KB indexer key and gate)), plus
  about 146 MB of fixed KDA state.
- **No re-run of tokens**: KDA state cannot rewind, so the backend keeps the last
  step's logits, prefix snapshots store them with the state, and a snapshot
  request stops prefill exactly at the snapshot position.
- **Layer split is sequential**: one GPU works at a time; ggml_backend_sched
  copies the mHC state across each boundary.
- **One graph per step**: rebuilt per decode token and per prefill chunk, no
  CUDA graphs yet. Decode is about 20 tok/s on 4x RTX PRO 6000.
- **GPU prefill and DSA ties**: ggml's cuBLAS runs F32 GEMMs in TF32, which can
  flip near-tie pool selections in batched prefill. Decode uses the F32 mat-vec
  kernels. Upstream avoids this with a fused F32 lightning-indexer kernel; porting
  it is a follow-up.
- **GPU MLA needs at least 4 heads**: the 512-wide latent flash-attention kernel
  rejects fewer (GLM-5.3-Flash has 64); the backend fails at startup instead.

## Validation

Against upstream llama.cpp on the same machine (per-token max |logit diff|,
compared with upstream's own prefill-vs-decode disagreement):

| Model | Device | Result |
|-------|--------|--------|
| Tiny random models, sparse and dense DSA | CPU, RTX 3050, H100 | Same argmax at every position |
| Release widths, 4 layers, Q4_K_M, 2887 tokens (DSA active) | CPU | Median 1.49e-2 vs upstream's own 1.48e-2 |
| Same | H100 decode | Median 1.16e-2 vs upstream's own 1.17e-2 |
| Released Q4_K weights (181 GB), 19 tokens | CPU | Same greedy token at all positions (" Tokyo" next) |
| Released Q4_K weights, `luce_server` | 4x RTX PRO 6000 | Correct chat answers |

## Build Targets

| Target | Backend | Purpose |
|--------|---------|---------|
| `luce_server` | CUDA | Production server |
| `test_glm5_unit` | CUDA build, runs on CPU | Unit tests on generated tiny models (no model files needed) |
| `test_server_unit` | CUDA build, runs on CPU | Chat template and reasoning-effort tests (`glm5` filter) |
