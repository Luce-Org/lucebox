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
| UD-Q4_K_XL | [Lucebox/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/Lucebox/Qwen3.8-Flash-Next-GGUF), `UD-Q4_K_XL/` (a byte-identical copy of [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), with the MTP drafter in `MTP/`) | 77 GB |
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
measured kernel profile through scoped code settings during loading and forward
evaluation. Device support is resolved once at load; forward scopes only exchange
the calling thread's profile flag. It does not read qwen4exp environment variables or change the process
environment; other models retain their own dispatch. Without `--chunk`, the
prefill chunk is the largest 256-row multiple that fits the memory left after
weights, caches and (with MTP) the draft and verify graphs, keeping 10% free:
UD at 262K context gets 7424 rows without the sidecar and 4864 with MTP. The
banner and `/props` report it. Prompt attention accumulates in F32: on UD an 18K
prompt prefilled in 2048- or 7424-row chunks gives logits bitwise equal to one
pass. Verify rows keep the T=1 attention path, so MTP output equals MTP off.

Build:

```bash
cmake -S server -B server/build-hip -DCMAKE_BUILD_TYPE=Release \
  -DLUCE_GPU_BACKEND=hip -DLUCE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build server/build-hip -j4 --target luce_server
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
| UD-Q4_K_XL, 16K prompt | 1,200-1,240 tok/s | 22.4 tok/s |
| UD-Q4_K_XL, 64K prompt | 1,200-1,250 tok/s | 20.9 tok/s |
| UD-Q4_K_XL, short prompt | - | 24.1 tok/s |
| IQ4_NL, 16K prompt | 950-1,010 tok/s | 28.5 tok/s (30.6 short) |
| GSQ-RCO IQ3_XXS, 16K prompt | ~475 tok/s | 25-26 tok/s |

Quality (lucebox gates, prompts above 512 tokens with a 1,500-token
preamble): HE / GSM / Math 10 / 10 / 10 and 16K planted-fact recall 3/3 on
all three quants.

## Running on an R9700 + Strix Halo

With a second GPU, split mode keeps the dense weights, the KV cache and the
MTP drafter on the target and loads the routed experts (71.7 GiB on
UD-Q4_K_XL) on the expert device:

```bash
./server/build-hip/luce_server \
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --profile qwen-next-r9700-strix --expert-placement routes.csv
```

The profile sets `--target-device hip:0` (the R9700, gfx1201),
`--expert-device hip:1` (the Strix Halo, gfx1151) and `--max-ctx auto`: the
model's 262,144-token context, lowered in steps of 4096 to what the R9700
holds beside the dense weights and every slot's cache.
Both GPUs run one scheduler: prompt chunks pipeline over 2 to 4 streams so
the Strix runs one stream's experts while the R9700 runs another's attention.

`--expert-placement` takes a routing-stats CSV (picks per layer and expert)
and copies the most-routed experts, up to `LUCE_EXPERT_BUDGET_MB` (default
8192), to the R9700: the same placement flag and budget the DeepSeek4 expert
tiers use. Decode and verify steps run those picks there while the Strix runs
the rest; prompt chunks keep every pick on the Strix. Without the flag every
pick runs on the Strix.

Build for both GPUs with HIP graph replay, which the retained MTP verify and
draft graphs use:

```bash
cmake -S server -B server/build-hip -DCMAKE_BUILD_TYPE=Release \
  -DLUCE_GPU_BACKEND=hip -DLUCE_HIP_ARCHITECTURES="gfx1151;gfx1201" \
  -DCMAKE_HIP_ARCHITECTURES="gfx1151;gfx1201" -DGGML_HIP_GRAPHS=ON
cmake --build server/build-hip -j8 --target luce_server
```

Measured on one box (UD-Q4_K_XL, adaptive MTP, greedy, 131K context, 8 GiB
of hot experts), against the Strix Halo alone:

| | Strix Halo | R9700 + Strix Halo |
| --- | --- | --- |
| Decode, short reply | 40.5 tok/s | 75.5 tok/s |
| Decode, code reply | 41.4 tok/s | 63.1 tok/s |
| Decode after a 16K prompt | 24-28 tok/s | 44-45 tok/s |
| Prefill, 16K prompt | 1,220-1,320 tok/s | 1,820 tok/s |
| Prefill, 55K prompt | 1,310 tok/s | 1,760 tok/s |
| Same 16K prompt sent again (prefix cache) | | 2.7 s instead of 10.6 s |

Without `--expert-placement` the split decodes a short reply at 69 tok/s and after
a 16K prompt at 44 tok/s. Greedy output is the same with adaptive MTP, a fixed
`--verify-width` or MTP off (`--verify-width 1`): every generated token runs
its hot experts on the R9700 with the same kernel.

With `--max-concurrency 2..4` split mode serves several requests at once.
Prefix snapshots live in system memory beside the R9700, so its memory holds
context; the server's resident limit bounds them. On the Strix alone they stay
in its memory, and `--max-ctx auto` keeps room for one full-context snapshot
per slot.

## Support

| Piece | State |
| --- | --- |
| Loader, graph and cache: Gated DeltaNet, full attention, hyper-connections, n-gram embeddings, MoE | done, matched byte for byte against upstream llama.cpp during the port (0 expert-ID mismatches across 48 layers on IQ4_NL and UD) |
| QSA at prefill and decode, keys pooled per 4-token block | done (exactly dense up to 2,051 tokens) |
| Attention block ratios other than 4, contexts of 2^24 tokens or more | refused at load |
| gfx1151 kernels: MMB bf16 and Q8_0 -> F16 WMMA GEMMs, fused HC / GDN / PLE, M-RoPE into the flash-attention layout | done |
| Chat template, reasoning effort, thinking budget, `preserve_thinking`, `sampling_no_thinking` | done |
| Concurrent serving (`--max-concurrency 2..4`) | supported on one GPU and in split mode, exact independent slots with full per-slot caches; MTP drafts for a request running alone; the prefix cache serves every slot; more than 4 is refused |
| MTP speculative decoding | sidecar discovered automatically; adaptive k=1..7 by default (code 16K / 64K 32.4 / 28.2 tok/s, counting 43.5), output identical to MTP off; `--verify-width 1` disables, `2..8` selects fixed k=1..7 |
| Prefix cache | done: snapshots at chat cut points, restored on hits; a 64K agent turn's first token in ~3.5 s instead of ~65 s (131K context). Prefill keeps a 4096-row chunk first (8192 in split mode). Snapshots share the Strix's memory with the caches (`--max-ctx auto` leaves one full-context snapshot per slot) and live in system memory beside a discrete GPU |
| Layer split | refused |
| Split mode (`--expert-device`, `--expert-placement`) | routed experts on a second GPU, pipelined prompt chunks, hot experts on the target for decode and verify; measured on R9700 + Strix Halo |
| Other GPUs | generic paths; kernels, defaults and quality gates are tuned and measured on gfx1151, and on gfx1201 as the split-mode target |

`--max-concurrency 1` keeps the single-sequence path. Values 2 through 4
enable the sequence engine and batched decode without environment settings.
Each slot prefills separately with the backend's chunk (auto-sized for every slot, at most 4,096 rows, or `--chunk`) and
owns a full F16 cache. Dense decode batches use the same MMVQ arithmetic as
solo decode; when any slot crosses the QSA boundary, decode runs per slot.
Paging, `--kv-pool-tokens`, and multi-device placement are unsupported.
The GPU concurrency checks use a 32,768-token context:

```bash
luce_server MODEL.gguf --max-concurrency 4 --max-ctx 32768 \
  --prefix-cache-slots 0 --disk-prefix-cache off
python3 server/test/qwen4exp_concurrency_http.py http://127.0.0.1:8080
```

The HTTP test checks eight responses (four near 2K and four near 16K) against
solo requests, including planted-fact recall, and reports aggregate prefill
and decode rates from `usage.timings`. GPU probes:
`smoke_qwen4exp_batched MODEL.gguf 2050 3072 1` and
`test_qwen4exp_seq_engine MODEL.gguf 32768` check logits, cache/indexer state,
contract/soak, the QSA boundary, and distinct concurrent token streams.

## Layout

| File | What |
| --- | --- |
| `src/qwen4exp/qwen4exp_loader.cpp` | GGUF keys and tensors |
| `src/qwen4exp/qwen4exp_graph.cpp` | prefill and decode graphs, QSA selection |
| `src/qwen4exp/qwen4exp_cache.cpp` | KV, recurrent state and QSA indexer cache |
| `src/qwen4exp/qwen4exp_backend.cpp` | `ModelBackend`: generation loop, gfx1151 profile, split-mode setup and hot experts |
| `deps/llama.cpp/ggml/src/ggml-cuda/` | QSA, MMB, HC / GDN / PLE and RoPE kernels |

Tests: `test_qwen4exp_qsa_ids` (QSA block selection, GPU and CPU),
`test_qwen4exp_indexer_score`, `test_rope_tail` (including the M-RoPE ->
CONT fusion alias), `test_backend_plan` (default prefill chunk), `test_qwen4exp_chunk`
(memory-sized chunk selection),
`test_qwen4exp_qsa_attn` (QSA attention rows), `test_server_unit`, and
`smoke_qwen4exp_forward` (split-prefill KLs and reference comparisons on a real
GGUF; `--expert-gpu N` runs it in split mode). `smoke_qwen4exp_batched` also checks
cancel/reset/reuse and slot isolation.

The smoke binary uses the same gfx1151 defaults as the server:

```bash
server/build-hip/smoke_qwen4exp_forward MODEL.gguf 6000 \
  --token-file tokens6000.txt --split 100:1
server/build-hip/smoke_qwen4exp_forward MODEL.gguf 16000 --compare-chunk 4096
```

`--split N[:chunk]` compares one prefill against a split suffix.
`--compare-chunk N` prefills in N-row chunks against `--chunk` (default 2048)
and reports the first greedy divergence and the teacher-forced KL over 128
generated tokens.

MTP uses the same scoped profile for drafting, verification, rollback and the
K/V fill inside trunk prefill. `--draft PATH` selects a sidecar explicitly;
without a sidecar the server decodes autoregressively. No MTP environment
variables are required. The existing global adaptive-width override remains
readable, but adaptive MTP is enabled by default without it.

```bash
server/build-hip/smoke_qwen4exp_forward MODEL.gguf 2200 --mtp 128 --mtp-all --chunk 2048
```

The smoke's `--mtp-draft 1..7` selects one fixed draft cap; `--mtp-all` checks
all seven at S=16 and the requested sequence length. `--chunk N` controls MTP
prefill chunks, `--tg N` controls ordinary decode length, and `--stable N`
checks the final N replayed QSA decode steps against graphs built fresh at each
step. `--draft PATH|0` selects or disables the smoke sidecar.
