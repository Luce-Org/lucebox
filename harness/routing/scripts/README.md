# Routing prototype: small-model backends

All small tiers (Qwen3.5-0.8B, Qwen3.5-2B, Brick classifiers) are served by
**luce_server** (qwen35 backend), not llama.cpp. The 27B tier is the existing
luce_server systemd service on `:8216` (`--model-name qwen3.8-27b`); nothing
here starts or stops it.

| file | purpose |
|---|---|
| `download_models.sh` | Fetch GGUFs into `~/models/routing/` with `uvx --from huggingface_hub hf download` (parallel, idempotent). `SKIP_EMBED=1` skips the embedding model. |
| `../config/backends.json` | Endpoint registry the router reads (`models` = generation tiers by `rank`, `services` = classifiers/embeddings). Per entry: `base_url`, `kind`, `gguf`, `served_name` (= `--model-name`), `context` (= `--max-ctx`), `capacity` (= `--max-concurrency`). |
| `stop_small.sh` | Stop instances by pidfile (`runs/servers/<name>.pid`); never `pkill -f`. |
| `launch_small.sh` | Not written yet (see "Status"). Intended: start every `kind: luce_server` entry that has a `gguf` on `--target-device hip:1` (Strix Halo), with pidfile + log in `runs/servers/`, args taken from backends.json and `--kv-pool-tokens = context*capacity` (the default sizes the pool from free device memory). |

`runs/` and `data/cache/` are gitignored.

## Models

| name | file | arch | notes |
|---|---|---|---|
| qwen35-0.8b | `Qwen3.5-0.8B-Q8_0.gguf` (unsloth) | qwen35, 24 blk, n_embd 1024 | tied embeddings |
| qwen35-2b | `Qwen3.5-2B-Q8_0.gguf` (unsloth) | qwen35, 24 blk, n_embd 2048 | tied embeddings |
| brick-max | `brick-complexity-2-max-Q8_0.gguf` (regolo, CC BY-NC) | qwen35 (0.8B merged LoRA) | tied embeddings |
| brick-eco | `brick-complexity-2-eco-Q8_0.gguf` (regolo, CC BY-NC) | qwen35 (0.8B merged LoRA) | tied embeddings |
| embed | `Qwen3-Embedding-0.6B-Q8_0.gguf` (Qwen) | qwen3, pooling_type=3 (last) | `kind: todo`: luce_server has no embedding endpoint |

The small qwen35 GGUFs use the same tensor names as the Qwen3.8-27B GGUF
except: no `output.weight` (LM head tied to `token_embd.weight`), no
`blk.N.nextn.*` MTP block, all matrices Q8_0. DeltaNet has
`ssm.time_step_rank = ssm.group_count = 16` (v heads == k heads; the 27B has
48 vs 16), which the graph already handles (repeat only when they differ).
The loader at `HEAD` (`server/src/qwen35/gguf_target_loader.cpp`) rejects
them with "missing top-level tensors"; tied-LM-head support is required.

## Brick readout

Prompt (the model card's; ChatML, no think block):

```
<|im_start|>system
You are a query difficulty classifier for an LLM routing system.
Classify each query as easy, medium, or hard based on the cognitive depth and domain expertise required to answer correctly.
Respond with ONLY one word: easy, medium, or hard.<|im_end|>
<|im_start|>user
Classify: {query}<|im_end|>
<|im_start|>assistant
```

First answer tokens (Qwen3.5 vocab, no leading space since they follow
`assistant\n`): `easy`=43518, `medium`=25252, `hard`=18140 (the `Ġ`-prefixed
variants are 3999/10728/2503; capitalised 35512/39640/26044).

Through `/v1/chat/completions` luce_server applies its qwen35 chat template,
which appends `<think>\n\n</think>\n\n` after `assistant\n` when
`enable_thinking` is false (and `<think>\n` when true). That differs from the
model-card prompt; which variant the LoRA was trained on is unverified. The
readout request needs `max_tokens: 1`, `temperature: 0`, and top-k
logprobs; renormalise over the three class tokens. luce_server at `HEAD` has
no logprobs support (being added on this branch in `token_logprobs.*`).
