# Recommended server setups

These are starting configurations for the model and hardware combinations we test. Replace the model paths and device indexes for your machine. The [server parameter reference](../README.md#server-parameter-reference) explains every flag.

## Single GPU

Entries are `luce_server` arguments unless the cell contains another command. Pass the target as the first positional argument and add `--draft <path>` for rows that use speculative decode.

| Model | RTX 3090 | Strix Halo `gfx1151` | R9700 `gfx1201` |
|---|---|---|---|
| **Qwen 3.8 27B IQ4_XS + DFlash2** | `--target-device cuda:0`<br>`--draft-device cuda:0`<br>`--draft-block-size 16`<br>`--cache-type-k q8_0`<br>`--cache-type-v q8_0` | `--target-device hip:0`<br>`--draft-device hip:0`<br>`--draft-block-size 16`<br>`--cache-type-k q8_0`<br>`--cache-type-v q8_0` | `--target-device hip:0`<br>`--draft-device hip:0`<br>`--draft-block-size 16`<br>`--max-ctx 131072`<br>`--cache-type-k q8_0`<br>`--cache-type-v q8_0`<br>[Measured profile](https://www.lucebox.com/blog/qwen38-r9700) |
| **Qwen 3.6 35B-A3B Q4_K_M** | `--target-device cuda:0`<br>`--spark`<br>`--kvflash auto` | `--target-device hip:0`<br>`--kvflash auto` | `--target-device hip:0`<br>`--kvflash auto` |
| **Laguna XS 2.1 33B Q4_K_M** | `--target-device cuda:0`<br>`--draft <path>`<br>`--prefill-drafter <path>`<br>`--max-ctx 262144`<br>`--kvflash 8192`<br>`--chunk 1024`<br>[Measured profile](https://www.lucebox.com/blog/laguna-xs21) | `--target-device hip:0`<br>`--kvflash auto` | `--target-device hip:0`<br>`--kvflash auto` |
| **Gemma 4 26B-A4B or 31B** | `--target-device cuda:0`<br>`--draft-device cuda:0`<br>`--kvflash auto` | `--target-device hip:0`<br>`--draft-device hip:0`<br>`--kvflash auto` | `--target-device hip:0`<br>`--draft-device hip:0`<br>`--kvflash auto` |
| **Qwen 3.5 0.8B Megakernel** | `uv run --directory optimizations/megakernel python final_bench.py --backend bf16` | Not supported | Not supported |

For long contexts on the 32 GB R9700, `--cache-type-k f16 --cache-type-v f16` decodes faster than `q8_0`. The verify batch's head-256 attention runs a kernel that reads f16, so a `q8_0` cache is converted to f16 in full on every verify step, a cost that grows with the context. Measured with the profile above on a 26.7K-token prompt: 73.3 ms per verify step with f16 versus 78.9 ms with `q8_0`, with no difference at short context (63.6 versus 63.4 ms). f16 is also the more precise cache. It doubles KV memory; at `--max-ctx 131072` the f16 cache fits next to the model and drafter on the 32 GB card.

If a machine has both Strix Halo and an R9700, set `HIP_VISIBLE_DEVICES=<r9700-index>` before using an R9700-only profile. The selected card is then `hip:0` inside the process.

## DeepSeek V4 on Strix Halo

The plain launch is the qualified one. The `gfx1151` device profile installs every kernel and policy default at start (fused five-row verifier, verify width from the DSpark confidence head, sparse prefill kernels), so there is nothing to tune. Use the adaptive ROCmFPX artifact with all six routed experts and `--chunk 8192` on the 128 GB part. Measured this way: 42 tok/s decode and 320 tok/s prefill at 8K, 36 tok/s at 123K, 39 tok/s on code and math, 25 tok/s on prose ([PR #729](https://github.com/Luce-Org/lucebox/pull/729); details in the [DeepSeek V4 guide](DS4.md#experimental-amd-q5-verifier)).

```bash
# LUCE_DS4_SPARSE_DECODE_FLASH=1 stays an explicit experimental opt-in
# (single HIP target; may change generated tokens — see DS4.md).
LUCE_DS4_SPEC=1 \
LUCE_DS4_DRAFT=/path/to/DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf \
luce_server /path/to/DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf \
  --target-device hip:0 \
  --max-ctx 131072 \
  --chunk 8192 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --ds4-fused-decode --ds4-fused-verify-f16-kv \
  --ds4-expert-top-k 6 \
  --ds4-prefill sparse
```

The earlier fixed-width recipe behind the [blog post](https://www.lucebox.com/blog/deepseek-v4-flash-0731) still runs, but it pins verify width 4 and exact prefill and misses the fast path.

## Multi-GPU

| Hardware and model | Configuration | Validation |
|---|---|---|
| **2x RTX 3090 + Qwen 3.8 27B** | `--target-devices cuda:0,cuda:1`<br>`--target-split-mode tensor`<br>`--peer-access`<br>`--cache-type-k q4_0`<br>`--cache-type-v q4_0`<br>`--verify-width 8`<br>Use the Qwen 3.8 DFlash2 drafter. | [PR #637](https://github.com/Luce-Org/lucebox/pull/637) |
| **RX 7900 XT + Strix Halo + DeepSeek V4** | Build for `gfx1100;gfx1151`, then run [`serve_ds4_dual_rocm_128k.sh`](../scripts/serve_ds4_dual_rocm_128k.sh) with the target and DSpark paths. The checked-in profile uses all six routed experts. | [PR #604](https://github.com/Luce-Org/lucebox/pull/604) |

The original setup matrix was introduced in [PR #602](https://github.com/Luce-Org/lucebox/pull/602). Keep a setting here only when it is still a useful starting point; measured claims belong beside their exact benchmark or qualification link.
