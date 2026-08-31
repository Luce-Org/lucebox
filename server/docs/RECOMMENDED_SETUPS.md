# Recommended server setups

These are starting configurations for the model and hardware combinations we test. Replace the model paths and device indexes for your machine. The [server parameter reference](../README.md#server-parameter-reference) explains every flag.

## Single GPU

Entries are `dflash_server` arguments unless the cell contains another command. Pass the target as the first positional argument and add `--draft <path>` for rows that use speculative decode.

| Model | RTX 3090 | Strix Halo `gfx1151` | R9700 `gfx1201` |
|---|---|---|---|
| **Qwen 3.8 27B IQ4_XS + DFlash2** | `--target-device cuda:0`<br>`--draft-device cuda:0`<br>`--draft-block-size 16`<br>`--cache-type-k q8_0`<br>`--cache-type-v q8_0` | `--target-device hip:0`<br>`--draft-device hip:0`<br>`--draft-block-size 16`<br>`--cache-type-k q8_0`<br>`--cache-type-v q8_0` | `--target-device hip:0`<br>`--draft-device hip:0`<br>`--draft-block-size 16`<br>`--max-ctx 131072`<br>`--cache-type-k q8_0`<br>`--cache-type-v q8_0`<br>[Measured profile](https://www.lucebox.com/blog/qwen38-r9700) |
| **Qwen 3.6 35B-A3B Q4_K_M** | `--target-device cuda:0`<br>`--spark`<br>`--kvflash auto` | `--target-device hip:0`<br>`--kvflash auto` | `--target-device hip:0`<br>`--kvflash auto` |
| **Laguna XS 2.1 33B Q4_K_M** | `--target-device cuda:0`<br>`--draft <path>`<br>`--prefill-drafter <path>`<br>`--max-ctx 262144`<br>`--kvflash 8192`<br>`--chunk 1024`<br>[Measured profile](https://www.lucebox.com/blog/laguna-xs21) | `--target-device hip:0`<br>`--kvflash auto` | `--target-device hip:0`<br>`--kvflash auto` |
| **Gemma 4 26B-A4B or 31B** | `--target-device cuda:0`<br>`--draft-device cuda:0`<br>`--kvflash auto` | `--target-device hip:0`<br>`--draft-device hip:0`<br>`--kvflash auto` | `--target-device hip:0`<br>`--draft-device hip:0`<br>`--kvflash auto` |
| **Qwen 3.5 0.8B Megakernel** | `uv run --directory optimizations/megakernel python final_bench.py --backend bf16` | Not supported | Not supported |

If a machine has both Strix Halo and an R9700, set `HIP_VISIBLE_DEVICES=<r9700-index>` before using an R9700-only profile. The selected card is then `hip:0` inside the process.

## DeepSeek V4 on Strix Halo

The current adaptive ROCmFPX artifact uses all six routed experts. This is the profile behind the [published Strix Halo results](https://www.lucebox.com/blog/deepseek-v4-flash-0731):

```bash
DFLASH_DS4_SPEC=1 \
DFLASH_DS4_SPEC_Q=4 \
DFLASH_DS4_FUSED_VERIFY=1 \
DFLASH_DS4_DRAFT=/path/to/dspark.gguf \
DFLASH_DS4_DRAFT_GPU=0 \
dflash_server /path/to/DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf \
  --target-device hip:0 \
  --ds4-fused-decode \
  --ds4-expert-top-k 6 \
  --ds4-prefill exact
```

## Multi-GPU

| Hardware and model | Configuration | Validation |
|---|---|---|
| **2x RTX 3090 + Qwen 3.8 27B** | `--target-devices cuda:0,cuda:1`<br>`--target-split-mode tensor`<br>`--peer-access`<br>`--cache-type-k q4_0`<br>`--cache-type-v q4_0`<br>`--verify-width 8`<br>Use the Qwen 3.8 DFlash2 drafter. | [PR #637](https://github.com/Luce-Org/lucebox/pull/637) |
| **RX 7900 XT + Strix Halo + DeepSeek V4** | Build for `gfx1100;gfx1151`, then run [`serve_ds4_dual_rocm_128k.sh`](../scripts/serve_ds4_dual_rocm_128k.sh) with the target and DSpark paths. The checked-in profile uses all six routed experts. | [PR #604](https://github.com/Luce-Org/lucebox/pull/604) |

The original setup matrix was introduced in [PR #602](https://github.com/Luce-Org/lucebox/pull/602). Keep a setting here only when it is still a useful starting point; measured claims belong beside their exact benchmark or qualification link.
