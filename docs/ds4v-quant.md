# DS4V-3 vision GGUF quant for Strix Halo plus 7900 XT

This file defines the quant recipe and the budget math for the DS4V vision GGUF.
It follows `docs/ds4v-uncensored-vision-plan.md` section DS4V-3.
The source weights are the OrcaRouter abliterated parent locked in `docs/ds4v-source.md`.
The rung sizes come from the prometheusAIR rung table for DeepSeek V4 Flash Vision.
This authoring machine has no GPU and holds no weights. The quant run stays pending on the operator machine.

## Rung table

prometheusAIR sizes for the vision parent. The mmproj at F16 adds 0.9 GiB on top of every rung.

| Rung | Weights GiB | Full load GiB |
|---|---|---|
| IQ1_M | 66.9 | 92.1 |
| IQ2_XXS | 78.8 | 104.0 |
| IQ2_S | 95.8 | 121.0 |
| IQ3_XXS | 108.7 | 133.9 |

Full load equals weights plus 0.9 mmproj plus 13 KV at 1M context plus 11.3 draft.
The total hardware budget is 152 GiB. That is 128 GiB Strix Halo plus 24 GiB discrete.

## Default rung

The default rung is IQ2_XXS.

Reason. Weights plus mmproj plus KV at full 1M context sit at 92.7 GiB. That fits one 96 GiB card with the full 1M context loaded. On the operator pair it leaves 48.0 GiB of headroom for the draft model plus KV plus serving overhead. Quality holds above IQ1_M on the abliterated source. IQ2_S also fits with 31.0 GiB left. IQ3_XXS fits with 18.1 GiB left but the margin gets thin under vision activations.

## Recipe

Run it with `scripts/ds4v-quant.sh`. The `plan` subcommand prints the exact commands. The `run` subcommand executes them. The `verify` subcommand checks the output.

1. Dequant. The FP8 source shards are dequantized to BF16 first. The abliteration is baked into the FP4 and FP8 expert shards and survives the dequant.
2. Convert. `convert_hf_to_gguf.py` emits the BF16 gguf. It also emits the mmproj at F16. That mmproj is the 0.9 GiB file.
3. Imatrix. Build the importance matrix from the abliterated source. Use text plus image calibration prompts. `llama-imatrix -m bf16.gguf --mmproj mmproj.gguf -f calibration.txt -o imatrix.gguf`.
4. Quantize. One `llama-quantize` pass per rung with `--imatrix`.

The type map inside the quantize pass.

- `token_embd` and `output` at Q8_0.
- Attention tensors and shared expert tensors at Q6_K.
- Router tensors `ffn_gate_inp` and `exp_probs_b` at BF16.
- `bias_vl` router bias at BF16. It stays on all 43 layers. The DS4V-2 manifest counted 43 of them.
- Routed expert tensors at the rung type from the table.
- The mmproj ships at F16. It is never quantized down.

The llama.cpp build is pinned at master `9400c894` or later with both vision PRs merged. Release `b10763` is refused. The script checks the version output and the git ancestry before it runs anything.

The script never deletes the source weights. The FP8 shards and the BF16 gguf both stay in place after the run.

## Budget math

Against the 152 GiB pair.

| Rung | Full load GiB | Margin GiB |
|---|---|---|
| IQ1_M | 92.1 | 59.9 |
| IQ2_XXS | 104.0 | 48.0 |
| IQ2_S | 121.0 | 31.0 |
| IQ3_XXS | 133.9 | 18.1 |

The placement follows PR 604 asymmetric expert parallelism. Dense work and hot experts run on the discrete GPU. Tail experts run on Strix Halo. The device join happens on the discrete GPU. The draft runs on Strix Halo. The card in `share/model_cards/ds4v-vision.json` records the per device budgets as 122880 MiB for Strix Halo and 24576 MiB for the discrete GPU.

The headline serving profile uses top_k 4 with the DSpark drafter at Q4RMFP4. Top_k 6 stays as the reference profile from `docs/ds4v-baseline.md`. Context is 135168 slots, same as the baseline. The KV figure above assumes full 1M context. The 135168 slot profile uses less.

## Verify

- The GGUF keeps `bias_vl` on all 43 layers. The script greps the gguf dump for `bias_vl` and counts 43.
- The mmproj loads. The script runs a one token generation through the mtmd path with the mmproj attached.
- Both checks need the weights. They run on the operator machine. No weights are needed for the script help and plan paths.

## Pending on the operator machine

- The weight download from `scripts/ds4v-fetch-source.sh`.
- The dequant, convert, imatrix and quant run. Record the GGUF SHA-256 values here once they exist.
- The unit verify box. Bias_vl on 43 layers and mmproj load.
- The ten live lanes and the perf lanes from the plan.
