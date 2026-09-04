# DS4V-2 abliterated vision source

This file locks the source weights for the DS4V vision GGUF.
It follows `docs/ds4v-uncensored-vision-plan.md` section DS4V-2.
The source is a drop in for `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`.

## Source repo

- Parent repo. `orcarouter/DeepSeek-V4-Flash-Vision-Uncensored` on huggingface.co.
- Base model. `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`. Relation is finetune.
- License. MIT. The repo ships a `LICENSE` file.
- Pipeline tag. image-text-to-text. Architecture `DeepseekV4ForCausalLM`.
- Download commands live in `scripts/ds4v-fetch-source.sh`.

## Provenance

- The OrcaRouter parent is abliterated. A rank 1 refusal direction was removed from the residual stream.
- The removal is baked into the FP4 and FP8 expert shards. No runtime patch is needed.
- The vision tower is preserved. The aligner is preserved. The MTP draft head is preserved.
- The repo card states measured evals for the abliteration. None were re-run here.

## Manifest numbers

- Shards. 48.
- Tensors. 72633.
- Vision tensors. 259. Keys prefixed `vision.`.
- Aligner tensors. 4. Keys prefixed `aligner.`.
- Router bias tensors. 43. Keys containing `bias_vl`.
- MTP blocks. 3. Keys prefixed `mtp.`.
- Status. These counts are upstream stated. They are pending verification against the real `model.safetensors.index.json`.
- The repo is gated on huggingface.co. The index download needs a token that has accepted the gate. `scripts/ds4v-fetch-source.sh manifest` accepts `HF_TOKEN` as bearer auth.
- Verification is pending on the operator machine together with the weight download and the shard checksums.

## Rejected alternatives

- OrcaRouter GGUF directly. Rejected. It is text only. It drops the vision tower.
- Fresh abliteration of the stock checkpoint. Rejected. The OrcaRouter parent already bakes the edit with measured evals.

## What runs where

- This authoring machine has no GPU and holds no weights. The 157 GiB parent download stays pending for the operator machine.
- The fetch script downloads the full parent repo into `models/DeepSeek-V4-Flash-Vision-Uncensored`. It never deletes weights.
- `scripts/ds4v-fetch-source.sh check-only` verifies shard SHA-256 checksums once the shards are local. It never downloads shards.
