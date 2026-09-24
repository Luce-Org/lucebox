# Image input

The server accepts JPEG and PNG images through OpenAI chat completions when a
model is started with its vision projector, `--mmproj <file>`. Without
`--mmproj` nothing in the text serving path changes.

| Model | Decoder | Projector | Runs on |
| --- | --- | --- | --- |
| Qwen3.8-27B | [`Qwen3.8-27B-IQ4_XS-pure.gguf`](https://huggingface.co/Lucebox/Qwen3.8-27B-IQ4_XS-fast-GGUF) (any Qwen3.5 / Qwen3.8 dense GGUF works) | [`Qwen3.8-27B-mmproj-Q8_0.gguf`](https://huggingface.co/Lucebox/Qwen3.8-27B-IQ4_XS-fast-GGUF), or any published `qwen3vl_merger` mmproj | one GPU, any backend |
| DeepSeek V4 Flash Vision (DS4V) | [`DeepSeek-V4-Flash-Vision-Exp-ROCMFPX-MIX-STRIX.gguf`](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3) | [`DeepSeek-V4-Flash-Vision-Exp-mmproj-BF16.gguf`](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3) | HIP: a Strix Halo alone, or R9700 + Strix Halo |

**Status: experimental.** Both models answer image questions correctly end to
end; see each model's verification notes for what has and has not been
measured.

## Quick start

Build the server as in the [README](../README.md#run-the-server). DS4V also
needs hipBLASLt at build time (`hipblaslt-dev` on ROCm; CMake prints
`hipBLASLt found: building the DS4V vision ops`).

### Qwen3.8-27B on one GPU (R9700)

```bash
hf download Lucebox/Qwen3.8-27B-IQ4_XS-fast-GGUF \
  Qwen3.8-27B-IQ4_XS-pure.gguf Qwen3.8-27B-mmproj-Q8_0.gguf --local-dir models
hf download Lucebox/Qwen3.8-27B-DFlash2-GGUF \
  Qwen3.8-27B-DFlash2-Q8_0.gguf --local-dir models

./server/build-hip/luce_server models/Qwen3.8-27B-IQ4_XS-pure.gguf \
  --target-device hip:0 \
  --draft models/Qwen3.8-27B-DFlash2-Q8_0.gguf --draft-device hip:0 \
  --draft-block-size 16 --max-ctx 32768 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --mmproj models/Qwen3.8-27B-mmproj-Q8_0.gguf \
  --port 8216
```

About 21 GiB of VRAM at the peak of an image request. Text and image requests
both decode with the DFlash2 drafter.

### DeepSeek V4 Flash Vision on a Strix Halo

```bash
hf download Lucebox/DeepSeek-V4-Flash-0731-ROCmFP3 \
  DeepSeek-V4-Flash-Vision-Exp-ROCMFPX-MIX-STRIX.gguf \
  DeepSeek-V4-Flash-Vision-Exp-mmproj-BF16.gguf --local-dir models
hf download Lucebox/DeepSeek-V4-Flash-0731-DSpark-GGUF \
  DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf --local-dir models

LUCE_DS4_SPEC=1 \
LUCE_DS4_DRAFT=models/DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf \
LUCE_DS4_SPARSE_DECODE_FLASH=1 \
./server/build-hip/luce_server models/DeepSeek-V4-Flash-Vision-Exp-ROCMFPX-MIX-STRIX.gguf \
  --target-device hip:0 --max-ctx 131072 --chunk 8192 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --ds4-fused-decode --ds4-fused-verify-f16-kv \
  --ds4-expert-top-k 6 --ds4-prefill sparse \
  --mmproj models/DeepSeek-V4-Flash-Vision-Exp-mmproj-BF16.gguf \
  --port 8216
```

`hip:0` must be the Strix Halo; on a host with a discrete GPU too, expose the
Strix Halo alone with `HIP_VISIBLE_DEVICES`. This is the text model's published
launch plus `--mmproj`: the Vision file replaces
`DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf` for text as well and decodes
at least as fast (numbers below). For R9700 + Strix Halo see [DS4V](#ds4v) below.

With an R9700 in the same box, run the image encoder there while the model
stays on the Strix Halo: expose both GPUs, point `--target-device` at the Strix
Halo and add `--mmproj-device` with the R9700 (on lucebox6, without
`HIP_VISIBLE_DEVICES`, that is `--target-device hip:1 --mmproj-device hip:0`).
The encoder then runs about twice as fast and streams each image into prefill
as soon as it is encoded, so the Strix Halo never waits for the next one:

| Images | Prompt tokens | Encoder on the Strix Halo | Encoder on the R9700 |
| --- | --- | --- | --- |
| 1 | 126 | 2.97 s | 2.94 s |
| 4 | 942 | 11.2 s | 9.1 s |
| 8 | 2,262 | 27.7 s | 19.2 s |
| 16 | 4,358 | 51.8 s | 34.6 s |

Time to the first token, with the published launch above and ChartQA charts.
Answers are identical in both layouts. `--mmproj-device` applies to this one-GPU
layout; with the experts split across both GPUs the encoder already runs on the
R9700.

### Send an image

```bash
IMG=$(base64 < photo.png | tr -d '\n')
curl -s http://127.0.0.1:8216/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":[
        {"type":"text","text":"What does this chart show?"},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$IMG"'"}}]}],
       "max_tokens":256}'
```

`GET /props` reports `capabilities.image_input_supported: true` once the
projector has loaded. In the Docker images, set `LUCE_MMPROJ` to the projector
path inside the container.

## Request contract

Use `POST /v1/chat/completions` with user-message content parts in display order:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }],
  "max_tokens": 128
}
```

Only base64 JPEG/PNG data URLs are supported. Remote URLs, images outside user
content arrays, and image parts through other API formats are rejected. A
request carries at most 16 images, 16 MiB encoded each and 32 MiB combined.
Decoder pixel and aspect limits also apply. A model's image marker cannot be supplied
as ordinary text.

The server expands image markers after final rendering and tokenization, and
the expanded image tokens count toward context and usage. Image requests
bypass the token-keyed prefix, disk and agent-turn caches and prompt
compression: tokens alone do not identify an image. Image requests decode
with the model's drafter like text requests.

Layer or tensor splitting across GPUs, remote target shards and upstream
forwarding do not support images. Qwen3.5 / Qwen3.8 serve images with
concurrent sequence scheduling (`--paged-attention --max-concurrency N`): each
image request is encoded when it is admitted and then prefills and decodes in
the shared batch like text, with the drafter. On one R9700, four concurrent
256-token image answers finish in 6.9 s (149 tok/s in total) against 13.3 s
(77 tok/s) one at a time.

DeepSeek V4 Flash Vision batches too, with the batched launch from the DeepSeek
guide plus `--mmproj` (and `--mmproj-device` for an R9700 encoder):

```
luce_server models/DeepSeek-V4-Flash-Vision-Exp-ROCMFPX-MIX-STRIX.gguf \
  --target-device hip:1 --mmproj-device hip:0 \
  --paged-attention --max-concurrency 4 --kv-pool-tokens 24576 --max-ctx 8192 \
  --ds4-prefill exact --prefix-cache-slots 0 --ds4-expert-top-k 6 \
  --mmproj models/DeepSeek-V4-Flash-Vision-Exp-mmproj-BF16.gguf
```

Its image blocks need whole-block bidirectional prefill, which the batched
engine's 16-row step cannot run. Image requests admitted since the last step
are therefore prefilled up to their last token together, in shared
layer-major sparse passes into per-request staging caches (each layer's
experts are read once for all of them); that state is copied into each
request's paged slot and the last token prefills in the batch, so the answers
decode alongside everyone else. On the Strix Halo with the encoder on the
R9700, four concurrent image answers of 256 tokens finish in 35 s (29 tok/s in
total), two images plus two text requests at 31 tok/s; four text requests
reach 38 tok/s. Image requests beyond the free slots wait in the queue. `/props` reports the
effective capability in `capabilities.image_input_supported` after backend
initialization.

## Qwen3.5 / Qwen3.8

The launch is in the [quick start](#qwen38-27b-on-one-gpu-r9700).

The projector is read directly from the published `clip` file, BF16, F16 or
Q8_0; Q8_0 is recommended (below). Projectors with
deepstack branches (Qwen3-VL) are refused. An image is resized the way the
model was trained (bicubic, both sides to a multiple of 32 pixels) and costs
one token per 32x32 pixels, between 64 and 1,024 tokens; larger images are
scaled down to the cap.

Image tokens take two-dimensional rotary positions, so positions run behind
token counts after an image. Prefill handles that in its normal chunk loop;
decoding carries the offset for the rest of the request.

### Verification status

Covered by `test_qwen35_image`: target sizes against the model's reference
resize rule, the tower's patch order and position table sampling, marker
expansion, rotary positions, and image rows that straddle prefill chunks.

Measured on an R9700 alone with the DFlash2 drafter, thinking off. With the
Lucebox `Qwen3.8-27B-IQ4_XS-pure` file and a Q8_0 projector:

- 220 seeded questions from `lmms-lab/ai2d` and `lmms-lab/ChartQA` with
  lmms-eval prompts: AI2D 90/100, ChartQA relaxed accuracy 56/60 (augmented)
  and 42/60 (human). Image prompts prefill in 0.56 s on average; one to four
  images per request all answer correctly (four images, 2,495 tokens: 3.2 s).
- Text decodes at 56 to 117 tok/s on 256-token answers (84 on average).
- Image requests decode with the drafter. On 12 images with 256-token
  answers: 4.0 s per answer (76 tok/s after the first token), against 5.5 s
  for llama.cpp with the same drafter (`--spec-type draft-dflash`) and 8.4 s
  without one; faster than llama.cpp with the drafter on every image, 1.21x
  to 1.58x. The 220-question score
  is unchanged (188, 218 answers identical to plain decode).

With unsloth's UD-IQ4_XS file and the published BF16 projector:

- 220 seeded questions from `lmms-lab/ai2d` and `lmms-lab/ChartQA` with
  lmms-eval prompts: AI2D 85/100, ChartQA relaxed accuracy 55/60 (augmented)
  and 43/60 (human), no errors. Image prompts average 448 tokens and prefill in
  0.71 s (largest 1,068 tokens, 1.8 s); decode runs at 31 to 35 tok/s (plain
  decode, measured before image requests used the drafter).
- A projector with its weight matrices in Q8_0 (rows that are not a multiple
  of 32 stay F16) encodes a 975-token image in 443 ms instead of 677 ms with
  the BF16 file, with the same scores on the 220 questions and 216 identical
  answers. Prefer one when available.
- llama.cpp (HIP build, `-fa on`, same GGUF, projector and image cap) answers
  the same on every test image; on a 1,012-token image prompt it prefills in
  1.65 s to our 1.68 s, on a 323-token one in 0.61 s to our 0.50 s.
- An image inside a 6,271-token prompt, two images in one request, and a
  follow-up turn after an image all answer correctly.
- Text requests are byte-identical to a build without image support, at the
  same speed, with or without a projector loaded (five prompts up to 19.6K
  tokens). The projector adds 0.9 GiB of VRAM; the peak during image requests
  was 21.6 GiB against 20.8 GiB for text.
- The same requests answer correctly on a Strix Halo alone, where a
  1,012-token image prompt prefills in 4.6 s and decodes at 14 tok/s (plain
  decode).

Not yet established: a comparison against the reference implementation on the
same questions, and CUDA. The tower uses only standard ggml
operators, so nothing in it is HIP specific.

## DS4V

The server must be built with hipBLASLt available (the `hipblaslt-dev` package
on ROCm). CMake reports `hipBLASLt found: building the DS4V vision ops`; a build
without it refuses `--mmproj` for this model at startup.

Image input needs Linux HIP, a DeepSeek4 decoder whose GGUF carries the image
router biases, `--ds4-prefill sparse`, and `--mmproj` pointing at the published projector (or one [exported with our
tool](ds4v-mmproj.md)). Two layouts work:

- **One GPU holding the whole model** (for example a Strix Halo): nothing else
  to set. The projector is loaded after the weights and must fit beside them.
- **Two GPUs splitting the experts in process** (for example R9700 + Strix
  Halo): `LUCE_DS4_MOE_TP=1`, `LUCE_DS4_MOE_TP_INPROC=1`, and
  `LUCE_DS4_MOE_TP_GPU` selecting the second device, with `--target-device`
  on the first. Device ordinals must match the host's actual topology.

Remote expert IPC, all-on-secondary placement, experts kept on the CPU and
dense prefill do not support images.

Published llama.cpp conversions of the decoder load directly (image router
bias named `blk.N.exp_probs_b_vl.bias`, no `deepseek4.vocab_size` key). Split
GGUF files and llama.cpp's `clip` projector files are not read yet.

A decoder in our own ROCMFP MIX format comes from `tools/ds4_mix_converter`
run on the Vision-Exp checkpoint. It follows the shipped DeepSeek-V4-Flash
recipe: routed gate and up experts in fp2, down experts in fp2 on the shipped
layer set and fp3 elsewhere, dense projections in ROCmFP4, the token embedding
in Q6_K, codebooks embedded in the GGUF (one file, about 100 GB). It keeps the
image router biases. Pass `--imatrix` with an importance matrix (llama.cpp's
per-expert layout is used expert by expert; the community publishes one for
this model) or `--absmax-only`. The converter uses every core: about 40 minutes
for this checkpoint on 32 cores.

Image requests wait in the same queue as text requests. A waiting request
holds only its preprocessed patches, a few MB per image; its encoded rows
exist only while it runs, so the number of slots bounds them. When host
memory is too short to prepare another image request, the server answers
HTTP 503 and the client should retry.

The server expands image markers after final rendering and tokenization.
Expanded image tokens count toward context and usage. Image blocks remain
whole during prefill, image rows use their learned routing bias, and raw
attention is bidirectional within each image's visible span. The projector's
tile permutation is applied once when assembling rows with named sentinel
embeddings. All chunks are capped at 1,024 tokens while a projector is loaded.

The image payload survives request copies and retry paths. Failed or cancelled
multi-image encoding publishes no partial embedding matrices.

### Memory

The projector is validated and loaded before expert placement. Admission counts
actual selected owner tensor sizes, allocation alignment, MIX tables, copy
staging, future KV, and explicit execution reserves. Host and integrated-device
charges share one physical-memory budget. Before image decoding, the server
checks host availability; before encoding, it synchronizes and releases
disposable decoder, owner, and draft graphs and checks live device/host
availability again. KV, saved snapshots, and draft weights remain reflected in
that live measurement. Reservations are conservative policy, not a guarantee
against unrelated concurrent allocations.

### Verification status

Covered by unit tests in the main build: image transport and request policy,
prompt expansion and ownership, embedding assembly and cancellation, image
spans and the expert budget, plus the decoder loader and image-batch admission
tests in `test_deepseek4_unit`.

Measured with the public `DeepSeek-V4-Flash-Vision-Exp` Q2_K_S decoder and
the exported projector, on a Strix Halo alone and on R9700 + Strix Halo, 220
seeded questions from `lmms-lab/ai2d` and `lmms-lab/ChartQA` with lmms-eval
prompts: AI2D 85/100, ChartQA relaxed accuracy 55/60 (augmented) and 43/60
(human). Both layouts score the same and give word-identical answers on 213 of
220 questions. An image request prefills in about 4 s and, without the
drafter, decodes at about
23 tok/s.

With our own ROCMFP MIX conversion of the same checkpoint (per-expert
importance matrix, the shipped recipe above), on a Strix Halo alone at top-k 6:

- Against the MXFP4 reference (native FP4 experts) on 8,176 wikitext-2 tokens:
  KL 0.464 mean, 0.102 median, top-1 agreement 78.4%, perplexity 4.14 against
  2.82. The community Q2_K_S scores KL 0.511 in our engine (0.523 in
  llama.cpp) and perplexity 4.22.
- AI2D 84/100, ChartQA 54/60 and 40/60 (the Q2_K_S: 85, 55, 43); the sanity
  and one-to-four-image sets are all correct.
- With the published DSpark drafter and fused decode and verify, text decodes
  at 25 to 37 tok/s on 256-token answers (30 mean), as fast as the shipped
  text model.
- Image requests decode with the DSpark drafter too: on 12 images with
  256-token answers, 13.3 s per answer (30 tok/s after the first token)
  against 15.7 s (22 tok/s) without it. Capturing the drafter's features
  during prefill adds about 0.7 s before the first token, so one-word answers
  come back slightly later. The 220 questions score AI2D 86, ChartQA 55 and
  40 with the drafter (209 answers identical to plain decode); one to four
  images all correct.

Not yet established:

- The vision tower misses the fixed 0.9995 feature-cosine gate against the
  reference implementation: 0.99906 on the Radeon RX 7900 XT it was developed
  on, 0.99823 on CPU. Embeddings pass; features do not.
- No comparison against the reference implementation on the same questions.

## Code layout

Shared by every model:

| Piece | Where |
| --- | --- |
| Reading images out of a request, limits, redaction | `server/src/server/image_input.*` |
| JPEG and PNG decoding to RGB | `server/src/common/vision/image_decode.*`, codecs in `server/cmake/ImageCodecs.cmake` |
| Bicubic resizing that matches Pillow byte for byte | `server/src/common/vision/image_resize.*` |
| Reading a published `clip`-format projector file | `server/src/common/vision/mmproj_file.*` |
| Image positions in a prompt, batches that keep an image whole | `server/src/common/vision/image_spans.h` |
| The backend contract | `supports_images`, `image_placeholder`, `prepare_images` in `server/src/common/model_backend.h`; `GenerateRequest::images` in `server/src/common/generation_types.h` |

DS4V only, all under `server/src/deepseek4/`: resizing and patching
(`deepseek4_vision_preprocess`), the vision tower (`deepseek4_vision`), marker
expansion and embedding assembly (`deepseek4_image_prompt`,
`deepseek4_image_assembly`), attention visibility and expert routing for image
rows (`deepseek4_image_policy`), and memory admission
(`deepseek4_image_admission`).

Qwen3.5 / Qwen3.8 only, all under `server/src/qwen35/`: the vision tower and
its preprocessing (`qwen35_vision`), marker expansion and rotary positions
(`qwen35_image_prompt`), what a request carries (`qwen35_image_request.h`), and
the backend's three contract methods (`qwen35_backend_images.cpp`). Prefill and
decode changes are a few lines in `qwen35_backend.cpp`.

Another model needs its own preprocessing, tower and prompt expansion, and its
backend implements the three contract methods. Nothing in the HTTP server or in
`common/vision` names a model.
