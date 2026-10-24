<p align="left">
  <a href="../../README.md">← lucebox</a>
</p>

<p align="center">
  <img src="hero.png" width="760" alt="Two 1024x1024 MOSS-Image outputs from luce_server at 8 steps of er_sde and CFG 2.5: a vintage travel poster with the word LUCEBOX over a mountain lake, and a red bicycle against a stone wall in a narrow street" />
</p>

<h1 align="center">MOSS-Image on one R9700</h1>

<p align="center">
  <strong>A 1024×1024 image in 2.74 s on one Radeon AI PRO R9700, from the full 5B model, served by luce_server.</strong><br/>
  MOSS-Image SFT, 8 steps of er_sde, int8 convrot weights, no distillation and no step caching; 5.20 s at CFG 2.5, as in the images above.<br/>
  A native engine in luce_server with our own int8 GEMMs, attention, sampler and VAE kernels for RDNA4, closer to the bf16 model than stable-diffusion.cpp.<br/>
  <strong>1.7×</strong> a DGX Spark running ComfyUI on the same diffusion weights and sampler, <strong>2.0×</strong> on a new prompt; 1.5× and 1.9× its faster ComfyUI setup, with int8 attention and the small VAE.
</p>

---

```
MOSS-Image SFT · 1024×1024 · 8 steps of er_sde · CFG 1 · warm server · median per image

  R9700, luce_server                      █████                             2.74 s
  DGX Spark, ComfyUI, same prompt again   █████████                         4.54 s
  DGX Spark, ComfyUI, new prompt          ███████████                       5.60 s
  R9700, stable-diffusion.cpp             ████████████████████████         12.19 s
  DGX Spark, stable-diffusion.cpp, Euler  ████████████████████████████████  16.1 s

  Same int8 convrot format, steps, CFG, size and seed in every row, and the same sampler
  except in the last row. Each luce_server image is a new prompt to the server, timed from
  prompt in to PNG out over HTTP. Method in RESULTS.md.
```

MOSS-Image is a 5B single-stream diffusion transformer (42 layers, 24 heads of 128) with a Qwen3.5 9B text encoder and the FLUX.2 VAE. luce_server runs it natively: give it the MOSS-Image GGUF and it answers the images API instead of chat. The model is untouched: same weights, steps, sampler and seed as the reference, with nothing distilled, cached or skipped. The speed comes from the kernels.

## Quick start

Needs ROCm 7.2 with the hipBLAS, hipBLASLt, rocBLAS and zlib development packages, cmake, ninja, and python3 with numpy and PyYAML for the conversion.

```bash
cmake -S ../../server -B ../../server/build -DCMAKE_BUILD_TYPE=Release \
  -DLUCE_GPU_BACKEND=hip -DLUCE_HIP_ARCHITECTURES=gfx1201
cmake --build ../../server/build --target luce_server -j
./fetch_models.sh   # pinned Hugging Face revisions into ./models, then writes moss-image.gguf (about 6 minutes)
../../server/build/luce_server models/moss-image.gguf --text-encoder models/Qwen3.5-9B-Q8_0.gguf --port 7860
```

The server is ready once both files are on the GPU, about 4.5 s from the page cache, and the first request runs at full speed. The first start takes 3 s more: two warm-up images tune hipBLASLt's kernels, which it keeps in `~/.cache/lucebox/moss-image-gemm.txt` (under `$XDG_CACHE_HOME` when set) for the starts after it.

On a machine with a second GPU, `--text-encoder-device hip:N` runs the text encoder there and hands its output over through host memory, so the two GPUs need no peer access. Build for both (`-DLUCE_HIP_ARCHITECTURES="gfx1201;gfx1151"` for the R9700 beside a Ryzen AI Max+ 395): the encoder then runs on the APU's Radeon 8060S and the R9700 loads only the 5.5 GB model. With the R9700 behind a PCIe Gen3 x4 link, the server is ready in 2.0-2.1 s instead of about 4.5 s, and a new prompt's encoding takes 78-98 ms instead of 25-28 ms. The two GPUs round the encoder differently, so the images are close to the R9700-encoded ones, not byte-identical, and on the APU a prompt of more than 128 tokens can encode differently from one run to the next, its image with it (see [RESULTS.md](RESULTS.md)).

With `--standby` the server shares its GPU with another model, such as a chat model on the same card. The weights wait in pinned host memory. `POST /memory/release` gives back the GPU memory the server holds in under 10 ms, leaving its context (about 0.6 GB with the text encoder on a second GPU), and `POST /memory/load` copies the weights onto the GPU while the other model is idle, in 1.6 s over the Gen3 x4 link. A request that finds the weights in host memory copies them during its first step, each layer waiting only for its own weights: 3.9 s an image instead of 2.8 s, and the same image bit for bit.

```bash
curl -s http://127.0.0.1:7860/v1/images/generations -H 'Content-Type: application/json' \
  -d '{"prompt": "a lovely cat holding a sign that says \"hello\"", "size": "1024x1024"}' |
  python3 -c 'import base64, json, sys; open("cat.png", "wb").write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))'
```

## API

| Route | Takes |
|---|---|
| `POST /v1/images/generations` | OpenAI's `prompt`, `n` and `size`; PNG as `b64_json` |
| `POST /sdapi/v1/txt2img` | AUTOMATIC1111's `prompt`, `negative_prompt`, `width`, `height`, `steps`, `cfg_scale`, `seed`, `batch_size`, `sampler_name`, `scheduler` |
| `GET /v1/models` | the model, under `--model-name` (default `moss-image`) |
| `GET /memory` | where the weights are (`gpu` or `host`), the MiB the server holds on the GPU and in pinned host memory, and the GPU's free MiB |
| `POST /memory/release` | with `--standby`: frees the work buffers and the weights, or only the work buffers with `{"keep_weights": true}`; stops the loads asked for before it, also one still waiting for the GPU |
| `POST /memory/load` | with `--standby`: copies the weights onto the GPU; 409 with the reason when the GPU lacks room for them plus 1 GiB, or a later release stops it |

A request without settings gets 1024×1024, 8 steps of er_sde on the simple schedule (the sampler ComfyUI runs for this model), CFG 1 and a new seed. stable-diffusion.cpp's `<sd_cpp_extra_args>{...}</sd_cpp_extra_args>` in an OpenAI prompt sets the seed, negative prompt, size, image count and `sample_params` (`sample_steps`, `sample_method` `euler` or `er_sde`, `scheduler` `discrete` or `simple`, `guidance.txt_cfg`, `eta`, `flow_shift`), so clients written for `sd-server` work unchanged. Sides are multiples of 16 from 256, up to 1024×1024 pixels in all; up to 100 steps and 8 images a request, and prompts of up to 16 KiB.

CFG 1 runs the model once a step (2.74 s an image), and at 8 steps some of its images come out misshapen: by eye, 4 of 8 in a check of two prompts with four seeds each, against 1 of 8 at CFG 2.5, where each step runs the model twice (5.20 s). Euler leaves 8-step images hazy on either schedule; er_sde does not.

## How it works

luce_server opens the GGUF, sees the `moss-image` architecture and starts the image engine (`server/src/moss_image/`) instead of the chat loader. A request runs in three parts:

1. **Text.** The engine's own Qwen3.5 loader and graph run the 9B encoder over the prompt in MOSS-Image's chat template, without the lm_head; the DiT's text embedder then runs in bf16 on hipBLASLt. The last 4 prompts' encodings stay on the GPU, so a prompt sent again with another seed skips this part.
2. **Sampling.** Each step runs the 42 transformer layers over the image's 4,096 tokens plus the prompt's. With CFG the two passes are one batch of rows, and the unconditional pass skips layer 22 as stable-diffusion.cpp does. The sampler runs on the GPU too: Euler and er_sde on stable-diffusion.cpp's schedules, with its Philox noise bit for bit, so a seed starts from the same noise.
3. **Decode.** The FLUX.2 decoder, with an implicit-GEMM convolution for RDNA4 that reads the nearest-upscale source and adds bias and skip connection in its epilogue, fed by a group norm and SiLU that write its f16 input directly. 0.18 s per image.

Each transformer layer is 9 kernel launches, 10 with CFG, where the two passes' attention runs as two:

| Kernel | Does the work of |
|---|---|
| norm + quantize | RMS norm, the step's modulation scale and shift, the 256-wide Hadamard rotation of the int8 convrot format, per-token int8 |
| int8 qkv GEMM + attention prologue | GEMM, dequantize, q and k RMS norm and RoPE, v cast to f16 |
| flash attention for RDNA4 | head size 128, no mask, the next K/V tile prefetched while the current one computes |
| quantize the attention output | rotation, per-token int8 |
| int8 o GEMM + gated residual | GEMM, dequantize, gate multiply, residual add |
| norm + quantize | as above, with the second modulation |
| int8 up/gate GEMM | GEMM, int32 sums for the gate |
| SiTU gate + quantize | dequantize, two tanh caps, sigmoid, gate, rotation, int8 |
| int8 down GEMM + gated residual | GEMM, dequantize, gate multiply, residual add |

- **Own int8 GEMMs** (`dit_kernels.cu`) for all four linears. 256×128×64 tiles on RDNA4 WMMA with the epilogue in registers, so the int32 sums of qkv, o and down never go to memory. The SiTU gate needs whole rows of the up/gate sums, and the card runs at its power limit, so it stays a separate memory-bound pass, where it costs less than inside the GEMM. The partial last block of text tokens splits its k loop four ways, so its tiles fill in as the others finish instead of holding the GPU for a round of their own.
- **hipBLASLt** (`blas.cpp`) runs the bf16 and f16 products: the text embedder, the DiT's first and last linears and the VAE's attention. On the first start, two warm-up images, with and without CFG, time hipBLASLt's 16 best candidates for each shape of a 1024×1024 request, one tuning per 256 tokens, and the tuning file keeps the fastest. A later start finds each among hipBLASLt's candidates again and runs it once while the weights upload, so it needs no warm-up image. Requests time nothing: a shape the warm-up did not tune (another size, or a prompt length its tuned kernel cannot run) takes hipBLASLt's own choice, so its products depend only on the request, the tuning file, the GPU and hipBLASLt, and come out the same bit for bit after any restart.
- **No graph runtime.** A step is a fixed sequence of kernel launches on one stream, with no allocations, copies or host waits in between.

## Correctness

- **Closer to the bf16 model than stable-diffusion.cpp.** Against the bf16 release weights run with f32 attention, luce_server's images score a PSNR of 50.7 dB after one Euler step and 35.0 dB after eight, the stable-diffusion.cpp build this engine replaced 47.3 and 35.2 dB, and stock stable-diffusion.cpp's int8 path 39.8 and 27.6 dB (mean of the three test prompts). After eight steps the per-image scores move both ways, as any two exact-but-different runs do.
- **Checked against stored images.** `CHECK=1 ./bench.sh ...` starts each server from the tuned kernels in `reference/` and fails when an image's md5 differs from the one stored there for the R9700 with hipBLASLt 100202, so a compiler or library update cannot change the pixels unnoticed.
- **The weights** are the int8 convrot format ComfyUI ships for this model: per-row int8 weights rotated in 256-wide Hadamard groups, activations quantized per token at run time. `server/scripts/convert_moss_image.py` writes them and the VAE into one GGUF from the bf16 release with numpy only (or from an int8 convrot file, to the same bytes). The result does not depend on the machine, and `fetch_models.sh` checks its sha256.

## Limits

- Measured on the Radeon AI PRO R9700 (RDNA4, gfx1201) only. The kernels are RDNA4's: on another GPU the engine refuses to load. The text encoder runs through ggml, on any GPU the build includes. Builds without hipBLASLt or zlib leave the engine out, and luce_server then says so for a MOSS-Image file.
- One request at a time; others wait their turn.
- The card ran at its 330 W power limit, the driver's maximum. At the 300 W default an image takes about 1% longer.

## Files

| Path | Contents |
|---|---|
| `server/src/moss_image/` | the engine: weights, text encoder, pipeline, kernels, sampler, PNG, HTTP |
| `server/scripts/convert_moss_image.py` | the GGUF from the release files |
| `fetch_models.sh` | the release files and the text encoder at pinned revisions, then the conversion |
| `bench.sh` | warm per-image times of one or more luce_server builds, and each image's md5; with `CHECK=1`, against the stored ones |
| `reference/` | the tuned GEMM kernels and the image md5s `CHECK=1` compares with |

Numbers and method: [RESULTS.md](RESULTS.md).
