# MOSS-Image results

**Setup.** AMD Radeon AI PRO R9700 (RDNA4 `gfx1201`, 32 GB) at its 330 W power limit, ROCm 7.2.4. luce_server built for `gfx1201`. MOSS-Image SFT as int8 convrot with the FLUX.2 VAE in one GGUF (`server/scripts/convert_moss_image.py` from the bf16 release), Qwen3.5 9B Q8_0 text encoder. 1024×1024, 8 steps of er_sde on the simple schedule (the sampler ComfyUI runs for this model), CFG 1, seed 42, unless a table says otherwise.

**Method.** `bench.sh`: three rounds; in each, the server starts (the first start tunes hipBLASLt with two warm-up images, later starts read the tuned kernels back), then takes the three test prompts, each new to it. Medians of 9 images. *Request* is timed by the client, prompt in and PNG out over HTTP; *generate_image* is the server's text encoding, sampling and VAE decode.

## Per image on the R9700

| Setting | Request | generate_image | Text | Sampling, 8 steps | Decode |
|---|---:|---:|---:|---:|---:|
| **CFG 1** | **2.74 s** | 2.71 s | 0.03 s | 2.53 s | 0.16 s |
| CFG 2.5 | 5.20 s | 5.17 s | 0.03 s | 4.98 s | 0.16 s |

A step takes 0.32 s at CFG 1 and 0.62 s at CFG 2.5, which runs the model twice. The PNG and the HTTP round trip add 0.03 s. The previous kernels took 3.06 s and 5.80 s in the same runs (sampling 2.83 s and 5.56 s, decode 0.18 s; bench.sh starts the two builds in turn each round), and made the same images bit for bit.

## Against the stable-diffusion.cpp build it replaces

This engine replaces a build of stable-diffusion.cpp `b13a482` with the same kernels as ggml patches, served by its `sd-server`. Both servers took the same requests in one session from one client script: per setting a warm-up request, then three rounds of the three test prompts, so from the second round each prompt repeats (both keep the text encoding of the last four prompts). Medians of 9 images:

These runs predate the current kernels, which take luce_server's er_sde rows to 2.74 s at CFG 1 and 5.20 s at CFG 2.5 with the same images (above).

| Setting | stable-diffusion.cpp build | luce_server |
|---|---:|---:|
| Euler, 1 step | 0.597 s | **0.541 s** |
| Euler, 8 steps | **3.028 s** | 3.040 s |
| er_sde, 8 steps | 3.109 s | **3.063 s** |
| er_sde, 8 steps, CFG 2.5 | 5.821 s | **5.766 s** |

- The model costs about the same in both: sampling took 2.79 s there and 2.84 s here with Euler, 2.86 and 2.87 s with er_sde, and 5.58 and 5.57 s at CFG 2.5. The decode takes 0.20 s there and 0.18 s here.
- Euler at 8 steps is the one setting where the old build was faster, by 12 ms: its Euler sampling was 0.05 s shorter, while its er_sde sampling took 0.07 s longer than its Euler. Here the two samplers differ by 0.03 s.
- With `bench.sh`'s method (new prompts, a fresh server each round), the old build took 3.16 s per request at CFG 1 (generate_image 3.11 s, sampling 2.87 s) and 5.92 s at CFG 2.5; stock stable-diffusion.cpp `b13a482` took 12.19 s at CFG 1.
- luce_server is ready 4.4-4.5 s after it starts, from the page cache: the model (2.9 s) and the text encoder (4.3 s) load at once while hipBLASLt loads the kernels tuned on the first start, so the first request runs at full speed. The first start tunes them with two warm-up images (3.0 s) and is ready after 7.2-7.5 s; the starts after it pick the same kernels, and their images match its bit for bit. `sd-server` was ready after 6.0 s and took 8.5 s on its first request.
- With the text encoder on the APU beside the R9700 (a Ryzen AI Max+ 395's Radeon 8060S, `--text-encoder-device hip:1`, built for `gfx1201;gfx1151`; the R9700 behind a PCIe Gen3 x4 link), luce_server is ready after 2.0-2.1 s: the model (2.0 s) has the R9700's link to itself while the encoder loads into the APU's memory (1.1 s). generate_image then takes 2.76-2.79 s for a new prompt, whose encoding takes 77-98 ms there instead of 25-28 ms, and 2.67-2.69 s for a repeated one. The APU rounds the encoder differently, so the images are close to the R9700-encoded ones, not byte-identical: PSNR 35.6, 44.0 and 37.9 dB on three prompts (seeds 71, 73 and 74), the same composition in each, with one local detail redrawn in the first.

## Accuracy

The reference is the bf16 release weights on stock stable-diffusion.cpp with f32 attention everywhere (`--flash-attn off`), the most exact run its code offers. Every run in this section samples with Euler on the discrete schedule. PSNR of each build's images against it, higher is closer; one step shows a single forward pass's error, eight steps show it compounded:

| Build | Weights | Steps | cat | poster | street | Mean |
|---|---|---:|---:|---:|---:|---:|
| stable-diffusion.cpp, stock | int8 | 1 | 40.56 | 39.13 | 39.73 | 39.8 |
| stable-diffusion.cpp build | int8 | 1 | 44.50 | 46.06 | 51.30 | 47.3 |
| **luce_server** | int8 | 1 | 51.72 | 47.90 | 52.61 | **50.7** |
| stable-diffusion.cpp, stock | bf16 | 8 | 21.81 | 27.12 | 31.54 | 26.8 |
| stable-diffusion.cpp, stock | int8 | 8 | 24.62 | 28.14 | 29.99 | 27.6 |
| stable-diffusion.cpp build | int8 | 8 | 33.92 | 35.26 | 36.47 | 35.2 |
| **luce_server** | int8 | 8 | 36.76 | 32.51 | 35.73 | **35.0** |

- After one step luce_server lands closest to the reference. After eight, the two int8 builds score the same within the spread of three images: the per-image scores move both ways (+2.8 dB for the cat, -2.8 dB for the poster).
- The two builds do not round alike everywhere (for example, the text encoder is the engine's own Qwen3.5 graph, and the VAE's attention runs in f16 where `sd-server`'s runs in f32), so their images are close, not byte-identical: PSNR 47-53 dB between them after one step, 34-45 dB after eight, and 27-29 dB with er_sde at CFG 2.5, whose noise each step compounds the differences. Composition and lettering agree.
- The initial noise is stable-diffusion.cpp's, bit for bit (its Philox generator and Box-Muller transform), and so is er_sde's noise at each step.
- Stock's own bf16 run with its default flash attention lands no closer than its int8 run, so on this card most of stock's error comes from its attention path, not from the int8 weights.

## Run to run

Requests no longer time kernels: with the text encoder on the R9700, a request's image depends only on the request, the tuning file, the GPU and hipBLASLt.

- Three starts from one tuning file ran the same 62 requests: sizes from 256×256 to 4096×256, prompts of 11 to 3,397 tokens around the edges of the text-token blocks, CFG 1 and 2.5, both samplers and schedules, 1 to 20 steps, and batches of 2. With the text encoder on the R9700 they made the same 62 images on every start, a request repeated in one server made the same image, and the tuning file stayed byte-identical.
- Before, the first request at a shape the warm-up had not tuned (another size, or a prompt past 256 tokens) timed hipBLASLt's candidates and kept the fastest, so its kernel, and with it the image's last bits, could change from one start to the next. Such a shape now takes hipBLASLt's own kernel, and its first image comes sooner: 2.55-2.58 s instead of 2.86 s at 1008×1008, and 2.65-2.69 s instead of 2.95-2.96 s at 768×1344 (generate_image, two starts of each build). The images after it take as long as before: 2.53-2.57 s against 2.55-2.56 s, and 2.59-2.62 s against 2.62-2.63 s. At the shapes the warm-up tunes, the images are the same as before, bit for bit.
- With the text encoder on the APU, a prompt of more than 128 tokens can encode differently from one run to the next, and its image with it: over three starts, 3 of the 62 requests (prompts of 129, 139 and 191 tokens) changed. Encoding the same prompt again in one server gave another result now and then (2 of 40 encodings), and more often while another program used the GPU. Given the same encoder output, every image repeated.
- `CHECK=1 ./bench.sh` compares every image with the md5s in `reference/`. On a second R9700, the previous build and this one both made the six stored images (three prompts at CFG 1 and 2.5) in all three rounds, at the same speed: 2.72 s at CFG 1 and 5.12 s at CFG 2.5 per request for each.

## Sampler and CFG

- At 8 steps, Euler leaves images hazy on its discrete schedule as on the simple one; er_sde does not, on either. Both make one model call a step.
- At CFG 1, some images come out misshapen. In a check by eye of two prompts (a fox in fresh snow, a fisherman mending a net) with seeds 1 to 4, 4 of the 8 images were misshapen at CFG 1 and 1 of 8 at CFG 2.5. The server defaults to CFG 1, which runs the model once a step; a request sets CFG 2.5 with `cfg_scale` or `guidance.txt_cfg`.

## Against a DGX Spark

The same prompts, seed, size, steps, CFG and sampler on a DGX Spark (GB10), warm, but Euler in the stable-diffusion.cpp row:

| DGX Spark setup | Prompt repeated | New prompt | luce_server is |
|---|---:|---:|---:|
| ComfyUI, int8 convrot diffusion weights from our converter, 16-bit attention, full VAE | 4.54 s | 5.60 s | **1.7× / 2.0×** |
| ComfyUI, its shipped files: int8 attention, small VAE | 4.22 s | 5.34 s | 1.5× / 1.9× |
| stable-diffusion.cpp `b13a482` (CUDA), same int8 weights, Euler | | 16.1 s | 5.9× |

- ComfyUI 0.42.0 (`b1a247c`), PyTorch 2.14.1 with CUDA 13.0, comfy-kitchen 0.2.39, its native MOSS-Image nodes and its default sampler for the model, er_sde on the simple schedule. Its text encoder is the shipped Qwen3.5 9B w4a8 in both ComfyUI rows; ours is Q8_0.
- Times are ComfyUI's own "Prompt executed in", which includes saving the PNG but not the HTTP round trip. ComfyUI keeps the previous request's text encoding, so the same prompt twice in a row skips it: one such request per setup, against five new-prompt requests (5.59-5.61 s).
- The R9700 side is luce_server's request time with a new prompt each time (2.74 s), so the comparison counts more work on our side, not less.
- ComfyUI per new-prompt image: text encoding 1.07 s, sampling 3.95 s (0.49 s a step), VAE 0.45 s, saving 0.12-0.29 s. luce_server: 0.03 s, 2.53 s (0.32 s a step), 0.16 s, and 0.03 s for the PNG and HTTP.
- The shipped files trade quality for speed: in our poster test they garbled the word "LUCEBOX", which the 16-bit attention run kept readable (one prompt, one seed).

## Standby

`--standby` with the text encoder on the APU (`--text-encoder-device hip:1`), the R9700 behind its PCIe Gen3 x4 link, and a chat model (Qwen3.8 27B, 22.8 GB) on the same R9700 between the image requests. Seeds 71, 73 and 74 at CFG 1, each prompt new to the server; generate_image per image, and with the previous kernels, at CFG 1 and (seed 75) at CFG 2.5:

| Weights when the request arrives | CFG 1 | Sampling, CFG 1 | Previous kernels, CFG 1 / 2.5 |
|---|---:|---:|---:|
| On the GPU, after `POST /memory/load` | 2.76-2.84 s | 2.50-2.53 s | 2.98-3.08 s / 5.63 s |
| In host memory, copied during the first step | 3.91-3.92 s | 3.49-3.50 s | 4.30-4.66 s / 7.15 s |

- `POST /memory/load` took 1.55-1.67 s for the 5,258 MiB of weights (3.3-3.6 GB/s). It needs room for them plus 1 GiB: beside the chat model about 4 GB stayed free with the weights on the card.
- `POST /memory/release` gave back 12,500 MiB (weights and work buffers) in 6-7 ms. One sent 0.4 s into a load stopped it at the next 128 MiB piece. Released, the server held about 0.6 GB of the R9700, its context.
- A release also stops a load still waiting for the GPU. With one load copying the weights and a second one queued behind it, a release sent 0.6 s in stopped both and answered in 25 ms; the previous build let the queued load copy all 5,258 MiB first, which held the release back 1.6 s.
- Every image matched, bit for bit, the one a server without `--standby` made from the same seed, whether its weights were loaded, kept or copied during its first step, at CFG 1 and at CFG 2.5.
- With a controller that releases the weights for each chat request and loads them again 2 s after the last one, an image asked for once the chat was idle took 2.98 s end to end over HTTP (3.22-3.27 s with the previous kernels), and one asked for right after a long chat reply, which took the weights off, 4.06-4.08 s. When a short text chat request alone on the card leaves the weights there, an image right after its reply took 2.92-3.02 s, instead of 4.32 s with the weights taken off and the previous kernels. Images back to back took 2.70 s for a repeated prompt and 2.80 s for a new one (medians of 5).

## Power

The R9700 runs at its power limit during every request. These numbers use 330 W, the most the driver allows; the card ships at 300 W, which makes an image about 1% slower.
