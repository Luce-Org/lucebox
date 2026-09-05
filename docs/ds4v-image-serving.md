# DS4V image serving

The DS4V integration accepts JPEG and PNG images through OpenAI chat
completions when the matching projector is supplied with `--mmproj`.
The current implementation has passed its remote HIP build and CPU integration
checks. Private paired-model HTTP qualification is still pending; a successful
projector export or standalone encoder check does not establish that result.

## Supported configuration

The initial serving path requires Linux HIP, a DeepSeek4 decoder with the
supported DS4V dimensions, and two distinct local HIP devices. The decoder uses
sparse prefill with in-process heterogeneous expert ownership:
`DFLASH_DS4_MOE_TP=1`, `DFLASH_DS4_MOE_TP_INPROC=1`, and
`DFLASH_DS4_MOE_TP_GPU` selecting the secondary device. Set `--target-device`
to the primary device, `--ds4-prefill sparse`, and `--mmproj` to the
[exported projector](ds4v-mmproj.md). Device ordinals must match the host's
actual topology.

Layer splitting, remote expert IPC, all-on-secondary placement, dense prefill,
concurrent sequence scheduling, and upstream forwarding do not support images.
`/props` reports the effective capability in
`capabilities.image_input_supported` after backend initialization.
Without `--mmproj`, text serving follows its existing path and image requests
are rejected.

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
content arrays, and image parts through other API formats are rejected.
Requests permit at most four images, 16 MiB encoded bytes per image, and
32 MiB combined encoded bytes. Decoder pixel and aspect limits also apply.
The reserved DS4 image marker cannot be supplied as ordinary text.

One image request may be outstanding per backend. Its admission lease remains
with the immutable payload through queueing and generation; another image
request is rejected until that payload is released. This bounds simultaneous
preprocessing and prepared-image memory. Text requests retain the normal queue.

The server expands image markers after final rendering and tokenization.
Expanded image tokens count toward context and usage. Image blocks remain
whole during prefill, image rows use their learned routing bias, and raw
attention is bidirectional within each image's visible span. The projector's
tile permutation is applied once when assembling rows with named sentinel
embeddings. All chunks are capped at 1,024 tokens while a projector is loaded.

Image requests use autoregressive decoding and bypass token-only prefix,
disk, and agent-turn caches, prompt compression, and speculative capture.
Their image payload survives request copies and retry paths. Failed or cancelled
multi-image encoding publishes no partial embedding matrices.

## Memory and verification

The projector is validated and loaded before expert placement. Admission counts
actual selected owner tensor sizes, allocation alignment, MIX tables, copy
staging, future KV, and explicit execution reserves. Host and integrated-device
charges share one physical-memory budget. Before image decoding, the server
checks host availability; before encoding, it synchronizes and releases
disposable decoder, owner, and draft graphs and checks live device/host
availability again. KV, saved snapshots, and draft weights remain reflected in
that live measurement. Reservations are conservative policy, not a guarantee
against unrelated concurrent allocations.

The remote checks include the server unit suite, decoder loader and image-batch
admission tests, synthetic allocation/UMA accounting, preprocessing/codec tests,
and standalone mixed embedding and cancellation tests. Native HIP encoder
comparisons for corn and carrots pass the unchanged feature/embedding gates;
corn also matches the source HIP output exactly and repeats byte for byte.
Full image HTTP behavior, paired runtime resource peaks, and performance require
their separate private serving proof.
