# DeepSeek-V4 vision projector export

`server/tools/export_ds4v_mmproj.py` extracts the vision tower and aligner from
the verified DeepSeek-V4 parent model without loading or changing tensor values.
It requires only Python's standard library.

```bash
python3 server/tools/export_ds4v_mmproj.py \
  /models/DeepSeek-V4-Flash-Vision-Uncensored \
  /models/ds4v-mmproj.gguf
```

The output path must not exist. The exporter validates `config.json`, the
safetensors index, all 267 required names and shapes, BF16 payload lengths,
source bounds, and shard paths before it writes. It publishes the finished file
atomically and removes its temporary file after any pre-publication failure.

The GGUF keeps every `vision.*`, `aligner.*`, and `image_*` name, shape, dtype,
and payload byte unchanged. `general.architecture` is `deepseek4_vision`.
Required `deepseek4.vision.*` metadata records the 32-block, width-1024,
16-head tower; head width 64; patch size 14; intermediate width 2816; 2D RoPE
layout and theta 10000; ratio-3, width-9216 aligner; language width 4096;
vocabulary 129280; RMS epsilon `1e-6`; image token/pixel/aspect bounds; RGB
mean/std `0.5`; channel-major patches; N-layout recipe version 1 with
compression alignment 4; bottom/right aligner padding; channel-first unfold;
and exact GELU.

Run the focused synthetic suite with:

```bash
python3 -m unittest -v server.tests.test_export_ds4v_mmproj
```

When NumPy is available, the suite also opens the result with llama.cpp's
vendored `GGUFReader`, independently of the exporter's writer.
