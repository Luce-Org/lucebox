# Native mmproj vision

Optional build (`-DDFLASH27B_MMPROJ=ON`) wires llama.cpp **mtmd** so
`dflash_server` can load a GGUF multimodal projector alongside the text model
and accept OpenAI-style `image_url` content in chat completions.

## Runtime

```bash
# Build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DDFLASH27B_MMPROJ=ON -DDFLASH27B_SERVER=ON
cmake --build build --target dflash_server -j"$(nproc)"

# Serve (CLI or entrypoint env)
dflash_server ... --mmproj /path/to/mmproj-F16.gguf
# Container: DFLASH_MMPROJ=/path/to/mmproj-F16.gguf
# Optional: --no-mmproj-offload / DFLASH_MMPROJ_NO_OFFLOAD=1
```

`/props` reports `capabilities.vision_supported: true` when the projector is
loaded. Multimodal turns run AR decode; text-only turns keep DFlash speculative
decode when configured.

Supported on both monolithic Qwen35 and layer-split backends
(`supports_multimodal()` is delegated through `LayerSplitBackend`).

## Example

Chat completion with an attached meme image — the model reads the visual
layout and answers in natural language:

![Native mmproj vision: meme interpretation](images/vision-mmproj-meme-example.png)

Request shape (abbreviated):

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What do you think this image means?"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
  }]
}
```
