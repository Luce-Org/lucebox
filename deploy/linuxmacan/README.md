# linuxmacan Lucebox deployment

`models.json` serves every public model name through the native Lucebox
GGUF/ROCm backend. The default is `qwen3.8-27b-swift`; the base,
PFlash-compatible, abliterated and Gemma names remain available through the
same automatic single-GPU model router.

Install:

- `models.json` at `/opt/lucebox/models.json`
- `swift-qwen.jinja` at `/opt/lucebox/templates/swift-qwen.jinja`
- `model_router.py` at `/opt/lucebox/model_router.py`
- `deploy/lucebox.service` as the enabled public service on port 8216

The vLLM and vLLM PFlash services must remain stopped and disabled while this
deployment is active. Model changes unload the current Lucebox target before
loading the selected GGUF, so the 27B models never contend for GPU memory.

Pi configuration: merge `integrations/pi/lucebox-models.example.json` into the
existing providers object, retaining local connection/auth settings. It uses
model-level `thinkingLevelMap`; xhigh is exposed for all Lucebox Qwen entries.
