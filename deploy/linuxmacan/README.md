# linuxmacan deployment snapshot

`models.json` records the verified configuration after adding `qwen3.8-27b-swift` with DFlash2 and auto caching/compression. Default remains `qwen3.8-27b`. Install the template at `/opt/lucebox/templates/swift-qwen.jinja` and config at `/opt/lucebox/models.json`; referenced model files must already exist. Swift intentionally does not inherit the original model's latency calibration.

The deployed executable SHA-256 is recorded with benchmark provenance. Source commits 2246420 and 3000d2a are preserved on this branch; the complete 2246420 memory/progress patch was not deployed by this registration. Do not treat this configuration snapshot as evidence that all branch source is running in production.

Pi configuration: merge `integrations/pi/lucebox-models.example.json` into the existing providers object, retaining local connection/auth settings. It uses model-level `thinkingLevelMap`; xhigh is exposed for all four Lucebox Qwen entries. The example key is a placeholder.
