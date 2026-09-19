# linuxmacan deployment snapshot

`models.json` exposes the production names `qwen3.8-27b-swift` and
`qwen3.8-27b-abliterated`. The compatibility names `qwen3.8-27b` and
`qwen3.8-27b-pflash` resolve to Swift. The model router starts the external
vLLM/PFlash services for Swift and releases them before loading a local GGUF,
so the two 27B targets never contend for the same GPU memory. Install the
template at `/opt/lucebox/templates/swift-qwen.jinja` and config at
`/opt/lucebox/models.json`; referenced model files must already exist.

Install `lucebox-model-router.service` as the public service on port 8216.
The PFlash proxy listens internally on 127.0.0.1:18217. Install
`vllm-swift-int4-run.sh` at `/root/vllm-native-test/run-concurrent-server.sh`.
The original `lucebox.service` is not used by this deployment.

The deployed executable SHA-256 is recorded with benchmark provenance. Source commits 2246420 and 3000d2a are preserved on this branch; the complete 2246420 memory/progress patch was not deployed by this registration. Do not treat this configuration snapshot as evidence that all branch source is running in production.

Pi configuration: merge `integrations/pi/lucebox-models.example.json` into the existing providers object, retaining local connection/auth settings. It uses model-level `thinkingLevelMap`; xhigh is exposed for all four Lucebox Qwen entries. The example key is a placeholder.
