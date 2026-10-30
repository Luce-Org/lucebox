# Native Vulkan launcher (experimental)

This opt-in Linux executable connects the existing Lucebox HTTP server and
LuceEngine to native ggml Vulkan ModelBackend implementations. It does not link
libllama, forward HTTP requests, launch an inference subprocess or schedule
unsupported graph operations on a CPU backend. Host tokenization, sampling and
optional diagnostic logprob normalization are expected CPU work.

The ordinary `cmake -S server` CUDA/HIP build and its vendored ggml are unchanged.
Vulkan has a separate build tree and pinned dependency (see PROVENANCE.md).

## Build

Prerequisites: CMake >=3.21, C++17 compiler, Ninja, Git, nlohmann-json CMake
package, Vulkan headers/loader, glslc and a supported Vulkan GPU/driver.

    cmake -S server/vulkan -B build-vulkan -G Ninja -DCMAKE_BUILD_TYPE=Release
    cmake --build build-vulkan -j1
    ctest --test-dir build-vulkan --output-on-failure

CMake downloads a SHA256-verified public llama.cpp archive and builds only its
ggml subdirectory. No llama model loader or inference implementation is linked.
Default CTest is GPU-free. To add the device K-quant test, opt in at CMake
configure time (not with an environment variable), only on an available,
monitored GPU host:

    cmake -S server/vulkan -B build-vulkan -DLUCE_VULKAN_GPU_TESTS=ON
    cmake --build build-vulkan -j1
    ctest --test-dir build-vulkan --output-on-failure

### Offline CPU regressions (no GPU SDK or device)

With the pinned llama.cpp source already present, build only the startup,
request, greedy, prefill and constructor-ownership regressions:

    cmake -S server/vulkan/tests -B build-vulkan-cpu -DLLAMA_SOURCE_DIR=/absolute/path/to/llama.cpp-c061df19838ff60970faf54fd7e414953590125d -DCMAKE_BUILD_TYPE=Release
    cmake --build build-vulkan-cpu -j2
    ctest --test-dir build-vulkan-cpu --output-on-failure

This Linux GCC/Clang test build disables device backends. Ownership tests run
both real constructors with injected CPU-only device allocation failures; they
are not a substitute for GPU initialization or inference validation. The shared
policy/startup tests use the same helpers as the HTTP server and launcher.

## Run

    GGML_VK_W9100_KQ_ROWS=1 build-vulkan/lucebox-vulkan MODEL.gguf --port 18080 --ctx 4096 --chunk 256 --flash 0

The launcher binds 127.0.0.1 only. GGUF `general.architecture` selects Qwen2 or
LFM2 MoE; other architectures fail clearly, without replacing the normal server.
Use `/v1/chat/completions`, e.g. `{"messages":[{"role":"user","content":"Compute 17 * 23. Reply with only the integer."}],"temperature":0,"max_tokens":96}`.
Stop with SIGINT/SIGTERM. No service or deployment is installed.

Validated LFM target: LFM2.5-8B-A1B Q6_K, SHA256
`7ccf57a2d410d8822d1560a1ca10c8318f3e15d6a8f6d42d1903d58e80ea20a6`.
The loader checks this model geometry and requires an unsplit GGUF. It does not
claim support for every lfm2moe model/quant. Qwen2 retains split-GGUF loading.

## Limits

The new LFM Vulkan route is serialized, greedy, non-streaming text chat, n=1,
output budget 1..1024, with FP16 attention KV and FP32 convolution state.
Flash attention, tools, constrained decoding, API logprobs, snapshots, prefix
caching, speculative decoding, compression and parking are not supported.
Concurrent/multisequence performance is not certified. The regular CUDA/HIP
LFM route and Qwen route do not inherit these LFM Vulkan HTTP restrictions.
LFM context capacity is limited to 128..4096; Qwen retains 128..32768.
Context overflow fails before native dispatch; every new request clears KV and
convolution state. LFM startup requires a nonempty GGUF chat template.
Request/response tokenization uses that template and BOS. Optional no-op fields
`tools: []`, `logprobs: false`, and `response_format: null` are accepted; enabled
features and other representations remain unsupported. Repetition-penalty
fields (`repetition_penalty`, `repeat_penalty`, `rep_pen`) are all rejected.

`LUCEBOX_TRACE=/absolute/path.jsonl` optionally records prompt/output token IDs,
top-five logprobs, graph execution counts and device residency. It can contain
private prompt data; leave it unset in normal use. `LUCEBOX_LAYER_DUMP` is a
diagnostic-only layer output dump and is also off by default.

## Correctness invariants

Keep the convolution projection -> reshape -> residual boundary and do not add
unnecessary contiguous copies. Vulkan fusion changes can cause logprob drift.
Prefill preserves the final chunk, then the final four-token boundary. A unit
test covers these partitions; model acceptance requires same-token, same-history
reference comparisons, including contexts crossing these boundaries. Arithmetic
success alone is not sufficient. Numerical tolerance is absolute 0.001.

The W9100-only row-count override accepts 1,2,4,8. Unset preserves upstream ggml
behavior; all other device IDs ignore the override. This is not a general speed
claim or an invitation to change driver, clocks, power or cooling settings.

## Cooling during validation

W9100 testing used a custom GPU-only fan curve managed externally through
CoolerControl. Chassis fans remained under factory automatic control. GPU clocks,
voltage, power limits, and VBIOS were unchanged. This contribution does not install
or modify fan controls.

See [validation notes](VALIDATION.md) for the exact fan curve, test conditions,
results, and coverage limitations. Cooling settings are part of the recorded test
environment, not a requirement to copy these settings onto other hardware.

The accompanying external validation report retains full raw evidence and
explicit skipped platform coverage.
