# W9100 validation environment

These notes describe the environment used to validate the native Vulkan LFM
backend. They are not a cooling utility or a universal fan-setting recommendation.
No fan-control code, service configuration, firmware, or model weights are shipped
by this contribution.

## Hardware and cooling

- Dell Precision 5820 with Intel Xeon W-2125 and AMD FirePro W9100.
- Linux 7.0.0-34-generic; ggml Vulkan on the W9100.
- GPU clocks, voltage, power limits, and VBIOS were unchanged.
- Dell chassis fans remained under factory automatic control.
- The W9100 GPU blower used an externally managed CoolerControl profile already
  established before the final candidate/reference tests. The same profile was
  maintained throughout those tests; it was not tuned between timed phases.

Recorded GPU temperature / blower duty control points:

- 30°C: 20%
- 50°C: 20%
- 60°C: 35%
- 70°C: 50%
- 80°C: 65%
- 85°C: 80%
- 95°C: 100%

The CoolerControl graph used the GPU temperature sensor. Its Standard function
was configured with `response_delay = 5`, `deviance = 2.0`,
`only_downward = true`, `duty_minimum = 2`, `duty_maximum = 100`, decreasing
step limits of 2 and 10, `threshold_hopping = true`, and
`bypass_min_at_extremes = true`. These are the recorded controller settings;
the duty percentages above are targets, not a fixed fan RPM.

The external controller's stop/crash fallback to automatic GPU fan control was
validated separately. This patch neither installs that controller/fallback nor
changes the chassis fan settings. Check the cooling appropriate to your own card
and enclosure when reproducing sustained measurements; the custom profile is not
proof that stock cooling will produce the same temperature or timing.

## Model and test conditions

- Exact LFM2.5-8B-A1B Q6_K; context capacity 4096.
- Five matched prompts, 96 generated tokens per prompt, greedy temperature 0,
  seed 0, prefill chunk 256, flash attention disabled.
- The W9100-specific `GGML_VK_W9100_KQ_ROWS=1` override was used.
- Reference: llama.cpp `c061df19838ff60970faf54fd7e414953590125d`, with the
  recorded W9100 Vulkan row-count delta. Candidate uses native LuceEngine and
  ModelBackend with ggml Vulkan, not libllama or a proxy to the reference.
- Both paths collected top-five diagnostic log probabilities. Numerical
  acceptance used the unchanged absolute tolerance of 0.001.
- Matched timing phases used 90 seconds of idle conditioning. The first
  reference phase overlapped a CPU-only compilation on the Dell; the second did
  not. This host-load confound limits interpretation of small timing differences.

## Results and limits

- 2400 matched score comparisons, 4320 supplemental context/boundary comparisons,
  and 157920 sustained score comparisons: zero failures. Input tokens, generated
  histories, and output text matched their respective reference traces.
- Context tests included prompt lengths 255/256/257 and a 4000-token prompt plus
  96 generated tokens, filling the validated 4096-token capacity exactly.
- Functional arithmetic and grounded-color answers, EOS, request-state reset,
  malformed/unsupported requests, and overflow rejection passed on the real
  executable. The 96-token timing traces are fixed-length generation samples,
  not a claim that every benchmark prompt completed its answer within 96 tokens.
- Sustained run: 329 requests over 605.854 seconds of active request time;
  602.907 seconds inside native prefill and decode.
- Native decode: 64.287 tokens/s matched and 64.256 tokens/s sustained.
  Reference decode: 67.299 and 66.060 tokens/s. No speedup is claimed.
- Sustained telemetry: 629 samples; peak GPU temperature 71°C; minimum blower
  speed 2227 RPM. No GPU faults were recorded in the test-window audit.
- Six Qwen2 regression requests and the monitored Vulkan K-quant test passed.
- Default CUDA compilation passed. CPU-compatible CTest recorded 725 process-level
  successes and 39 explicit skips; 14 unavailable-device failures from the full
  discovery run were separately excluded and retained in the external receipts.
  Some upstream tests exit successfully when hardware is absent. No CUDA/HIP GPU
  inference validation or HIP build coverage is claimed.

The new LFM Vulkan path is limited to the tested geometry and quantization,
serialized greedy non-streaming text chat, and the validated context capacity.
Streaming, tools, API logprobs, flash attention, and concurrent/multisequence
certification are outside this contribution. See [README.md](README.md) for build,
launch, diagnostics, and capability limits.
