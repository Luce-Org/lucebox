# lucebox — CLI for the Lucebox inference appliance

This Python package ships inside the `ghcr.io/luce-org/lucebox-hub` image. Most
users do not install it directly: they install the small
[`lucebox` host wrapper](https://github.com/Luce-Org/lucebox/blob/main/lucebox.sh),
which invokes the package in the appropriate container:

    lucebox                         # branded interactive menu
    lucebox setup                   # guided first run
    lucebox check
    lucebox models select           # numbered picker + download + activate
    lucebox optimize                # review/apply the Automatic profile
    lucebox optimize --advanced     # override individual optimizations
    lucebox start

The wrapper inventories the host and selects CUDA for NVIDIA builds (including
RTX 3090 + Strix) or ROCm for AMD builds (including R9700 + Strix). The package
then handles readiness checks, TOML configuration, model selection and
download, optimization and placement settings, and construction of the final
server command. Host facts are passed through `LUCEBOX_HOST_*` environment
variables so the package never has to guess the host configuration.

The guided picker leads with Lucebox's four featured model paths: Qwen3.6
27B, Qwen3.6 35B-A3B, Laguna XS.2, and DeepSeek V4 Flash. It checks the
resolved placement before downloading, so a machine without enough compatible
accelerator memory cannot accidentally begin DeepSeek's roughly 114 GB
target-plus-draft download. Older supported Gemma presets remain available by
name and through `lucebox models list`.

Automatic mode combines a typed model capability contract with the complete
accelerator topology. It prints the selected prefill, decode, and KV-cache
strategies, then explains each DFlash, PFlash, KVFlash, Spark, and placement
decision. A fitting target stays on the faster primary GPU; qualified
secondary-device and memory-saving paths activate only when needed. Advanced
mode exposes only policies that are legal for that model and backend while
placement, context, cache, and DDTree invariants remain guarded. Preview paths
are labeled and never enabled silently. In particular, DeepSeek sparse prefill
is an explicit approximate HIP preview and exact MLA prefill remains Automatic.
A factory-preloaded Lucebox can ship the models, shared optimizer scorer, and
paired runtime, so a buyer does not download or compile during first setup.

Spark is enabled automatically only when an MoE model is under GPU-memory
pressure and the machine reports at least 32 GB of host RAM; its cold experts
live in system memory. Lower-memory or unknown hosts keep Spark off unless an
advanced user explicitly opts in.

On a same-backend multi-GPU host, supported models can use a layer split when
the target does not fit one device. On R9700 + Strix, ROCm prefers the discrete
R9700 and can use Strix as a companion. RTX + Strix cross-vendor execution uses
a validated CUDA-main/HIP-companion native package; it is selected only when
that paired runtime is actually installed. Cross-GPU transfers default to safe
host staging, with peer access left as an explicit expert opt-in.

Inside a source checkout, the same wrapper also exposes `lucebox build`,
`lucebox native`, and `lucebox harness` for contributors. `lucebox build hybrid`
builds both native backends and `lucebox package-runtime` stages the factory
layout. Single-backend buyer installations keep using the prebuilt container;
heterogeneous buyers can receive the paired runtime preinstalled. Neither path
requires a source checkout during normal use.

See the [project README](https://github.com/Luce-Org/lucebox#readme) for the
installation and user flow. Contributors can find the CLI implementation in
[`src/lucebox`](https://github.com/Luce-Org/lucebox/tree/main/lucebox/src/lucebox).
