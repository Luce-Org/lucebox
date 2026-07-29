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

The wrapper detects the host and selects CUDA for NVIDIA builds (including RTX
3090 + Strix) or ROCm for AMD builds (including R9700 + Strix). The package then
handles readiness checks, TOML configuration, model selection and download,
optimization settings, and construction of the final server command. Host
facts are passed through `LUCEBOX_HOST_*` environment variables so the
container never has to guess the host configuration.

Automatic mode considers the selected model and the primary GPU. It shows a
plain-language decision for DFlash, PFlash, KVFlash, and Spark, prefers exact
all-GPU execution when it fits, and enables memory-saving paths only under
pressure. Advanced mode lets experienced users override those four product
choices while the CLI retains safe context, cache, and DDTree defaults. A
factory-preloaded Lucebox can ship the models and shared optimizer scorer, so a
buyer does not need to download anything during first setup.

Spark is enabled automatically only when an MoE model is under GPU-memory
pressure and the machine reports at least 32 GB of host RAM; its cold experts
live in system memory. Lower-memory or unknown hosts keep Spark off unless an
advanced user explicitly opts in.

Heterogeneous machines use one detected primary accelerator by default. The
CLI pins ROCm to the largest-VRAM AMD device and never sums memory across GPUs;
automatic multi-GPU layer sharding is deliberately not claimed by this release.

Inside a source checkout, the same wrapper also exposes `lucebox build`,
`lucebox native`, and `lucebox harness` for contributors. Buyer installations
keep using the prebuilt container, so no compiler or source checkout is needed.

See the [project README](https://github.com/Luce-Org/lucebox#readme) for the
installation and user flow. Contributors can find the CLI implementation in
[`src/lucebox`](https://github.com/Luce-Org/lucebox/tree/main/lucebox/src/lucebox).
