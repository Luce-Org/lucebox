# lucebox — CLI for the Lucebox inference appliance

This Python package ships inside the `ghcr.io/luce-org/lucebox-hub` image. Most
users do not install it directly: they install the small
[`lucebox` host wrapper](https://github.com/Luce-Org/lucebox/blob/main/lucebox.sh),
which invokes the package in the appropriate container:

    lucebox check
    lucebox models list
    lucebox models download qwen3.6-27b --activate
    lucebox start

The wrapper detects the host and selects CUDA for NVIDIA builds (including RTX
3090 + Strix) or ROCm for AMD builds (including R9700 + Strix). The package then
handles readiness checks, TOML configuration, model selection and download,
optimization settings, and construction of the final server command. Host
facts are passed through `LUCEBOX_HOST_*` environment variables so the
container never has to guess the host configuration.

See the [project README](https://github.com/Luce-Org/lucebox#readme) for the
installation and user flow. Contributors can find the CLI implementation in
[`src/lucebox`](https://github.com/Luce-Org/lucebox/tree/main/lucebox/src/lucebox).
