#!/usr/bin/env python3
"""Pick which model e2e jobs a change needs and print the GitHub Actions matrix.

Changed paths come on stdin, one per line. A path under a model's own sources
selects that model; a path in shared server code selects every model; anything
else (docs, scripts, other models) selects nothing. With --fallback-all, used
for a PR a maintainer labelled `e2e`, an empty selection runs every model: the
label is an explicit request, so a PR that only touches another model's code
still gets both runs. Other-model paths still keep a mixed PR (e.g. laguna +
qwen35) to the models it touches, and keep merges to main from re-running them.

Each matrix entry carries everything its job needs: the GPU on lucebox3, the
model files (in the models directory) and the server flags.

    gh api repos/OWNER/REPO/pulls/N/files --paginate --jq '.[].filename' \\
        | python3 .github/ci/e2e/select_models.py --mode paths
"""

from __future__ import annotations

import argparse
import json
import sys

# HIP indices as on lucebox3 (see gpu-tests-amd in ci.yml).
MODELS = {
    "qwen": {
        "device": "r9700",
        "device_name": "Radeon AI PRO R9700",
        "arch": "gfx1201",
        "hip_index": 0,
        "target": "Qwen3.8-27B-UD-IQ4_XS.gguf",
        "draft": "qwen38-dflash2-q8_0.gguf",
        "server_args": "--max-ctx 8192 --disk-prefix-cache off",
    },
    "ds4": {
        "device": "strix-halo",
        "device_name": "Strix Halo Radeon 8060S",
        "arch": "gfx1151",
        "hip_index": 1,
        "target": "DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf",
        "draft": "",
        # Sparse (batched) prefill: exact prefill runs at ~21 tok/s on the Strix
        # Halo, too slow for the 2.5K-token needle prompt within the job budget.
        "server_args": "--max-ctx 8192 --ds4-fused-decode --ds4-expert-top-k 6"
        " --ds4-prefill sparse --prefix-cache-slots 0 --prefill-cache-slots 0"
        " --disk-prefix-cache off",
    },
}

MODEL_PREFIXES = {
    "ds4": ("server/src/deepseek4/",),
    "qwen": (
        "server/src/qwen35/",
        "server/src/qwen3/",
        "server/src/draft/",
        "server/src/delta_net",
        "server/src/flashprefill",
        "server/src/pflash_",
    ),
}
# Other model families: changes there cannot affect Qwen or DS4.
OTHER_MODEL_PREFIXES = (
    "server/src/bailingmoe3/",
    "server/src/gemma4/",
    "server/src/laguna/",
    "server/src/qwen35moe/",
)
SHARED_PREFIXES = (
    "server/src/",
    "server/include/",
    "server/deps/",
    "server/cmake/",
    "server/hip_compat/",
    "server/CMakeLists.txt",
    ".github/ci/e2e/",
    ".github/ci/kfd_health.sh",
    ".github/workflows/model-e2e.yml",
)


def models_for(paths: list[str]) -> list[str]:
    selected: set[str] = set()
    for path in paths:
        owners = [m for m, prefixes in MODEL_PREFIXES.items() if path.startswith(prefixes)]
        if owners:
            selected.update(owners)
        elif path.startswith(OTHER_MODEL_PREFIXES):
            continue
        elif path.startswith(SHARED_PREFIXES):
            selected.update(MODELS)
    return sorted(selected)


def matrix(models: list[str]) -> dict:
    return {"include": [{"model": model, **MODELS[model]} for model in models]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--mode",
        default="paths",
        help="'paths' reads changed paths from stdin, 'all' runs every model, "
        "or a comma-separated list of models",
    )
    parser.add_argument(
        "--fallback-all",
        action="store_true",
        help="run every model when the paths select none (an explicit request)",
    )
    args = parser.parse_args(argv)

    if args.mode == "paths":
        models = models_for([line.strip() for line in sys.stdin if line.strip()])
        if not models and args.fallback_all:
            models = sorted(MODELS)
    elif args.mode == "all":
        models = sorted(MODELS)
    else:
        models = sorted({m.strip() for m in args.mode.split(",") if m.strip()})
        unknown = [m for m in models if m not in MODELS]
        if unknown:
            parser.error(f"unknown model(s): {', '.join(unknown)}")
    print(json.dumps(matrix(models), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
