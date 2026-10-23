#!/usr/bin/env bash
# Downloads MOSS-Image SFT at pinned revisions and writes the GGUF luce_server serves images from:
#   moss_image_sft_bf16.safetensors  the diffusion transformer, bf16 (converted, then no longer needed)
#   flux2-vae.safetensors            the image decoder (converted, then no longer needed)
#   moss-image.gguf                  both, the transformer's big linears as int8 convrot
#                                    (server/scripts/convert_moss_image.py)
#   Qwen3.5-9B-Q8_0.gguf             the text encoder
#
#   ./fetch_models.sh [DIR]     (default ./models: 26 GB, 15 GB once the two safetensors files are deleted)
#
# Needs curl and python3 with numpy and PyYAML.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
dir=${1:-$here/models}
mkdir -p "$dir"

get() {  # repo revision path name
    if [ -s "$dir/$4" ]; then
        echo "have $4"
        return
    fi
    echo "fetching $4"
    curl -L --fail --retry 3 -C - -o "$dir/$4.part" "https://huggingface.co/$1/resolve/$2/$3"
    mv "$dir/$4.part" "$dir/$4"
}
get Comfy-Org/MOSS-Image 3284a1dbb351e3396d70f39f2a71cbf6ed2ef1bb \
    diffusion_models/moss_image_sft_bf16.safetensors moss_image_sft_bf16.safetensors
get Comfy-Org/MOSS-Image 3284a1dbb351e3396d70f39f2a71cbf6ed2ef1bb vae/flux2-vae.safetensors flux2-vae.safetensors
# Our byte-for-byte copy of unsloth/Qwen3.5-9B-GGUF at 3885219b, so the file stays downloadable.
get Lucebox/Qwen3.5-9B-GGUF 4577338b9093794ef7006ff1e41bffc095af4246 Qwen3.5-9B-Q8_0.gguf Qwen3.5-9B-Q8_0.gguf

if [ ! -s "$dir/moss-image.gguf" ]; then
    python3 "$here/../../server/scripts/convert_moss_image.py" --dit "$dir/moss_image_sft_bf16.safetensors" \
        --vae "$dir/flux2-vae.safetensors" --out "$dir/moss-image.gguf.part"
    # The conversion is exact (float64 rotation, numpy only): every machine writes these bytes.
    sum=$(sha256sum "$dir/moss-image.gguf.part" | cut -d' ' -f1)
    if [ "$sum" != 3fe066fb7c99dd14b26cb7aee5d3c4703d3031ec5a68bef8556225d53dc13ccb ]; then
        echo "moss-image.gguf.part has sha256 $sum, not the expected 3fe066fb...; keeping it for a look" >&2
        exit 1
    fi
    mv "$dir/moss-image.gguf.part" "$dir/moss-image.gguf"
fi
echo "models in $dir"
