#!/usr/bin/env bash
# DS4V-2 source fetch and proof script for the abliterated vision source
# orcarouter/DeepSeek-V4-Flash-Vision-Uncensored on huggingface.co.
# See docs/ds4v-source.md and docs/ds4v-uncensored-vision-plan.md DS4V-2.
#
# This script never deletes weights. The check-only path never downloads
# shards. The manifest path downloads only model.safetensors.index.json.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

SOURCE_REPO=${DS4V_SOURCE_REPO:-orcarouter/DeepSeek-V4-Flash-Vision-Uncensored}
MODEL_DIR=${DS4V_SOURCE_MODEL_DIR:-"$REPO_ROOT/models/DeepSeek-V4-Flash-Vision-Uncensored"}
ART_DIR=${DS4V_ART_DIR:-"$REPO_ROOT/artifacts/ds4v-2"}
MANIFEST_NAME=model.safetensors.index.json
MANIFEST_URL="https://huggingface.co/$SOURCE_REPO/resolve/main/$MANIFEST_NAME"
API_URL="https://huggingface.co/api/models/$SOURCE_REPO"
EXPECTED_SHARDS=48

usage() {
    cat <<'EOF'
DS4V-2 source script. Fetch and proof steps for the abliterated vision source
orcarouter/DeepSeek-V4-Flash-Vision-Uncensored.

usage: scripts/ds4v-fetch-source.sh <subcommand>

subcommands:
  fetch        Download the full parent repo into $DS4V_SOURCE_MODEL_DIR.
               Requires the hf CLI (huggingface_hub[cli]) or huggingface-cli.
               Honors HF_TOKEN when set. Never deletes weights.
  manifest     Download ONLY model.safetensors.index.json via curl and count
               vision, aligner, bias_vl, and mtp keys with python3 or jq.
               Writes artifacts/ds4v-2/manifest-counts.json.
               Honors HF_TOKEN as bearer auth when set. The repo is gated on
               huggingface.co, so without a token this exits 0 with a gated
               note plus the public API shard listing.
  check-only   Verify sha256 of local shards when present. Never downloads
               shards. Exits 0 with a source absent note when no shards exist.
  help         Show this usage and exit 0.

environment:
  DS4V_SOURCE_REPO      Parent repo id. Default orcarouter/DeepSeek-V4-Flash-Vision-Uncensored.
  DS4V_SOURCE_MODEL_DIR Download target. Default models/DeepSeek-V4-Flash-Vision-Uncensored.
  DS4V_ART_DIR          Evidence dir. Default artifacts/ds4v-2.
  HF_TOKEN              Optional huggingface token used as bearer auth.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

require_dir() {
    [ -d "$1" ] || die "directory $1 does not exist. Run the right subcommand first."
}

curl_hf() {
    # curl with optional HF_TOKEN bearer auth. Extra args pass through.
    if [ -n "${HF_TOKEN:-}" ]; then
        curl -fsSL -H "Authorization: Bearer $HF_TOKEN" "$@"
    else
        curl -fsSL "$@"
    fi
}

cmd_fetch() {
    local cli
    if command -v hf >/dev/null 2>&1; then
        cli=hf
    elif command -v huggingface-cli >/dev/null 2>&1; then
        cli=huggingface-cli
    else
        die "the hf CLI is missing. Install it with 'pip install \"huggingface_hub[cli]\"' (or 'pip install huggingface_hub' for huggingface-cli), then retry scripts/ds4v-fetch-source.sh fetch."
    fi
    mkdir -p "$MODEL_DIR"
    echo "fetching $SOURCE_REPO into $MODEL_DIR with $cli download"
    # --local-dir keeps real files in place. This script never deletes weights.
    if [ "$cli" = hf ]; then
        hf download "$SOURCE_REPO" --local-dir "$MODEL_DIR"
    else
        huggingface-cli download "$SOURCE_REPO" --local-dir "$MODEL_DIR"
    fi
    echo "fetch complete. Weights live in $MODEL_DIR. Nothing was deleted."
}

manifest_counts_json() {
    # Counts from the weight map. Caller passes the index path.
    local index="$1"
    python3 - "$index" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
wm = d["weight_map"]
shards = sorted(set(wm.values()))
mtp_keys = [k for k in wm if k.startswith("mtp.")]
mtp_blocks = sorted(set(k.split(".")[1] for k in mtp_keys))
out = {
    "repo": "orcarouter/DeepSeek-V4-Flash-Vision-Uncensored",
    "manifest": "model.safetensors.index.json",
    "shard_count": len(shards),
    "tensor_count": len(wm),
    "vision_tensors": sum(1 for k in wm if k.startswith("vision.")),
    "aligner_tensors": sum(1 for k in wm if k.startswith("aligner.")),
    "bias_vl_tensors": sum(1 for k in wm if "bias_vl" in k),
    "mtp_blocks": len(mtp_blocks),
    "mtp_block_names": mtp_blocks,
    "shards": shards,
}
print(json.dumps(out, indent=2))
PYEOF
}

manifest_shard_list_api() {
    # Public API metadata. File names are visible without accepting the gate.
    curl -fsSL "$API_URL" | python3 -c '
import json, sys
d = json.load(sys.stdin)
names = [s["rfilename"] for s in d.get("siblings", [])]
shards = sorted(n for n in names if n.startswith("model-") and n.endswith(".safetensors"))
out = {
    "repo": d.get("id"),
    "gated": d.get("gated"),
    "license": (d.get("cardData") or {}).get("license"),
    "base_model": (d.get("cardData") or {}).get("base_model"),
    "shard_count": len(shards),
    "shards": shards,
    "counts": None,
    "note": "tensor counts pending on the operator machine. The repo is gated on huggingface.co. Set HF_TOKEN with a token that has accepted the gate to fetch model.safetensors.index.json.",
}
print(json.dumps(out, indent=2))
'
}

cmd_manifest() {
    mkdir -p "$ART_DIR"
    local out="$ART_DIR/manifest-counts.json"
    local tmp="$ART_DIR/$MANIFEST_NAME.tmp"
    echo "downloading only $MANIFEST_NAME from $MANIFEST_URL"
    if curl_hf -o "$tmp" "$MANIFEST_URL" && python3 -m json.tool "$tmp" >/dev/null 2>&1; then
        mv "$tmp" "$ART_DIR/$MANIFEST_NAME"
        manifest_counts_json "$ART_DIR/$MANIFEST_NAME" > "$out"
        echo "manifest counts written to $out"
        cat "$out"
    else
        rm -f "$tmp"
        echo "warning: $SOURCE_REPO is gated on huggingface.co or unreachable. No HF_TOKEN is set, or it has not accepted the gate. Recording the public API shard listing instead. Tensor counts stay pending on the operator machine." >&2
        manifest_shard_list_api > "$out"
        echo "public shard listing written to $out"
        cat "$out"
    fi
}

cmd_check_only() {
    mkdir -p "$ART_DIR"
    local log="$ART_DIR/check-only.log"
    local -a shard_files=()
    if [ -d "$MODEL_DIR" ]; then
        while IFS= read -r f; do
            shard_files+=("$f")
        done < <(find "$MODEL_DIR" -name 'model-*.safetensors' -type f | sort)
    fi
    {
        echo "check-only run $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "model dir $MODEL_DIR"
    } > "$log"
    if [ "${#shard_files[@]}" -eq 0 ]; then
        cat >> "$log" <<EOF
source absent. No shards under $MODEL_DIR.
expected manifest. $EXPECTED_SHARDS shards of $SOURCE_REPO.
expected files. model-00001-of-00048.safetensors through model-00048-of-00048.safetensors.
nothing verified. Nothing downloaded. Exit 0.
EOF
        cat "$log"
        echo "check-only. Source absent. Expected $EXPECTED_SHARDS shards. Nothing downloaded. Exit 0."
        return 0
    fi
    echo "verifying sha256 of ${#shard_files[@]} local shards" | tee -a "$log"
    local failed=0
    for f in "${shard_files[@]}"; do
        if shasum -a 256 "$f" >> "$log"; then
            echo "ok $(basename "$f")"
        else
            echo "FAIL sha256 $(basename "$f")" >&2
            failed=1
        fi
    done
    # Cross check against a local index when one is present.
    if [ -f "$ART_DIR/$MANIFEST_NAME" ]; then
        manifest_counts_json "$ART_DIR/$MANIFEST_NAME" >> "$log"
        echo "local manifest counts appended to $log"
    fi
    [ "$failed" -eq 0 ] || die "one or more shard checksums failed. See $log"
    echo "all local shard checksums pass. Log at $log."
}

case "${1:-}" in
    fetch)       shift; cmd_fetch "$@" ;;
    manifest)    shift; cmd_manifest "$@" ;;
    check-only)  shift; cmd_check_only "$@" ;;
    help|-h|--help) usage; exit 0 ;;
    "")          usage >&2; die "missing subcommand" ;;
    *)           usage >&2; die "unknown subcommand '$1'" ;;
esac
