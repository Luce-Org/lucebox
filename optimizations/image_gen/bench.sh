#!/usr/bin/env bash
# Warm per-image time of one or more luce_server builds at 1024x1024 and seed 42, 8 steps of er_sde on the simple
# schedule (the sampler ComfyUI runs for MOSS-Image) and CFG 1 unless set. Each round starts the builds' servers in
# turn, so clocks and temperature hit them alike. A server's first start tunes its GEMMs with two warm-up images and
# later starts read them back from its tuning file (see the README); the three test prompts are each new to it (it
# reuses the text encoding of a repeated prompt). Prints the median times (from the server's launch until it answers;
# the whole request, prompt in and PNG out; the server's own time per image, then its text, sampling and decode
# phases) and each image's md5: the same hashes from two builds mean the same pixels.
#
#   ./bench.sh LABEL=COMMAND ...
#   ./bench.sh ours=../../server/build/luce_server
#   ./bench.sh new=../../server/build/luce_server old=/path/to/other/luce_server
#   CHECK=1 CFG=2.5 ./bench.sh ours=../../server/build/luce_server
#
# Environment: MODELS (default ./models), STEPS (default 8), CFG (default 1), SAMPLER (default er_sde), SCHEDULER
# (default simple), ROUNDS (default 3), PORT (default 7861), OUT (default ./bench-out).
#
# CHECK=1 checks the pixels too, so that a compiler or library update cannot change the images unnoticed: every server
# starts from reference/moss-image-gemm.txt as its tuning file (a copy in OUT), and every image of every round must
# have the md5 that reference/md5.txt stores for its prompt and settings. The exit status is 1 when one differs, and 2
# when the images are not comparable: settings with no stored images, the text encoder on another GPU, or a server
# that tuned its own kernels instead, as on another GPU or hipBLASLt build, whose kernels round differently.
set -euo pipefail
ulimit -c 0
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
models=${MODELS:-$here/models}
steps=${STEPS:-8}
cfg=${CFG:-1.0}
sampler=${SAMPLER:-er_sde}
scheduler=${SCHEDULER:-simple}
rounds=${ROUNDS:-3}
port=${PORT:-7861}
out=${OUT:-$here/bench-out}
check=${CHECK:-0}
mkdir -p "$out"
out=$(cd "$out" && pwd)  # XDG_CACHE_HOME must be absolute

names=(cat poster street)
prompts=(
    "a lovely cat holding a sign says 'moss-image.cpp'"
    'A vintage travel poster of a mountain lake at sunrise, with the word "LUCEBOX" in large bold letters at the top and "Local AI" in smaller letters at the bottom'
    'A photo of a red bicycle leaning against a stone wall in a narrow Italian street, soft morning light'
)

request() {  # prompt png-path; prints the request's wall time in seconds
    python3 - "$port" "$1" "$2" "$steps" "$cfg" "$sampler" "$scheduler" <<'PY'
import base64, json, sys, time, urllib.request
port, prompt, path, steps, cfg, sampler, scheduler = sys.argv[1:8]
body = json.dumps({"prompt": prompt, "width": 1024, "height": 1024, "seed": 42, "steps": int(steps),
                   "cfg_scale": float(cfg), "sampler_name": sampler, "scheduler": scheduler}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/sdapi/v1/txt2img", body, {"Content-Type": "application/json"})
start = time.perf_counter()
image = base64.b64decode(json.load(urllib.request.urlopen(req, timeout=900))["images"][0])
print(f"{time.perf_counter() - start:.3f}")
open(path, "wb").write(image)
PY
}

for round in $(seq "$rounds"); do
    for spec in "$@"; do
        label=${spec%%=*}
        command=${spec#*=}
        log=$out/$label-$round.log
        if [[ $check == 1 ]]; then
            export XDG_CACHE_HOME=$out/$label-cache
            mkdir -p "$XDG_CACHE_HOME/lucebox"
            cp "$here/reference/moss-image-gemm.txt" "$XDG_CACHE_HOME/lucebox/moss-image-gemm.txt"
        fi
        launched=$(date +%s.%N)
        # shellcheck disable=SC2086  # COMMAND may carry an env prefix
        $command "$models/moss-image.gguf" --text-encoder "$models/Qwen3.5-9B-Q8_0.gguf" \
            --host 127.0.0.1 --port "$port" > "$log" 2>&1 &
        pid=$!
        trap 'kill "$pid" 2>/dev/null || true' EXIT  # a failed request must not leave the server on the GPU
        for _ in $(seq 12000); do
            curl -s -o /dev/null "http://127.0.0.1:$port/v1/models" && break
            kill -0 "$pid" 2>/dev/null || { echo "$label: server exited, see $log" >&2; exit 1; }
            sleep 0.05
        done
        awk -v a="$launched" -v b="$(date +%s.%N)" 'BEGIN { printf "%.2f\n", b - a }' > "$out/$label-$round.ready"
        for i in "${!names[@]}"; do
            request "${prompts[$i]}" "$out/$label-${names[$i]}.png"
        done > "$out/$label-$round.wall"
        for name in "${names[@]}"; do
            echo "$name $(md5sum < "$out/$label-$name.png" | cut -c1-32)"
        done > "$out/$label-$round.md5"
        kill "$pid"
        wait "$pid" 2>/dev/null || true
        trap - EXIT
        echo "$label round $round: $(grep -oE 'generate_image completed in [0-9.]+' "$log" | grep -oE '[0-9.]+$' | tr '\n' ' ')s"
    done
done

median() {  # [printf format, default %.2f]
    sort -n | awk -v f="${1:-%.2f}" '{ v[NR] = $1 } END {
        printf f " (min " f ", max " f ", n=%d)", NR % 2 ? v[(NR + 1) / 2] : (v[NR / 2] + v[NR / 2 + 1]) / 2, v[1], v[NR], NR }'
}
phase() {  # label phase: the phase's milliseconds from every image's "[moss-image] ..." line
    for f in "$out/$1"-[0-9]*.log; do grep -oE "$2 [0-9]+ ms" "$f" | grep -oE '[0-9]+'; done | median %.0f
}
# The stored md5 of a prompt at this run's settings, if any.
stored() {
    awk -v k="$steps $sampler $scheduler $(awk -v c="$cfg" 'BEGIN { printf "%g", c }') $1" \
        '!/^#/ && $1 " " $2 " " $3 " " $4 " " $5 == k { print $6 }' "$here/reference/md5.txt"
}
# Why a label's images are not comparable with the stored ones; nothing when they are.
incomparable() {
    for round in $(seq "$rounds"); do
        if grep -q 'text encoder on GPU' "$out/$1-$round.log"; then
            echo "not comparable: the text encoder ran on another GPU"
            return
        fi
        if ! grep -q 'GEMM kernels tuned in' "$out/$1-$round.log"; then
            echo "not comparable: the server did not take reference/moss-image-gemm.txt's kernels (another GPU or hipBLASLt)"
            return
        fi
    done
}
echo
status=0
for spec in "$@"; do
    label=${spec%%=*}
    per_image=$(for f in "$out/$label"-[0-9]*.log; do grep -oE 'generate_image completed in [0-9.]+' "$f" | grep -oE '[0-9.]+$'; done | median)
    echo "$label: ready $(cat "$out/$label"-[0-9]*.ready | median) s, request $(cat "$out/$label"-[0-9]*.wall | median) s"
    echo "  generate_image $per_image s; text $(phase "$label" text) ms, sampling $(phase "$label" sampling) ms, decode $(phase "$label" decode) ms"
    why=$([[ $check == 1 ]] && incomparable "$label" || true)
    for name in "${names[@]}"; do
        sums=$(for round in $(seq "$rounds"); do awk -v n="$name" '$1 == n { print $2 }' "$out/$label-$round.md5"; done | sort -u)
        shown=$sums
        if [[ $(wc -l <<< "$sums") -gt 1 ]]; then
            shown="differs between rounds: $(paste -sd ' ' - <<< "$sums")"
        fi
        verdict=""
        if [[ $check == 1 ]]; then
            want=$(stored "$name")
            if [[ -z $want ]]; then
                verdict="not comparable: no stored image for these settings"
            elif [[ -n $why ]]; then
                verdict=$why
            elif [[ $sums == "$want" ]]; then
                verdict="= stored"
            else
                verdict="DIFFERS from the stored $want"
                status=1
            fi
            if [[ $verdict == "not comparable"* && $status == 0 ]]; then
                status=2
            fi
        fi
        printf '  %-7s %s %s\n' "$name" "$shown" "$verdict"
    done
done
exit "$status"
