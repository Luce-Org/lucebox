#!/usr/bin/env bash

# Shared DGX Spark safety helpers for Ling 3 benchmark launchers. This file is
# sourced by the launch scripts; it intentionally does not enable shell flags.

ling3_validate_positive_integer() {
    local name=$1
    local value=$2
    if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value <= 0 )); then
        printf '%s must be a positive integer, got %q\n' "$name" "$value" >&2
        return 1
    fi
}

ling3_refuse_existing_artifacts() {
    local output_json=$1
    local artifact
    for artifact in "$output_json" "${output_json}.server.log" "${output_json}.host.log"; do
        if [[ -e "$artifact" ]]; then
            printf 'Refusing to overwrite %s\n' "$artifact" >&2
            return 1
        fi
    done
}

ling3_host_snapshot() {
    local host_log=$1
    local phase=$2
    {
        printf 'phase=%s\n' "$phase"
        date --iso-8601=seconds
        awk '/^(MemAvailable|SwapTotal|SwapFree):/ { print }' /proc/meminfo
        free -h
        ps -eo pid,etimes,rss,comm,args | grep -E '[d]flash_server|[l]lama-server' || true
        ip -brief address 2>/dev/null || true
        ip route show default 2>/dev/null || true
        if command -v nmcli >/dev/null 2>&1; then
            nmcli -t -f GENERAL.STATE,GENERAL.CONNECTION,GENERAL.DEVICE,IP4.ADDRESS \
                device show 2>/dev/null || true
        fi
        printf '\n'
    } >>"$host_log"
}

ling3_safe_host_preflight() {
    local output_json=$1
    local host_log="${output_json}.host.log"
    local min_available_gib=${LING_BENCH_MIN_AVAILABLE_GIB:-96}
    local max_swap_mib=${LING_BENCH_MAX_SWAP_USED_MIB:-512}
    local available_kib swap_total_kib swap_free_kib swap_used_kib

    ling3_validate_positive_integer LING_BENCH_MIN_AVAILABLE_GIB \
        "$min_available_gib"
    ling3_validate_positive_integer LING_BENCH_MAX_SWAP_USED_MIB \
        "$max_swap_mib"

    available_kib=$(awk '/^MemAvailable:/ { print $2 }' /proc/meminfo)
    swap_total_kib=$(awk '/^SwapTotal:/ { print $2 }' /proc/meminfo)
    swap_free_kib=$(awk '/^SwapFree:/ { print $2 }' /proc/meminfo)
    swap_used_kib=$((swap_total_kib - swap_free_kib))

    if (( available_kib < min_available_gib * 1024 * 1024 )); then
        printf 'Refusing model load: MemAvailable is %.1f GiB; require at least %d GiB.\n' \
            "$(awk -v kib="$available_kib" 'BEGIN { print kib / 1048576 }')" \
            "$min_available_gib" >&2
        return 1
    fi
    if (( swap_used_kib > max_swap_mib * 1024 )); then
        printf 'Refusing model load: swap use is %.1f MiB; limit is %d MiB.\n' \
            "$(awk -v kib="$swap_used_kib" 'BEGIN { print kib / 1024 }')" \
            "$max_swap_mib" >&2
        return 1
    fi
    if pgrep -x dflash_server >/dev/null || pgrep -x llama-server >/dev/null; then
        printf 'Refusing model load while another inference server is running.\n' >&2
        return 1
    fi
    ling3_host_snapshot "$host_log" preflight
}

ling3_start_guarded_server() {
    local max_seconds=${LING_SERVER_MAX_RUNTIME_SECONDS:-1800}
    ling3_validate_positive_integer LING_SERVER_MAX_RUNTIME_SECONDS \
        "$max_seconds"
    if ! command -v timeout >/dev/null 2>&1; then
        printf 'Refusing unguarded model load: GNU timeout is unavailable.\n' >&2
        return 1
    fi

    timeout --signal=TERM --kill-after=30s "${max_seconds}s" "$@" &
    LING3_GUARDED_SERVER_PID=$!
}
