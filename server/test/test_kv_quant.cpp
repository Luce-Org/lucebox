// Unit tests for dflash::kv_quant (parse_kv_type, resolve_kv_types,
// is_supported_kv_pair). Plain int main(), no frameworks.
//
// T8 (unsupported pair aborts) is not tested in-process because std::abort()
// terminates the test runner. Manual verification:
//   DFLASH27B_KV_K=tq3_0 DFLASH27B_KV_V=q5_0 ./dflash/build/test_kv_quant
// Expected: prints "[dflash] KV pair …" message and aborts.

#include "kv_quant.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#define setenv(name, value, overwrite) _putenv_s(name, value)
#define unsetenv(name) _putenv_s(name, "")
#endif

// ─── helpers ────────────────────────────────────────────────────────────────

static void clear_kv_env() {
    unsetenv("DFLASH27B_KV_F16");
    unsetenv("DFLASH27B_KV_Q4");
    unsetenv("DFLASH27B_KV_TQ3");
    unsetenv("DFLASH27B_KV_K");
    unsetenv("DFLASH27B_KV_V");
}

// ─── T1: parse_kv_type ──────────────────────────────────────────────────────

static void t1_parse_kv_type() {
    // Case-insensitive successes
    assert(dflash::parse_kv_type("f16")   == GGML_TYPE_F16);
    assert(dflash::parse_kv_type("F16")   == GGML_TYPE_F16);
    assert(dflash::parse_kv_type("bf16")  == GGML_TYPE_BF16);
    assert(dflash::parse_kv_type("BF16")  == GGML_TYPE_BF16);
    assert(dflash::parse_kv_type("q4_0")  == GGML_TYPE_Q4_0);
    assert(dflash::parse_kv_type("Q4_0")  == GGML_TYPE_Q4_0);
    assert(dflash::parse_kv_type("q4_1")  == GGML_TYPE_Q4_1);
    assert(dflash::parse_kv_type("q5_0")  == GGML_TYPE_Q5_0);
    assert(dflash::parse_kv_type("q5_1")  == GGML_TYPE_Q5_1);
    assert(dflash::parse_kv_type("q8_0")  == GGML_TYPE_Q8_0);
    assert(dflash::parse_kv_type("Q8_0")  == GGML_TYPE_Q8_0);
    assert(dflash::parse_kv_type("tq3_0") == GGML_TYPE_TQ3_0);
    assert(dflash::parse_kv_type("TQ3_0") == GGML_TYPE_TQ3_0);

    // Unknown / empty inputs
    assert(dflash::parse_kv_type("garbage") == GGML_TYPE_COUNT);
    assert(dflash::parse_kv_type("")         == GGML_TYPE_COUNT);
    assert(dflash::parse_kv_type("q9_0")     == GGML_TYPE_COUNT);

    std::puts("T1 PASS");
}

// ─── T2: resolve_kv_types precedence ────────────────────────────────────────

static void t2_resolve_kv_types() {
    ggml_type k, v;

    // 2a: no env → default Q8_0/Q8_0
    clear_kv_env();
    dflash::resolve_kv_types(k, v);
    assert(k == GGML_TYPE_Q8_0 && v == GGML_TYPE_Q8_0);

    // 2b: F16 shorthand
    clear_kv_env();
    setenv("DFLASH27B_KV_F16", "1", 1);
    dflash::resolve_kv_types(k, v);
    assert(k == GGML_TYPE_F16 && v == GGML_TYPE_F16);
    clear_kv_env();

    // 2c: Q4 shorthand wins over F16 (set both; Q4 is checked last in layer 2)
    clear_kv_env();
    setenv("DFLASH27B_KV_F16", "1", 1);
    setenv("DFLASH27B_KV_Q4",  "1", 1);
    dflash::resolve_kv_types(k, v);
    assert(k == GGML_TYPE_Q4_0 && v == GGML_TYPE_Q4_0);
    clear_kv_env();

    // 2d: per-axis overrides override legacy shorthand
    clear_kv_env();
    setenv("DFLASH27B_KV_Q4", "1", 1);
    setenv("DFLASH27B_KV_K",  "q8_0", 1);
    setenv("DFLASH27B_KV_V",  "q4_0", 1);
    dflash::resolve_kv_types(k, v);
    assert(k == GGML_TYPE_Q8_0 && v == GGML_TYPE_Q4_0);
    clear_kv_env();

    // 2e: TQ3_0/Q8_0 asymmetric via per-axis
    clear_kv_env();
    setenv("DFLASH27B_KV_K", "tq3_0", 1);
    setenv("DFLASH27B_KV_V", "q8_0",  1);
    dflash::resolve_kv_types(k, v);
    assert(k == GGML_TYPE_TQ3_0 && v == GGML_TYPE_Q8_0);
    clear_kv_env();

    std::puts("T2 PASS");
}

// ─── T3: is_supported_kv_pair matrix ─────────────────────────────────────────

static void t3_is_supported_kv_pair() {
    // K ∈ {F16,BF16,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0} × V ∈ {F16,BF16,Q4_0,Q4_1,Q5_0,Q5_1,Q8_0,TQ3_0}
    const ggml_type k_generic[] = {
        GGML_TYPE_F16, GGML_TYPE_BF16,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0
    };
    const ggml_type v_full[] = {
        GGML_TYPE_F16, GGML_TYPE_BF16,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0, GGML_TYPE_TQ3_0
    };

    // All generic K × full V should be supported
    for (ggml_type k : k_generic) {
        for (ggml_type v : v_full) {
            assert(dflash::is_supported_kv_pair(k, v));
        }
    }

    // K = TQ3_0 × V ∈ {F16, BF16, Q4_0, Q8_0, TQ3_0}  → supported
    const ggml_type v_tq3_ok[] = {
        GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_0,
        GGML_TYPE_Q8_0, GGML_TYPE_TQ3_0
    };
    for (ggml_type v : v_tq3_ok) {
        assert(dflash::is_supported_kv_pair(GGML_TYPE_TQ3_0, v));
    }

    // K = TQ3_0 × V ∈ {Q4_1, Q5_0, Q5_1} → NOT supported
    assert(!dflash::is_supported_kv_pair(GGML_TYPE_TQ3_0, GGML_TYPE_Q4_1));
    assert(!dflash::is_supported_kv_pair(GGML_TYPE_TQ3_0, GGML_TYPE_Q5_0));
    assert(!dflash::is_supported_kv_pair(GGML_TYPE_TQ3_0, GGML_TYPE_Q5_1));

    // Explicit negatives from the task spec
    assert(!dflash::is_supported_kv_pair(GGML_TYPE_TQ3_0, GGML_TYPE_Q5_0));  // K=TQ3, V=Q5_0 false
    assert(!dflash::is_supported_kv_pair(GGML_TYPE_TQ3_0, GGML_TYPE_Q4_1));  // K=TQ3, V=Q4_1 false
    assert( dflash::is_supported_kv_pair(GGML_TYPE_Q5_0,  GGML_TYPE_TQ3_0)); // K=Q5_0,V=TQ3_0 true

    std::puts("T3 PASS");
}

// ─── T4: TQ3 backward-compat shorthand ───────────────────────────────────────

static void t4_tq3_shorthand() {
    clear_kv_env();
    setenv("DFLASH27B_KV_TQ3", "1", 1);
    ggml_type k, v;
    dflash::resolve_kv_types(k, v);
    assert(k == GGML_TYPE_TQ3_0 && v == GGML_TYPE_TQ3_0);
    clear_kv_env();

    std::puts("T4 PASS");
}

// ─── T5: hybrid KV reservation uses n_full + cache-type (not n_layer × f16) ──
// The bug at qwen35moe_backend.cpp:2595 reserved n_layer(40) × hardcoded f16,
// over-reserving ~7× and forcing experts cold. The shared helper must count
// only full-attention layers (n_full) and honor the resolved cache type.
static void t5_kv_reservation() {
    // Qwen3.6-35B-A3B geometry: 40 layers, full_attention_interval=4 → n_full=10.
    const int n_layer = 40, interval = 4, n_head_kv = 10, dk = 128, dv = 128;
    const uint64_t got = dflash::kv_reservation_bytes_per_token(
        n_layer, interval, n_head_kv, GGML_TYPE_Q4_0, dk, GGML_TYPE_Q4_0, dv);
    const uint64_t expect = (uint64_t)(n_layer / interval) * (uint64_t)n_head_kv *
        (uint64_t)(ggml_row_size(GGML_TYPE_Q4_0, dk) + ggml_row_size(GGML_TYPE_Q4_0, dv));
    assert(got == expect);

    // Regression guard vs the old bug (n_layer × hardcoded f16 = 2 B/elem):
    const uint64_t buggy = (uint64_t)n_layer * 2 * (uint64_t)n_head_kv * (uint64_t)dk * 2;
    assert(got < buggy / 3);  // ÷4 (layers) × (q4_0 < f16) → well over 3× smaller

    // cache-type is honored: q4_0 reservation is strictly smaller than f16.
    const uint64_t f16 = dflash::kv_reservation_bytes_per_token(
        n_layer, interval, n_head_kv, GGML_TYPE_F16, dk, GGML_TYPE_F16, dv);
    assert(got < f16);

    // full_attention_interval=0 guard: no div-by-zero, falls back to n_layer.
    const uint64_t fallback = dflash::kv_reservation_bytes_per_token(
        n_layer, 0, n_head_kv, GGML_TYPE_Q4_0, dk, GGML_TYPE_Q4_0, dv);
    assert(fallback == (uint64_t)n_layer * (uint64_t)n_head_kv *
        (uint64_t)(ggml_row_size(GGML_TYPE_Q4_0, dk) + ggml_row_size(GGML_TYPE_Q4_0, dv)));

    std::puts("T5 PASS");
}

// ─── main ────────────────────────────────────────────────────────────────────

int main() {
    clear_kv_env();  // start clean regardless of calling environment

    t1_parse_kv_type();
    t2_resolve_kv_types();
    t3_is_supported_kv_pair();
    t4_tq3_shorthand();
    t5_kv_reservation();

    std::puts("ALL TESTS PASS");
    return 0;
}
