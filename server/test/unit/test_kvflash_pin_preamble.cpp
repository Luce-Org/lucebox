// Unit tests for LUCE_KVFLASH_PREAMBLE_ROWS, the pin of the system+tools preamble outside the history budget
// (no model, CPU ggml only). Off by default: the off cases must reproduce the feature-off numbers exactly.

#include "CppUnitTestFramework.hpp"
#include "scoped_env.h"

#include "kvflash_pager.h"
#include "kvflash_prefill_reselect.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPinPreambleFixture {};

// Token ids used by the synthetic prompts.
constexpr int32_t S = 100, U = 101, SYS = 102, NL = 103, END = 104;

struct PagerHarness {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;
    PagerHarness() {
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = 16 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < 2; l++) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, 512, 2));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, 512, 2));
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }
    ~PagerHarness() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};
void must(bool ok) { if (!ok) throw std::runtime_error("pin helper precondition failed"); }
}  // namespace

// ---- preamble end ----------------------------------------------------------
TEST_CASE(KvflashPinPreambleFixture, preamble_end_system_with_tools_first) {
    // <|im_start|>system \n [tools...] <|im_end|> \n <|im_start|>user ...
    const std::vector<int32_t> ids = {S, SYS, NL, 7, 8, 9, END, NL, S, U, NL, 5, END};
    CHECK(kvflash_preamble_end(ids, S, SYS) == 8);
}
TEST_CASE(KvflashPinPreambleFixture, preamble_end_no_system_is_zero) {
    CHECK(kvflash_preamble_end({S, U, NL, 5, END, NL}, S, SYS) == 0);   // user first
    CHECK(kvflash_preamble_end({}, S, SYS) == 0);                       // empty
    CHECK(kvflash_preamble_end({7, S, SYS, NL, 1, S, U}, S, SYS) == 0); // system not at the start
    CHECK(kvflash_preamble_end({S, SYS, NL, 7, END, NL}, S, SYS) == 0); // system only: no next turn
    CHECK(kvflash_preamble_end({S, SYS, NL, 7, S, U}, -1, SYS) == 0);   // unknown ids
    CHECK(kvflash_preamble_end({S, SYS, NL, 7, S, U}, S, -1) == 0);
}

// ---- reserve sizing -----------------------------------------------------------
TEST_CASE(KvflashPinPreambleFixture, preamble_reserve_rounds_up_to_a_chunk) {
    CHECK(kvflash_preamble_reserve_tokens(0, 64) == 0);
    CHECK(kvflash_preamble_reserve_tokens(-5, 64) == 0);
    CHECK(kvflash_preamble_reserve_tokens(1, 64) == 64);
    CHECK(kvflash_preamble_reserve_tokens(64, 64) == 64);
    CHECK(kvflash_preamble_reserve_tokens(7168, 64) == 7168);
    CHECK(kvflash_preamble_reserve_tokens(7169, 64) == 7232);
    CHECK(kvflash_preamble_reserve_tokens(100, 0) == 0);
}
TEST_CASE(KvflashPinPreambleFixture, preamble_rows_env) {
    {
        luce_test::ScopedEnvVar e("LUCE_KVFLASH_PREAMBLE_ROWS", nullptr);
        CHECK(kvflash_preamble_rows_from_env() == 0);
    }
    {
        luce_test::ScopedEnvVar e("LUCE_KVFLASH_PREAMBLE_ROWS", "7168");
        CHECK(kvflash_preamble_rows_from_env() == 7168);
    }
    {
        luce_test::ScopedEnvVar e("LUCE_KVFLASH_PREAMBLE_ROWS", "-3");
        CHECK(kvflash_preamble_rows_from_env() == 0);
    }
    {
        luce_test::ScopedEnvVar e("LUCE_KVFLASH_PREAMBLE_ROWS", "junk");
        CHECK(kvflash_preamble_rows_from_env() == 0);
    }
}

// ---- pin rows / recall floor / budget -------------------------------------
TEST_CASE(KvflashPinPreambleFixture, pin_rows_never_exceed_the_reserve) {
    CHECK(kvflash_preamble_pin_rows(500, 1000) == 500);
    CHECK(kvflash_preamble_pin_rows(1500, 1000) == 1000);
    CHECK(kvflash_preamble_pin_rows(0, 1000) == 0);
    CHECK(kvflash_preamble_pin_rows(500, 0) == 0);
}
TEST_CASE(KvflashPinPreambleFixture, pin_chunks_counts_the_covered_chunks) {
    CHECK(kvflash_pin_chunks(0, 7168, 64) == 0);
    CHECK(kvflash_pin_chunks(7000, 0, 64) == 0);          // feature off
    CHECK(kvflash_pin_chunks(7000, 7168, 64) == 110);     // ceil(7000 / 64)
    CHECK(kvflash_pin_chunks(9000, 7168, 64) == 112);     // clipped to the reserve
}

// ---- pager: pin, truncate, re-pin ------------------------------------------
TEST_CASE(KvflashPinPreambleFixture, pin_preamble_truncate_below_then_repin) {
    PagerHarness h;
    KvFlashConfig cfg;
    cfg.pool_tokens = 512; cfg.chunk_tokens = 64; cfg.sink_chunks = 1; cfg.tail_window_chunks = 4;
    KvFlashPager p;
    must(p.attach(cfg, h.k, h.v));
    must(p.alloc_span(0, 6 * 64));
    // chunk 0 is also the sink; a 64-row reserve pins exactly one chunk.
    CHECK(kvflash_pin_preamble(p, 300, 64) == 1);
    CHECK(p.is_pinned(0));
    CHECK(!p.is_pinned(1));
    p.truncate(0);                       // below the preamble: pins are clipped away
    CHECK(!p.is_pinned(0));
    must(p.alloc_span(0, 3 * 64));
    CHECK(kvflash_pin_preamble(p, 300, 64) == 1);   // re-applied
    CHECK(p.is_pinned(0));
    p.reset();                           // request start: reset clears pins
    CHECK(!p.is_pinned(0));
    CHECK(kvflash_pin_preamble(p, 300, 64) == 1);
    CHECK(kvflash_pin_preamble(p, 0, 64) == 0);    // no preamble: nothing pinned
    CHECK(kvflash_pin_preamble(p, 300, 0) == 0);   // feature off: nothing pinned
}

TEST_CASE(KvflashPinPreambleFixture, pin_survives_the_dense_reset_path) {
    // kvflash_sync_prefill: reset(), identity slot_for over the prompt, then the pin is re-applied.
    PagerHarness h;
    KvFlashConfig cfg;
    cfg.pool_tokens = 512; cfg.chunk_tokens = 64; cfg.sink_chunks = 1; cfg.tail_window_chunks = 4;
    KvFlashPager p;
    must(p.attach(cfg, h.k, h.v));
    must(p.alloc_span(0, 3 * 64));
    kvflash_pin_preamble(p, 300, 64);
    p.reset();
    for (int i = 0; i < 4 * 64; i++) must(p.slot_for(i) == i);
    CHECK(!p.is_pinned(0));                                   // reset drops it ...
    CHECK(kvflash_pin_preamble(p, 300, 64) == 1);             // ... and the re-apply restores it
    must(p.alloc_span(4 * 64, 8 * 64));                       // long LRU growth fills the pool
    CHECK(p.is_resident(0));                                  // the system chunk is never the victim
}

TEST_CASE(KvflashPinPreambleFixture, preamble_gate_requires_every_id) {
    CHECK(kvflash_preamble_gate(S, SYS).empty());
    CHECK(kvflash_preamble_gate(-1, SYS) == "<|im_start|> id missing");
    CHECK(kvflash_preamble_gate(S, -1) == "system id missing");
}
