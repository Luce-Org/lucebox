// Pager pieces the serial recall path leans on (CPU ggml backend, no GPU): the recency reselect that
// resets the pool before the scoring pass, and the decode mask bound that hides rows above the position.

#include "CppUnitTestFramework.hpp"

#include "kvflash_pager.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPagerRecallFixture {};
constexpr int kChunk = 64, kPool = 512;   // 8 blocks; sink 1 + tail window 4 + 2 spare = 7

struct Harness {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;
    Harness() {
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = 16 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < 2; l++) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, kPool, 2));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, kPool, 2));
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }
    ~Harness() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};
KvFlashConfig cfg() {
    KvFlashConfig c;
    c.pool_tokens = kPool; c.chunk_tokens = kChunk; c.sink_chunks = 1; c.tail_window_chunks = 4;
    return c;
}
void must(bool ok) { if (!ok) throw std::runtime_error("pager recall helper precondition failed"); }
std::vector<bool> resident_set(const KvFlashPager & p, int n) {
    std::vector<bool> v;
    for (int c = 0; c < n; c++) v.push_back(p.is_resident(c));
    return v;
}
// The previous request: 14 chunks, a hook-picked old chunk (2) recalled, the hook gone.
void leave_stale_pick(KvFlashPager & p) {
    must(p.alloc_span(0, 14 * kChunk));
    p.score_hook = [](int c) { return c == 2 ? 100.0f : (float)c; };
    p.reselect(0);
    p.score_hook = nullptr;
}
}  // namespace

// truncate(C) with C not fitting keeps every current resident, so the previous request's recalled
// chunks stay resident into the next request's scoring pass. reselect_recent() restores the pool a
// pager that only ever saw C tokens under plain LRU holds.
TEST_CASE(KvflashPagerRecallFixture, recent_pool_after_truncate_equals_a_fresh_lru_pager) {
    const int C = 10 * kChunk;   // 10 chunks > 8 blocks
    Harness F;
    KvFlashPager fresh;
    REQUIRE(fresh.attach(cfg(), F.k, F.v));
    REQUIRE(fresh.alloc_span(0, C));
    const std::vector<bool> want = resident_set(fresh, 10);

    Harness A;
    KvFlashPager stale;
    REQUIRE(stale.attach(cfg(), A.k, A.v));
    leave_stale_pick(stale);
    stale.truncate(C);
    CHECK(stale.is_resident(2));                       // without the recency step the stale pick survives
    CHECK(resident_set(stale, 10) != want);
    CHECK(stale.reselect_recent() > 0);
    CHECK(!stale.score_hook);                          // no hook is left behind: later evictions are plain LRU
    CHECK(resident_set(stale, 10) == want);
}

// A prefix that fits is fully resident: the recency step has nothing to do, so applying it on every
// pooled request changes no request.
TEST_CASE(KvflashPagerRecallFixture, reselect_recent_on_a_fitting_prefix_is_a_noop) {
    Harness A;
    KvFlashPager p;
    REQUIRE(p.attach(cfg(), A.k, A.v));
    leave_stale_pick(p);
    p.truncate(6 * kChunk);                            // 6 chunks <= 8 blocks
    const std::vector<bool> before = resident_set(p, 6);
    CHECK(p.reselect_recent() == 0);
    CHECK(resident_set(p, 6) == before);
    CHECK(!p.score_hook);
}

// AR decode builds its attention mask from fill_slot_mask. Without a position bound it exposes every
// row of a resident chunk, including rows above cur_pos that hold a previous turn's tokens.
TEST_CASE(KvflashPagerRecallFixture, slot_mask_hides_rows_above_the_decode_position) {
    Harness A;
    KvFlashPager p;
    REQUIRE(p.attach(cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 3 * kChunk));              // chunks 0..2 resident
    const int cur_pos = 2 * kChunk + 10;               // inside the tail chunk
    std::vector<uint16_t> open((size_t)p.arena_tokens()), cut((size_t)p.arena_tokens());
    p.fill_slot_mask(open.data());
    const int tail_row = p.block_of(2) * kChunk;
    CHECK(open[(size_t)tail_row + 40] == 0);           // unbounded: the row above cur_pos is attendable (the defect)
    p.fill_slot_mask(cut.data(), cur_pos + 1);         // a decode step at cur_pos attends rows <= cur_pos
    for (int i = 0; i < kChunk; i++) {
        const bool visible = 2 * kChunk + i <= cur_pos;
        CHECK(cut[(size_t)tail_row + i] == (visible ? 0 : 0xFC00));
    }
    CHECK(cut[(size_t)p.block_of(0) * kChunk + 5] == 0);   // older chunks stay visible
    CHECK(cut[(size_t)p.block_of(1) * kChunk + 63] == 0);
}
