// Clean-chunk page-out: a chunk recalled from host and not written since has a
// host copy identical to its block, so the next page_out skips the D2H. Any row
// write into the chunk (alloc_span/slot_for) makes it dirty again. CPU ggml
// backend, no GPU.
#include "CppUnitTestFramework.hpp"
#include "kvflash_pager_test_host.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPagerCleanEvictFixture {};
constexpr int kChunk = 64, kPool = 512;   // 8 blocks

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
void must(bool ok) { if (!ok) throw std::runtime_error("clean-evict helper precondition failed"); }

// Write `val` over every byte of the token row at logical position pos, all tensors and heads.
void write_row(Harness & H, const KvFlashPager & p, int64_t pos, uint8_t val) {
    const int slot = p.slot_of(pos);
    must(slot >= 0);
    for (auto * set : {&H.k, &H.v}) {
        for (ggml_tensor * t : *set) {
            std::vector<uint8_t> row(t->nb[1], val);
            for (int h = 0; h < (int)t->ne[2]; h++)
                ggml_backend_tensor_set(t, row.data(), (size_t)h * t->nb[2] + (size_t)slot * t->nb[1], t->nb[1]);
        }
    }
}
std::vector<uint8_t> bytes_of(const KvFlashPager & p, int c) {
    std::vector<uint8_t> b(p.chunk_bytes());
    p.read_chunk(c, b.data());
    return b;
}
// 10 chunks in 8 blocks; chunks 1 and 2 got evicted by the append, so make room by evicting 3.
void fill(Harness & H, KvFlashPager & p) {
    must(p.alloc_span(0, 10 * kChunk));
    for (int c = 0; c < 10; c++)
        if (p.is_resident(c)) for (int i = 0; i < kChunk; i++) write_row(H, p, (int64_t)c * kChunk + i, (uint8_t)(c * 16 + (i & 7) + 1));
}
}  // namespace

TEST_CASE(KvflashPagerCleanEvictFixture, first_page_out_copies_and_a_clean_re_evict_skips) {
    Harness H; TestPager p;
    REQUIRE(p.attach(cfg(), H.k, H.v));
    fill(H, p);
    REQUIRE(p.is_resident(5));
    const std::vector<uint8_t> want = bytes_of(p, 5);
    REQUIRE(p.page_out(5));
    p.synchronize_paging();
    CHECK(p.stats().d2h_skipped == 0);            // dirty: the first page-out must copy
    CHECK(bytes_of(p, 5) == want);
    REQUIRE(p.page_in(5));
    p.synchronize_paging();
    REQUIRE(p.page_out(5));                        // clean: host copy already equals the block
    p.synchronize_paging();
    CHECK(p.stats().d2h_skipped == 1);
    CHECK(bytes_of(p, 5) == want);                 // host bytes equal what a real copy writes
}

TEST_CASE(KvflashPagerCleanEvictFixture, a_write_after_page_in_forces_the_copy) {
    Harness H; TestPager p;
    REQUIRE(p.attach(cfg(), H.k, H.v));
    fill(H, p);
    REQUIRE(p.page_out(5));
    REQUIRE(p.page_in(5));
    p.synchronize_paging();
    REQUIRE(p.alloc_span(5 * kChunk + 3, 1));      // a rewrite of a row inside the recalled chunk
    write_row(H, p, 5 * kChunk + 3, 0xEE);
    const std::vector<uint8_t> want = bytes_of(p, 5);
    const int64_t skipped = p.stats().d2h_skipped;
    REQUIRE(p.page_out(5));
    p.synchronize_paging();
    CHECK(p.stats().d2h_skipped == skipped);       // dirty again: copied, not skipped
    CHECK(bytes_of(p, 5) == want);
}

TEST_CASE(KvflashPagerCleanEvictFixture, truncate_rows_into_a_recalled_chunk_makes_it_dirty) {
    Harness H; TestPager p;
    REQUIRE(p.attach(cfg(), H.k, H.v));
    fill(H, p);
    REQUIRE(p.page_out(5));
    REQUIRE(p.page_in(5));
    p.synchronize_paging();
    p.truncate(5 * kChunk + 10);                   // zeroes rows >= 10 of resident chunk 5
    p.synchronize_paging();
    const std::vector<uint8_t> want = bytes_of(p, 5);
    const int64_t skipped = p.stats().d2h_skipped;
    REQUIRE(p.page_out(5));
    p.synchronize_paging();
    CHECK(p.stats().d2h_skipped == skipped);
    CHECK(bytes_of(p, 5) == want);
}

TEST_CASE(KvflashPagerCleanEvictFixture, reselect_cycles_skip_clean_evictions_and_keep_bytes) {
    Harness H; TestPager p;
    REQUIRE(p.attach(cfg(), H.k, H.v));
    fill(H, p);
    std::vector<std::vector<uint8_t>> want;
    for (int c = 0; c < 10; c++) want.push_back(bytes_of(p, c));
    for (int cycle = 0; cycle < 3; cycle++) {
        p.score_hook = [](int c) { return c == 2 ? 100.0f : (float)c; };
        p.reselect(0);
        p.reselect_recent();
        p.synchronize_paging();
    }
    CHECK(p.stats().d2h_skipped > 0);
    for (int c = 0; c < 10; c++) CHECK(bytes_of(p, c) == want[(size_t)c]);
}
