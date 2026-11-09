// A reselect recalls its page-ins as one batch (CPU ggml backend, no GPU): the block hand-outs inside it do not
// fence the page stream one by one (a host wait each on HIP); the one wait at its end covers them all. Residency
// and page counts are unchanged, so the pool a request sees is bit for bit the one it saw before.

#include "CppUnitTestFramework.hpp"

#include "kvflash_pager.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPagerReselectFenceFixture {};
constexpr int kChunk = 64, kPool = 512;   // 8 blocks; sink 1 + tail window 4 + 3 evictable

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
std::vector<bool> resident_set(const KvFlashPager & p, int n) {
    std::vector<bool> v;
    for (int c = 0; c < n; c++) v.push_back(p.is_resident(c));
    return v;
}
}  // namespace

// 14 chunks through 8 blocks leave chunks 1..9 host-backed. A pick of three of them pages three out and three in;
// none of the three hand-outs fences on its own, and every allocation outside a reselect still does.
TEST_CASE(KvflashPagerReselectFenceFixture, reselect_page_ins_fence_once_at_the_end) {
    Harness A;
    KvFlashPager p;
    REQUIRE(p.attach(cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 14 * kChunk));
    const int64_t fences0 = p.stats().fence_points, ins0 = p.stats().page_ins;
    CHECK(fences0 == 14);                                        // one hand-out per fresh chunk
    p.score_hook = [](int c) { return c >= 1 && c <= 3 ? 100.0f : (float)c; };
    const int events = p.reselect(0);
    CHECK(events == 6);
    CHECK(p.stats().page_ins - ins0 == 3);
    CHECK(p.stats().fence_points == fences0);                    // the recalls fenced as one batch
    CHECK(p.is_resident(1)); CHECK(p.is_resident(2)); CHECK(p.is_resident(3));
    p.score_hook = nullptr;
    REQUIRE(p.alloc_span(14 * kChunk, kChunk));                  // a plain allocation fences as before
    CHECK(p.stats().fence_points == fences0 + 1);
}

// The recency reselect is the same batch: its residency equals the per-chunk path's, fence count aside.
TEST_CASE(KvflashPagerReselectFenceFixture, reselect_recent_batches_and_keeps_the_recency_set) {
    Harness A;
    KvFlashPager p;
    REQUIRE(p.attach(cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 14 * kChunk));
    p.score_hook = [](int c) { return c >= 1 && c <= 3 ? 100.0f : (float)c; };
    REQUIRE(p.reselect(0) == 6);
    p.score_hook = nullptr;
    Harness B;
    KvFlashPager fresh;
    REQUIRE(fresh.attach(cfg(), B.k, B.v));
    REQUIRE(fresh.alloc_span(0, 14 * kChunk));
    const int64_t fences0 = p.stats().fence_points;
    CHECK(p.reselect_recent() == 6);
    CHECK(p.stats().fence_points == fences0);
    CHECK(resident_set(p, 14) == resident_set(fresh, 14));
}
