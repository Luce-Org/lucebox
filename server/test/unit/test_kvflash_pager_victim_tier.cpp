// The pager's victim tier: a paged-out chunk's bytes are kept in a tier beside the pool (device memory on GPU
// builds) and a recall of a chunk still there copies from the tier instead of the host copy. Residency, page counts
// and the bytes a recall lands are unchanged; only where they come from differs. The tier is FIFO by page-out, a
// chunk written after a recall loses its stale tier copy, and truncate drops the dropped.
// GPU builds run the cases on device tensors (the tier is device memory and only engages for them; skipped without
// a device); CPU builds run them on CPU tensors with the tier in host memory.

#include "CppUnitTestFramework.hpp"

#include "kvflash_pager.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef KVFLASH_HAS_ASYNC_DMA
#include "ggml-cuda.h"
#endif

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPagerVictimTierFixture {};
constexpr int kChunk = 64, kPool = 512;   // 8 blocks; sink 1 + tail window 4 + 3 evictable

struct Harness {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> k, v;
    // on_device: the tensors the tier pairs with on GPU builds (ignored on CPU builds, where only the CPU exists).
    explicit Harness(bool on_device = true) {
#ifdef KVFLASH_HAS_ASYNC_DMA
        if (on_device) {
            backend = ggml_backend_cuda_init(0);
            if (!backend) SKIP("no GPU device: the victim tier pairs with device tensors");
        }
#else
        (void)on_device;
#endif
        if (!backend) backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = 16 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < 2; l++) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, kPool, 2));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, kPool, 2));
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");
    }
    ~Harness() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
    Harness(const Harness &) = delete;
    Harness & operator=(const Harness &) = delete;
};
KvFlashConfig cfg(int victim_blocks) {
    KvFlashConfig c;
    c.pool_tokens = kPool; c.chunk_tokens = kChunk; c.sink_chunks = 1; c.tail_window_chunks = 4;
    c.victim_blocks = victim_blocks;
    return c;
}
// Layer-0 K rows of chunk c, head 0: the segment a page-out and a recall move.
std::vector<uint8_t> k_rows(const Harness & h, const KvFlashPager & p, int c) {
    std::vector<uint8_t> out((size_t)kChunk * h.k[0]->nb[1]);
    ggml_backend_tensor_get(h.k[0], out.data(), (size_t)p.block_of(c) * kChunk * h.k[0]->nb[1], out.size());
    return out;
}
void tag_k_rows(const Harness & h, const KvFlashPager & p, int c, uint8_t tag) {
    std::vector<uint8_t> v((size_t)kChunk * h.k[0]->nb[1], tag);
    ggml_backend_tensor_set(h.k[0], v.data(), (size_t)p.block_of(c) * kChunk * h.k[0]->nb[1], v.size());
}
std::vector<uint8_t> tagged(const Harness & h, uint8_t tag) { return std::vector<uint8_t>((size_t)kChunk * h.k[0]->nb[1], tag); }
// 14 chunks, each tagged while resident, so chunks 1..9 reach the host (and the tier) with known bytes.
void fill_tagged(Harness & h, KvFlashPager & p) {
    for (int c = 0; c < 14; c++) {
        if (!p.alloc_span(c * kChunk, kChunk)) throw std::runtime_error("fill_tagged: alloc_span failed");
        tag_k_rows(h, p, c, (uint8_t)(c + 1));
    }
}
std::vector<bool> resident_set(const KvFlashPager & p, int n) {
    std::vector<bool> v;
    for (int c = 0; c < n; c++) v.push_back(p.is_resident(c));
    return v;
}
}  // namespace

// A tier large enough keeps every paged-out chunk: the three recalls of a pick come from it, with the bytes the
// host copy holds, and the residency equals a pager without a tier.
TEST_CASE(KvflashPagerVictimTierFixture, recalls_come_from_the_tier_with_identical_bytes_and_residency) {
    Harness A, B;
    KvFlashPager p, plain;
    REQUIRE(p.attach(cfg(16), A.k, A.v));
    REQUIRE(p.victim_blocks() == 16);                    // the tier engaged for these tensors
    REQUIRE(plain.attach(cfg(0), B.k, B.v));
    fill_tagged(A, p);
    fill_tagged(B, plain);
    auto pick = [](int c) { return c >= 1 && c <= 3 ? 100.0f : (float)c; };
    p.score_hook = pick; plain.score_hook = pick;
    CHECK(p.reselect(0) == 6);
    CHECK(plain.reselect(0) == 6);
    CHECK(resident_set(p, 14) == resident_set(plain, 14));
    CHECK(p.stats().victim_hits == 3);
    CHECK(p.stats().victim_misses == 0);
    CHECK(plain.stats().victim_hits == 0);
    CHECK(plain.stats().victim_misses == 3);
    for (int c = 1; c <= 3; c++) {
        CHECK(k_rows(A, p, c) == tagged(A, (uint8_t)(c + 1)));
        CHECK(k_rows(A, p, c) == k_rows(B, plain, c));
    }
}

// A tier of four holds the newest four page-outs: a reselect pages out first, so after it the tier holds its three
// page-outs and the newest earlier one; older recalls fall back to the host copy.
TEST_CASE(KvflashPagerVictimTierFixture, a_full_tier_drops_the_oldest_page_out_first) {
    Harness A;
    KvFlashPager p;
    REQUIRE(p.attach(cfg(4), A.k, A.v));
    REQUIRE(p.victim_blocks() == 4);
    fill_tagged(A, p);                                   // page-outs 1..6 in order; the tier ends holding 3,4,5,6
    p.score_hook = [](int c) { return c >= 1 && c <= 3 ? 100.0f : (float)c; };
    CHECK(p.reselect(0) == 6);                           // out 7,8,9 (tier: 6,7,8,9), in 1,2,3 from the host
    CHECK(p.stats().victim_hits == 0);
    CHECK(p.stats().victim_misses == 3);
    p.score_hook = nullptr;
    CHECK(p.reselect_recent() == 6);                     // out 1,2,3 (tier: 9,1,2,3), back in 7,8,9: only 9 hits
    CHECK(p.stats().victim_hits == 1);
    CHECK(p.stats().victim_misses == 5);
    for (int c = 7; c <= 9; c++) CHECK(k_rows(A, p, c) == tagged(A, (uint8_t)(c + 1)));
}

// A recalled chunk keeps its tier copy while clean, so paging it out again copies nothing; a row write makes the
// copy stale and the next page-out replaces it, so the following recall lands the new bytes.
TEST_CASE(KvflashPagerVictimTierFixture, a_write_after_a_recall_invalidates_the_tier_copy) {
    Harness A;
    KvFlashPager p;
    REQUIRE(p.attach(cfg(16), A.k, A.v));
    REQUIRE(p.victim_blocks() == 16);
    fill_tagged(A, p);
    p.score_hook = [](int c) { return c == 2 ? 100.0f : (float)c; };
    CHECK(p.reselect(0) == 2);                           // 2 in (tier hit), 9 out
    CHECK(p.stats().victim_hits == 1);
    REQUIRE(p.slot_for(2 * kChunk + 5) >= 0);            // a row write: the tier copy of 2 is stale now
    tag_k_rows(A, p, 2, 0xAB);
    p.score_hook = nullptr;
    CHECK(p.reselect_recent() == 2);                     // 2 out (copied again), 9 in
    p.score_hook = [](int c) { return c == 2 ? 100.0f : (float)c; };
    CHECK(p.reselect(0) == 2);                           // 2 back from the tier, with the written bytes
    CHECK(p.stats().victim_hits == 3);
    CHECK(k_rows(A, p, 2) == tagged(A, 0xAB));
}

// truncate drops the tier copies of the dropped chunks: the chunks written at the same ids afterwards recall their
// own bytes.
TEST_CASE(KvflashPagerVictimTierFixture, truncate_drops_tier_copies_of_dropped_chunks) {
    Harness A;
    KvFlashPager p;
    REQUIRE(p.attach(cfg(16), A.k, A.v));
    REQUIRE(p.victim_blocks() == 16);
    fill_tagged(A, p);
    p.truncate(5 * kChunk);                              // chunks 5..13 gone, 0..4 fit: no tier copy may survive for 5+
    for (int c = 5; c < 14; c++) {
        REQUIRE(p.alloc_span(c * kChunk, kChunk));
        tag_k_rows(A, p, c, (uint8_t)(c + 101));
    }
    p.score_hook = [](int c) { return c >= 5 && c <= 7 ? 100.0f : (float)c; };
    CHECK(p.reselect(0) == 4);                           // 7 is still resident: out 8,9, in 5,6
    for (int c = 5; c <= 7; c++) CHECK(k_rows(A, p, c) == tagged(A, (uint8_t)(c + 101)));
}

// CPU-backend tensors are paged by memcpy; on a GPU build they get no device tier (there is no device peer for
// them), and the pager says so. On a CPU build the tier is host memory and engages.
TEST_CASE(KvflashPagerVictimTierFixture, host_tensors_on_a_gpu_build_get_no_device_tier) {
    Harness A(/*on_device=*/false);
    KvFlashPager p;
    p.set_pinned_host_backing(false);  // CPU tensors: pageable staging, no cudaMallocHost (stub-driver CI)
    REQUIRE(p.attach(cfg(16), A.k, A.v));
#ifdef KVFLASH_HAS_ASYNC_DMA
    CHECK(p.victim_blocks() == 0);
    fill_tagged(A, p);
    p.score_hook = [](int c) { return c >= 1 && c <= 3 ? 100.0f : (float)c; };
    CHECK(p.reselect(0) == 6);                           // recalls still land, from the host copy
    CHECK(p.stats().victim_hits == 0);
    CHECK(p.stats().victim_misses == 3);
    for (int c = 1; c <= 3; c++) CHECK(k_rows(A, p, c) == tagged(A, (uint8_t)(c + 1)));
#else
    CHECK(p.victim_blocks() == 16);
#endif
}

// Every page-out queues a FIFO entry; entries whose slot was dropped or reassigned are dead. Thousands of
// page-out/recall cycles must not grow the queue past a small multiple of the tier.
TEST_CASE(KvflashPagerVictimTierFixture, the_fifo_does_not_grow_with_the_number_of_page_out_recall_cycles) {
    Harness A;
    KvFlashPager p;
    REQUIRE(p.attach(cfg(16), A.k, A.v));
    for (int c = 0; c < 8; c++) REQUIRE(p.alloc_span(c * kChunk, kChunk));
    for (int i = 0; i < 2000; i++) {
        p.page_out(1);              // takes a tier slot and queues an entry
        p.slot_for(kChunk);         // recall plus a row write: the slot is freed, the entry is dead
    }
    CHECK(p.victim_fifo_size() <= 3 * p.victim_blocks());
}
