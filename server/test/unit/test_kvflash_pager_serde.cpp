// Unit tests for KvFlashPager serialize/deserialize, floor_to_chunk, critical
// chunk pinning and the per-chunk score ledger (kvflash_pager.h). CPU ggml
// backend, no GPU: the cache tensors live in a CPU buffer, so every code path
// that moves bytes between pool and host backing is exercised for real.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_pager.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {

struct KvflashPagerSerdeFixture {};

constexpr int kHeadDim = 4, kPool = 512, kNHeadKv = 2, kNLayer = 2, kChunk = 64;
// KVFLASH1 blob header: 8-byte magic + 8 x uint32.
constexpr size_t kHdr = 8 + 8 * 4;
constexpr size_t kLedgerEntry = 8;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// pool 512 / chunk 64 = 8 blocks; sink 1 + tail 4 + 2 = 7 <= 8 blocks.
struct Harness {
    ggml_context *          ctx = nullptr;
    ggml_backend_t          backend = nullptr;
    ggml_backend_buffer_t   buf = nullptr;
    std::vector<ggml_tensor *> k, v;

    Harness() {
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = (size_t)(kNLayer * 2 + 8) * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < kNLayer; l++) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, kPool, kNHeadKv));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, kPool, kNHeadKv));
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }
    ~Harness() {
        if (buf)     ggml_backend_buffer_free(buf);
        if (ctx)     ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
    Harness(const Harness &) = delete;
    Harness & operator=(const Harness &) = delete;
};

KvFlashConfig make_cfg() {
    KvFlashConfig c;
    c.pool_tokens = kPool;
    c.chunk_tokens = kChunk;
    c.sink_chunks = 1;
    c.tail_window_chunks = 4;
    return c;
}

// REQUIRE is a fixture member; free helpers throw instead (the framework
// reports an exception as a test failure).
void must(bool ok) { if (!ok) throw std::runtime_error("kvflash serde helper precondition failed"); }

// Every (tensor, head) segment that belongs to block `b`, in a fixed order.
template <class F>
void for_each_segment(Harness & h, int block, F && fn) {
    for (int l = 0; l < kNLayer; l++) {
        for (int kv = 0; kv < 2; kv++) {
            ggml_tensor * t = kv == 0 ? h.k[l] : h.v[l];
            for (int hh = 0; hh < kNHeadKv; hh++) {
                const size_t off = (size_t)block * kChunk * t->nb[1] + (size_t)hh * t->nb[2];
                fn(t, off, (size_t)kChunk * t->nb[1]);
            }
        }
    }
}

// Deterministic per-(logical chunk, seed) byte pattern, never all-zero.
uint8_t pat(int c, int seed, size_t i) { return (uint8_t)(c * 53 + seed * 29 + i * 31 + 7); }

void stamp_chunk(Harness & h, const KvFlashPager & p, int c, int seed) {
    const int block = p.block_of(c);
    must(block >= 0);
    size_t n = 0;
    for_each_segment(h, block, [&](ggml_tensor * t, size_t off, size_t len) {
        std::vector<uint8_t> b(len);
        for (size_t i = 0; i < len; i++) b[i] = pat(c, seed, n + i);
        ggml_backend_tensor_set(t, b.data(), off, len);
        n += len;
    });
}

bool chunk_matches(Harness & h, const KvFlashPager & p, int c, int seed) {
    const int block = p.block_of(c);
    if (block < 0) return false;
    size_t n = 0;
    bool ok = true;
    for_each_segment(h, block, [&](ggml_tensor * t, size_t off, size_t len) {
        std::vector<uint8_t> b(len);
        ggml_backend_tensor_get(t, b.data(), off, len);
        for (size_t i = 0; i < len; i++) ok = ok && b[i] == pat(c, seed, n + i);
        n += len;
    });
    return ok;
}

// A: 6 chunks (0..5) stamped, chunks 2 and 3 paged out, chunk 2 recalled into a
// different block (non-identity), chunk 3 left on host. Scores 10*c on 0..5.
void build_source(Harness & A, KvFlashPager & pa, int seed) {
    must(pa.attach(make_cfg(), A.k, A.v));
    must(pa.alloc_span(0, 6 * kChunk));
    for (int c = 0; c < 6; c++) stamp_chunk(A, pa, c, seed);
    must(pa.page_out(2));
    must(pa.page_out(3));
    must(pa.page_in(2));
    pa.synchronize_paging();
    for (int c = 0; c < 6; c++) pa.set_chunk_score(c, 10.0f * (float)c);
}

} // namespace

TEST_CASE(KvflashPagerSerdeFixture, floor_to_chunk_aligns_down) {
    CHECK(floor_to_chunk(0, 64) == 0);
    CHECK(floor_to_chunk(63, 64) == 0);
    CHECK(floor_to_chunk(64, 64) == 64);
    CHECK(floor_to_chunk(65, 64) == 64);
    CHECK(floor_to_chunk(127, 64) == 64);
    CHECK(floor_to_chunk(34077, 64) == 34048);
}

TEST_CASE(KvflashPagerSerdeFixture, roundtrip_preserves_residency_bytes_and_host_copies) {
    Harness A;
    KvFlashPager pa;
    build_source(A, pa, 1);
    REQUIRE(pa.block_of(2) >= 0 && pa.block_of(2) != 2);  // relocated, not identity
    REQUIRE(!pa.is_resident(3));                          // paged out at serialize time

    const std::vector<uint8_t> blob = pa.serialize();
    REQUIRE(blob.size() == kHdr + 6 * (size_t)(kChunk * kHeadDim * 2 * kNHeadKv * kNLayer * 2) +
                           6 * kLedgerEntry);
    REQUIRE(std::memcmp(blob.data(), "1HSALFVK", 8) == 0);  // "KVFLASH1" little-endian u64

    Harness B;
    KvFlashPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    // Pre-existing state must be discarded by deserialize.
    REQUIRE(pb.alloc_span(0, 2 * kChunk));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));

    REQUIRE(pb.n_chunks() == 6);
    for (int c = 0; c < 6; c++) CHECK(pb.is_resident(c) == pa.is_resident(c));
    CHECK(!pb.is_resident(3));
    CHECK(pb.resident_blocks() == 5);
    for (int c : {0, 1, 2, 4, 5}) CHECK(chunk_matches(B, pb, c, 1));
    // Positions: logical -> slot lands inside the chunk's block, in order.
    for (int c : {0, 1, 2, 4, 5}) {
        for (int i : {0, 17, kChunk - 1}) {
            CHECK(pb.slot_of((int64_t)c * kChunk + i) == pb.block_of(c) * kChunk + i);
        }
    }
    CHECK(pb.slot_of(3 * kChunk) == -1);
    // The host copy of chunk 3 survived: recalling it yields the source bytes.
    REQUIRE(pb.page_in(3));
    pb.synchronize_paging();
    CHECK(chunk_matches(B, pb, 3, 1));
    // Source pager is untouched by serialize().
    CHECK(!pa.is_resident(3));
    CHECK(pa.n_chunks() == 6);
}

TEST_CASE(KvflashPagerSerdeFixture, roundtrip_of_all_host_backed_chunks) {
    Harness A;
    KvFlashPager pa;
    build_source(A, pa, 2);
    for (int c : {0, 1, 2, 4, 5}) REQUIRE(pa.page_out(c));
    REQUIRE(pa.resident_blocks() == 0);
    const std::vector<uint8_t> blob = pa.serialize();

    Harness B;
    KvFlashPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    CHECK(pb.resident_blocks() == 0);
    CHECK(pb.n_chunks() == 6);
    for (int c = 0; c < 6; c++) {
        REQUIRE(pb.page_in(c));
        pb.synchronize_paging();
        CHECK(chunk_matches(B, pb, c, 2));
    }
}

TEST_CASE(KvflashPagerSerdeFixture, partial_serialize_keeps_only_first_chunks) {
    Harness A;
    KvFlashPager pa;
    build_source(A, pa, 3);
    const std::vector<uint8_t> full = pa.serialize();
    const std::vector<uint8_t> part = pa.serialize(4);
    REQUIRE(part.size() < full.size());

    Harness B;
    KvFlashPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(part.data(), part.size()));
    CHECK(pb.n_chunks() == 4);
    for (int c : {0, 1, 2}) { CHECK(pb.is_resident(c)); CHECK(chunk_matches(B, pb, c, 3)); }
    CHECK(!pb.is_resident(3));
    CHECK(pb.block_of(4) == -1);
    CHECK(pb.slot_of(4 * kChunk) == -1);
    REQUIRE(pb.page_in(3));
    pb.synchronize_paging();
    CHECK(chunk_matches(B, pb, 3, 3));

    // max_chunks beyond n_chunks and the default both mean "everything".
    CHECK(pa.serialize(100) == full);
    CHECK(pa.serialize(-1) == full);
    // max_chunks == 0 is a valid, empty snapshot.
    const std::vector<uint8_t> none = pa.serialize(0);
    CHECK(none.size() == kHdr);
    REQUIRE(pb.deserialize(none.data(), none.size()));
    CHECK(pb.n_chunks() == 0);
    CHECK(pb.resident_blocks() == 0);
}

TEST_CASE(KvflashPagerSerdeFixture, ledger_scores_roundtrip) {
    Harness A;
    KvFlashPager pa;
    build_source(A, pa, 4);
    pa.set_chunk_score(4, -3.5f);
    CHECK(pa.chunk_score(5) == 50.0f);
    CHECK(pa.chunk_score(99) == kNegInf);   // out of range: unset sentinel
    pa.set_chunk_score(99, 1.0f);           // out of range: ignored, no crash
    pa.set_chunk_score(-1, 1.0f);

    const std::vector<uint8_t> blob = pa.serialize();
    Harness B;
    KvFlashPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    for (int c = 0; c < 6; c++) CHECK(pb.chunk_score(c) == pa.chunk_score(c));
    CHECK(pb.chunk_score(3) == 30.0f);      // host-backed chunk keeps its score
    CHECK(pb.chunk_score(4) == -3.5f);
    // A chunk never scored stays at the unset sentinel through the roundtrip.
    Harness C;
    KvFlashPager pc;
    REQUIRE(pc.attach(make_cfg(), C.k, C.v));
    REQUIRE(pc.alloc_span(0, 2 * kChunk));
    const std::vector<uint8_t> blob2 = pc.serialize();
    REQUIRE(pb.deserialize(blob2.data(), blob2.size()));
    CHECK(pb.chunk_score(0) == kNegInf);
    CHECK(pb.chunk_score(1) == kNegInf);
}

TEST_CASE(KvflashPagerSerdeFixture, reset_pager_is_identity_so_dense_snapshots_ignore_stale_mapping) {
    Harness A;
    KvFlashPager pa;
    build_source(A, pa, 3);
    REQUIRE(!pa.is_identity());   // previous request left a paged mapping
    pa.reset();
    CHECK(pa.is_identity());
    CHECK(pa.n_chunks() == 0);
}

TEST_CASE(KvflashPagerSerdeFixture, serialized_dtype_is_recorded) {
    Harness A;
    KvFlashPager pa;
    build_source(A, pa, 5);
    const std::vector<uint8_t> blob = pa.serialize();
    Harness B;
    KvFlashPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    CHECK(pb.serialized_kv_k_type() == (uint32_t)GGML_TYPE_F16);
}

TEST_CASE(KvflashPagerSerdeFixture, malformed_blobs_are_rejected_without_touching_state) {
    Harness A;
    KvFlashPager pa;
    build_source(A, pa, 6);
    const std::vector<uint8_t> good = pa.serialize();

    Harness B;
    KvFlashPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.alloc_span(0, 3 * kChunk));
    for (int c = 0; c < 3; c++) stamp_chunk(B, pb, c, 9);
    const uint64_t epoch0 = pb.epoch();

    auto rejected = [&](const std::vector<uint8_t> & blob) {
        return !pb.deserialize(blob.data(), blob.size());
    };
    auto untouched = [&]() {
        bool ok = pb.n_chunks() == 3 && pb.epoch() == epoch0;
        for (int c = 0; c < 3; c++) ok = ok && pb.is_resident(c) && chunk_matches(B, pb, c, 9);
        return ok;
    };

    // Empty / null / short at every structural boundary.
    CHECK(!pb.deserialize(nullptr, 0));
    CHECK(rejected(std::vector<uint8_t>{}));
    for (size_t len : {size_t(1), size_t(7), size_t(8), kHdr - 1, kHdr, kHdr + 1,
                       good.size() / 2, good.size() - 1}) {
        CHECK(rejected(std::vector<uint8_t>(good.begin(), good.begin() + len)));
    }
    CHECK(untouched());

    { auto b = good; b[0] ^= 0xFF;  CHECK(rejected(b)); }     // wrong magic
    { auto b = good; b[7] ^= 0x01;  CHECK(rejected(b)); }     // wrong magic (last byte)
    { auto b = good; b[8 + 4] ^= 0xFF; CHECK(rejected(b)); }  // chunk_tokens mismatch
    { auto b = good; b[8 + 8] ^= 0xFF; CHECK(rejected(b)); }  // n_head_kv mismatch
    { auto b = good; b[8 + 12] ^= 0xFF; CHECK(rejected(b)); } // k_seg_bytes mismatch
    { auto b = good; b[8 + 20] ^= 0xFF; CHECK(rejected(b)); } // chunk_bytes mismatch
    { auto b = good; b[8 + 24] ^= 0x01; CHECK(rejected(b)); } // KV dtype mismatch
    { auto b = good; b[8 + 28] = 7;     CHECK(rejected(b)); } // unknown ledger flag
    { auto b = good; std::memset(&b[8], 0xFF, 4); CHECK(rejected(b)); }  // nc = 0xFFFFFFFF (overflow)
    { auto b = good; b[8] = 7;          CHECK(rejected(b)); } // nc claims more chunks than the blob holds
    {   // ledger claims more resident chunks (40) than the pool has blocks (8)
        const size_t cb = (good.size() - kHdr - 6 * kLedgerEntry) / 6;
        std::vector<uint8_t> b(good.begin(), good.begin() + kHdr);
        b[8] = 40;  // nc = 40 (little-endian u32)
        b.resize(kHdr + 40 * cb + 40 * kLedgerEntry, 0);
        for (size_t c = 0; c < 40; c++) b[kHdr + 40 * cb + c * kLedgerEntry] = 1;
        CHECK(rejected(b));
    }
    CHECK(untouched());

    // Trailing bytes after a valid blob are tolerated; the blob still loads.
    { auto b = good; b.push_back(0xAB); CHECK(pb.deserialize(b.data(), b.size())); }
    CHECK(pb.n_chunks() == 6);
}

TEST_CASE(KvflashPagerSerdeFixture, pin_range_is_half_open_and_pinned_chunks_are_never_evicted) {
    Harness A;
    KvFlashPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));

    // [64, 128) is exactly chunk 1; chunk 2 (token 128) is outside.
    p.pin_range(1 * kChunk, 2 * kChunk);
    CHECK(!p.is_pinned(0));
    CHECK(p.is_pinned(1));
    CHECK(!p.is_pinned(2));
    // A range ending mid-chunk pins that chunk too; empty / inverted is a no-op.
    p.unpin_all();
    p.pin_range(kChunk, kChunk + 1);
    CHECK(p.is_pinned(1));
    CHECK(!p.is_pinned(2));
    p.unpin_all();
    p.pin_range(2 * kChunk, 2 * kChunk);
    p.pin_range(3 * kChunk, 2 * kChunk);
    p.pin_range(-5, 0);
    for (int c = 0; c < 8; c++) CHECK(!p.is_pinned(c));
    CHECK(!p.is_pinned(-1));
    CHECK(!p.is_pinned(1000));

    // Deadlock guard: sink 1 + tail 4 + pinned + 2 must fit in 8 blocks, so
    // at most one chunk can be pinned. A 2-chunk span is refused whole.
    p.pin_range(1 * kChunk, 3 * kChunk);
    CHECK(!p.is_pinned(1));
    CHECK(!p.is_pinned(2));
    p.pin_range(1 * kChunk, 2 * kChunk);
    CHECK(p.is_pinned(1));
    p.pin_range(2 * kChunk, 3 * kChunk);   // a second pin would deadlock: refused
    CHECK(!p.is_pinned(2));
    p.pin_range(1 * kChunk, 2 * kChunk);   // re-pinning an already pinned chunk is fine
    CHECK(p.is_pinned(1));

    // Drive eviction far past the pool: chunk 1 stays resident throughout,
    // with both LRU and score-driven victim selection.
    REQUIRE(p.alloc_span(0, 8 * kChunk));
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) p.score_hook = [](int c) { return c == 1 ? -1e9f : (float)c; };
        for (int c = 8 + pass * 8; c < 16 + pass * 8; c++) {
            REQUIRE(p.alloc_span(c * kChunk, kChunk));
            CHECK(p.is_resident(1));
            CHECK(p.is_resident(0));  // sink
        }
    }
    // reselect() with a score hook that hates chunk 1 still keeps it.
    p.score_hook = [](int c) { return c == 1 ? -1e9f : (float)c; };
    p.reselect();
    CHECK(p.is_resident(1));

    // reset() clears pins.
    p.reset();
    CHECK(!p.is_pinned(1));
}

TEST_CASE(KvflashPagerSerdeFixture, no_new_api_use_keeps_eviction_unchanged) {
    // Without pins or scores the victim is the oldest unprotected chunk.
    Harness A;
    KvFlashPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 8 * kChunk));
    REQUIRE(p.alloc_span(8 * kChunk, kChunk));   // evicts chunk 1 (sink 0 protected)
    CHECK(p.is_resident(0));
    CHECK(!p.is_resident(1));
    CHECK(p.is_resident(8));
}
