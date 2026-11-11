// Unit tests for KvFlashPager serialize/deserialize, floor_to_chunk, critical
// chunk pinning and the per-chunk score ledger (kvflash_pager.h). CPU ggml
// backend, no GPU: the cache tensors live in a CPU buffer, so every code path
// that moves bytes between pool and host backing is exercised for real.
#include "CppUnitTestFramework.hpp"
#include "kvflash_pager_test_host.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {

struct KvflashPagerSerdeFixture {};

constexpr int kHeadDim = 4, kPool = 512, kNHeadKv = 2, kNLayer = 2, kChunk = 64;
// KVFLASH3 blob header: 8-byte magic + 9 x uint32. After the KV bytes and the
// ledger comes the pool layout: 16-byte header, 16 bytes per chunk, 4 per free block.
constexpr size_t kHdr = 8 + 9 * 4;
constexpr size_t kLedgerEntry = 8;
constexpr size_t kPoolHdr = 16, kPoolEntry = 16;

// chunk_bytes as the blob header records it (header field 5).
size_t blob_chunk_bytes(const std::vector<uint8_t> & b) {
    uint32_t cb = 0;
    std::memcpy(&cb, b.data() + 8 + 5 * 4, 4);
    return cb;
}
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// pool 512 / chunk 64 = 8 blocks; sink 1 + tail 4 + 2 = 7 <= 8 blocks.
struct Harness {
    ggml_context *          ctx = nullptr;
    ggml_backend_t          backend = nullptr;
    ggml_backend_buffer_t   buf = nullptr;
    std::vector<ggml_tensor *> k, v;

    explicit Harness(ggml_type vtype = GGML_TYPE_F16, int pool = kPool) {
        backend = ggml_backend_cpu_init();
        ggml_init_params ip{};
        ip.mem_size = (size_t)(kNLayer * 2 + 8) * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        for (int l = 0; l < kNLayer; l++) {
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, pool, kNHeadKv));
            v.push_back(ggml_new_tensor_3d(ctx, vtype, kHeadDim, pool, kNHeadKv));
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

// Deterministic per-(logical chunk, seed, byte offset in chunk) pattern. `i` is
// the offset across the whole chunk, so every segment gets different bytes (a
// linear pattern with a short period repeated identically in each segment).
uint8_t pat(int c, int seed, size_t i) {
    uint32_t x = (uint32_t)c * 2654435761u ^ (uint32_t)seed * 40503u ^ (uint32_t)i * 2246822519u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    return (uint8_t)(x >> 24) | 1u;   // never zero: a zeroed segment is detected too
}

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
    TestPager pa;
    build_source(A, pa, 1);
    REQUIRE(pa.block_of(2) >= 0 && pa.block_of(2) != 2);  // relocated, not identity
    REQUIRE(!pa.is_resident(3));                          // paged out at serialize time

    const std::vector<uint8_t> blob = pa.serialize();
    // 5 resident chunks in an 8-block pool leave 3 free blocks.
    REQUIRE(blob.size() == kHdr + 6 * (size_t)(kChunk * kHeadDim * 2 * kNHeadKv * kNLayer * 2) +
                           6 * kLedgerEntry + kPoolHdr + 6 * kPoolEntry + 3 * 4);
    REQUIRE(std::memcmp(blob.data(), "3HSALFVK", 8) == 0);  // "KVFLASH3" little-endian u64

    Harness B;
    TestPager pb;
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
    TestPager pa;
    build_source(A, pa, 2);
    for (int c : {0, 1, 2, 4, 5}) REQUIRE(pa.page_out(c));
    REQUIRE(pa.resident_blocks() == 0);
    const std::vector<uint8_t> blob = pa.serialize();

    Harness B;
    TestPager pb;
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
    TestPager pa;
    build_source(A, pa, 3);
    const std::vector<uint8_t> full = pa.serialize();
    const std::vector<uint8_t> part = pa.serialize(4);
    REQUIRE(part.size() < full.size());

    Harness B;
    TestPager pb;
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
    CHECK(none.size() == kHdr + kPoolHdr + 8 * 4);   // no chunks, all 8 blocks free
    REQUIRE(pb.deserialize(none.data(), none.size()));
    CHECK(pb.n_chunks() == 0);
    CHECK(pb.resident_blocks() == 0);
}

TEST_CASE(KvflashPagerSerdeFixture, ledger_scores_roundtrip) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 4);
    pa.set_chunk_score(4, -3.5f);
    CHECK(pa.chunk_score(5) == 50.0f);
    CHECK(pa.chunk_score(99) == kNegInf);   // out of range: unset sentinel
    pa.set_chunk_score(99, 1.0f);           // out of range: ignored, no crash
    pa.set_chunk_score(-1, 1.0f);

    const std::vector<uint8_t> blob = pa.serialize();
    Harness B;
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    for (int c = 0; c < 6; c++) CHECK(pb.chunk_score(c) == pa.chunk_score(c));
    CHECK(pb.chunk_score(3) == 30.0f);      // host-backed chunk keeps its score
    CHECK(pb.chunk_score(4) == -3.5f);
    // A chunk never scored stays at the unset sentinel through the roundtrip.
    Harness C;
    TestPager pc;
    REQUIRE(pc.attach(make_cfg(), C.k, C.v));
    REQUIRE(pc.alloc_span(0, 2 * kChunk));
    const std::vector<uint8_t> blob2 = pc.serialize();
    REQUIRE(pb.deserialize(blob2.data(), blob2.size()));
    CHECK(pb.chunk_score(0) == kNegInf);
    CHECK(pb.chunk_score(1) == kNegInf);
}

TEST_CASE(KvflashPagerSerdeFixture, reset_pager_is_identity_so_dense_snapshots_ignore_stale_mapping) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 3);
    REQUIRE(!pa.is_identity());   // previous request left a paged mapping
    pa.reset();
    CHECK(pa.is_identity());
    CHECK(pa.n_chunks() == 0);
}

TEST_CASE(KvflashPagerSerdeFixture, serialized_dtype_is_recorded) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 5);
    const std::vector<uint8_t> blob = pa.serialize();
    Harness B;
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    CHECK(pb.serialized_kv_k_type() == (uint32_t)GGML_TYPE_F16);
}

TEST_CASE(KvflashPagerSerdeFixture, deserialize_rejects_a_changed_v_dtype_and_the_old_format) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 7);
    const std::vector<uint8_t> blob = pa.serialize();

    // Same K, V changed F16 -> BF16: identical byte geometry, different values.
    Harness B(GGML_TYPE_BF16);
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    CHECK(!pb.deserialize(blob.data(), blob.size()));
    CHECK(pb.n_chunks() == 0);

    // KVFLASH1 (8-field header) is refused. KVFLASH2 is this blob without the
    // pool layout, so the same bytes under that magic load (the layout becomes
    // tolerated trailing bytes); kvflash2_blob_loads_in_identity_order covers it.
    Harness C;
    TestPager pc;
    REQUIRE(pc.attach(make_cfg(), C.k, C.v));
    auto old = blob;
    std::memcpy(old.data(), "1HSALFVK", 8);
    CHECK(!pc.deserialize(old.data(), old.size()));
    CHECK(pc.n_chunks() == 0);
    CHECK(pc.deserialize(blob.data(), blob.size()));
}

TEST_CASE(KvflashPagerSerdeFixture, malformed_blobs_are_rejected_without_touching_state) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 6);
    const std::vector<uint8_t> good = pa.serialize();

    Harness B;
    TestPager pb;
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
    { auto b = good; b[8 + 32] ^= 0x01; CHECK(rejected(b)); } // V dtype mismatch
    { auto b = good; b[8 + 28] = 7;     CHECK(rejected(b)); } // unknown ledger flag
    { auto b = good; std::memset(&b[8], 0xFF, 4); CHECK(rejected(b)); }  // nc = 0xFFFFFFFF (overflow)
    { auto b = good; b[8] = 7;          CHECK(rejected(b)); } // nc claims more chunks than the blob holds
    {   // ledger claims more resident chunks (40) than the pool has blocks (8)
        const size_t cb = blob_chunk_bytes(good);
        std::vector<uint8_t> b(good.begin(), good.begin() + kHdr);
        b[8] = 40;  // nc = 40 (little-endian u32)
        b.resize(kHdr + 40 * cb + 40 * kLedgerEntry + kPoolHdr + 40 * kPoolEntry, 0);
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
    TestPager p;
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
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 8 * kChunk));
    REQUIRE(p.alloc_span(8 * kChunk, kChunk));   // evicts chunk 1 (sink 0 protected)
    CHECK(p.is_resident(0));
    CHECK(!p.is_resident(1));
    CHECK(p.is_resident(8));
}

TEST_CASE(KvflashPagerSerdeFixture, attach_clears_pins_from_a_previous_attach) {
    // 16-block pool: the deadlock guard allows 9 pins (1 sink + 4 tail + 9 + 2 <= 16).
    Harness Big(GGML_TYPE_F16, 2 * kPool);
    KvFlashConfig big = make_cfg();
    big.pool_tokens = 2 * kPool;
    TestPager p;
    REQUIRE(p.attach(big, Big.k, Big.v));
    p.pin_range(1 * kChunk, 9 * kChunk);   // chunks 1..8
    REQUIRE(p.is_pinned(1));
    REQUIRE(p.is_pinned(8));

    // Re-attach to the 8-block pool: stale pins would protect every chunk and
    // starve allocation (sink + 7 pinned = the whole pool, no victim left).
    Harness Small;
    REQUIRE(p.attach(make_cfg(), Small.k, Small.v));
    for (int c = 0; c < 9; c++) CHECK(!p.is_pinned(c));
    CHECK(p.alloc_span(0, 12 * kChunk));
}

TEST_CASE(KvflashPagerSerdeFixture, pin_range_rejects_positions_beyond_the_chunk_index_range) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    // Chunk index 2^31 used to narrow to a negative int, skip the resize and
    // write pinned_ out of bounds.
    const int64_t c31 = (int64_t)1 << 31;
    p.pin_range(c31 * kChunk, c31 * kChunk + 1);
    p.pin_range(c31 * kChunk - 1, c31 * kChunk + 1);
    // Huge span: bounded scan, refused whole (it could never fit the pool).
    p.pin_range(0, (int64_t)1 << 40);
    p.pin_range(0, std::numeric_limits<int64_t>::max());
    // In int range but absurd: must not allocate a gigabyte of pin flags.
    p.pin_range((int64_t)1 << 36, ((int64_t)1 << 36) + 1);
    for (int c : {0, 1, 7, 1 << 20, 1 << 30, std::numeric_limits<int>::max()}) CHECK(!p.is_pinned(c));
    CHECK(!p.is_pinned(-1));
    // Valid pins still work afterwards.
    p.pin_range(1 * kChunk, 2 * kChunk);
    CHECK(p.is_pinned(1));
}

TEST_CASE(KvflashPagerSerdeFixture, never_materialized_holes_stay_holes_through_a_roundtrip) {
    Harness A;
    TestPager pa;
    REQUIRE(pa.attach(make_cfg(), A.k, A.v));
    REQUIRE(pa.alloc_span(0, 2 * kChunk));
    for (int c = 0; c < 2; c++) stamp_chunk(A, pa, c, 8);
    REQUIRE(pa.alloc_span(6 * kChunk, 1));   // jump: chunks 2..5 are holes, never written
    stamp_chunk(A, pa, 6, 8);
    REQUIRE(pa.n_chunks() == 7);
    for (int c = 2; c < 6; c++) REQUIRE(pa.block_of(c) == -1);
    const std::vector<uint8_t> blob = pa.serialize();

    Harness B;
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    // A hole is not recallable zero KV, and holds no host copy.
    for (int c = 2; c < 6; c++) CHECK(!pb.page_in(c));
    CHECK(pb.stats().host_bytes == 3 * (int64_t)blob_chunk_bytes(blob));
    // reselect() must not pick them either, however well they score.
    pb.score_hook = [](int c) { return c >= 2 && c < 6 ? 1e6f : (float)c; };
    pb.reselect();
    for (int c = 2; c < 6; c++) CHECK(!pb.is_resident(c));
    for (int c : {0, 1, 6}) { CHECK(pb.is_resident(c)); CHECK(chunk_matches(B, pb, c, 8)); }
    // The hole survives a second serialize unchanged.
    pb.score_hook = nullptr;
    CHECK(pb.serialize() == blob);

    // A blob claiming a resident chunk that was never materialized is corrupt.
    auto bad = blob;
    const size_t cb = blob_chunk_bytes(blob);
    bad[kHdr + 7 * cb + 3 * kLedgerEntry] = 1;       // chunk 3 "resident"...
    bad[kHdr + 7 * cb + 3 * kLedgerEntry + 1] = 0;   // ...but not materialized
    CHECK(!pb.deserialize(bad.data(), bad.size()));
}

// The fill pattern must tell segments apart: a segment is 512 bytes here, so a
// pattern that repeats every 256 bytes made whole K/V/head/layer segments
// interchangeable and a swapped-segment bug invisible.
TEST_CASE(KvflashPagerSerdeFixture, fill_pattern_detects_swapped_segments) {
    Harness A;
    TestPager pa;
    REQUIRE(pa.attach(make_cfg(), A.k, A.v));
    REQUIRE(pa.alloc_span(0, kChunk));
    struct Seg { ggml_tensor * t; size_t off, len; };
    std::vector<Seg> segs;
    for_each_segment(A, pa.block_of(0), [&](ggml_tensor * t, size_t off, size_t len) {
        segs.push_back({t, off, len});
    });
    REQUIRE(segs.size() == (size_t)(kNLayer * 2 * kNHeadKv));
    // Swap segments i and j in place, check, swap back.
    auto swapped_is_caught = [&](size_t i, size_t j) {
        stamp_chunk(A, pa, 0, 3);
        std::vector<uint8_t> a(segs[i].len), b(segs[j].len);
        ggml_backend_tensor_get(segs[i].t, a.data(), segs[i].off, a.size());
        ggml_backend_tensor_get(segs[j].t, b.data(), segs[j].off, b.size());
        ggml_backend_tensor_set(segs[i].t, b.data(), segs[i].off, b.size());
        ggml_backend_tensor_set(segs[j].t, a.data(), segs[j].off, a.size());
        return !chunk_matches(A, pa, 0, 3);
    };
    stamp_chunk(A, pa, 0, 3);
    REQUIRE(chunk_matches(A, pa, 0, 3));
    for (size_t i = 0; i < segs.size(); i++)
        for (size_t j = i + 1; j < segs.size(); j++)
            CHECK(swapped_is_caught(i, j));   // head, K<->V and layer swaps alike
}

TEST_CASE(KvflashPagerSerdeFixture, pageable_host_backing_pages_out_and_in_bit_exact) {
    // Explicit pageable mode: the path hosted CI (no pinned memory) relies on.
    Harness A;
    KvFlashPager p;
    p.set_pinned_host_backing(false);
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 3 * kChunk));
    for (int c = 0; c < 3; c++) stamp_chunk(A, p, c, 11);
    REQUIRE(p.page_out(1));
    REQUIRE(!p.is_resident(1));
    CHECK(p.stats().host_bytes > 0);
    REQUIRE(p.page_in(1));
    p.synchronize_paging();
    CHECK(chunk_matches(A, p, 1, 11));
    const std::vector<uint8_t> blob = p.serialize();
    Harness B;
    KvFlashPager q;
    q.set_pinned_host_backing(false);
    REQUIRE(q.attach(make_cfg(), B.k, B.v));
    REQUIRE(q.deserialize(blob.data(), blob.size()));
    for (int c = 0; c < 3; c++) CHECK(chunk_matches(B, q, c, 11));
    q.truncate(kChunk);   // frees pageable host buffers; reset frees the rest
    CHECK(q.n_chunks() == 1);
}

TEST_CASE(KvflashPagerSerdeFixture, blob_fields_are_little_endian) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 2 * kChunk));
    p.set_chunk_score(1, 1.0f);   // IEEE-754 0x3f800000
    const std::vector<uint8_t> blob = p.serialize();
    const size_t cb = blob_chunk_bytes(blob);
    const uint8_t nc_le[4] = {2, 0, 0, 0}, chunk_tokens_le[4] = {kChunk, 0, 0, 0};
    CHECK(std::memcmp(&blob[8], nc_le, 4) == 0);
    CHECK(std::memcmp(&blob[12], chunk_tokens_le, 4) == 0);
    const uint8_t one_le[4] = {0x00, 0x00, 0x80, 0x3f};
    CHECK(std::memcmp(&blob[kHdr + 2 * cb + kLedgerEntry + 4], one_le, 4) == 0);
}

// A restored pager is the saved one: same block per chunk, same free list and
// LRU order, so from the same point it evicts the same victims and puts new
// chunks in the same blocks as the source. Attention then reads the same slots
// in the same order, which is what lets a suffix-only restore match a cold
// prefill byte for byte.
TEST_CASE(KvflashPagerSerdeFixture, roundtrip_keeps_blocks_and_lru_so_later_evictions_match) {
    Harness A, B;
    TestPager pa, pb;
    build_source(A, pa, 4);                       // chunk 2 relocated, chunk 3 on host
    must(pa.alloc_span(6 * kChunk, 2 * kChunk));  // 7 resident, 1 block free
    for (int c = 6; c < 8; c++) stamp_chunk(A, pa, c, 4);
    const std::vector<uint8_t> blob = pa.serialize();
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.alloc_span(0, 3 * kChunk));        // state that deserialize must replace
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    CHECK(pb.serialize() == blob);
    auto same_layout = [&]() {
        bool ok = pa.n_chunks() == pb.n_chunks();
        for (int c = 0; ok && c < pa.n_chunks(); c++) ok = pa.block_of(c) == pb.block_of(c);
        return ok;
    };
    CHECK(same_layout());
    for (int c = 0; c < 8; c++) if (pa.is_resident(c)) CHECK(chunk_matches(B, pb, c, 4));
    // Six more chunks: one into the free block, then five LRU evictions.
    for (int c = 8; c < 14; c++) {
        must(pa.alloc_span(c * kChunk, kChunk));
        must(pb.alloc_span(c * kChunk, kChunk));
        stamp_chunk(A, pa, c, 4);
        stamp_chunk(B, pb, c, 4);
        CHECK(same_layout());
    }
    CHECK(pa.serialize() == pb.serialize());
}

// Into a pool of a different size the saved blocks do not apply: the chunks
// load in identity order, as KVFLASH2 did, with the same residency and bytes.
TEST_CASE(KvflashPagerSerdeFixture, roundtrip_into_another_pool_size_keeps_residency) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 5);
    const std::vector<uint8_t> blob = pa.serialize();
    Harness B(GGML_TYPE_F16, 2 * kPool);
    TestPager pb;
    KvFlashConfig cfg = make_cfg();
    cfg.pool_tokens = 2 * kPool;
    REQUIRE(pb.attach(cfg, B.k, B.v));
    REQUIRE(pb.deserialize(blob.data(), blob.size()));
    CHECK(pb.n_chunks() == 6);
    CHECK(pb.resident_blocks() == 5);
    int next = 0;
    for (int c = 0; c < 6; c++) {
        CHECK(pb.is_resident(c) == pa.is_resident(c));
        if (!pb.is_resident(c)) continue;
        CHECK(pb.block_of(c) == next++);
        CHECK(chunk_matches(B, pb, c, 5));
    }
}
TEST_CASE(KvflashPagerSerdeFixture, reselect_recalls_top_scored_host_chunk_and_keeps_tail) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 12 * kChunk));       // 8 blocks: 4 chunks go to host
    int cold = -1;
    for (int c = 1; c < 8 && cold < 0; c++) if (!p.is_resident(c)) cold = c;
    REQUIRE(cold > 0);
    p.score_hook = [cold](int c) { return c == cold ? 1e9f : (float)-c; };
    CHECK(p.reselect() > 0);
    CHECK(p.is_resident(cold));
    CHECK(p.is_resident(0));                      // sink
    for (int c = 8; c < 12; c++) CHECK(p.is_resident(c));   // tail window
}

TEST_CASE(KvflashPagerSerdeFixture, reselect_tail_override_protects_only_n_chunks) {
    auto run = [](int tail_override) {
        Harness A;
        TestPager p;
        if (!p.attach(make_cfg(), A.k, A.v) || !p.alloc_span(0, 12 * kChunk)) return std::vector<int>{};
        // Tail chunks score lowest; old chunks score by index.
        p.score_hook = [](int c) { return c >= 8 ? -1000.0f + c : (float)c; };
        if (tail_override < 0) p.reselect(); else p.reselect(tail_override);
        std::vector<int> res;
        for (int c = 0; c < 12; c++) if (p.is_resident(c)) res.push_back(c);
        return res;
    };
    // Default (cfg tail = 4): 8..11 protected.
    const std::vector<int> def = run(-1);
    REQUIRE(!def.empty());
    for (int c = 8; c < 12; c++) CHECK(std::find(def.begin(), def.end(), c) != def.end());
    // Override 1: protects c > cur-1-1, i.e. the current chunk 11 plus 10; 8, 9 lose to the higher-scored old chunks.
    const std::vector<int> two = run(1);
    CHECK(std::find(two.begin(), two.end(), 10) != two.end());
    CHECK(std::find(two.begin(), two.end(), 11) != two.end());
    CHECK(std::find(two.begin(), two.end(), 8) == two.end());
    CHECK(std::find(two.begin(), two.end(), 9) == two.end());
    CHECK(std::find(two.begin(), two.end(), 0) != two.end());   // sink
}

TEST_CASE(KvflashPagerSerdeFixture, host_k_seg_points_at_the_paged_out_k_rows) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 8 * kChunk));
    for (int c = 0; c < 8; c++) stamp_chunk(A, p, c, 3);
    REQUIRE(p.alloc_span(8 * kChunk, 2 * kChunk));       // pages two chunks out
    int cold = -1;
    for (int c = 1; c < 8 && cold < 0; c++) if (!p.is_resident(c)) cold = c;
    REQUIRE(cold > 0);
    CHECK(p.host_k_seg(0, 0, 0) == nullptr);              // resident: no host copy to read
    CHECK(p.cur_chunk() == 9);
    CHECK(p.n_pool_blocks() == 8);
    const size_t len = p.k_seg_bytes();
    CHECK(len == (size_t)kChunk * kHeadDim * 2);
    for (int l = 0; l < kNLayer; l++) {
        for (int h = 0; h < kNHeadKv; h++) {
            const uint8_t * s = p.host_k_seg(cold, l, h);
            REQUIRE(s != nullptr);
            const size_t n = (size_t)((l * 2 + 0) * kNHeadKv + h) * len;   // stamp order: layer, K|V, head
            bool same = true;
            for (size_t i = 0; i < len; i++) same = same && s[i] == pat(cold, 3, n + i);
            CHECK(same);
        }
    }
}

TEST_CASE(KvflashPagerSerdeFixture, reselect_with_full_tail_keeps_a_non_recent_resident_set_exactly) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 12 * kChunk));
    // Move the pool to a resident set that is NOT the newest chunks.
    p.score_hook = [](int c) { return (float)-c; };
    p.reselect(0);                                  // sink + current chunk + oldest-scored-best chunks 1..6
    std::vector<int> before;
    for (int c = 0; c < 12; c++) if (p.is_resident(c)) before.push_back(c);
    REQUIRE(before.size() == 8);
    CHECK(!p.is_resident(8));                       // a recent chunk is out, older ones are in
    CHECK(p.is_resident(2));
    // Every chunk protected: more tied candidates than blocks. Residents win the ties.
    p.score_hook = [](int c) { return (float)c; };
    CHECK(p.reselect(12) == 0);
    std::vector<int> after;
    for (int c = 0; c < 12; c++) if (p.is_resident(c)) after.push_back(c);
    CHECK(after == before);
}

namespace {
// Block map after: alloc 12 chunks, install the base hook, let `mid` run
// (it may install a "reselect" hook), then truncate (refill_after_truncate_
// ranks the host chunks by whatever hook is installed then).
template <class Mid>
std::vector<int> blocks_after_truncate(Mid && mid) {
    Harness A;
    TestPager p;
    must(p.attach(make_cfg(), A.k, A.v));
    must(p.alloc_span(0, 12 * kChunk));
    p.score_hook = [](int c) { return (float)-c; };     // base policy: older host chunks first
    mid(p);
    p.truncate(10 * kChunk);                             // frees two blocks; host chunks refill them
    std::vector<int> blocks;
    for (int c = 0; c < p.n_chunks(); c++) blocks.push_back(p.block_of(c));
    return blocks;
}
}

TEST_CASE(KvflashPagerSerdeFixture, scoped_score_hook_restores_the_base_hook_so_truncate_recalls_alike) {
    const std::vector<int> ref = blocks_after_truncate([](KvFlashPager &) {});
    const std::vector<int> scoped = blocks_after_truncate([](KvFlashPager & p) {
        ScopedScoreHook scope(p);
        p.score_hook = [](int c) { return (float)c; };   // the reselect's hook: newest host chunks first
    });
    const std::vector<int> leaked = blocks_after_truncate([](KvFlashPager & p) {
        p.score_hook = [](int c) { return (float)c; };   // no restore: what apply did before
    });
    CHECK(scoped == ref);
    CHECK(leaked != ref);                                // the test can see the contamination
}

TEST_CASE(KvflashPagerSerdeFixture, scoped_score_hook_restores_an_empty_hook_too) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    {
        ScopedScoreHook scope(p);
        p.score_hook = [](int c) { return (float)c; };
        CHECK((bool)p.score_hook);
    }
    CHECK(!p.score_hook);
}

// Mode 4 depends on this: without a hook, truncating back to a position and
// appending the same tokens again evicts the same chunks as the first time.
TEST_CASE(KvflashPagerSerdeFixture, truncate_back_restores_residency_and_replays_the_same_evictions) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    auto resident = [&]() {
        std::vector<int> r;
        for (int c = 0; c < 32; c++) if (p.is_resident(c)) r.push_back(c);
        return r;
    };
    REQUIRE(p.alloc_span(0, 6 * kChunk));
    const std::vector<int> pre = resident();
    std::vector<int> first_grown;
    for (int pass = 0; pass < 2; pass++) {
        // Append past the pool so old chunks are evicted to host.
        for (int c = 6; c < 14; c++) REQUIRE(p.alloc_span(c * kChunk, kChunk));
        const std::vector<int> grown = resident();
        if (pass == 0) CHECK(grown != pre);
        if (pass == 0) first_grown = grown;
        else CHECK(grown == first_grown);                    // same appends evict the same chunks
        p.truncate(6 * kChunk);                              // rewind: free the new chunks, recall evicted ones
        CHECK(resident() == pre);                            // residency equals the pre-append residency
    }
    // A mid-chunk rewind point (the question starts inside a chunk) behaves the same.
    p.truncate(0);
    REQUIRE(p.alloc_span(0, 5 * kChunk + 10));
    const std::vector<int> pre2 = resident();
    for (int c = 6; c < 14; c++) REQUIRE(p.alloc_span(c * kChunk, kChunk));
    p.truncate(5 * kChunk + 10);
    CHECK(resident() == pre2);
}

// ── truncate() layout contract ───────────────────────────────────────────
// truncate() promises "the result matches a pager that only ever saw P
// tokens". Pool attention reduces over physical slots in order, so the
// logical-to-physical map (chunks_[c].block) is part of that result: the same
// content in a different layout rounds differently. Scope: only when the kept
// prefix fits the pool (ceil(P/chunk) <= n_blocks). When it does not fit, the
// layout is history-dependent by design and is not tested here.
namespace {
constexpr int kPoolBlocks = kPool / kChunk;   // 8

std::vector<int> layout_of(const KvFlashPager & p, int n_chunks) {
    std::vector<int> v;
    for (int c = 0; c < n_chunks; c++) v.push_back(p.block_of(c));
    return v;
}

// What a pager that only ever saw P tokens gives: the prefix map, then the
// blocks handed out for M more chunks appended after it.
struct FreshMaps { std::vector<int> prefix, after_append; bool ok = false; };

FreshMaps fresh_maps(int P, int M) {
    FreshMaps f;
    Harness A;
    TestPager p;
    if (!p.attach(make_cfg(), A.k, A.v)) return f;
    if (!p.alloc_span(0, P)) return f;
    const int keep = (P + kChunk - 1) / kChunk;
    f.prefix = layout_of(p, keep);
    if (!p.alloc_span((int64_t)keep * kChunk, (int64_t)M * kChunk)) return f;
    f.after_append = layout_of(p, keep + M);
    f.ok = true;
    return f;
}

// truncate(P) on `p`, then append M chunks; same two maps as fresh_maps().
FreshMaps truncated_maps(KvFlashPager & p, int P, int M) {
    FreshMaps f;
    p.truncate(P);
    const int keep = (P + kChunk - 1) / kChunk;
    f.prefix = layout_of(p, keep);
    if (!p.alloc_span((int64_t)keep * kChunk, (int64_t)M * kChunk)) return f;
    f.after_append = layout_of(p, keep + M);
    f.ok = true;
    return f;
}

// Append chunks [from, to) one at a time (the prefill pattern).
bool append_chunks(KvFlashPager & p, int from, int to) {
    for (int c = from; c < to; c++)
        if (!p.alloc_span((int64_t)c * kChunk, kChunk)) return false;
    return true;
}
}

// Control: a pager that never overflowed. Must already pass.
TEST_CASE(KvflashPagerSerdeFixture, truncate_fitting_prefix_gives_fresh_layout_without_overflow) {
    for (int P : {3 * kChunk, 5 * kChunk + 10}) {
        const FreshMaps fresh = fresh_maps(P, 2);
        REQUIRE(fresh.ok);
        Harness A;
        TestPager p;
        REQUIRE(p.attach(make_cfg(), A.k, A.v));
        REQUIRE(append_chunks(p, 0, 7));                 // 7 <= 8 blocks: never evicts
        const FreshMaps got = truncated_maps(p, P, 2);
        REQUIRE(got.ok);
        CHECK(got.prefix == fresh.prefix);
        CHECK(got.after_append == fresh.after_append);
    }
}

TEST_CASE(KvflashPagerSerdeFixture, truncate_fitting_prefix_gives_fresh_layout_after_reselect) {
    for (int P : {3 * kChunk, 5 * kChunk + 10}) {
        const FreshMaps fresh = fresh_maps(P, 2);
        REQUIRE(fresh.ok);
        Harness A;
        TestPager p;
        REQUIRE(p.attach(make_cfg(), A.k, A.v));
        REQUIRE(append_chunks(p, 0, 14));                // past the pool: LRU evicts and pages out
        // A reselect with a score hook permutes residency and layout.
        p.score_hook = [](int c) { return (float)-c; };
        p.reselect(0);
        p.score_hook = nullptr;
        const FreshMaps got = truncated_maps(p, P, 2);
        REQUIRE(got.ok);
        CHECK(got.prefix == fresh.prefix);               // block of every chunk c < keep
        CHECK(got.after_append == fresh.after_append);   // later appends hand out the same blocks
    }
}

TEST_CASE(KvflashPagerSerdeFixture, truncate_fitting_prefix_gives_fresh_layout_after_any_lru_history) {
    const int P = 3 * kChunk;
    const FreshMaps fresh = fresh_maps(P, 2);
    REQUIRE(fresh.ok);
    {   // History A: a long straight append, LRU only.
        Harness A;
        TestPager p;
        REQUIRE(p.attach(make_cfg(), A.k, A.v));
        REQUIRE(append_chunks(p, 0, 16));
        const FreshMaps got = truncated_maps(p, P, 2);
        REQUIRE(got.ok);
        CHECK(got.prefix == fresh.prefix);
        CHECK(got.after_append == fresh.after_append);
    }
    {   // History B: a different length with an interleaved truncate, LRU only.
        Harness A;
        TestPager p;
        REQUIRE(p.attach(make_cfg(), A.k, A.v));
        REQUIRE(append_chunks(p, 0, 12));
        p.truncate(9 * kChunk);                          // 9 > 8 blocks: layout is history-dependent here
        REQUIRE(append_chunks(p, 9, 15));
        const FreshMaps got = truncated_maps(p, P, 2);
        REQUIRE(got.ok);
        CHECK(got.prefix == fresh.prefix);
        CHECK(got.after_append == fresh.after_append);
    }
}

// ── truncate() identity relayout moves real bytes ────────────────────────
namespace {
// Append chunks [from, to) one at a time and stamp each with `seed`.
bool append_stamped(Harness & h, KvFlashPager & p, int from, int to, int seed) {
    for (int c = from; c < to; c++) {
        if (!p.alloc_span((int64_t)c * kChunk, kChunk)) return false;
        stamp_chunk(h, p, c, seed);
    }
    return true;
}

bool all_match(Harness & h, KvFlashPager & p, int n, int seed) {
    p.synchronize_paging();
    for (int c = 0; c < n; c++)
        if (!chunk_matches(h, p, c, seed)) return false;
    return true;
}
}

// Codex's case: 8 blocks, chunks 0-5, then 6-13, then truncate to 6 chunks.
// The retained map used to be [0,7,1,2,3,4]; every retained chunk must now sit
// in block c with its own bytes, and the next appends get blocks 6, 7.
TEST_CASE(KvflashPagerSerdeFixture, truncate_relayout_codex_case_moves_bytes_to_identity_blocks) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(append_stamped(A, p, 0, 6, 1));
    REQUIRE(append_stamped(A, p, 6, 14, 1));
    p.truncate(6 * kChunk);
    CHECK(layout_of(p, 6) == (std::vector<int>{0, 1, 2, 3, 4, 5}));
    CHECK(all_match(A, p, 6, 1));
    REQUIRE(append_chunks(p, 6, 8));
    CHECK(layout_of(p, 8) == (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
    CHECK(all_match(A, p, 6, 1));
}

// A pure swap (a cycle with no free block to start from) plus a partial last
// chunk: bytes survive, rows >= P are zero, and the vacated blocks are zero.
TEST_CASE(KvflashPagerSerdeFixture, truncate_relayout_resolves_a_swap_cycle_without_losing_bytes) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(append_stamped(A, p, 0, 8, 2));
    REQUIRE(p.page_out(1));
    REQUIRE(p.page_out(2));
    REQUIRE(p.page_in(1));
    REQUIRE(p.page_in(2));
    p.synchronize_paging();
    REQUIRE(p.block_of(1) == 2);
    REQUIRE(p.block_of(2) == 1);                         // chunks 1 and 2 swapped
    p.truncate(5 * kChunk);
    CHECK(layout_of(p, 5) == (std::vector<int>{0, 1, 2, 3, 4}));
    CHECK(all_match(A, p, 5, 2));
    for (int b = 5; b < 8; b++) {                        // vacated blocks hold zeros
        bool zero = true;
        for_each_segment(A, b, [&](ggml_tensor * t, size_t off, size_t len) {
            std::vector<uint8_t> v(len);
            ggml_backend_tensor_get(t, v.data(), off, len);
            for (uint8_t x : v) zero = zero && x == 0;
        });
        CHECK(zero);
    }
}


// ── KVFLASH2 beside KVFLASH3 ─────────────────────────────────────────────
// serialize() keeps writing main's KVFLASH3 (pool layout included).
// serialize_to()/blob_header()/read_chunk()/ledger_blob() give KVFLASH2, the
// same blob without the pool layout, for a writer that keeps chunks apart.
// deserialize() takes both.
namespace {
std::vector<uint8_t> strip_pool_layout(const std::vector<uint8_t> & k3, int nc) {
    std::vector<uint8_t> k2(k3.begin(),
                            k3.begin() + kHdr + (size_t)nc * blob_chunk_bytes(k3) + (size_t)nc * kLedgerEntry);
    k2[0] = '2';   // "KVFLASH2" as a little-endian u64 starts with '2'
    return k2;
}
}

TEST_CASE(KvflashPagerSerdeFixture, blob_parts_concatenate_to_the_streamed_kvflash2_blob) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 1);
    pa.synchronize_paging();
    const std::vector<uint8_t> k2 = strip_pool_layout(pa.serialize(), 6);
    std::vector<uint8_t> parts = pa.blob_header();
    REQUIRE(parts.size() == kHdr);
    REQUIRE(std::memcmp(parts.data(), "2HSALFVK", 8) == 0);
    std::vector<uint8_t> chunk(pa.chunk_bytes());
    for (int c = 0; c < pa.n_chunks(); ++c) {
        pa.read_chunk(c, chunk.data());
        parts.insert(parts.end(), chunk.begin(), chunk.end());
    }
    const std::vector<uint8_t> ledger = pa.ledger_blob();
    REQUIRE(ledger.size() == (size_t)pa.n_chunks() * kLedgerEntry);
    CHECK(ledger[3 * kLedgerEntry] == 0);        // chunk 3 is paged out: not resident
    CHECK(ledger[3 * kLedgerEntry + 1] == 1);    // but materialized on the host
    parts.insert(parts.end(), ledger.begin(), ledger.end());
    CHECK(parts == k2);
    std::vector<uint8_t> streamed;
    REQUIRE(pa.serialize_to([&](const uint8_t * d, size_t n) { streamed.insert(streamed.end(), d, d + n); return true; }));
    CHECK(streamed == k2);
    CHECK(!pa.serialize_to([&](const uint8_t *, size_t) { return false; }));   // a sink failure is reported
}

// A KVFLASH2 blob carries no pool layout: the chunks load with the residency,
// bytes and scores of the KVFLASH3 blob, in identity block order.
TEST_CASE(KvflashPagerSerdeFixture, kvflash2_blob_loads_in_identity_order_with_the_same_residency) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 5);
    const std::vector<uint8_t> k2 = strip_pool_layout(pa.serialize(), 6);
    Harness B;
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.alloc_span(0, 3 * kChunk));        // state that deserialize must replace
    REQUIRE(pb.deserialize(k2.data(), k2.size()));
    CHECK(pb.n_chunks() == 6);
    CHECK(pb.resident_blocks() == 5);
    int next = 0;
    for (int c = 0; c < 6; c++) {
        CHECK(pb.is_resident(c) == pa.is_resident(c));
        CHECK(pb.chunk_score(c) == 10.0f * (float)c);
        if (!pb.is_resident(c)) continue;
        CHECK(pb.block_of(c) == next++);
        CHECK(chunk_matches(B, pb, c, 5));
    }
    REQUIRE(pb.page_in(3));                       // the host copy came along
    pb.synchronize_paging();
    CHECK(chunk_matches(B, pb, 3, 5));
}

// A bundle adoption admits a chunk count first: deserialize refuses a blob of
// another count before it touches any state, in both formats.
TEST_CASE(KvflashPagerSerdeFixture, deserialize_checks_the_expected_chunk_count_before_any_state_moves) {
    Harness A;
    TestPager pa;
    build_source(A, pa, 1);
    const std::vector<uint8_t> k3 = pa.serialize();
    const std::vector<uint8_t> k2 = strip_pool_layout(k3, 6);
    Harness B;
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.alloc_span(0, 2 * kChunk));
    CHECK(!pb.deserialize(k3.data(), k3.size(), 5));
    CHECK(!pb.deserialize(k2.data(), k2.size(), 5));
    CHECK(pb.n_chunks() == 2);                                                 // untouched
    CHECK(pb.is_resident(0) && pb.is_resident(1));
    REQUIRE(pb.deserialize(k2.data(), k2.size(), 6));
    CHECK(pb.n_chunks() == 6);
    REQUIRE(pb.deserialize(k3.data(), k3.size(), 6));
    CHECK(pb.serialize() == k3);
}

// A KVFLASH3 blob written by main (89fddc3d0): this geometry, build_source(seed
// 7), captured from that commit's serializer. The KV bytes are the stamps, so
// only the header, ledger and pool layout are kept here. It loads with its
// pool layout and serializes back byte for byte, and today's writer gives the
// same bytes for the same state: the pooled snapshots main wrote still restore
// exactly, and the ones this pager writes still load on main.
namespace {
constexpr uint8_t kMainHdr[44] = {
    0x33, 0x48, 0x53, 0x41, 0x4c, 0x46, 0x56, 0x4b, 0x06, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
};
constexpr uint8_t kMainLedger[48] = {
    0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x20, 0x41,
    0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0xa0, 0x41, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x41,
    0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x20, 0x42, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x48, 0x42,
};
constexpr uint8_t kMainPool[124] = {
    0x08, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x07, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
};
constexpr int kMainBlockOfChunk2 = 3;   // blocks after build_source: 0 1 3 -1 4 5
}

TEST_CASE(KvflashPagerSerdeFixture, kvflash3_blob_written_by_main_loads_and_reserializes_byte_for_byte) {
    const size_t cb = (size_t)kChunk * kHeadDim * 2 * kNHeadKv * kNLayer * 2;
    std::vector<uint8_t> golden(kMainHdr, kMainHdr + sizeof kMainHdr);
    for (int c = 0; c < 6; c++)
        for (size_t i = 0; i < cb; i++) golden.push_back(pat(c, 7, i));
    golden.insert(golden.end(), kMainLedger, kMainLedger + sizeof kMainLedger);
    golden.insert(golden.end(), kMainPool, kMainPool + sizeof kMainPool);
    REQUIRE(std::memcmp(golden.data(), "3HSALFVK", 8) == 0);
    REQUIRE(blob_chunk_bytes(golden) == cb);

    Harness B;
    TestPager pb;
    REQUIRE(pb.attach(make_cfg(), B.k, B.v));
    REQUIRE(pb.alloc_span(0, 3 * kChunk));
    REQUIRE(pb.deserialize(golden.data(), golden.size()));
    CHECK(pb.n_chunks() == 6);
    for (int c = 0; c < 6; c++) {
        CHECK(pb.is_resident(c) == (c != 3));
        CHECK(pb.chunk_score(c) == 10.0f * (float)c);
        if (c != 3) CHECK(chunk_matches(B, pb, c, 7));
    }
    CHECK(pb.block_of(2) == kMainBlockOfChunk2);   // main's page_in put chunk 2 here; the layout keeps it
    CHECK(pb.serialize() == golden);

    Harness A;
    TestPager pa;
    build_source(A, pa, 7);
    CHECK(pa.serialize() == golden);
}

// ── slot mask ────────────────────────────────────────────────────────────
// fill_slot_mask(valid_upto): resident rows below the bound are visible, rows at or above it and
// every paged-out or free row are -inf. The default bound shows every resident row.
TEST_CASE(KvflashPagerSerdeFixture, slot_mask_is_bounded_by_valid_upto) {
    constexpr uint16_t kZero = 0x0000, kMinusInf = 0xFC00;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), {}, {}));        // no tensors: bookkeeping only
    REQUIRE(p.alloc_span(0, 4 * kChunk));         // chunks 0..3 resident
    std::vector<uint16_t> m((size_t)p.arena_tokens());
    auto row = [&](int c, int i) { return m[(size_t)p.block_of(c) * kChunk + i]; };

    p.fill_slot_mask(m.data());                   // default: every resident row visible
    for (int c = 0; c < 4; c++) CHECK(row(c, 0) == kZero && row(c, kChunk - 1) == kZero);

    // A decode step at position 2*kChunk passes p + 1: rows 0..p visible, the rest hidden.
    p.fill_slot_mask(m.data(), 2 * kChunk + 1);
    CHECK(row(1, kChunk - 1) == kZero);
    CHECK(row(2, 0) == kZero);
    CHECK(row(2, 1) == kMinusInf);
    CHECK(row(3, 0) == kMinusInf);

    p.fill_slot_mask(m.data(), 0);                // a bound of 0 hides everything
    for (int c = 0; c < 4; c++) CHECK(row(c, 0) == kMinusInf);

    const int b1 = p.block_of(1);
    REQUIRE(p.page_out(1));                       // a paged-out chunk stays hidden under any bound
    p.fill_slot_mask(m.data());
    CHECK(m[(size_t)b1 * kChunk] == kMinusInf);
    CHECK(row(0, 0) == kZero);
}

// ── pins, step fences, deferred free ─────────────────────────────────────
// A pinned chunk's rows belong to the step (or request) that
// pinned it; an explicit page_out refuses it the way eviction already skips it.
TEST_CASE(KvflashPagerSerdeFixture, page_out_refuses_a_pinned_chunk_until_it_is_unpinned) {
    Harness A;
    TestPager p;
    REQUIRE(p.attach(make_cfg(), A.k, A.v));
    REQUIRE(p.alloc_span(0, 4 * kChunk));
    p.pin_range(2 * kChunk, 3 * kChunk);          // chunk 2, until unpin_all
    CHECK(!p.page_out(2));
    CHECK(p.is_resident(2));
    CHECK(p.page_out(1));                         // an unpinned chunk still goes
    p.unpin_all();
    CHECK(p.page_out(2));
    CHECK(!p.is_resident(2));
    REQUIRE(p.page_in(2));
    p.pin_step_range(2 * kChunk, 3 * kChunk);     // chunk 2, until end_step
    CHECK(p.is_pinned(2));
    CHECK(!p.page_out(2));
    p.end_step();
    CHECK(!p.is_pinned(2));
    CHECK(p.page_out(2));
}

// Step pins expire at end_step(); pin_range pins survive it. Both keep the
// chunk out of lru_victims(), the order eviction takes.
TEST_CASE(KvflashPagerSerdeFixture, step_pins_expire_at_end_step_and_hard_pins_do_not) {
    TestPager p;
    KvFlashConfig cfg = make_cfg();
    cfg.tail_window_chunks = 1;                   // room for two pins: sink + tail + pins + 2 <= 8 blocks
    REQUIRE(p.attach(cfg, {}, {}));               // no tensors: bookkeeping only
    REQUIRE(p.alloc_span(0, 8 * kChunk));         // chunks 0..7 fill the 8 blocks
    // sink 0 and the tail 6..7 (window 1 + the current chunk) are protected:
    // 1..5 are the victims, oldest first.
    CHECK(p.lru_victims(8) == (std::vector<int>{1, 2, 3, 4, 5}));
    p.pin_range(1 * kChunk, 2 * kChunk);          // hard pin on 1
    p.pin_step_range(2 * kChunk, 3 * kChunk);     // step pin on 2
    CHECK(p.lru_victims(8) == (std::vector<int>{3, 4, 5}));
    p.pin_step_range(1 * kChunk, 2 * kChunk);     // a step pin over a hard pin leaves it hard
    p.end_step();
    CHECK(p.is_pinned(1));
    CHECK(!p.is_pinned(2));
    CHECK(p.lru_victims(8) == (std::vector<int>{2, 3, 4, 5}));
    // A ninth chunk evicts the first victim, never the pinned one.
    REQUIRE(p.alloc_span(8 * kChunk, kChunk));
    CHECK(!p.is_resident(2));
    CHECK(p.is_resident(1));
    p.abort_step();                               // idempotent; hard pins stay
    CHECK(p.is_pinned(1));
}

// Deferred free: a paged-out block is allocatable only after complete(record)
// reaches its page-out record (or synchronize_paging(), the host wait). An
// alloc that finds only pending blocks fails instead of taking a second victim.
TEST_CASE(KvflashPagerSerdeFixture, deferred_free_holds_blocks_until_their_record_completes) {
    TestPager p;
    REQUIRE(p.attach(make_cfg(), {}, {}));
    REQUIRE(p.alloc_span(0, 8 * kChunk));
    CHECK(p.resident_blocks() == 8);
    p.set_deferred_free(true);
    const uint64_t r0 = p.last_record();
    REQUIRE(p.page_out(1));
    REQUIRE(p.page_out(2));
    CHECK(p.last_record() == r0 + 2);
    CHECK(p.pending_free_blocks() == 2);
    CHECK(p.resident_blocks() == 6);              // pending blocks are neither resident nor free
    CHECK(!p.alloc_span(8 * kChunk, kChunk));     // nothing allocatable yet, no extra victim
    CHECK(p.is_resident(3));
    p.complete(r0 + 1);                           // the first page-out landed
    CHECK(p.pending_free_blocks() == 1);
    REQUIRE(p.alloc_span(8 * kChunk, kChunk));    // one block back
    CHECK(!p.alloc_span(9 * kChunk, kChunk));
    p.synchronize_paging();                       // completes every record
    CHECK(p.pending_free_blocks() == 0);
    REQUIRE(p.alloc_span(9 * kChunk, kChunk));
    // Eviction on demand with deferral on: the victim's block is queued and
    // the alloc fails once; after completion it succeeds without a new victim.
    CHECK(!p.alloc_span(10 * kChunk, kChunk));
    CHECK(p.pending_free_blocks() == 1);
    p.complete(p.last_record());
    REQUIRE(p.alloc_span(10 * kChunk, kChunk));
    CHECK(p.pending_free_blocks() == 0);
    p.set_deferred_free(false);
    REQUIRE(p.alloc_span(11 * kChunk, kChunk));   // immediate reuse again
}

// The blob is a layout snapshot: a block waiting on its page-out record holds no
// resident chunk, so it is written as free and a same-size pool takes the blob back.
TEST_CASE(KvflashPagerSerdeFixture, kvflash3_blob_with_deferred_free_pending_loads_back) {
    TestPager p;
    REQUIRE(p.attach(make_cfg(), {}, {}));
    REQUIRE(p.alloc_span(0, 8 * kChunk));
    p.set_deferred_free(true);
    REQUIRE(p.page_out(1));
    REQUIRE(p.page_out(2));
    REQUIRE(p.pending_free_blocks() == 2);
    const std::vector<uint8_t> blob = p.serialize();
    TestPager q;
    REQUIRE(q.attach(make_cfg(), {}, {}));
    REQUIRE(q.deserialize(blob.data(), blob.size()));
    CHECK(!q.is_resident(1));
    CHECK(!q.is_resident(2));
    CHECK(q.is_resident(3));
    CHECK(q.resident_blocks() == 6);
    CHECK(q.serialize() == blob);
}

// A pending block joins the free list at its back when its record completes, so the layout lists it there:
// the blob written while it waits is the blob of the same pager once it has settled.
TEST_CASE(KvflashPagerSerdeFixture, kvflash3_free_list_puts_pending_blocks_where_the_live_pager_frees_them) {
    TestPager p;
    REQUIRE(p.attach(make_cfg(), {}, {}));
    REQUIRE(p.alloc_span(0, 8 * kChunk));
    REQUIRE(p.page_out(5));                       // an older free block already waits in the list
    p.set_deferred_free(true);
    REQUIRE(p.page_out(1));
    REQUIRE(p.page_out(2));
    REQUIRE(p.pending_free_blocks() == 2);
    const std::vector<uint8_t> waiting = p.serialize();
    p.synchronize_paging();
    CHECK(p.pending_free_blocks() == 0);
    CHECK(p.serialize() == waiting);
}

// A page-out planned against an earlier generation is stale after reset(),
// attach() or abort_step(): it must not evict the new request's chunk.
TEST_CASE(KvflashPagerSerdeFixture, page_out_checked_refuses_a_stale_generation_and_protected_chunks) {
    TestPager p;
    REQUIRE(p.attach(make_cfg(), {}, {}));
    REQUIRE(p.alloc_span(0, 8 * kChunk));
    const uint64_t g = p.generation();
    CHECK(!p.page_out_checked(0, g));             // sink: not evictable
    CHECK(!p.page_out_checked(7, g));             // tail window
    CHECK(p.page_out_checked(2, g));
    CHECK(!p.is_resident(2));
    p.reset();
    REQUIRE(p.alloc_span(0, 8 * kChunk));
    CHECK(p.generation() != g);
    CHECK(!p.page_out_checked(2, g));             // planned against the old request
    CHECK(p.is_resident(2));
    const uint64_t g2 = p.generation();
    p.abort_step();
    CHECK(p.generation() != g2);
    CHECK(!p.page_out_checked(2, g2));
    CHECK(p.page_out_checked(2, p.generation()));
    CHECK(!p.is_resident(2));
}

// ── row_base: a pool that is a slice of a larger KV allocation ───────────
// Block ids and rows are arena space (row_base/chunk_tokens + local), tensor
// bytes are offset by row_base rows, and a serial consumer refuses a slice.
namespace {
constexpr int kPoolChunks = 8, kBaseChunks = 5, kTotalChunks = 14;

KvFlashConfig cfg_at(int base_chunks) {
    KvFlashConfig c;
    c.pool_tokens = kPoolChunks * kChunk;
    c.row_base = base_chunks * kChunk;
    return c;
}

struct TallHarness {   // one layer, two heads, kTotalChunks rows
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * k = nullptr, * v = nullptr;
    explicit TallHarness(int rows) {
        ggml_init_params ip{};
        ip.mem_size = 8 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, rows, kNHeadKv);
        v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, rows, kNHeadKv);
        buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        for (ggml_tensor * t : {k, v}) ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    }
    ~TallHarness() {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(be);
    }
};
}

TEST_CASE(KvflashPagerSerdeFixture, row_base_rows_and_block_ids_are_both_arena_space) {
    TestPager p;
    REQUIRE(p.attach(cfg_at(kBaseChunks), {}, {}));
    CHECK(p.row_base() == kBaseChunks * kChunk);
    CHECK(p.arena_tokens() == (kBaseChunks + kPoolChunks) * kChunk);
    REQUIRE(p.alloc_span(0, 3 * kChunk));
    for (int c = 0; c < 3; c++) {
        CHECK(p.block_of(c) >= kBaseChunks && p.block_of(c) < kBaseChunks + kPoolChunks);
        CHECK(p.slot_of((int64_t)c * kChunk + 7) == p.block_of(c) * kChunk + 7);
    }
    CHECK(p.slot_of(3 * kChunk) == -1);
    CHECK(!p.is_identity());                      // "chunk c in block c" names another slot's rows
    std::vector<int32_t> pos((size_t)p.arena_tokens(), 7);
    p.fill_slot_pos(pos.data());
    for (int i = 0; i < kBaseChunks * kChunk; i++) CHECK(pos[(size_t)i] == -1);   // below the slice: empty
    CHECK(pos[(size_t)p.slot_of(kChunk + 1)] == kChunk + 1);
    CHECK(p.set_block_order({kBaseChunks + 2, kBaseChunks + 7}));
    CHECK(!p.set_block_order({2}));               // a pool-local id is not an arena id here
    CHECK(!p.set_block_order({kBaseChunks + 1, kBaseChunks + 1}));
}

TEST_CASE(KvflashPagerSerdeFixture, row_base_attach_refuses_bad_base_and_short_tensors) {
    TestPager p;
    KvFlashConfig c = cfg_at(kBaseChunks);
    c.row_base = 5;                                   // not a chunk multiple
    CHECK(!p.attach(c, {}, {}));
    c.row_base = -kChunk;
    CHECK(!p.attach(c, {}, {}));
    TallHarness t((kBaseChunks + kPoolChunks) * kChunk - 1);   // one row short of base + pool
    REQUIRE(p.attach(cfg_at(0), {t.k}, {t.v}));
    REQUIRE(p.alloc_span(0, 2 * kChunk));
    CHECK(!p.attach(cfg_at(kBaseChunks), {t.k}, {t.v}));
    CHECK(p.row_base() == 0);                         // the failed attach left the previous one intact
    CHECK(p.n_chunks() == 2);
}

TEST_CASE(KvflashPagerSerdeFixture, attach_refuses_a_later_layer_tensor_too_short_for_the_slice) {
    TallHarness a(kTotalChunks * kChunk), b((kBaseChunks + kPoolChunks) * kChunk - 1);
    TestPager p;
    REQUIRE(p.attach(cfg_at(0), {a.k}, {a.v}));
    REQUIRE(p.alloc_span(0, 2 * kChunk));
    CHECK(!p.attach(cfg_at(kBaseChunks), {a.k, b.k}, {a.v, a.v}));   // layer 1 K is short
    CHECK(!p.attach(cfg_at(kBaseChunks), {a.k, a.k}, {a.v, b.v}));   // layer 1 V is short
    CHECK(p.row_base() == 0);                         // the failed attach left the previous one intact
    CHECK(p.n_chunks() == 2);
}

TEST_CASE(KvflashPagerSerdeFixture, row_base_page_out_and_in_move_bytes_at_the_offset_rows) {
    TallHarness t(kTotalChunks * kChunk);
    TestPager p;
    REQUIRE(p.attach(cfg_at(kBaseChunks), {t.k}, {t.v}));
    REQUIRE(p.alloc_span(0, 2 * kChunk));
    const int64_t row = p.slot_of(0);
    REQUIRE(row >= kBaseChunks * kChunk);
    const size_t len = (size_t)kChunk * t.k->nb[1];
    std::vector<uint8_t> stamp(len, 0xA5);
    ggml_backend_tensor_set(t.k, stamp.data(), (size_t)row * t.k->nb[1], len);
    REQUIRE(p.page_out(0));
    p.synchronize_paging();
    std::vector<uint8_t> got(len, 1);
    ggml_backend_tensor_get(t.k, got.data(), (size_t)row * t.k->nb[1], len);
    CHECK(got == std::vector<uint8_t>(len, 0));        // zeroed in the offset block
    REQUIRE(p.page_in(0));
    p.synchronize_paging();
    ggml_backend_tensor_get(t.k, got.data(), (size_t)p.slot_of(0) * t.k->nb[1], len);
    CHECK(got == stamp);
    std::vector<uint8_t> below((size_t)kBaseChunks * kChunk * t.k->nb[1], 1);   // nothing below the base was touched
    ggml_backend_tensor_get(t.k, below.data(), 0, below.size());
    CHECK(below == std::vector<uint8_t>(below.size(), 0));
}

// ── reselect() parity for the serial backends ────────────────────────────
// gemma4 and laguna call reselect() with no arguments, never pin a step, and
// install score_hook only around the call (eviction during alloc_span is
// LRU). The arrays are what main's pager at 89fddc3d0 did on this exact flow
// (block of every chunk, -1 = host), captured from that commit's header.
namespace {
constexpr float kScores0[12] = {0.f, 0.9f, 0.1f, 0.8f, 0.2f, 0.3f, 0.7f, 0.05f, 0.6f, 0.f, 0.f, 0.f};
constexpr float kScores1[14] = {0.f, 0.15f, 0.95f, 0.12f, 0.85f, 0.5f, 0.4f, 0.45f, 0.35f, 0.02f, 0.01f, 0.f, 0.f, 0.f};
constexpr float kTies[12]    = {0.f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.f, 0.f, 0.f};
constexpr int kMainAfterAlloc[12]     = {0, -1, -1, -1, -1, 5, 6, 7, 1, 2, 3, 4};               // events 0
constexpr int kMainAfterReselect1[12] = {0, 7, -1, 5, -1, -1, 6, -1, 1, 2, 3, 4};               // events 4
constexpr int kMainAfterAlloc2[14]    = {0, 7, -1, 5, -1, -1, -1, -1, -1, 2, 3, 4, 6, 1};       // events 0
constexpr int kMainAfterReselect2[14] = {0, -1, 3, -1, 2, 5, -1, 7, -1, -1, -1, 4, 6, 1};       // events 8
constexpr int kMainAfterTieReselect[12] = {0, 1, 7, 6, 5, -1, -1, -1, -1, 2, 3, 4};             // events 8

KvFlashConfig serial_cfg() { KvFlashConfig c = make_cfg(); c.tail_window_chunks = 2; return c; }
std::vector<int> blocks(const TestPager & p, int n) {
    std::vector<int> out;
    for (int c = 0; c < n; c++) out.push_back(p.block_of(c));
    return out;
}
template <size_t N> std::vector<int> vec(const int (&a)[N]) { return {a, a + N}; }
int reselect_with(TestPager & p, const float * s, int n) {   // the gemma4/laguna call shape
    p.score_hook = [=](int c) { return c < n ? s[c] : 1e30f; };
    const int e = p.reselect();
    p.score_hook = nullptr;
    p.synchronize_paging();
    return e;
}
}

TEST_CASE(KvflashPagerSerdeFixture, reselect_without_pins_matches_main_for_gemma4_and_laguna) {
    Harness h;
    TestPager p;
    REQUIRE(p.attach(serial_cfg(), h.k, h.v));
    REQUIRE(p.alloc_span(0, 12 * kChunk));                 // LRU evicts 1..4
    CHECK(blocks(p, 12) == vec(kMainAfterAlloc));
    CHECK(reselect_with(p, kScores0, 12) == 4);            // out 5, 7; in 1, 3
    CHECK(blocks(p, 12) == vec(kMainAfterReselect1));
    REQUIRE(p.alloc_span(12 * kChunk, 2 * kChunk));        // LRU again, hook removed
    CHECK(blocks(p, 14) == vec(kMainAfterAlloc2));
    CHECK(reselect_with(p, kScores1, 14) == 8);
    CHECK(blocks(p, 14) == vec(kMainAfterReselect2));
    CHECK(reselect_with(p, kScores1, 14) == 0);            // settled
    CHECK(blocks(p, 14) == vec(kMainAfterReselect2));
}

// The one place the serial backends can see a difference: exact score ties
// across the resident/host boundary. Main sorted by score alone, so ties fell
// to chunk-id order and it recalled 1..4 over the resident 5..8 (8 page events,
// kMainAfterTieReselect). This pager keeps the resident chunk on a tie: same
// pool contents, no traffic.
TEST_CASE(KvflashPagerSerdeFixture, reselect_score_ties_keep_resident_chunks_where_main_recalled_by_chunk_id) {
    Harness h;
    TestPager p;
    REQUIRE(p.attach(serial_cfg(), h.k, h.v));
    REQUIRE(p.alloc_span(0, 12 * kChunk));
    CHECK(blocks(p, 12) == vec(kMainAfterAlloc));
    CHECK(reselect_with(p, kTies, 12) == 0);
    CHECK(blocks(p, 12) == vec(kMainAfterAlloc));
    CHECK(vec(kMainAfterTieReselect) != vec(kMainAfterAlloc));   // main did move chunks here
}

// ── restore == continue ──────────────────────────────────────────────────
// A session parked to disk and restored must continue like the live slot: the chunk -> block table, the free list
// and its order and the LRU order the later evictions read are the live pager's, not the identity layout a
// KVFLASH2 blob falls back to (prs/evidence/G6/RESULTS.md section 6).
namespace {
// 10 chunks through an 8 block pool, then two recalls: chunks 1 and 2 return to the pool newer than 5..9, chunks 3 and 4
// stay on the host. Blocks are not the identity, the LRU order is not the chunk order, the free list is shuffled.
void build_lived_in(Harness & h, KvFlashPager & p) {
    must(p.attach(make_cfg(), h.k, h.v));
    for (int c = 0; c < 10; c++) {
        must(p.alloc_span(c * kChunk, kChunk));
        stamp_chunk(h, p, c, 3);
    }
    must(p.page_in(2));
    must(p.page_in(1));
    p.synchronize_paging();
    for (int c = 0; c < 10; c++) p.set_chunk_score(c, 10.0f * (float)c);
}

// What a park writes and a restore feeds the pager: the parts a bundle keeps apart, the pool layout among them.
std::vector<uint8_t> bundle_blob(const KvFlashPager & p) {
    std::vector<uint8_t> blob;
    must(p.serialize_to([&](const uint8_t * d, size_t n) { blob.insert(blob.end(), d, d + n); return true; }));
    const std::vector<uint8_t> pool = p.pool_layout_blob();
    blob.insert(blob.end(), pool.begin(), pool.end());
    blob[0] = '3';   // KVFLASH3
    return blob;
}

std::vector<int> block_table(const KvFlashPager & p, int n) {
    std::vector<int> t;
    for (int c = 0; c < n; c++) t.push_back(p.block_of(c));
    return t;
}
}  // namespace

TEST_CASE(KvflashPagerSerdeFixture, a_restored_pager_is_the_live_pager_and_evicts_alike) {
    Harness A;
    TestPager live;
    build_lived_in(A, live);
    REQUIRE(live.n_chunks() == 10);
    REQUIRE(!live.is_resident(3));
    REQUIRE(!live.is_resident(4));
    REQUIRE(live.block_of(2) != 2);
    REQUIRE((live.lru_victims(8) == std::vector<int>{2, 1}));   // chunk 2 is older than 1; the identity order would say {1, 2}

    const std::vector<uint8_t> blob = bundle_blob(live);
    Harness B;
    TestPager back;
    REQUIRE(back.attach(make_cfg(), B.k, B.v));
    REQUIRE(back.deserialize(blob.data(), blob.size(), 10));

    // field by field
    REQUIRE(back.n_chunks() == live.n_chunks());
    CHECK(back.cur_chunk() == live.cur_chunk());
    CHECK(back.resident_blocks() == live.resident_blocks());
    CHECK(block_table(back, 10) == block_table(live, 10));
    for (int c = 0; c < 10; c++) CHECK(back.chunk_score(c) == live.chunk_score(c));
    CHECK(back.lru_victims(8) == live.lru_victims(8));
    CHECK(back.serialize() == live.serialize());                // pool layout: blocks, LRU ranks, use clock, free list order

    // the same later life: pins (the binder re-applies them), appends that evict, then a reselect
    live.pin_range(2 * kChunk, 3 * kChunk);
    back.pin_range(2 * kChunk, 3 * kChunk);
    for (int c = 10; c < 14; c++) {
        REQUIRE(live.alloc_span(c * kChunk, kChunk));
        REQUIRE(back.alloc_span(c * kChunk, kChunk));
        CHECK(block_table(back, c + 1) == block_table(live, c + 1));
        CHECK(back.lru_victims(8) == live.lru_victims(8));
    }
    const auto score = [](int c) { return (float)((c * 7) % 5); };
    live.score_hook = score;
    back.score_hook = score;
    CHECK(back.reselect() == live.reselect());
    CHECK(block_table(back, 14) == block_table(live, 14));
    live.score_hook = nullptr;
    back.score_hook = nullptr;
    CHECK(back.serialize() == live.serialize());
}

// Hard pins are state: a pinned chunk is no eviction victim and reselect() keeps it, and a resume reselects before the turn
// re-pins its preamble (the G6 evidence: live kept preamble chunks 1-9 resident, the restored slot only chunk 0). The pin set
// rides in the pool layout, so the restored pager protects what the live one does and the digest sees a lost pin.
TEST_CASE(KvflashPagerSerdeFixture, a_restored_pager_keeps_the_hard_pins_and_reselects_like_the_live_one) {
    Harness A;
    TestPager live;
    build_lived_in(A, live);
    live.pin_range(2 * kChunk, 3 * kChunk);                      // chunk 2: resident, older than 5..9, the first LRU victim
    REQUIRE(live.is_pinned(2));

    Harness C;                                                   // a pin lost in the round trip is another layout: another digest
    TestPager unpinned;
    build_lived_in(C, unpinned);
    CHECK(unpinned.pool_layout_blob() != live.pool_layout_blob());

    const std::vector<uint8_t> blob = bundle_blob(live);
    Harness B;
    TestPager back;
    REQUIRE(back.attach(make_cfg(), B.k, B.v));
    REQUIRE(back.deserialize(blob.data(), blob.size(), 10));

    for (int c = 0; c < 10; c++) CHECK(back.is_pinned(c) == live.is_pinned(c));
    CHECK(back.lru_victims(8) == live.lru_victims(8));
    CHECK(back.pool_layout_blob() == live.pool_layout_blob());

    // the resume: a reselect before anything re-pins, then appends that evict
    const auto score = [](int c) { return (float)((c * 7) % 5); };
    live.score_hook = score;
    back.score_hook = score;
    CHECK(back.reselect_recent() == live.reselect_recent());
    CHECK(block_table(back, 10) == block_table(live, 10));
    CHECK(back.is_resident(2) == live.is_resident(2));
    for (int c = 10; c < 14; c++) {
        REQUIRE(live.alloc_span(c * kChunk, kChunk));
        REQUIRE(back.alloc_span(c * kChunk, kChunk));
        CHECK(block_table(back, c + 1) == block_table(live, c + 1));
    }
    live.score_hook = nullptr;
    back.score_hook = nullptr;
    CHECK(back.serialize() == live.serialize());
}
