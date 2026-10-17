// Unit tests for KvFlashPager::truncate(P) (kvflash_pager.h): drop every
// logical position >= P, free the blocks, keep pins/ledger consistent, and
// leave a state indistinguishable from a pager that only ever saw P tokens.
// CPU ggml backend, no GPU.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_pager.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {

struct KvflashPagerTruncateFixture {};

constexpr int kHeadDim = 4, kPool = 512, kNHeadKv = 2, kNLayer = 2, kChunk = 64;
constexpr size_t kHdr = 8 + 8 * 4;
constexpr size_t kChunkBytes = (size_t)kChunk * kHeadDim * 2 * kNHeadKv * kNLayer * 2;

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

void must(bool ok) { if (!ok) throw std::runtime_error("kvflash truncate helper precondition failed"); }

uint8_t pat(int c, size_t i) { return (uint8_t)(c * 53 + i * 31 + 7); }

// Write logical chunk c's rows [0, nrows) with a deterministic pattern; rows
// >= nrows stay zero (what a pager that only saw nrows tokens holds).
void stamp_rows(Harness & h, const KvFlashPager & p, int c, int nrows) {
    const int block = p.block_of(c);
    must(block >= 0);
    size_t n = 0;
    for (int l = 0; l < kNLayer; l++) {
        for (int kv = 0; kv < 2; kv++) {
            ggml_tensor * t = kv == 0 ? h.k[l] : h.v[l];
            for (int hh = 0; hh < kNHeadKv; hh++) {
                const size_t off = (size_t)block * kChunk * t->nb[1] + (size_t)hh * t->nb[2];
                const size_t len = (size_t)kChunk * t->nb[1];
                const size_t live = (size_t)nrows * t->nb[1];
                std::vector<uint8_t> b(len, 0);
                for (size_t i = 0; i < live; i++) b[i] = pat(c, n + i);
                ggml_backend_tensor_set(t, b.data(), off, len);
                n += len;
            }
        }
    }
}

// Append tokens [from, to) chunk by chunk (allocating, stamping live rows).
void append(Harness & h, KvFlashPager & p, int from, int to) {
    for (int pos = from; pos < to;) {
        const int c = pos / kChunk;
        const int end = std::min(to, (c + 1) * kChunk);
        must(p.alloc_span(pos, end - pos));
        stamp_rows(h, p, c, end - c * kChunk);
        pos = end;
    }
}

// Chunk-level bytes only: header + KV payload (no residency ledger).
std::vector<uint8_t> kv_blob(const KvFlashPager & p) {
    std::vector<uint8_t> b = p.serialize();
    uint32_t nc = 0;
    std::memcpy(&nc, b.data() + 8, 4);
    b.resize(kHdr + (size_t)nc * kChunkBytes);
    return b;
}

std::vector<int32_t> slot_pos(const KvFlashPager & p) {
    std::vector<int32_t> v(kPool);
    p.fill_slot_pos(v.data());
    return v;
}

} // namespace

TEST_CASE(KvflashPagerTruncateFixture, at_chunk_edge_frees_trailing_blocks) {
    Harness A, R;
    KvFlashPager pa, pr;
    must(pa.attach(make_cfg(), A.k, A.v));
    must(pr.attach(make_cfg(), R.k, R.v));
    append(A, pa, 0, 5 * kChunk);
    append(R, pr, 0, 3 * kChunk);
    for (int c = 0; c < 5; c++) pa.set_chunk_score(c, 10.f * c);
    for (int c = 0; c < 3; c++) pr.set_chunk_score(c, 10.f * c);

    pa.truncate(3 * kChunk);
    CHECK(pa.n_chunks() == 3);
    CHECK(pa.resident_blocks() == 3);
    CHECK(pa.slot_of(3 * kChunk) == -1);
    CHECK(!pa.is_resident(4));
    CHECK(pa.chunk_score(4) < -1e30f);
    CHECK(pa.serialize() == pr.serialize());
    CHECK(slot_pos(pa) == slot_pos(pr));
}

TEST_CASE(KvflashPagerTruncateFixture, mid_chunk_zeroes_tail_rows_and_append_resumes) {
    Harness A, R;
    KvFlashPager pa, pr;
    must(pa.attach(make_cfg(), A.k, A.v));
    must(pr.attach(make_cfg(), R.k, R.v));
    const int P = 2 * kChunk + 17;
    append(A, pa, 0, 4 * kChunk + 5);
    append(R, pr, 0, P);

    pa.truncate(P);
    CHECK(pa.n_chunks() == 3);
    CHECK(pa.resident_blocks() == 3);
    CHECK(pa.serialize() == pr.serialize());   // rows >= P zeroed in the partial chunk
    CHECK(slot_pos(pa) == slot_pos(pr));

    // Appending from P behaves as if the cache only ever held P tokens.
    append(A, pa, P, 5 * kChunk);
    append(R, pr, P, 5 * kChunk);
    CHECK(pa.serialize() == pr.serialize());
}

TEST_CASE(KvflashPagerTruncateFixture, inside_tail_window) {
    Harness A, R;
    KvFlashPager pa, pr;
    must(pa.attach(make_cfg(), A.k, A.v));
    must(pr.attach(make_cfg(), R.k, R.v));
    append(A, pa, 0, 7 * kChunk);
    const int P = 6 * kChunk + 1;        // tail window = chunks 2..6
    append(R, pr, 0, P);
    pa.truncate(P);
    CHECK(pa.serialize() == pr.serialize());
    append(A, pa, P, 8 * kChunk);
    append(R, pr, P, 8 * kChunk);
    CHECK(pa.serialize() == pr.serialize());
}

TEST_CASE(KvflashPagerTruncateFixture, after_eviction_and_page_out) {
    Harness A, R;
    KvFlashPager pa, pr;
    must(pa.attach(make_cfg(), A.k, A.v));
    must(pr.attach(make_cfg(), R.k, R.v));
    // 12 chunks on an 8-block pool: eviction has run, some chunks are host-only.
    append(A, pa, 0, 12 * kChunk);
    REQUIRE(pa.stats().page_outs > 0);
    const int P = 10 * kChunk + 9;
    append(R, pr, 0, P);
    const int64_t host_before = pa.stats().host_bytes;
    REQUIRE(host_before > 0);

    pa.truncate(P);
    CHECK(pa.n_chunks() == 11);
    CHECK(pa.resident_blocks() <= 8);
    CHECK(pa.stats().host_bytes <= host_before);
    CHECK(kv_blob(pa) == kv_blob(pr));        // evicted chunks keep their bytes

    // Truncate down into the evicted region: chunk 3 may live on host only.
    const int P2 = 3 * kChunk + 40;
    Harness R2;
    KvFlashPager pr2;
    must(pr2.attach(make_cfg(), R2.k, R2.v));
    append(R2, pr2, 0, P2);
    pa.truncate(P2);
    CHECK(pa.n_chunks() == 4);
    CHECK(kv_blob(pa) == kv_blob(pr2));
    CHECK(pa.stats().host_bytes <= (int64_t)(4 * kChunkBytes));
    // The pool is whole again: every freed block is reusable.
    append(A, pa, P2, 9 * kChunk);
    CHECK(pa.n_chunks() == 9);
}

TEST_CASE(KvflashPagerTruncateFixture, to_zero_and_beyond_length) {
    Harness A, R;
    KvFlashPager pa, pr;
    must(pa.attach(make_cfg(), A.k, A.v));
    must(pr.attach(make_cfg(), R.k, R.v));
    append(A, pa, 0, 3 * kChunk);
    append(R, pr, 0, 3 * kChunk);

    pa.truncate(10 * kChunk);               // beyond the length: no-op
    CHECK(pa.serialize() == pr.serialize());
    CHECK(pa.resident_blocks() == 3);

    pa.truncate(0);
    CHECK(pa.n_chunks() == 0);
    CHECK(pa.resident_blocks() == 0);
    CHECK(pa.stats().host_bytes == 0);
    Harness E;
    KvFlashPager pe;
    must(pe.attach(make_cfg(), E.k, E.v));
    CHECK(pa.serialize() == pe.serialize());
    append(A, pa, 0, 2 * kChunk);           // usable again from scratch
    CHECK(pa.n_chunks() == 2);
}

TEST_CASE(KvflashPagerTruncateFixture, pins_beyond_p_are_dropped) {
    Harness A;
    KvFlashPager pa;
    must(pa.attach(make_cfg(), A.k, A.v));
    append(A, pa, 0, 6 * kChunk);
    pa.pin_range(3 * kChunk, 4 * kChunk);   // chunk 3 (guard allows one pin on an 8-block pool)
    REQUIRE(pa.is_pinned(3));
    pa.truncate(3 * kChunk + 1);            // chunk 3 survives as the partial chunk
    CHECK(pa.is_pinned(3));
    pa.truncate(3 * kChunk);                // chunk 3 is now entirely >= P
    CHECK(!pa.is_pinned(3));
    // Re-growing must not resurrect the dropped pin.
    append(A, pa, 3 * kChunk, 6 * kChunk);
    CHECK(!pa.is_pinned(3));
}

TEST_CASE(KvflashPagerTruncateFixture, serialize_truncate_serialize_roundtrip) {
    Harness A, R, B;
    KvFlashPager pa, pr, pb;
    must(pa.attach(make_cfg(), A.k, A.v));
    must(pr.attach(make_cfg(), R.k, R.v));
    must(pb.attach(make_cfg(), B.k, B.v));
    const int P = 3 * kChunk + 30;
    append(A, pa, 0, 6 * kChunk);
    append(R, pr, 0, P);
    for (int c = 0; c < 6; c++) pa.set_chunk_score(c, 1.f + c);
    for (int c = 0; c < 4; c++) pr.set_chunk_score(c, 1.f + c);

    pa.truncate(P);
    const std::vector<uint8_t> blob = pa.serialize();
    CHECK(blob == pr.serialize());
    must(pb.deserialize(blob.data(), blob.size()));
    CHECK(pb.serialize() == blob);
    CHECK(pb.n_chunks() == 4);
}

// S9b follow-up: after a long decode the resident set holds the NEWEST chunks
// (older prompt chunks went to host). truncate(C) must leave the resident set
// of a pager that only ever appended C tokens, or the replay after a live
// restore attends over a different chunk set than a cold prefill does.
TEST_CASE(KvflashPagerTruncateFixture, truncate_leaves_the_resident_set_a_fresh_pager_reaches) {
    for (int P : {12 * kChunk + 10, 12 * kChunk, 9 * kChunk + 63, 5 * kChunk}) {
        Harness A, R;
        KvFlashPager pa, pr;
        must(pa.attach(make_cfg(), A.k, A.v));
        must(pr.attach(make_cfg(), R.k, R.v));
        append(A, pa, 0, 20 * kChunk);          // long decode: tail is resident, prompt chunks on host
        append(R, pr, 0, P);
        pa.truncate(P);
        pa.synchronize_paging();
        REQUIRE(pa.n_chunks() == pr.n_chunks());
        std::vector<int> ra, rr;
        for (int c = 0; c < pa.n_chunks(); c++) {
            if (pa.is_resident(c)) ra.push_back(c);
            if (pr.is_resident(c)) rr.push_back(c);
        }
        CHECK(ra == rr);
        CHECK(kv_blob(pa) == kv_blob(pr));
        // ledger: untouched scores of kept chunks stay -inf like the fresh pager's
        for (int c = 0; c < pa.n_chunks(); c++) CHECK(pa.chunk_score(c) == pr.chunk_score(c));
    }
}
