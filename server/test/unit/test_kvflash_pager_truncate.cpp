// Unit tests for KvFlashPager::truncate(P) (kvflash_pager.h): drop every
// logical position >= P, free the blocks, keep pins/ledger consistent, and
// leave a state indistinguishable from a pager that only ever saw P tokens.
// CPU ggml backend, no GPU.
#include "CppUnitTestFramework.hpp"
#include "kvflash_pager_test_host.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {

struct KvflashPagerTruncateFixture {};

constexpr int kHeadDim = 4, kPool = 512, kNHeadKv = 2, kNLayer = 2, kChunk = 64;
constexpr size_t kHdr = 8 + 9 * 4;
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

// Mixed per (chunk, byte offset in chunk) so no two segments share bytes.
uint8_t pat(int c, size_t i) {
    uint32_t x = (uint32_t)c * 2654435761u ^ (uint32_t)i * 2246822519u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    return (uint8_t)(x >> 24) | 1u;
}

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
    TestPager pa, pr;
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
    TestPager pa, pr;
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
    TestPager pa, pr;
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
    TestPager pa, pr;
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
    TestPager pr2;
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
    TestPager pa, pr;
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
    TestPager pe;
    must(pe.attach(make_cfg(), E.k, E.v));
    CHECK(pa.serialize() == pe.serialize());
    append(A, pa, 0, 2 * kChunk);           // usable again from scratch
    CHECK(pa.n_chunks() == 2);
}

TEST_CASE(KvflashPagerTruncateFixture, pins_beyond_p_are_dropped) {
    Harness A;
    TestPager pa;
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
    TestPager pa, pr, pb;
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
        TestPager pa, pr;
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

// A score ledger must not crowd protected chunks (sink, tail window, pins) out
// of the refill: a fresh pager always holds them, so truncate must too.
TEST_CASE(KvflashPagerTruncateFixture, refill_keeps_protected_chunks_resident_under_a_scorer) {
    Harness A;
    TestPager pa;
    must(pa.attach(make_cfg(), A.k, A.v));
    append(A, pa, 0, 20 * kChunk);
    pa.score_hook = [](int c) { return -(float)c; };   // older chunks outrank newer ones
    pa.truncate(12 * kChunk);
    pa.synchronize_paging();
    REQUIRE(pa.n_chunks() == 12);
    std::vector<int> resident;
    for (int c = 0; c < pa.n_chunks(); c++) if (pa.is_resident(c)) resident.push_back(c);
    // sink 0 + tail window 7..11, then the two best discretionary chunks (1, 2).
    CHECK(resident == (std::vector<int>{0, 1, 2, 7, 8, 9, 10, 11}));
}

// A forward's own span must survive the allocation of the span: with a scorer installed (the recall hook of a
// two-pass tail prefill) the earlier chunks of the same alloc_span call are the lowest scored, and must not be
// evicted to make room for the later ones (their slot would be -1: rows written to row -1, keys masked).
TEST_CASE(KvflashPagerTruncateFixture, alloc_span_never_evicts_a_chunk_of_its_own_span) {
    for (const int span_chunks : {2, 4, 6}) {
        Harness A;
        TestPager pa;
        must(pa.attach(make_cfg(), A.k, A.v));
        append(A, pa, 0, 20 * kChunk);
        // Recalled chunks (the old ones) outrank every chunk the forward is about to write.
        pa.score_hook = [](int c) { return c < 12 ? 100.f + (float)c : 1.f + (float)c * 1e-3f; };
        pa.reselect(2);
        pa.truncate(12 * kChunk);
        pa.synchronize_paging();
        const int from = 12 * kChunk;
        must(pa.alloc_span(from, span_chunks * kChunk));
        for (int c = 12; c < 12 + span_chunks; c++) {
            CHECK(pa.is_resident(c));
            CHECK(pa.slot_of(c * kChunk) >= 0);
        }
        // The next forward of the same prefill: its predecessors are old now, and may be victims again.
        must(pa.alloc_span(from + span_chunks * kChunk, span_chunks * kChunk));
        for (int c = 12 + span_chunks; c < 12 + 2 * span_chunks; c++) CHECK(pa.slot_of(c * kChunk) >= 0);
    }
}

// The two-pass tail's second prefill at serving scale (tensor-less pager, logic only): LRU first pass over the whole
// prompt, the recall scorer and reselect, a rollback to the tail's start, then the tail in forwards of 4..31 chunks
// with the scorer still installed and one poorly voted tail chunk. No chunk of a forward may lose its slot.
TEST_CASE(KvflashPagerTruncateFixture, tail_forwards_under_a_scorer_keep_every_inflight_chunk) {
    constexpr int pool_chunks = 256, n_chunks = 900, tail_chunks = 32;
    for (const int fwd : {4, 8, 16, 31}) {
        KvFlashConfig cfg;
        cfg.pool_tokens = pool_chunks * kChunk;
        TestPager p;
        must(p.attach(cfg, {}, {}));
        const int P = n_chunks * kChunk, tail_pos = P - tail_chunks * kChunk;
        for (int pos = 0; pos < P; pos += fwd * kChunk) must(p.alloc_span(pos, std::min(fwd * kChunk, P - pos)));
        std::vector<float> score(n_chunks);
        for (int c = 0; c < n_chunks; ++c) score[(size_t)c] = (float)((c * 7919) % 500);
        for (int c = n_chunks - tail_chunks; c < n_chunks; ++c) score[(size_t)c] = 400;
        score[(size_t)(n_chunks - tail_chunks + 3)] = 1;
        p.score_hook = [&](int c) { return c < n_chunks ? score[(size_t)c] : 1e6f + (float)c; };
        p.reselect(5);
        p.truncate(tail_pos);
        for (int pos = tail_pos; pos < P; pos += fwd * kChunk) {
            const int n = std::min(fwd * kChunk, P - pos);
            must(p.alloc_span(pos, n));
            for (int q = pos; q < pos + n; q += kChunk) CHECK(p.slot_of(q) >= 0);
        }
    }
}

// A tensor-less pager of `blocks` chunks (default sink and tail window), filled to its head with LRU appends.
namespace {
KvFlashConfig logic_cfg(int blocks) {
    KvFlashConfig cfg;
    cfg.pool_tokens = blocks * kChunk;
    return cfg;
}
// Append [0, tokens) one chunk per forward, as a prompt prefill does.
void fill(TestPager & p, int tokens) {
    for (int pos = 0; pos < tokens; pos += kChunk) must(p.alloc_span(pos, std::min(kChunk, tokens - pos)));
}
}

// The protection is scoped: a scorer that throws mid-allocation must not leave the span's chunks unevictable.
TEST_CASE(KvflashPagerTruncateFixture, alloc_span_protection_is_cleared_when_a_scorer_throws) {
    TestPager p;
    must(p.attach(logic_cfg(8), {}, {}));
    fill(p, 8 * kChunk);
    bool armed = false;
    p.score_hook = [&](int c) {
        if (armed && p.is_resident(12) && !p.is_resident(13)) throw std::runtime_error("scorer failed");
        return (float)c;
    };
    armed = true;
    bool threw = false;
    try { p.alloc_span(8 * kChunk, 6 * kChunk); } catch (const std::runtime_error &) { threw = true; }
    CHECK(threw);
    // The head sits at chunk 13, so chunk 8 is past the tail window: an eviction may take it again.
    CHECK(p.is_resident(8));
    CHECK(p.evictable_(8));
}

// The documented limit: a span that cannot sit in the pool beside the sink is not protected (its early chunks stay
// evictable, as for bulk bookkeeping callers), so a forward that big is not possible. Pinned here so a change to it
// is a decision.
TEST_CASE(KvflashPagerTruncateFixture, oversized_span_keeps_its_early_chunks_evictable) {
    TestPager p;
    must(p.attach(logic_cfg(8), {}, {}));
    fill(p, 8 * kChunk);
    p.score_hook = [](int c) { return c >= 8 ? (float)c : 100.f + (float)c; };   // the span's chunks rank lowest
    p.alloc_span(8 * kChunk, 8 * kChunk);   // 8 chunks + the sink do not fit 8 blocks
    bool lost = false;
    for (int c = 8; c < 16; c++) lost |= !p.is_resident(c);
    CHECK(lost);
}

// qwen4exp caps a forward at pool - 2 chunks of rows (qwen4exp_max_forward_rows): even unaligned that touches at most
// pool_blocks - 1 chunks, which sit beside the sink, so every qwen4exp span is protected.
TEST_CASE(KvflashPagerTruncateFixture, qwen4exp_forward_row_cap_keeps_every_span_protectable) {
    for (const int blocks : {8, 9, 16, 31, 64}) {
        for (const int offset : {0, 1, kChunk / 2, kChunk - 1}) {
            TestPager p;
            must(p.attach(logic_cfg(blocks), {}, {}));
            const int head = (blocks + 3) * kChunk;   // a pool that has been evicting
            fill(p, head);
            p.score_hook = [&](int c) { return c >= blocks + 3 ? (float)c : 100.f + (float)c; };
            const int rows = (blocks - 2) * kChunk;
            must(p.alloc_span(head + offset, rows));
            for (int q = head + offset; q < head + offset + rows; q += kChunk) CHECK(p.slot_of(q) >= 0);
            CHECK(p.slot_of(head + offset + rows - 1) >= 0);
        }
    }
}

namespace {

// The eviction rules of slot_for/alloc_span as they were before the span protection (victim: resident, past the sink
// and the tail window, lowest score or oldest use, lower chunk id on ties), over chunk residency only.
struct PreFixModel {
    int blocks, cur = 0, resident = 0;
    uint64_t clock = 0;
    std::vector<uint8_t> res;
    std::vector<uint64_t> use;
    std::function<float(int)> score;
    bool evicted_touched = false, evicted_untouched = false;   // a victim inside the span: already written / not yet
    PreFixModel(int b, int max_chunks) : blocks(b), res((size_t)max_chunks, 0), use((size_t)max_chunks, 0) {}

    int victim() const {
        int best = -1;
        for (int c = 0; c < (int)res.size(); c++) {
            if (!res[(size_t)c] || c < 1 || c > cur - 1 - 4) continue;
            if (best < 0) { best = c; continue; }
            const bool less = score ? score(c) < score(best) : use[(size_t)c] < use[(size_t)best];
            if (less) best = c;
        }
        return best;
    }
    int slot_for(int pos, int span_first) {
        const int c = pos / kChunk, prev = cur;
        cur = std::max(cur, c);
        if (!res[(size_t)c]) {
            if (resident == blocks) {
                const int v = victim();
                if (v < 0) { cur = prev; return -1; }
                (v < c ? evicted_touched : evicted_untouched) |= v >= span_first;
                res[(size_t)v] = 0;
                resident--;
            }
            res[(size_t)c] = 1;
            resident++;
        }
        use[(size_t)c] = ++clock;
        return 0;
    }
    bool alloc_span(int start, int n) {
        for (int i = 0; i < n; i++) if (slot_for(start + i, start / kChunk) < 0) return false;
        return true;
    }
};

}  // namespace

// Evidence that the span protection leaves every other KVFlash caller alone: random forwards that fit the pool (appends
// with an unaligned restart, and re-forwards of older rows), run on the pager and on the pre-fix rules, keep the same
// resident set and the same result. A run ends at the first forward where the pre-fix rules would have taken a victim
// inside the forward's own span. For an append that is only a chunk already written (the bug the fix is for, only with
// a scorer: LRU ranks a forward's chunks newest). A re-forward over resident rows far behind the head can also lose a
// chunk it has not reached yet under the pre-fix rules (it would page back in); the fix keeps it, the one intended
// difference.
TEST_CASE(KvflashPagerTruncateFixture, alloc_span_victims_match_the_pre_fix_selection_for_normal_forwards) {
    constexpr int blocks = 24, max_chunks = 400;
    int compared = 0, ended_by_touched = 0, ended_by_untouched = 0;
    for (const bool scored : {false, true}) {
        for (uint32_t seed = 1; seed <= 150; seed++) {
            std::mt19937 rng(seed * 2654435761u + scored);
            const auto sc = [seed](int c) { return (float)(((uint32_t)c * 2246822519u ^ seed * 3266489917u) % 1000); };
            TestPager p;
            must(p.attach(logic_cfg(blocks), {}, {}));
            PreFixModel m(blocks, max_chunks);
            if (scored) { p.score_hook = sc; m.score = sc; }
            int head = 0;
            for (int step = 0; step < 60 && head < (max_chunks - 30) * kChunk; step++) {
                const int cap = (blocks - 2) * kChunk;
                const bool append = step == 0 || rng() % 12 != 0;
                const int start = append ? std::max(0, head - (int)(rng() % kChunk)) : (int)(rng() % head);
                const int n = 1 + (int)(rng() % (append ? cap : 4 * kChunk));
                m.evicted_touched = m.evicted_untouched = false;
                const bool want = m.alloc_span(start, n);
                if (m.evicted_touched || m.evicted_untouched) {
                    CHECK(scored || !m.evicted_touched);
                    CHECK(append ? !m.evicted_untouched : true);
                    (m.evicted_touched ? ended_by_touched : ended_by_untouched)++;
                    break;
                }
                CHECK(p.alloc_span(start, n) == want);
                for (int c = 0; c < max_chunks; c++) CHECK(p.is_resident(c) == (bool)m.res[(size_t)c]);
                head = std::max(head, start + n);
                compared++;
            }
        }
    }
    CHECK(compared > 2000);
    CHECK(ended_by_touched > 0);   // the scored runs do reach the case the fix exists for
    std::printf("[differential] compared=%d ended_by_touched=%d ended_by_untouched=%d\n", compared, ended_by_touched,
                ended_by_untouched);
}
