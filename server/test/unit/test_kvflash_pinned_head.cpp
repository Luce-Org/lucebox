// KVFlash pinned head: the shared-prefix head a concurrent slot pins, engine-free, CPU only. The head is #941's cut
// rounded down to the engine's chunk grid; this pins the rounding, the decision, the binder (which chunks a slot holds
// pinned, the count recall reads), the bounded capture, the blob an attach adopts, the re-pin helper and the
// snapshot a head is saved to the disk cache as.

#include "CppUnitTestFramework.hpp"
#include "kvflash_pager_test_host.h"

#include "kvflash_pinned_head.h"
#include "kvflash_head_snapshot.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

using namespace luce::common;

namespace {
struct KvflashPinnedHeadFixture {};

constexpr int kCt = 64;

std::vector<int32_t> ids_of(int n, int salt = 0) {
    std::vector<int32_t> v((size_t)n);
    for (int i = 0; i < n; i++) v[(size_t)i] = i * 7 + 1 + salt;
    return v;
}

// A fake pager with the surface the binder uses.
struct FakePager {
    int ct = kCt, held = 10;
    std::vector<int> pinned;
    int chunk_tokens() const { return ct; }
    int n_chunks() const { return held; }
    void pin_range(int64_t, int64_t hi) { pinned.assign((size_t)((hi + ct - 1) / ct), 1); }
    bool is_pinned(int c) const { return c < (int)pinned.size(); }
    void unpin_all() { pinned.clear(); }
    void synchronize_paging() const {}
    size_t chunk_bytes() const { return 100; }
    std::vector<uint8_t> blob_header(int k) const { return {(uint8_t)k, 0}; }
    void read_chunk(int c, uint8_t * dst) const { std::fill(dst, dst + 100, (uint8_t)(c + 1)); }
};

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
            k.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, 512, 2));
            v.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 4, 512, 2));
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    }
    ~Harness() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
};
KvFlashConfig cfg8() {   // 8 blocks of 64 tokens
    KvFlashConfig c;
    c.pool_tokens = 512; c.chunk_tokens = 64; c.sink_chunks = 1; c.tail_window_chunks = 2;
    return c;
}
void must(bool ok) { if (!ok) throw std::runtime_error("pinned head test precondition failed"); }
// Two KVFLASH2 blobs of k chunks equal except the f32 score of each ledger entry (recency in a pager that served a turn;
// zero in a head, as in a fresh pager).
bool same_but_scores(std::vector<uint8_t> a, std::vector<uint8_t> b, int k, size_t chunk_bytes) {
    constexpr size_t kHdr = 8 + 9 * 4;
    if (a.size() != b.size() || a.size() != kHdr + (size_t)k * (chunk_bytes + 8)) return false;
    for (int c = 0; c < k; c++)
        for (size_t i = 4; i < 8; i++) a[kHdr + (size_t)k * chunk_bytes + (size_t)c * 8 + i] = b[kHdr + (size_t)k * chunk_bytes + (size_t)c * 8 + i] = 0;
    return a == b;
}
// The KVFLASH2 blob of the pager's first k chunks, assembled the way a slot bundle does.
std::vector<uint8_t> kvflash2_of(const TestPager & p, int k) {
    std::vector<uint8_t> out = p.blob_header(k);
    const size_t cb = p.chunk_bytes();
    out.resize(out.size() + (size_t)k * cb);
    for (int c = 0; c < k; c++) p.read_chunk(c, out.data() + out.size() - (size_t)(k - c) * cb);
    const std::vector<uint8_t> led = p.ledger_blob(k);
    out.insert(out.end(), led.begin(), led.end());
    return out;
}
}  // namespace

// ---- the grid ------------------------------------------------------------------------------------------------

TEST_CASE(KvflashPinnedHeadFixture, the_unit_is_a_chunk_or_the_lcm_with_a_larger_prefill_slice) {
    CHECK(kvflash_head_unit(64, 0) == 64);
    CHECK(kvflash_head_unit(64, 64) == 64);
    CHECK(kvflash_head_unit(64, 512) == 512);
    CHECK(kvflash_head_unit(64, 96) == 192);     // lcm
    CHECK(kvflash_head_unit(0, 512) == 1);
}

TEST_CASE(KvflashPinnedHeadFixture, the_boundary_is_the_servers_cut_rounded_down_to_the_grid_and_the_role_end_only_when_sharing_is_off) {
    CHECK(kvflash_head_boundary(1900, 700, 64, 512) == 1536);   // #941's cut wins, floored to the grid
    CHECK(kvflash_head_boundary(1536, 700, 64, 512) == 1536);   // on the grid: unchanged
    CHECK(kvflash_head_boundary(0, 700, 64, 512) == 0);         // no cut found: no head
    CHECK(kvflash_head_boundary(-1, 1100, 64, 512) == 1024);    // sharing off: the role end
    CHECK(kvflash_head_boundary(-1, 0, 64, 512) == 0);
    CHECK(kvflash_head_boundary(300, 700, 64, 512) == 0);       // under one unit
}

TEST_CASE(KvflashPinnedHeadFixture, k_is_a_whole_number_of_cut_units_capped_by_k_max_and_never_one_chunk) {
    CHECK(kvflash_head_k(1536, kCt, 512, 100) == 24);
    CHECK(kvflash_head_k(1536, kCt, 512, 20) == 16);            // k_max 20 rounds down to two units of 8 chunks
    CHECK(kvflash_head_k(1536, kCt, 512, 7) == 0);              // under one unit
    CHECK(kvflash_head_k(256, kCt, 64, 100) == 4);
    CHECK(kvflash_head_k(64, kCt, 64, 100) == 0);               // one chunk is not worth a head
    CHECK(kvflash_head_k(0, kCt, 512, 100) == 0);
}

TEST_CASE(KvflashPinnedHeadFixture, the_hash_chain_names_a_prefix_by_its_whole_chunks_only) {
    const auto ids = ids_of(5 * kCt + 10);
    const auto h = kvflash_head_hashes(ids, 5, kCt);
    REQUIRE(h.size() == 5);
    CHECK(kvflash_head_hashes(ids, 3, kCt).back() == h[2]);
    CHECK(kvflash_head_hashes(ids, 9, kCt).size() == 5);        // clamped to the whole chunks the ids hold
    auto other = ids; other[2 * kCt + 1] ^= 1;
    const auto g = kvflash_head_hashes(other, 5, kCt);
    CHECK(g[1] == h[1]);                                        // the chain agrees up to the changed chunk
    CHECK(g[2] != h[2]);
    CHECK(g[4] != h[4]);                                        // and never again
    CHECK(kvflash_head_hashes(ids, 0, kCt).empty());
}

TEST_CASE(KvflashPinnedHeadFixture, decide_attaches_a_known_head_captures_an_unknown_one_and_k_does_not_depend_on_what_is_known) {
    const auto ids = ids_of(3000);
    const uint64_t known = kvflash_head_hashes(ids, 24, kCt).back();
    const auto none = [](uint64_t, int) { return false; };
    const auto has = [known](uint64_t h, int k) { return h == known && k == 24; };
    const auto cold = kvflash_head_decide(ids, 1536, kCt, 512, 100, none);
    CHECK(cold.kind == KvFlashPreambleDecision::Capture);
    CHECK(cold.k == 24);
    const auto warm = kvflash_head_decide(ids, 1536, kCt, 512, 100, has);
    CHECK(warm.kind == KvFlashPreambleDecision::Attach);
    CHECK(warm.k == cold.k);
    CHECK(kvflash_head_decide(ids, 0, kCt, 512, 100, has).kind == KvFlashPreambleDecision::Cold);       // no boundary
    CHECK(kvflash_head_decide(ids, 1536, kCt, 512, 7, has).kind == KvFlashPreambleDecision::Cold);      // k_max under a unit
    CHECK(kvflash_head_decide(ids_of(1000), 1536, kCt, 512, 100, has).kind == KvFlashPreambleDecision::Cold);   // ids shorter than the head
}

TEST_CASE(KvflashPinnedHeadFixture, k_max_is_a_quarter_of_the_slot_pool) {
    CHECK(kvflash_head_k_max(256) == 64);
    CHECK(kvflash_head_k_max(10) == 2);
    CHECK(kvflash_head_k_max(0) == 0);
    CHECK(kvflash_head_k_max(3) == 0);                          // too small a pool to pin a head at all
}

// ---- the binder -------------------------------------------------------------------------------------------

TEST_CASE(KvflashPinnedHeadFixture, ensure_pins_records_the_count_and_a_binding_the_pager_cannot_hold_is_released) {
    KvFlashHeadBinder bd(2);
    FakePager p;
    CHECK(bd.bind(0, 0xabc, 3));
    CHECK(bd.ensure(0, p) == 3);
    CHECK(bd.pins(0) == 3);
    CHECK(bd.binding(0).hash == 0xabc);
    p.held = 2;                                                 // the slot's length fell below the head
    CHECK(bd.ensure(0, p) == 0);
    CHECK(bd.pins(0) == 0);
    CHECK(bd.binding(0).k == 0);
    FakePager q;
    CHECK(bd.ensure(1, q) == 0);                                // no binding: nothing to pin
    CHECK(!bd.bind(5, 1, 3));                                   // out of range
    CHECK(!bd.bind(1, 1, 0));                                   // k = 0 binds nothing
}

TEST_CASE(KvflashPinnedHeadFixture, rebinding_a_slot_replaces_its_binding_and_unbind_is_idempotent) {
    KvFlashHeadBinder bd(1);
    FakePager p;
    REQUIRE(bd.bind(0, 1, 3));
    REQUIRE(bd.bind(0, 2, 4));
    CHECK(bd.binding(0).hash == 2);
    CHECK(bd.binding(0).k == 4);
    bd.ensure(0, p);
    bd.unbind(0);
    bd.unbind(0);
    CHECK(bd.binding(0).k == 0);
    CHECK(bd.pins(0) == 0);
}

// ---- capture and the blob an attach adopts --------------------------------------------------------------------

TEST_CASE(KvflashPinnedHeadFixture, a_capture_is_the_pager_chunks_plus_the_slab_and_binds_and_pins_the_capturing_slot) {
    Harness h; TestPager p;
    must(p.attach(cfg8(), h.k, h.v));
    must(p.alloc_span(0, 5 * 64));
    for (ggml_tensor * t : h.k) { std::vector<uint8_t> junk(ggml_nbytes(t)); for (size_t i = 0; i < junk.size(); i++) junk[i] = (uint8_t)(i * 31 + 7); ggml_backend_tensor_set(t, junk.data(), 0, junk.size()); }
    const auto ids = ids_of(5 * kCt);
    KvFlashHeadBinder bd(1);
    const KvFlashHead head = kvflash_head_capture(bd, 0, p, ids, 3, 3, std::vector<uint8_t>{9, 9, 9});
    REQUIRE(head.k == 3);
    CHECK(head.hash == kvflash_head_hashes(ids, 3, kCt).back());
    CHECK(head.slab == std::vector<uint8_t>({9, 9, 9}));
    CHECK(head.chunks.size() == 3 * p.chunk_bytes());
    // the bytes a slot bundle writes for those chunks: header, K/V and ledger flags exact, scores zero
    CHECK(same_but_scores(kvflash_head_blob(head), kvflash2_of(p, 3), 3, p.chunk_bytes()));
    CHECK(bd.binding(0).k == 3);
    CHECK(bd.pins(0) == 3);
    // and the blob restores into a fresh pager with the same K/V bytes
    Harness h2; TestPager q;
    must(q.attach(cfg8(), h2.k, h2.v));
    const auto blob = kvflash_head_blob(head);
    REQUIRE(q.deserialize(blob.data(), blob.size(), 3));
    CHECK(same_but_scores(kvflash2_of(q, 3), kvflash2_of(p, 3), 3, p.chunk_bytes()));
}

TEST_CASE(KvflashPinnedHeadFixture, a_capture_is_refused_past_k_max_below_two_chunks_and_when_the_pager_or_the_ids_lack_the_chunks) {
    KvFlashHeadBinder bd(1);
    FakePager p;                                                // 10 chunks held
    const auto ids = ids_of(12 * kCt);
    CHECK(kvflash_head_capture(bd, 0, p, ids, 6, /*k_max=*/5, {}).k == 0);
    CHECK(kvflash_head_capture(bd, 0, p, ids, 1, 5, {}).k == 0);          // one chunk is not worth a head
    p.held = 3;
    CHECK(kvflash_head_capture(bd, 0, p, ids, 4, 5, {}).k == 0);          // chunk 3 does not exist
    CHECK(kvflash_head_capture(bd, 0, p, ids_of(2 * kCt), 3, 5, {}).k == 0);   // ids shorter than k chunks
    CHECK(bd.binding(0).k == 0);                                          // a refusal binds nothing
    FakePager q;
    CHECK(kvflash_head_capture(bd, 0, q, ids, 3, 5, {1}).k == 3);         // the same call, accepted
    CHECK(bd.binding(0).k == 3);
}

// ---- re-pin --------------------------------------------------------------------------------------------------

TEST_CASE(KvflashPinnedHeadFixture, repin_is_a_no_op_without_a_binding_and_pins_k_chunks_with_one) {
    Harness h; TestPager p;
    must(p.attach(cfg8(), h.k, h.v));
    must(p.alloc_span(0, 4 * 64));
    int k = 0;
    const KvFlashRepin none = kvflash_repin_head(p, k);
    CHECK(none.accepted == 0);
    CHECK(!none.released);
    CHECK(!p.is_pinned(0));
    k = 3;
    const KvFlashRepin r = kvflash_repin_head(p, k);
    CHECK(r.accepted == 3);
    CHECK(!r.released);
    CHECK(p.is_pinned(0) && p.is_pinned(2) && !p.is_pinned(3));
    CHECK(k == 3);
}

TEST_CASE(KvflashPinnedHeadFixture, repin_releases_the_binding_when_the_length_fell_below_the_head_and_holds_at_exactly_the_head) {
    Harness h; TestPager p;
    must(p.attach(cfg8(), h.k, h.v));
    must(p.alloc_span(0, 4 * 64));
    int k = 3;
    must(kvflash_repin_head(p, k).accepted == 3);
    p.truncate(3 * 64);
    CHECK(kvflash_repin_head(p, k).accepted == 3);              // exactly the head: held
    p.truncate(64 + 10);
    const KvFlashRepin r = kvflash_repin_head(p, k);
    CHECK(r.released);
    CHECK(k == 0);
    CHECK(!p.is_pinned(0));
}

// ---- the snapshot a head is saved as --------------------------------------------------------------------------

TEST_CASE(KvflashPinnedHeadFixture, a_head_round_trips_through_its_snapshot_and_a_snapshot_of_another_shape_is_refused) {
    KvFlashHead head;
    head.k = 3; head.hash = kvflash_head_hashes(ids_of(5 * kCt), 3, kCt).back();
    head.hdr = {1, 2, 3, 4}; head.slab = {7, 8, 9, 10, 11}; head.chunks.resize(3 * 100);
    for (size_t i = 0; i < head.chunks.size(); i++) head.chunks[i] = (uint8_t)(i * 13 + 1);
    const auto ids = ids_of(5 * kCt);
    OwnedSnapshot snap = kvflash_head_to_snapshot(head, kCt);
    REQUIRE(!snap.empty());
    CHECK(snap.cur_pos() == 3 * kCt);
    KvFlashHead back;
    REQUIRE(kvflash_head_from_snapshot(snap, ids, kCt, /*chunk_bytes=*/100, back));
    CHECK(back.k == 3);
    CHECK(back.hash == head.hash);
    CHECK(back.hdr == head.hdr);
    CHECK(back.chunks == head.chunks);
    CHECK(back.slab == head.slab);
    KvFlashHead bad;
    CHECK(!kvflash_head_from_snapshot(snap, ids, kCt, /*another geometry*/ 128, bad));   // chunks != k x chunk_bytes
    CHECK(bad.k == 0);
    CHECK(!kvflash_head_from_snapshot(snap, ids_of(2 * kCt), kCt, 100, bad));            // key shorter than the head
    CHECK(!kvflash_head_from_snapshot(OwnedSnapshot{}, ids, kCt, 100, bad));
}

TEST_CASE(KvflashPinnedHeadFixture, the_layout_id_follows_the_execution_identity) {
    const auto a = kvflash_head_layout_id("model-a|q8|ctx4096");
    CHECK(a == kvflash_head_layout_id("model-a|q8|ctx4096"));
    CHECK(a != kvflash_head_layout_id("model-a|q8|ctx8192"));
    CHECK(a != kvflash_head_layout_id(""));
}
