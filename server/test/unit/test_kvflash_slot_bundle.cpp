// Pure unit tests for the C1.5 F3 per-slot bundle directory (kvflash_slot_bundle.h): the record
// and log format, generation naming, the dirty range, head pack/unpack, assembly from a log with
// holes and a torn tail, commit + load on a directory, a crash after every commit step, compaction,
// the file helpers and the writer thread. No GPU.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_slot_bundle.h"
#include "../../src/common/kvflash_bundle_digest.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <filesystem>
#include <unistd.h>

using namespace luce::common;
namespace fs = std::filesystem;

namespace {
struct KvflashSlotBundleFixture {};

constexpr size_t CB = 8;                                  // a toy chunk of 8 bytes
const size_t RB = kvflash_record_bytes(CB);               // 16 + 8
const KvFlashBundleLimits LIM{1 << 20, 1 << 20, 1 << 20};  // history, chunks, slab bytes: no limit for these toys

std::vector<uint8_t> chunk(uint8_t fill) { return std::vector<uint8_t>(CB, fill); }
std::vector<uint8_t> hdr44() { std::vector<uint8_t> h(kKvFlashPagerHdr); for (size_t i = 0; i < h.size(); ++i) h[i] = (uint8_t)(0xa0 + i); return h; }

KvFlashSlotHead sample_head() {
    KvFlashSlotHead h;
    h.identity = kvflash_fnv1a("model|size|mtime|q8_0|q8_0|rot0|64|32|8192");
    h.gen = 3; h.log_gen = 2; h.log_bytes = kKvFlashLogHeader + 3 * RB;
    h.ckpt_pos = 3; h.trunk_pos = 1;
    h.end_gen = 3; h.ckpt_gen = 2; h.trunk_gen = 1;
    h.extra_pos = 2; h.extra_gen = 4; h.pin_pos = 2;     // the third rung (v5), between the trunk and the checkpoint, a pinned branch point
    h.history = {11, 22, 33, 44, 55};
    h.pager_hdr = hdr44();
    h.map = {kKvFlashLogHeader, kKvFlashLogHeader + RB, 0, kKvFlashLogHeader + 2 * RB};
    h.ledger.assign(4 * 8, 0); h.ledger[0] = 1; h.ledger[1] = 1; h.ledger[9] = 1; h.ledger[25] = 1;   // chunks 0, 1, 3 materialized, 2 a hole
    return h;
}

bool same(const KvFlashSlotHead & a, const KvFlashSlotHead & b) {
    return a.identity == b.identity && a.gen == b.gen && a.log_gen == b.log_gen && a.log_bytes == b.log_bytes &&
           a.ckpt_pos == b.ckpt_pos && a.trunk_pos == b.trunk_pos && a.end_gen == b.end_gen && a.ckpt_gen == b.ckpt_gen &&
           a.trunk_gen == b.trunk_gen && a.extra_pos == b.extra_pos && a.extra_gen == b.extra_gen && a.pin_pos == b.pin_pos && a.history == b.history && a.pager_hdr == b.pager_hdr && a.map == b.map && a.ledger == b.ledger;
}

inline std::vector<uint8_t> kvflash_record(uint32_t chunk, uint32_t gen, const uint8_t * payload, size_t n) {
    const auto t = kvflash_record_tag(chunk, gen, payload, n);
    std::vector<uint8_t> out(t.begin(), t.end());
    out.insert(out.end(), payload, payload + n);
    return out;
}
// An in-memory log: header + records for chunks 0, 1, 3 (chunk 2 is a hole in sample_head's map).
std::vector<uint8_t> sample_log() {
    std::vector<uint8_t> log = kvflash_log_header(2);
    for (uint32_t c : {0u, 1u, 3u}) { const auto r = kvflash_record(c, 3, chunk((uint8_t)(0x10 + c)).data(), CB); log.insert(log.end(), r.begin(), r.end()); }
    return log;
}
auto mem_reader(const std::vector<uint8_t> & log) {
    return [&log](uint64_t off, uint8_t * dst, size_t n) { if (off + n > log.size()) return false; std::memcpy(dst, log.data() + off, n); return true; };
}
std::vector<uint8_t> expected_blob(const KvFlashSlotHead & h) {
    std::vector<uint8_t> b = h.pager_hdr;
    for (uint8_t c : {0x10, 0x11}) { const auto k = chunk(c); b.insert(b.end(), k.begin(), k.end()); }
    b.insert(b.end(), CB, 0);                                                                       // the hole
    const auto k = chunk(0x13); b.insert(b.end(), k.begin(), k.end());
    b.insert(b.end(), h.ledger.begin(), h.ledger.end());
    return b;
}

// A commit of a whole park: a fresh log with records for chunks [0, n), an end slab (id 2 gen), a
// ckpt slab (id 2 gen + 1), the head.
KvFlashBundleCommit park(const std::string & dir, uint32_t gen, int n, uint8_t fill, uint64_t identity) {
    KvFlashBundleCommit c;
    c.dir = dir; c.gen = gen; c.log_gen = gen; c.new_log = true; c.new_dir = true; c.record_bytes = RB;
    KvFlashSlotHead h;
    h.identity = identity; h.gen = gen; h.log_gen = gen; h.end_gen = 2 * gen; h.ckpt_gen = 2 * gen + 1; h.ckpt_pos = 2;
    h.history.assign((size_t)n * 2, (int32_t)fill); h.pager_hdr = hdr44(); h.ledger.assign((size_t)n * 8, fill);
    for (int i = 0; i < n; ++i) { c.records.push_back({(uint32_t)i, chunk((uint8_t)(fill + i))}); h.map.push_back(kKvFlashLogHeader + (uint64_t)i * RB); }
    h.log_bytes = kKvFlashLogHeader + (uint64_t)n * RB;
    c.files = {{kvflash_bundle_slab_name(h.end_gen), kvflash_slab_pack({{fill, 1}}, {{fill, 2}})},
               {kvflash_bundle_slab_name(h.ckpt_gen), kvflash_slab_pack({{fill, 3}}, {{fill, 4}})}};
    c.head = kvflash_head_pack(h);
    return c;
}
// A park over `prev` resumed at chunk `from`: chunks [from, nc) rewritten, a new end slab (id 2 gen), the old one unlinked;
// appended to prev's log, or (compacted) copied with prev's live records into a new log that replaces it.
KvFlashBundleCommit repark(const std::string & dir, const KvFlashSlotHead & prev, uint32_t gen, int from, uint8_t fill, bool compacted) {
    KvFlashBundleCommit c;
    c.dir = dir; c.gen = gen; c.record_bytes = RB;
    KvFlashSlotHead h = prev;
    h.gen = gen; h.end_gen = 2 * gen;
    if (compacted) {
        c.new_log = true; c.log_gen = gen; c.copy_from = prev.log_gen; h.log_gen = gen; h.log_bytes = kKvFlashLogHeader;
        for (int i = 0; i < from; ++i) if (prev.map[(size_t)i]) { c.copies.push_back(prev.map[(size_t)i]); h.map[(size_t)i] = h.log_bytes; h.log_bytes += RB; }
        c.unlink.push_back(kvflash_bundle_log_name(prev.log_gen));
    } else {
        c.log_gen = prev.log_gen; c.log_bytes = prev.log_bytes;
    }
    for (int i = from; i < (int)prev.map.size(); ++i) { c.records.push_back({(uint32_t)i, chunk((uint8_t)(fill + i))}); h.map[(size_t)i] = h.log_bytes; h.log_bytes += RB; }
    c.files = {{kvflash_bundle_slab_name(h.end_gen), kvflash_slab_pack({{fill, 1}}, {{fill, 2}})}};
    c.head = kvflash_head_pack(h);
    c.unlink.push_back(kvflash_bundle_slab_name(prev.end_gen));
    return c;
}
}  // namespace

TEST_CASE(KvflashSlotBundleFixture, record_tag_log_header_and_crc_round_trip) {
    const auto payload = chunk(0x5a);
    const std::vector<uint8_t> r = kvflash_record(5, 7, payload.data(), payload.size());
    REQUIRE(r.size() == kKvFlashRecordTag + CB);
    CHECK(RB == r.size());
    KvFlashRecordTag t;
    REQUIRE(kvflash_record_tag_parse(r.data(), r.size(), t));
    CHECK(t.chunk == 5u); CHECK(t.gen == 7u); CHECK(t.crc == kvflash_crc64(payload.data(), payload.size()));
    CHECK(!kvflash_record_tag_parse(r.data(), kKvFlashRecordTag - 1, t));
    CHECK(kvflash_crc64(payload.data(), payload.size()) != 0);
    CHECK(kvflash_crc64((const uint8_t *)"123456789", 9) == 0x995DC9BBDF1939FAULL);           // CRC-64/XZ check value
    auto flipped = payload; flipped[3] ^= 0x80;
    CHECK(kvflash_crc64(flipped.data(), flipped.size()) != t.crc);
    const std::vector<uint8_t> lh = kvflash_log_header(9);
    REQUIRE(lh.size() == kKvFlashLogHeader);
    uint32_t gen = 0;
    REQUIRE(kvflash_log_header_parse(lh.data(), lh.size(), gen));
    CHECK(gen == 9u);
    auto bad = lh; bad[0] ^= 1;
    CHECK(!kvflash_log_header_parse(bad.data(), bad.size(), gen));
    CHECK(!kvflash_log_header_parse(lh.data(), lh.size() - 1, gen));
}

TEST_CASE(KvflashSlotBundleFixture, generation_naming_and_the_dirty_range) {
    CHECK(kvflash_bundle_log_name(4) == "chunks.4.log");
    CHECK(kvflash_bundle_slab_name(4) == "slab.4");
    CHECK(kvflash_bundle_dirty_chunks(0, 5, 64) == std::make_pair(0, 5));         // a cold admit: everything
    CHECK(kvflash_bundle_dirty_chunks(64, 5, 64) == std::make_pair(1, 5));        // resumed at a chunk boundary
    CHECK(kvflash_bundle_dirty_chunks(100, 5, 64) == std::make_pair(1, 5));       // inside chunk 1: the chunk is rewritten
    CHECK(kvflash_bundle_dirty_chunks(320, 5, 64) == std::make_pair(5, 5));       // resumed at the end: nothing sealed changed
    CHECK(kvflash_bundle_dirty_chunks(1000, 5, 64) == std::make_pair(5, 5));      // never past n_chunks
    CHECK(kvflash_bundle_dirty_chunks(-1, 5, 64) == std::make_pair(0, 5));
}

TEST_CASE(KvflashSlotBundleFixture, head_pack_unpack_is_byte_exact_and_refuses_bad_heads) {
    const KvFlashSlotHead h = sample_head();
    const std::vector<uint8_t> bytes = kvflash_head_pack(h);
    CHECK(std::memcmp(bytes.data(), "KVFSLOT3", 8) == 0);
    KvFlashSlotHead out;
    REQUIRE(kvflash_head_unpack(bytes.data(), bytes.size(), h.identity, out));
    CHECK(same(out, h));
    CHECK(kvflash_head_pack(out) == bytes);
    out = KvFlashSlotHead{};
    CHECK(!kvflash_head_unpack(bytes.data(), bytes.size(), h.identity + 1, out));          // another server
    CHECK(out.history.empty());                                                            // untouched on refusal
    for (size_t n = 0; n < bytes.size(); n += 7) if (n + 12 != bytes.size() && n + 8 != bytes.size() && n + 4 != bytes.size()) CHECK(!kvflash_head_unpack(bytes.data(), n, h.identity, out));   // those two lengths are the older heads, below
    auto longer = bytes; longer.push_back(0);
    CHECK(!kvflash_head_unpack(longer.data(), longer.size(), h.identity, out));
    auto bad = bytes; bad[0] ^= 1;
    CHECK(!kvflash_head_unpack(bad.data(), bad.size(), h.identity, out));
    CHECK(!kvflash_head_unpack(nullptr, 0, h.identity, out));
    auto t = h; t.history.clear();
    auto tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));
    t = h; t.ckpt_pos = 6;  tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));
    t = h; t.trunk_pos = 6; tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));
    t = h; t.end_gen = 0;   tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));
    t = h; t.ledger.pop_back(); tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));
    t = h; t.pager_hdr.pop_back(); tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));
    t = h; t.ckpt_pos = 0; t.ckpt_gen = 0; t.trunk_pos = 0; t.trunk_gen = 0;             // no rungs: fine
    tb = kvflash_head_pack(t); REQUIRE(kvflash_head_unpack(tb.data(), tb.size(), h.identity, out)); CHECK(same(out, t));
    t = h; t.ckpt_pos = 0;  tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));   // a rung needs both
    // the ladder: a trunk is below the checkpoint, and the three slabs are three files (a damaged head that names the
    // state-at-4000 file for a trunk at 3000 would restore it as position 3000)
    t = h; t.trunk_pos = 3;  tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));   // a trunk at the checkpoint
    t = h; t.trunk_pos = 4;  tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));   // a trunk above it
    t = h; t.ckpt_pos = 0; t.ckpt_gen = 0; tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));   // a trunk alone
    t = h; t.trunk_gen = h.ckpt_gen; tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));   // two rungs, one file
    t = h; t.ckpt_gen = h.end_gen;   tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));   // a rung is not the end
    t = h; t.trunk_gen = h.end_gen;  tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));
    t = h; t.trunk_pos = 0; t.trunk_gen = 0; tb = kvflash_head_pack(t); REQUIRE(kvflash_head_unpack(tb.data(), tb.size(), h.identity, out)); CHECK(same(out, t));   // a checkpoint alone: fine
    // the third rung and its pin end the head (+12 bytes). An older head (the base, the branch_pos u32 of the first B4 try, or the
    // third without a pin) reads as no third rung and no pin. A head cut short by exactly those bytes reads as the older one: accepted,
    // the head is written by an atomic rename.
    CHECK(bytes.size() == kvflash_head_bytes(h.history.size(), h.map.size()));
    for (size_t tail : {(size_t)0, (size_t)4, (size_t)8}) {                                  // base head, branch_pos head, third without pin
        auto old = std::vector<uint8_t>(bytes.begin(), bytes.end() - (tail == 8 ? 4 : 12));
        if (tail == 4) old.insert(old.end(), 4, 0);
        REQUIRE(kvflash_head_unpack(old.data(), old.size(), h.identity, out));
        auto want = h; want.pin_pos = 0;
        if (tail != 8) { want.extra_pos = 0; want.extra_gen = 0; }
        CHECK(same(out, want));
    }
    // the pin: a position of a rung of the head, or none; with a separate third it names the third
    t = h; t.pin_pos = 6;             tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));   // above the history
    t = h; t.pin_pos = 1;             tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));   // names the trunk while a third exists
    t = h; t.pin_pos = 4;             tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));   // names no rung
    t = h; t.pin_pos = 0;             tb = kvflash_head_pack(t); REQUIRE(kvflash_head_unpack(tb.data(), tb.size(), h.identity, out)); CHECK(same(out, t));   // a demoted third: unpinned
    t = h; t.extra_pos = 0; t.extra_gen = 0; t.pin_pos = h.ckpt_pos; tb = kvflash_head_pack(t); REQUIRE(kvflash_head_unpack(tb.data(), tb.size(), h.identity, out)); CHECK(same(out, t));   // the pin at the checkpoint, no separate slab
    t = h; t.extra_pos = 0; t.extra_gen = 0; t.pin_pos = h.trunk_pos; tb = kvflash_head_pack(t); REQUIRE(kvflash_head_unpack(tb.data(), tb.size(), h.identity, out)); CHECK(same(out, t));   // ... or at the trunk
    t = h; t.extra_pos = 0; t.extra_gen = 0; t.pin_pos = 4; tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));
    t = h; t.extra_gen = h.end_gen;   tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));   // the third is not the end
    t = h; t.extra_gen = h.ckpt_gen;  tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));
    t = h; t.extra_gen = h.trunk_gen; tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));
    t = h; t.extra_pos = 6;           tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));   // above the history
    t = h; t.extra_pos = 0;           tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));   // a rung needs both
    t = h; t.extra_gen = 0;           tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));
    t = h; t.extra_pos = h.ckpt_pos;  tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));   // a third at a rung's position adds nothing
    t = h; t.extra_pos = h.trunk_pos; tb = kvflash_head_pack(t); CHECK(!kvflash_head_unpack(tb.data(), tb.size(), h.identity, out));
    t = h; t.extra_pos = 0; t.extra_gen = 0; t.pin_pos = 0; tb = kvflash_head_pack(t); REQUIRE(kvflash_head_unpack(tb.data(), tb.size(), h.identity, out)); CHECK(same(out, t));   // no third: fine
    CHECK(kvflash_head_max_bytes(5, 4) >= bytes.size());
    const auto sb = kvflash_slab_pack({{1, 2}, {}}, {{3}});
    std::vector<std::vector<uint8_t>> ssm, conv;
    REQUIRE(kvflash_slab_unpack(sb.data(), sb.size(), ssm, conv));
    CHECK((ssm == std::vector<std::vector<uint8_t>>{{1, 2}, {}})); CHECK((conv == std::vector<std::vector<uint8_t>>{{3}}));
    CHECK(!kvflash_slab_unpack(sb.data(), sb.size() - 1, ssm, conv));
    CHECK(kvflash_slab_max_bytes(3) >= sb.size());
    CHECK(sb.capacity() == sb.size());                                                     // packed at its exact size: one slab of staging, not 1.5
    CHECK(kvflash_slab_bytes({{1, 2}, {}}, {{3}}) == sb.size());
}

TEST_CASE(KvflashSlotBundleFixture, a_map_that_disagrees_with_the_ledger_is_refused) {
    const KvFlashSlotHead h = sample_head();
    const std::vector<uint8_t> log = sample_log();
    std::vector<uint8_t> blob;
    std::string why;
    auto hole = h; hole.map[0] = 0;                                                         // materialized, no record: would restore as zero KV
    CHECK(!kvflash_bundle_assemble(hole, CB, mem_reader(log), blob, why)); CHECK(why.find("chunk 0") != std::string::npos);
    auto ghost = h; ghost.ledger[25] = 0;                                                   // a record for a chunk never materialized
    CHECK(!kvflash_bundle_assemble(ghost, CB, mem_reader(log), blob, why)); CHECK(why.find("chunk 3") != std::string::npos);
    auto odd = h; odd.ledger.pop_back();                                                    // a ledger of the wrong size
    CHECK(!kvflash_bundle_assemble(odd, CB, mem_reader(log), blob, why));
    CHECK(blob.empty());
}

TEST_CASE(KvflashSlotBundleFixture, the_files_of_a_failed_park_the_durable_head_does_not_name_are_leftovers) {
    const KvFlashSlotHead d = sample_head();                                                // log 2, slabs 3 (end), 2 (ckpt), 1 (trunk)
    KvFlashSlotHead f = d; f.gen = 4; f.log_gen = 4; f.end_gen = 5; f.ckpt_gen = 6;         // a compaction with a new ckpt, the trunk kept
    CHECK((kvflash_bundle_leftovers(d, f) == std::vector<std::string>{kvflash_bundle_log_name(4), kvflash_bundle_slab_name(5), kvflash_bundle_slab_name(6)}));
    f = d; f.gen = 4; f.end_gen = 5;                                                        // an append: the shared log is no leftover
    CHECK((kvflash_bundle_leftovers(d, f) == std::vector<std::string>{kvflash_bundle_slab_name(5)}));
    f = d; f.gen = 4; f.end_gen = 5; f.trunk_gen = d.ckpt_gen; f.ckpt_gen = 6;              // the ckpt moved to the trunk: its file stays
    CHECK((kvflash_bundle_leftovers(d, f) == std::vector<std::string>{kvflash_bundle_slab_name(5), kvflash_bundle_slab_name(6)}));
    CHECK(kvflash_bundle_leftovers(d, d).empty());
    f = d; f.gen = 4; f.end_gen = 5; f.extra_gen = 7;                                       // a new third rung: its file is a leftover
    CHECK((kvflash_bundle_leftovers(d, f) == std::vector<std::string>{kvflash_bundle_slab_name(5), kvflash_bundle_slab_name(7)}));
    f = d; f.gen = 4; f.end_gen = 5; f.ckpt_gen = d.extra_gen; f.ckpt_pos = d.extra_pos; f.extra_gen = 0; f.extra_pos = 0;   // the third became the ckpt: its file stays
    CHECK((kvflash_bundle_leftovers(d, f) == std::vector<std::string>{kvflash_bundle_slab_name(5)}));
}

TEST_CASE(KvflashSlotBundleFixture, the_third_rung_survives_a_commit_a_load_a_validate_a_restart_cleanup_and_a_remove) {
    const std::string dir = "kvflash_f3_unit_third";
    fs::remove_all(dir);
    const uint64_t id = 78;
    auto c = park(dir, 1, 3, 0x20, id);                                                     // end slab 2, ckpt slab 3, ckpt_pos 2
    KvFlashSlotHead h;
    REQUIRE(kvflash_head_unpack(c.head.data(), c.head.size(), id, h));
    h.ckpt_pos = 4; h.trunk_pos = 0; h.trunk_gen = 0; h.extra_pos = 2; h.extra_gen = 9; h.pin_pos = 2;     // a pinned third below the ckpt
    c.files.push_back({kvflash_bundle_slab_name(9), kvflash_slab_pack({{7, 7}}, {{6, 6}})});
    c.head = kvflash_head_pack(h);
    REQUIRE(kvflash_bundle_commit(c));
    KvFlashBundleLoaded l; std::string why;
    REQUIRE(kvflash_bundle_load(dir, id, CB, LIM, l, why));
    CHECK(l.head.extra_pos == 2u); CHECK(l.head.extra_gen == 9u);
    CHECK((l.extra_ssm == std::vector<std::vector<uint8_t>>{{7, 7}})); CHECK((l.extra_conv == std::vector<std::vector<uint8_t>>{{6, 6}}));
    CHECK(!kvflash_bundle_load(dir, id, CB, KvFlashBundleLimits{1 << 20, 1 << 20, 8}, l, why));   // its file counts against the cap too
    REQUIRE(kvflash_bundle_validate(dir, h, CB, 1 << 20, why));
    CHECK(kvflash_head_gens(h)[3] == 9u);
    kvflash_bundle_write(dir + "/stray", {1});                                              // a stray: the cleanup takes it, not the third
    kvflash_bundle_clean(dir, &h);
    CHECK(fs::exists(dir + "/" + kvflash_bundle_slab_name(9))); CHECK(!fs::exists(dir + "/stray"));
    fs::remove(dir + "/" + kvflash_bundle_slab_name(9));                                    // the third's file lost: not loadable, not valid
    CHECK(!kvflash_bundle_load(dir, id, CB, LIM, l, why));
    CHECK(!kvflash_bundle_validate(dir, h, CB, 1 << 20, why));
    kvflash_bundle_remove(dir);
}

TEST_CASE(KvflashSlotBundleFixture, the_crash_hook_names_a_step_and_a_commit) {
    CHECK(kvflash_bundle_crash_at(nullptr) == std::make_pair(0, 0));
    CHECK(kvflash_bundle_crash_at("") == std::make_pair(0, 0));
    CHECK(kvflash_bundle_crash_at("3") == std::make_pair(3, 1));                            // the first commit
    CHECK(kvflash_bundle_crash_at("2:5") == std::make_pair(2, 5));
    CHECK(kvflash_bundle_crash_at("6") == std::make_pair(0, 0));                            // steps are 1-5
    CHECK(kvflash_bundle_crash_at("1:0") == std::make_pair(0, 0));
}

TEST_CASE(KvflashSlotBundleFixture, assembly_reads_the_mapped_records_fills_holes_and_ignores_a_torn_tail) {
    const KvFlashSlotHead h = sample_head();
    std::vector<uint8_t> log = sample_log();
    std::vector<uint8_t> blob;
    std::string why;
    REQUIRE(kvflash_bundle_assemble(h, CB, mem_reader(log), blob, why));
    CHECK(blob == expected_blob(h));
    log.insert(log.end(), 5, 0xee);                                                         // a torn record past log_bytes
    blob.clear();
    REQUIRE(kvflash_bundle_assemble(h, CB, mem_reader(log), blob, why));                    // ignored: not mapped
    CHECK(blob == expected_blob(h));
    auto past = h; past.map[3] = h.log_bytes;                                               // a record past the committed length
    CHECK(!kvflash_bundle_assemble(past, CB, mem_reader(log), blob, why)); CHECK(!why.empty());
    auto corrupt = log; corrupt[kKvFlashLogHeader + kKvFlashRecordTag + 2] ^= 1;            // a payload bit of chunk 0
    CHECK(!kvflash_bundle_assemble(h, CB, mem_reader(corrupt), blob, why));
    auto swapped = h; std::swap(swapped.map[0], swapped.map[1]);                            // the tag names another chunk
    CHECK(!kvflash_bundle_assemble(swapped, CB, mem_reader(log), blob, why));
    auto short_log = std::vector<uint8_t>(log.begin(), log.begin() + (long)(h.log_bytes - 3));   // the reader cannot give the bytes
    CHECK(!kvflash_bundle_assemble(h, CB, mem_reader(short_log), blob, why));
    auto wrong_gen = log; wrong_gen[8] ^= 1;                                                // the log is another generation
    CHECK(!kvflash_bundle_assemble(h, CB, mem_reader(wrong_gen), blob, why));
}

TEST_CASE(KvflashSlotBundleFixture, commit_then_load_round_trips_and_an_append_rewrites_only_the_dirty_chunks) {
    const std::string dir = "kvflash_f3_unit_commit";
    fs::remove_all(dir);                                                                    // the first park creates the directory
    const uint64_t id = 77;
    REQUIRE(kvflash_bundle_commit(park(dir, 1, 3, 0x20, id)));
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(dir, id, CB, LIM, l, why));
    CHECK(l.head.gen == 1u); CHECK(l.head.map.size() == 3u); CHECK(l.head.history.size() == 6u);
    std::vector<uint8_t> want = hdr44();
    for (uint8_t c : {0x20, 0x21, 0x22}) { const auto k = chunk(c); want.insert(want.end(), k.begin(), k.end()); }
    want.insert(want.end(), l.head.ledger.begin(), l.head.ledger.end());
    CHECK(l.pager == want);
    CHECK((l.end_ssm == std::vector<std::vector<uint8_t>>{{0x20, 1}})); CHECK((l.ckpt_conv == std::vector<std::vector<uint8_t>>{{0x20, 4}}));
    CHECK(l.trunk_ssm.empty());
    CHECK(!kvflash_bundle_load(dir, id + 1, CB, LIM, l, why));                 // another server
    CHECK(!kvflash_bundle_load(dir, id, CB, KvFlashBundleLimits{5, 1 << 20, 1 << 20}, l, why));   // 6 tokens over a 5-token context
    CHECK(!kvflash_bundle_load(dir, id, CB, KvFlashBundleLimits{1 << 20, 2, 1 << 20}, l, why));   // 3 chunks over a 2-chunk context
    CHECK(!kvflash_bundle_load(dir, id, CB, KvFlashBundleLimits{1 << 20, 1 << 20, 8}, l, why));   // a slab file over the cap
    CHECK(why.find("slab") != std::string::npos);
    // park 2: resumed at chunk 1 (dirty [1, 3)), a new end slab (id 4), a new ckpt slab (id 5) with the old
    // ckpt moved down to the trunk (its file, id 3, keeps its name), the old end slab (id 2) unlinked
    KvFlashBundleCommit c;
    c.dir = dir; c.gen = 2; c.log_gen = 1; c.log_bytes = l.head.log_bytes; c.record_bytes = RB;
    KvFlashSlotHead h = l.head;
    h.gen = 2; h.end_gen = 4; h.trunk_pos = 1; h.trunk_gen = l.head.ckpt_gen; h.ckpt_pos = 4; h.ckpt_gen = 5;
    for (int i = 1; i < 3; ++i) { c.records.push_back({(uint32_t)i, chunk((uint8_t)(0x30 + i))}); h.map[(size_t)i] = l.head.log_bytes + (uint64_t)(i - 1) * RB; }
    h.log_bytes = l.head.log_bytes + 2 * RB;
    c.files = {{kvflash_bundle_slab_name(4), kvflash_slab_pack({{9}}, {{9}})}, {kvflash_bundle_slab_name(5), kvflash_slab_pack({{8}}, {{8}})}};
    c.head = kvflash_head_pack(h);
    c.unlink = {kvflash_bundle_slab_name(2)};
    REQUIRE(kvflash_bundle_commit(c));
    REQUIRE(kvflash_bundle_load(dir, id, CB, LIM, l, why));
    want = hdr44();
    for (uint8_t ch : {0x20, 0x31, 0x32}) { const auto k = chunk(ch); want.insert(want.end(), k.begin(), k.end()); }
    want.insert(want.end(), l.head.ledger.begin(), l.head.ledger.end());
    CHECK(l.pager == want);
    CHECK((l.end_ssm == std::vector<std::vector<uint8_t>>{{9}})); CHECK((l.ckpt_ssm == std::vector<std::vector<uint8_t>>{{8}}));
    CHECK((l.trunk_ssm == std::vector<std::vector<uint8_t>>{{0x20, 3}}));                    // the moved rung: park 1's ckpt slab
    CHECK(l.head.trunk_pos == 1u);
    CHECK(!fs::exists(dir + "/" + kvflash_bundle_slab_name(2)));
    CHECK(fs::exists(dir + "/" + kvflash_bundle_slab_name(3)));
    CHECK(fs::file_size(dir + "/" + kvflash_bundle_log_name(1)) == h.log_bytes);
    std::vector<uint8_t> log;
    REQUIRE(kvflash_bundle_read(dir + "/" + kvflash_bundle_log_name(1), log));
    KvFlashRecordTag t;
    REQUIRE(kvflash_record_tag_parse(log.data() + l.head.map[1], RB, t));
    CHECK(t.chunk == 1u); CHECK(t.gen == 2u);                                                 // the commit stamps its generation
    kvflash_bundle_remove(dir);
    CHECK(!fs::exists(dir));
}

TEST_CASE(KvflashSlotBundleFixture, a_crash_after_any_commit_step_leaves_the_previous_or_the_new_park_loadable) {
    const std::string dir = "kvflash_f3_unit_crash";
    const uint64_t id = 5;
    for (int step = 1; step <= 5; ++step) {
        fs::remove_all(dir); fs::create_directory(dir);
        REQUIRE(kvflash_bundle_commit(park(dir, 1, 2, 0x40, id)));
        KvFlashBundleCommit b = park(dir, 2, 2, 0x50, id);                                  // a cold re-park: new log, new slabs
        b.unlink = {kvflash_bundle_log_name(1), kvflash_bundle_slab_name(2), kvflash_bundle_slab_name(3)};
        CHECK(!kvflash_bundle_commit(b, step));                                              // "killed" after this step
        KvFlashBundleLoaded l;
        std::string why;
        REQUIRE(kvflash_bundle_load(dir, id, CB, LIM, l, why));
        kvflash_bundle_clean(dir, &l.head);                                                  // the restart's cleanup
        REQUIRE(kvflash_bundle_load(dir, id, CB, LIM, l, why));
        const uint8_t fill = step < 4 ? 0x40 : 0x50;                                         // before the head rename: the old park
        CHECK(l.head.gen == (step < 4 ? 1u : 2u));
        CHECK(l.pager[kKvFlashPagerHdr] == fill);
        CHECK((l.end_ssm == std::vector<std::vector<uint8_t>>{{fill, 1}}));
        for (const auto & e : fs::directory_iterator(dir)) {                                 // nothing the head does not name survives
            const std::string f = e.path().filename().string();
            CHECK((f == "head" || f == kvflash_bundle_log_name(l.head.log_gen) || f == kvflash_bundle_slab_name(l.head.end_gen) ||
                   f == kvflash_bundle_slab_name(l.head.ckpt_gen)));
        }
    }
    // a torn tail: bytes past the committed length are cut by the cleanup, and the next append lands at log_bytes
    {
        KvFlashBundleLoaded l;
        std::string why;
        REQUIRE(kvflash_bundle_load(dir, id, CB, LIM, l, why));
        FILE * f = std::fopen((dir + "/" + kvflash_bundle_log_name(l.head.log_gen)).c_str(), "ab");
        REQUIRE(f != nullptr);
        std::fputs("torn", f); std::fclose(f);
        REQUIRE(kvflash_bundle_load(dir, id, CB, LIM, l, why));               // still loads: the tail is not mapped
        kvflash_bundle_clean(dir, &l.head);
        CHECK(fs::file_size(dir + "/" + kvflash_bundle_log_name(l.head.log_gen)) == l.head.log_bytes);
    }
    kvflash_bundle_remove(dir);
    // no head at all (killed inside the first park): the cleanup empties the directory
    fs::create_directory(dir);
    KvFlashBundleCommit first = park(dir, 1, 1, 0x60, id);
    CHECK(!kvflash_bundle_commit(first, 2));
    KvFlashBundleLoaded l;
    std::string why;
    CHECK(!kvflash_bundle_load(dir, id, CB, LIM, l, why));
    kvflash_bundle_clean(dir, nullptr);
    CHECK(fs::is_empty(dir));
    kvflash_bundle_remove(dir);
}

TEST_CASE(KvflashSlotBundleFixture, a_crash_in_an_append_or_a_compaction_leaves_the_previous_or_the_new_park_loadable) {
    const std::string dir = "kvflash_f3_unit_crash2";
    const uint64_t id = 6;
    for (bool compacted : {false, true}) for (int step = 1; step <= 5; ++step) {
        fs::remove_all(dir);
        REQUIRE(kvflash_bundle_commit(park(dir, 1, 3, 0x40, id)));
        KvFlashBundleLoaded l;
        std::string why;
        REQUIRE(kvflash_bundle_load(dir, id, CB, LIM, l, why));
        CHECK(!kvflash_bundle_commit(repark(dir, l.head, 2, 1, 0x50, compacted), step));    // "killed" after this step
        REQUIRE(kvflash_bundle_load(dir, id, CB, LIM, l, why));
        kvflash_bundle_clean(dir, &l.head);                                                  // the restart's cleanup
        REQUIRE(kvflash_bundle_load(dir, id, CB, LIM, l, why));
        const bool old = step < 4;
        CHECK(l.head.gen == (old ? 1u : 2u));
        CHECK(l.pager[kKvFlashPagerHdr] == 0x40);                                            // chunk 0: kept by either park
        CHECK(l.pager[kKvFlashPagerHdr + CB] == (old ? 0x41 : 0x51));                        // chunk 1: rewritten by park 2
        CHECK((l.end_ssm == std::vector<std::vector<uint8_t>>{{(uint8_t)(old ? 0x40 : 0x50), 1}}));
        CHECK(fs::file_size(dir + "/" + kvflash_bundle_log_name(l.head.log_gen)) == l.head.log_bytes);   // a torn tail is cut
        for (const auto & e : fs::directory_iterator(dir)) {                                 // nothing the head does not name survives
            const std::string f = e.path().filename().string();
            CHECK((f == "head" || f == kvflash_bundle_log_name(l.head.log_gen) || f == kvflash_bundle_slab_name(l.head.end_gen) ||
                   f == kvflash_bundle_slab_name(l.head.ckpt_gen)));
        }
    }
    kvflash_bundle_remove(dir);
}

TEST_CASE(KvflashSlotBundleFixture, compaction_copies_the_live_records_into_a_new_log_generation) {
    const std::string dir = "kvflash_f3_unit_compact";
    fs::remove_all(dir); fs::create_directory(dir);
    const uint64_t id = 9;
    REQUIRE(kvflash_bundle_commit(park(dir, 1, 3, 0x70, id)));
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(dir, id, CB, LIM, l, why));
    CHECK(!kvflash_bundle_needs_compaction(l.head.log_bytes, 3, RB));                      // 3 records, 3 live
    CHECK(kvflash_bundle_needs_compaction(kKvFlashLogHeader + 7 * RB, 3, RB));             // 4 dead > 3 live
    // a park's verdict counts the log after its append: 6 records, 1 kept + 2 written = 3 live, 5 dead after the append
    CHECK(kvflash_bundle_compacts(kKvFlashLogHeader + 6 * RB, 1, 2, RB));
    CHECK(!kvflash_bundle_needs_compaction(kKvFlashLogHeader + 6 * RB, 3, RB));            // the same log judged before the append: 3 dead, not 5
    CHECK(!kvflash_bundle_compacts(kKvFlashLogHeader + 3 * RB, 1, 2, RB));                 // 5 records after, 3 live, 2 dead
    CHECK(!kvflash_bundle_compacts(0, 0, 4, RB));                                          // a fresh log: nothing dead
    // park 2 resumed at chunk 1 and compacted: chunk 0 is copied from log 1, chunks 1-2 are new, log 2 replaces log 1
    KvFlashBundleCommit c;
    c.dir = dir; c.gen = 2; c.log_gen = 2; c.new_log = true; c.copy_from = 1;
    c.copies = {l.head.map[0]}; c.record_bytes = RB;
    KvFlashSlotHead h = l.head;
    h.gen = 2; h.log_gen = 2; h.end_gen = 4;
    h.map = {kKvFlashLogHeader, kKvFlashLogHeader + RB, kKvFlashLogHeader + 2 * RB};
    for (int i = 1; i < 3; ++i) c.records.push_back({(uint32_t)i, chunk((uint8_t)(0x80 + i))});
    h.log_bytes = kKvFlashLogHeader + 3 * RB;
    c.files = {{kvflash_bundle_slab_name(4), kvflash_slab_pack({{2}}, {{2}})}};
    c.head = kvflash_head_pack(h);
    c.unlink = {kvflash_bundle_log_name(1), kvflash_bundle_slab_name(2)};
    REQUIRE(kvflash_bundle_commit(c));
    REQUIRE(kvflash_bundle_load(dir, id, CB, LIM, l, why));
    std::vector<uint8_t> want = hdr44();
    for (uint8_t ch : {0x70, 0x81, 0x82}) { const auto k = chunk(ch); want.insert(want.end(), k.begin(), k.end()); }
    want.insert(want.end(), l.head.ledger.begin(), l.head.ledger.end());
    CHECK(l.pager == want);
    CHECK(!fs::exists(dir + "/" + kvflash_bundle_log_name(1)));
    CHECK(fs::file_size(dir + "/" + kvflash_bundle_log_name(2)) == h.log_bytes);
    kvflash_bundle_remove(dir);
}

TEST_CASE(KvflashSlotBundleFixture, write_is_tmp_then_rename_and_read_has_a_cap) {
    const std::string path = "kvflash_slot_bundle_unit_test.bin";
    const std::vector<uint8_t> bytes = {1, 2, 3, 4, 5};
    REQUIRE(kvflash_bundle_write(path, bytes));
    std::vector<uint8_t> back = {7};
    CHECK(!kvflash_bundle_read(path, back, bytes.size() - 1));
    CHECK(back == std::vector<uint8_t>{7});                                                  // untouched on refusal
    REQUIRE(kvflash_bundle_read(path, back));
    CHECK(back == bytes);
    CHECK(std::fopen((path + ".tmp." + std::to_string(::getpid())).c_str(), "rb") == nullptr);   // no leftover tmp
    CHECK(!kvflash_bundle_write_stream(path, bytes, [](auto &) { return false; }));          // a failed part: no file change
    REQUIRE(kvflash_bundle_read(path, back));
    CHECK(back == bytes);
    std::remove(path.c_str());
    CHECK(!kvflash_bundle_read(path, back));
    CHECK(!kvflash_bundle_write("/nonexistent-dir-kvflash/x", bytes));
}

TEST_CASE(KvflashSlotBundleFixture, the_writer_thread_commits_jobs_in_order_and_reports_each) {
    const std::string dir = "kvflash_f3_unit_writer";
    fs::remove_all(dir); fs::create_directory(dir);
    std::vector<std::pair<int, bool>> done;
    {
        KvFlashBundleWriter w([&](const KvFlashBundleWriter::Job & j, bool ok, double) { done.push_back({j.slot, ok}); });
        w.submit({1, park(dir, 1, 1, 0x11, 3)});                                               // an older park of the slot
        w.submit({1, park(dir, 2, 1, 0x22, 3)});                                               // the newer one lands last
        w.submit({2, park("/nonexistent-dir-kvflash", 1, 1, 0, 3)});                           // reported, does not stop the writer
        w.wait(1);                                                                              // both parks of slot 1 are on disk (FIFO)
        KvFlashBundleLoaded l;
        std::string why;
        REQUIRE(kvflash_bundle_load(dir, 3, CB, LIM, l, why));
        CHECK(l.head.gen == 2u);
        w.wait(42);                                                                             // no job of that slot: at once
        w.drain();
        CHECK(w.pending() == 0);
        REQUIRE(done.size() == 3);
        CHECK(done[0] == std::make_pair(1, true));
        CHECK(done[1] == std::make_pair(1, true));
        CHECK(done[2] == std::make_pair(2, false));
        w.submit({1, park(dir, 3, 1, 0x33, 3)});                                               // the destructor drains
    }
    REQUIRE(done.size() == 4);
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(dir, 3, CB, LIM, l, why));
    CHECK(l.head.gen == 3u);
    kvflash_bundle_remove(dir);
}

TEST_CASE(KvflashSlotBundleFixture, the_writer_reports_a_slot_in_flight_until_its_job_is_done) {
    const std::string dir = "kvflash_f3_unit_writer_flight";
    fs::remove_all(dir); fs::create_directory(dir);
    std::mutex m; std::condition_variable cv; bool release = false, started = false;
    {
        KvFlashBundleWriter w([&](const KvFlashBundleWriter::Job &, bool, double) {
            std::unique_lock<std::mutex> lk(m); started = true; cv.notify_all(); cv.wait(lk, [&] { return release; });
        });
        CHECK(!w.in_flight(1));
        w.submit({1, park(dir, 1, 1, 0x11, 3)});
        CHECK(w.in_flight(1));                                       // queued or being written
        CHECK(!w.in_flight(2));                                      // another slot's park is not this one's
        { std::unique_lock<std::mutex> lk(m); cv.wait(lk, [&] { return started; }); }
        CHECK(w.in_flight(1));                                       // the job is in its Done callback: still owned
        { std::lock_guard<std::mutex> lk(m); release = true; } cv.notify_all();
        w.wait(1);
        CHECK(!w.in_flight(1));
    }
    fs::remove_all(dir);
}

// A bundle holds KV that one paged-attention route computed (WMMA or V_DOT2); the other route
// rounds differently, so a bundle saved under one must not be adopted under the other.
TEST_CASE(KvflashSlotBundleFixture, the_identity_differs_when_only_the_paged_attention_route_differs) {
    KvFlashIdentityParts a;
    a.model = "model.gguf"; a.model_size = 1234; a.model_mtime = 99;
    a.kv_k_type = 8; a.kv_v_type = 8; a.chunk_tokens = 64; a.n_pool_blocks = 32; a.max_context = 8192;
    KvFlashIdentityParts same_route = a;
    KvFlashIdentityParts other_route = a;
    a.paged_wmma_route = true; same_route.paged_wmma_route = true; other_route.paged_wmma_route = false;
    CHECK(kvflash_identity_key(a) == kvflash_identity_key(same_route));
    CHECK(kvflash_identity_key(a) != kvflash_identity_key(other_route));
    CHECK(kvflash_fnv1a(kvflash_identity_key(a)) != kvflash_fnv1a(kvflash_identity_key(other_route)));
}

// The route is what dispatch picks for the model, not what the device could run: the WMMA kernel
// takes a GQA ratio up to 8 only, F16/Q8_0/Q4_0 KV, head dim 256, 16-token blocks and no tree, and
// every other graph runs V_DOT2 whatever LUCE_PAGED_WMMA says.
TEST_CASE(KvflashSlotBundleFixture, the_paged_attention_route_follows_the_models_dispatch) {
    const auto route = [](bool env, bool rdna4, int n_head, int n_head_kv, ggml_type kt = GGML_TYPE_F16,
                          ggml_type vt = GGML_TYPE_F16, bool tree = false, int head_dim = 256, int block = 16) {
        return ggml_cuda_paged_attn_wmma_route_for(env, rdna4, tree, head_dim, n_head, n_head_kv, kt, vt, block);
    };
    CHECK(route(true, true, 16, 2));                          // GQA 8: the eligible case
    CHECK(!route(true, true, 24, 2));                         // GQA 12 > 8: V_DOT2 on a WMMA-capable device, env unset
    CHECK(!route(true, true, 25, 2));                         // not a multiple
    CHECK(!route(false, true, 16, 2));                        // LUCE_PAGED_WMMA=0
    CHECK(!route(true, false, 16, 2));                        // not RDNA4
    CHECK(!route(true, true, 16, 2, GGML_TYPE_Q5_0));         // KV type the kernel has no instantiation for
    CHECK(!route(true, true, 16, 2, GGML_TYPE_F16, GGML_TYPE_Q5_0));
    CHECK(route(true, true, 16, 2, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0));
    CHECK(!route(true, true, 16, 2, GGML_TYPE_F16, GGML_TYPE_F16, /*tree=*/true));
    CHECK(!route(true, true, 16, 2, GGML_TYPE_F16, GGML_TYPE_F16, false, /*head_dim=*/128));
    CHECK(!route(true, true, 16, 2, GGML_TYPE_F16, GGML_TYPE_F16, false, 256, /*block=*/32));
}

// Both env settings dispatch V_DOT2 for a GQA-12 model, so a bundle saved under one is compatible with
// the other; for an eligible model the two settings are different routes and different identities.
TEST_CASE(KvflashSlotBundleFixture, the_identity_follows_the_dispatched_route_not_the_env) {
    const auto key = [](bool env, int n_head, int n_head_kv) {
        KvFlashIdentityParts p;
        p.model = "model.gguf"; p.model_size = 1234; p.model_mtime = 99;
        p.kv_k_type = (int)GGML_TYPE_F16; p.kv_v_type = (int)GGML_TYPE_F16;
        p.chunk_tokens = 64; p.n_pool_blocks = 32; p.max_context = 8192;
        p.paged_wmma_route = ggml_cuda_paged_attn_wmma_route_for(
            env, /*device_rdna4=*/true, false, 256, n_head, n_head_kv, GGML_TYPE_F16, GGML_TYPE_F16, 16);
        return kvflash_identity_key(p);
    };
    CHECK(key(true, 24, 2) == key(false, 24, 2));
    CHECK(key(true, 16, 2) != key(false, 16, 2));
}

// ---- R3 digest mode: the digest of the whole saved state, per component ----------------------

namespace {
struct DigestState {
    std::vector<int32_t> hist{11, 22, 33, 44, 55};
    std::vector<uint8_t> slab = kvflash_slab_pack({{1, 2, 3}}, {{4, 5}});
    std::vector<uint8_t> hdr = hdr44();
    std::vector<uint8_t> ledger;
    std::vector<uint8_t> ckpt = kvflash_slab_pack({{6, 7}}, {{8}}), trunk = kvflash_slab_pack({{9}}, {{10, 11}}), extra;   // two rungs, no third unless extra_pos
    int32_t ckpt_pos = 4, trunk_pos = 2, extra_pos = 0, pin = 0;
    std::vector<std::vector<uint8_t>> chunks;   // chunk 2 is not materialized: no bytes digested
    DigestState() {
        ledger.assign(4 * kKvFlashLedgerEntry, 0);
        for (int c : {0, 1, 3}) ledger[(size_t)c * kKvFlashLedgerEntry + 1] = 1;
        chunks = {chunk(0x10), chunk(0x11), chunk(0), chunk(0x13)};
    }
    KvFlashStateDigest digest() const {
        KvFlashDigestRungs rg = kvflash_digest_rungs(ckpt_pos, &ckpt, trunk_pos, &trunk, extra_pos, pin > 0 && pin == extra_pos, &extra);
        if (extra_pos == 0) rg.pin = pin;   // a pin with no third: the position alone
        return kvflash_state_digest(hist, slab, hdr, ledger, rg, CB, [&](int ci, uint8_t * dst) { std::memcpy(dst, chunks[(size_t)ci].data(), CB); });
    }
};
}  // namespace

TEST_CASE(KvflashSlotBundleFixture, state_digest_is_deterministic_and_counts_every_digested_byte) {
    const DigestState s;
    const KvFlashStateDigest a = s.digest(), b = s.digest();
    CHECK(a.all != 0); CHECK(a.hist != 0); CHECK(a.slab != 0); CHECK(a.kv != 0); CHECK(a.ledger != 0); CHECK(a.pager != 0); CHECK(a.rung != 0);
    CHECK(a.all == b.all); CHECK(a.hist == b.hist); CHECK(a.slab == b.slab); CHECK(a.kv == b.kv); CHECK(a.ledger == b.ledger); CHECK(a.pager == b.pager); CHECK(a.rung == b.rung);
    CHECK(a.bytes == s.hist.size() * 4 + s.slab.size() + 3 * CB + s.ledger.size() + s.hdr.size() + s.ckpt.size() + s.trunk.size());   // the rung slabs count, the presence and position folds do not
}

TEST_CASE(KvflashSlotBundleFixture, state_digest_flipping_one_byte_moves_that_component_and_all_only) {
    const DigestState base;
    const KvFlashStateDigest d0 = base.digest();
    auto moved = [&](const KvFlashStateDigest & d) {   // which of the six components differ, as a bit mask
        return (d.hist != d0.hist) | (d.slab != d0.slab) << 1 | (d.kv != d0.kv) << 2 | (d.ledger != d0.ledger) << 3 | (d.pager != d0.pager) << 4 | (d.rung != d0.rung) << 5;
    };
    DigestState s = base; s.hist[2] ^= 1;               CHECK(moved(s.digest()) == 1);  CHECK(s.digest().all != d0.all);
    s = base; s.slab[5] ^= 1;                           CHECK(moved(s.digest()) == 2);  CHECK(s.digest().all != d0.all);
    s = base; s.chunks[3][7] ^= 1;                      CHECK(moved(s.digest()) == 4);  CHECK(s.digest().all != d0.all);
    s = base; s.chunks[0].swap(s.chunks[1]);            CHECK(moved(s.digest()) == 4);  CHECK(s.digest().all != d0.all);   // order is part of the digest
    s = base; s.chunks[2][0] = 0x77;                    CHECK(s.digest().all == d0.all);   // a chunk the ledger does not materialize is not state
    s = base; s.ledger[4] ^= 1;                         CHECK(moved(s.digest()) == 8);  CHECK(s.digest().all != d0.all);
    s = base; s.ckpt[9] ^= 1;                           CHECK(moved(s.digest()) == 32); CHECK(s.digest().all != d0.all);   // a rung slab
    s = base; s.trunk_pos = 3;                          CHECK(moved(s.digest()) == 32); CHECK(s.digest().all != d0.all);   // a rung position
    s = base; s.pin = 4;                                CHECK(moved(s.digest()) == 32); CHECK(s.digest().all != d0.all);   // the pin
    s = base; s.trunk_pos = 0;                          CHECK(moved(s.digest()) == 32); CHECK(s.digest().all != d0.all);   // a rung gone
    s = base; s.hdr[20] ^= 1;                           CHECK(moved(s.digest()) == 16); CHECK(s.digest().all != d0.all);
}

TEST_CASE(KvflashSlotBundleFixture, state_digest_survives_a_commit_and_a_load_and_the_head_generations_are_not_in_it) {
    const std::string dir = "kvflash_f3_unit_digest";
    fs::remove_all(dir);
    const uint64_t id = 78;
    KvFlashBundleCommit c = park(dir, 1, 3, 0x20, id);
    REQUIRE(kvflash_bundle_commit(c));
    KvFlashBundleLoaded l;
    std::string why;
    REQUIRE(kvflash_bundle_load(dir, id, CB, LIM, l, why));
    // what the save stages: its history, its end slab, the chunks the pager holds, its ledger and header
    const std::vector<int32_t> hist(6, 0x20);
    const std::vector<uint8_t> slab = kvflash_slab_pack({{0x20, 1}}, {{0x20, 2}}), ledger(3 * kKvFlashLedgerEntry, 0x20);
    auto live = [&](int ci, uint8_t * dst) { std::memcpy(dst, chunk((uint8_t)(0x20 + ci)).data(), CB); };
    const std::vector<uint8_t> ckpt = kvflash_slab_pack({{0x20, 3}}, {{0x20, 4}});   // the ckpt rung park() wrote at position 2
    KvFlashDigestRungs want_r; want_r.r[0] = {2, &ckpt};
    const KvFlashStateDigest saved = kvflash_state_digest(hist, slab, hdr44(), ledger, want_r, CB, live);
    // what the restore stages: the loaded head, the loaded end slab, the chunks of the assembled pager blob
    auto blob = [&](int ci, uint8_t * dst) { std::memcpy(dst, l.pager.data() + kKvFlashPagerHdr + (size_t)ci * CB, CB); };
    const std::vector<uint8_t> lck = kvflash_slab_pack(l.ckpt_ssm, l.ckpt_conv);
    KvFlashDigestRungs got_r; got_r.r[0] = {(int32_t)l.head.ckpt_pos, &lck};
    const KvFlashStateDigest back = kvflash_state_digest(l.head.history, kvflash_slab_pack(l.end_ssm, l.end_conv), l.head.pager_hdr, l.head.ledger, got_r, CB, blob);
    CHECK(saved.all != 0);
    CHECK(back.all == saved.all); CHECK(back.hist == saved.hist); CHECK(back.slab == saved.slab); CHECK(back.kv == saved.kv);
    CHECK(back.ledger == saved.ledger); CHECK(back.pager == saved.pager); CHECK(back.rung == saved.rung); CHECK(saved.rung != 0); CHECK(back.bytes == saved.bytes);
    // a later park of the same state: other gen, log gen, end slab id and record offsets, same digest
    KvFlashBundleCommit c2 = repark(dir, l.head, 2, 3, 0x20, true);
    REQUIRE(kvflash_bundle_commit(c2));
    KvFlashBundleLoaded l2;
    REQUIRE(kvflash_bundle_load(dir, id, CB, LIM, l2, why));
    CHECK(l2.head.gen != l.head.gen); CHECK(l2.head.map != l.head.map || l2.head.end_gen != l.head.end_gen);
    auto blob2 = [&](int ci, uint8_t * dst) { std::memcpy(dst, l2.pager.data() + kKvFlashPagerHdr + (size_t)ci * CB, CB); };
    const std::vector<uint8_t> lck2 = kvflash_slab_pack(l2.ckpt_ssm, l2.ckpt_conv);
    KvFlashDigestRungs got2; got2.r[0] = {(int32_t)l2.head.ckpt_pos, &lck2};
    CHECK(kvflash_state_digest(l2.head.history, kvflash_slab_pack(l2.end_ssm, l2.end_conv), l2.head.pager_hdr, l2.head.ledger, got2, CB, blob2).all == saved.all);
    kvflash_bundle_remove(dir);
}

TEST_CASE(KvflashSlotBundleFixture, state_digest_line_has_the_fixed_format) {
    KvFlashStateDigest d;
    d.all = 0x0123456789abcdefULL; d.hist = 1; d.slab = 2; d.kv = 3; d.ledger = 4; d.pager = 0xffULL; d.rung = 5; d.bytes = 4096;
    CHECK(kvflash_state_digest_line("save", "9f8e7d6c5b4a3921", 7, d, 12.34) ==
          "[kvflash-digest] save sid=9f8e7d6c5b4a3921 gen=7 all=0123456789abcdef hist=0000000000000001 slab=0000000000000002 kv=0000000000000003 ledger=0000000000000004 pager=00000000000000ff rung=0000000000000005 bytes=4096 ms=12.3");
}

TEST_CASE(KvflashSlotBundleFixture, a_pinned_third_at_the_ckpt_position_is_hashed_raw_and_an_unpinned_one_is_not) {
    DigestState s;
    s.ckpt_pos = 4096;
    s.extra = kvflash_slab_pack({{12, 13}}, {{14}}); s.extra_pos = 4096; s.pin = 4096;   // a pinned third at the ckpt's position: a copy of its slab
    const KvFlashStateDigest d0 = s.digest();
    s.extra[9] ^= 1;
    CHECK(s.digest().rung != d0.rung); CHECK(s.digest().all != d0.all);
    // an unpinned third at the ckpt's position: restore never rebuilds it, so only whether it differs from the rung is folded in
    DigestState none;
    none.ckpt_pos = 4096;
    DigestState eq = none;
    eq.extra = eq.ckpt; eq.extra_pos = 4096;                                              // an equal duplicate
    CHECK(eq.digest().rung == none.digest().rung); CHECK(eq.digest().all == none.digest().all);
    DigestState ne = eq;
    ne.extra[9] ^= 1;                                                                     // a differing duplicate, at the ckpt
    const KvFlashStateDigest dn = ne.digest(), de = eq.digest();
    CHECK(dn.rung != de.rung); CHECK(dn.all != de.all);
    CHECK(dn.hist == de.hist); CHECK(dn.slab == de.slab); CHECK(dn.kv == de.kv); CHECK(dn.ledger == de.ledger); CHECK(dn.pager == de.pager);
    DigestState tr = none;
    tr.extra = tr.trunk; tr.extra_pos = tr.trunk_pos;                                     // the same at the trunk
    CHECK(tr.digest().rung == none.digest().rung);
    tr.extra[9] ^= 1;
    CHECK(tr.digest().rung != none.digest().rung);
    DigestState own = none;
    own.extra = kvflash_slab_pack({{12}}, {{14}}); own.extra_pos = 5000;                  // a third of its own is state
    const KvFlashStateDigest o0 = own.digest();
    own.extra[9] ^= 1;
    CHECK(own.digest().rung != o0.rung);
}

TEST_CASE(KvflashSlotBundleFixture, regen_and_fault_lines_have_the_fixed_format) {
    CHECK(kvflash_regen_line("9f8e7d6c5b4a3921", 12, 9) == "[kvflash-digest] regen sid=9f8e7d6c5b4a3921 gen=12 from=9");
    CHECK(kvflash_fault_line("kv", "9f8e7d6c5b4a3921", 7) == "[kvflash-digest] fault kv sid=9f8e7d6c5b4a3921 gen=7");
}
