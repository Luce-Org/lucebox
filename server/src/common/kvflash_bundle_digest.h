// Digest of a session's whole saved state (R3 digest mode, LUCE_KVFLASH_BUNDLE_DIGEST=1): one FNV-1a (the slot-state
// hash) per component and one over the seven, from host bytes alone, so the save (the bytes being staged) and the
// restore (the live slot, read back through the same staging) are compared on equal terms.
//
// Digested: the token history, the end slab (ssm + conv pack), every materialized chunk's KV bytes in chunk order
// (each folded with its index), the ledger, the pager header (magic, chunk count, chunk tokens, head count, K/V segment
// bytes, chunk bytes, K/V types: geometry only, all fixed for a session), and the rungs a next turn can rewind to
// (ckpt, trunk, third: a presence byte, the position and the slab pack; then the pin position), and the slot's user-question
// query rows (the window lo/hi, n, the key hash and the row floats; none: nothing folded). A restored slot that scores a question
// with fewer rows than the live one picks differently while every other component is equal, so the rows are state.
// Excluded, because a later park of the same state legitimately differs there: the head's gen, log_gen, end_gen and
// rung gens (a counter per park), log_bytes and the chunk map offsets (where a record landed in an appended or
// compacted log), the record tags' park generation and crc, and the slab file ids. A chunk the ledger does not
// materialize is zeros at load and is not digested.
#pragma once

#include "kvflash_slot_bundle.h"
#include "fnv.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace luce::common {

// The ladder a restored session can rewind to: ckpt, trunk and third rung (a position and its slab pack; no slab or a
// non-positive position: absent) and the pin position (0: none).
struct KvFlashDigestRungs {
    struct Rung { int32_t pos = 0; const std::vector<uint8_t> * slab = nullptr; };
    Rung r[3];
    int32_t pin = 0;
    bool dup_differs = false;   // an unpinned third at the ckpt's or trunk's position whose slab differs from that rung's
};

struct KvFlashStateDigest {
    uint64_t all = 0, hist = 0, slab = 0, kv = 0, ledger = 0, pager = 0, rung = 0, uq = 0;
    uint64_t bytes = 0;   // bytes digested, the chunk index folds not counted
};

// chunk_at(ci, dst) writes chunk_bytes of chunk ci (KvFlashSlotPager::read_chunk, or a slice of a blob).
template <class ChunkAt>
inline KvFlashStateDigest kvflash_state_digest(const std::vector<int32_t> & hist, const std::vector<uint8_t> & slab, const std::vector<uint8_t> & pager_hdr,
                                               const std::vector<uint8_t> & ledger, const KvFlashDigestRungs & rungs, size_t chunk_bytes, ChunkAt && chunk_at, const KvFlashUqStore * uq = nullptr) {
    KvFlashStateDigest d;
    d.hist = fnv1a_bytes(kFnvBasis, (const uint8_t *)hist.data(), hist.size() * 4);
    d.slab = fnv1a_bytes(kFnvBasis, slab.data(), slab.size());
    d.ledger = fnv1a_bytes(kFnvBasis, ledger.data(), ledger.size());
    d.pager = fnv1a_bytes(kFnvBasis, pager_hdr.data(), pager_hdr.size());
    d.bytes = hist.size() * 4 + slab.size() + ledger.size() + pager_hdr.size();
    d.rung = kFnvBasis;
    for (const auto & r : rungs.r) {
        const uint8_t present = r.pos > 0 && r.slab;
        d.rung = fnv1a_bytes(d.rung, &present, 1);
        if (!present) continue;
        d.rung = fnv1a_bytes(d.rung, (const uint8_t *)&r.pos, 4);
        d.rung = fnv1a_bytes(d.rung, r.slab->data(), r.slab->size());
        d.bytes += r.slab->size();
    }
    if (rungs.dup_differs) { const uint8_t one = 1; d.rung = fnv1a_bytes(d.rung, &one, 1); }   // folded only when 1: an equal duplicate hashes as none
    const uint8_t pinned = rungs.pin > 0;
    d.rung = fnv1a_bytes(d.rung, &pinned, 1);
    if (pinned) d.rung = fnv1a_bytes(d.rung, (const uint8_t *)&rungs.pin, 4);
    d.kv = kFnvBasis;
    std::vector<uint8_t> buf(chunk_bytes);
    for (size_t ci = 0; ci * kKvFlashLedgerEntry + 1 < ledger.size(); ++ci) {
        if (ledger[ci * kKvFlashLedgerEntry + 1] == 0) continue;
        chunk_at((int)ci, buf.data());
        const uint32_t idx = (uint32_t)ci;
        d.kv = fnv1a_bytes(d.kv, (const uint8_t *)&idx, 4);
        d.kv = fnv1a_bytes(d.kv, buf.data(), buf.size());
        d.bytes += chunk_bytes;
    }
    d.uq = kFnvBasis;
    if (uq && !uq->empty()) {
        const int32_t meta[3] = {uq->lo, uq->hi, uq->n};
        d.uq = fnv1a_bytes(d.uq, (const uint8_t *)meta, 12);
        d.uq = fnv1a_bytes(d.uq, (const uint8_t *)&uq->hash, 8);
        d.uq = fnv1a_bytes(d.uq, (const uint8_t *)uq->rows.data(), uq->rows.size() * sizeof(float));
        d.bytes += uq->rows.size() * sizeof(float);
    }
    const uint64_t parts[7] = {d.hist, d.slab, d.kv, d.ledger, d.pager, d.rung, d.uq};
    d.all = fnv1a_bytes(kFnvBasis, (const uint8_t *)parts, sizeof parts);
    return d;
}

// LUCE_KVFLASH_BUNDLE_DIGEST=1, read once: the one gate of every digest line (save, restore, regen, fault).
inline bool kvflash_bundle_digest_on() {
    static const bool on = [] { const char * e = std::getenv("LUCE_KVFLASH_BUNDLE_DIGEST"); return e && e[0] == '1'; }();
    return on;
}
// R3 negative control (test hook): LUCE_KVFLASH_DIGEST_FAULT=kv|slab, read once, "" unless digest mode is on.
inline const std::string & kvflash_digest_fault() {
    static const std::string f = [] {
        const char * e = kvflash_bundle_digest_on() ? std::getenv("LUCE_KVFLASH_DIGEST_FAULT") : nullptr;
        return e && (std::string(e) == "kv" || std::string(e) == "slab") ? std::string(e) : std::string();
    }();
    return f;
}

// The rungs as the session holds them, raw: a third at a rung's position is hashed when it is pinned (restore rebuilds it from
// the rung it names, so a corrupted copy must show). An unpinned one is not rebuilt by restore, yet a branch there prefers it
// over the rung (kvflash_ladder_resume), so one byte says whether its slab differs from that rung's: folded only when it does.
inline KvFlashDigestRungs kvflash_digest_rungs(int ck_pos, const std::vector<uint8_t> * ck, int tr_pos, const std::vector<uint8_t> * tr,
                                               int ex_pos, bool ex_pinned, const std::vector<uint8_t> * ex) {
    KvFlashDigestRungs g;
    g.r[0] = {ck_pos, ck}; g.r[1] = {tr_pos, tr};
    if (ex_pos > 0 && (ex_pinned || (ex_pos != ck_pos && ex_pos != tr_pos))) g.r[2] = {ex_pos, ex};
    g.pin = ex_pos > 0 && ex_pinned ? ex_pos : 0;
    if (ex_pos > 0 && !ex_pinned && (ex_pos == ck_pos || ex_pos == tr_pos)) {
        const std::vector<uint8_t> * rung = ex_pos == ck_pos ? ck : tr;
        g.dup_differs = !ex || !rung || *ex != *rung;
    }
    return g;
}

// Storage compaction copies a session's records under a new gen without a save: this line, flushed before its head commit,
// lets D follow a restore of the new gen back to the gen whose content the new head carries.
inline std::string kvflash_regen_line(const std::string & sid, uint32_t gen, uint32_t from) {
    return "[kvflash-digest] regen sid=" + sid + " gen=" + std::to_string(gen) + " from=" + std::to_string(from);
}
// The negative control's mark: one byte of the live restored slot was flipped before the restore digest.
inline std::string kvflash_fault_line(const std::string & kind, const std::string & sid, uint32_t gen) {
    return "[kvflash-digest] fault " + kind + " sid=" + sid + " gen=" + std::to_string(gen);
}

// The one log line of both sides; tag is "save" or "restore", sid the session's store key.
inline std::string kvflash_state_digest_line(const char * tag, const std::string & sid, uint32_t gen, const KvFlashStateDigest & d, double ms) {
    char b[512];
    std::snprintf(b, sizeof b, "[kvflash-digest] %s sid=%s gen=%u all=%016llx hist=%016llx slab=%016llx kv=%016llx ledger=%016llx pager=%016llx rung=%016llx uq=%016llx bytes=%llu ms=%.1f",
                  tag, sid.c_str(), gen, (unsigned long long)d.all, (unsigned long long)d.hist, (unsigned long long)d.slab, (unsigned long long)d.kv,
                  (unsigned long long)d.ledger, (unsigned long long)d.pager, (unsigned long long)d.rung, (unsigned long long)d.uq, (unsigned long long)d.bytes, ms);
    return b;
}

}  // namespace luce::common
