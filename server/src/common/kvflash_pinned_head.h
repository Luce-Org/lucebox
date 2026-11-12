// KVFlash pinned head: the shared system+tools prefix of a concurrent slot, held pinned in the slot's pool and kept
// once in host memory so a second slot attaches it instead of prefilling it. The head is the cut #941's
// SharedPrefixes found, rounded down to the engine's chunk grid. Pure and engine-free: the grid, the decision, the
// slot -> head binder with the one re-pin, and the bounded capture / attach / rebind over a pager and an allocator.
//
// Why the grid: a turn is cut at multiples of the cut unit U (kvflash_slot_turn_next_cut). A head that ends on a
// multiple of U adds no cut to a cold prefill, so a hit runs the same chunks a cold run would.
#pragma once

#include "kvflash_admit_plan.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace luce::common {

// The grid heads live on: whole chunks, and with `slice` > ct whole prefill slices too.
inline int kvflash_head_unit(int ct, int slice) {
    return ct <= 0 ? 1 : slice > ct ? std::lcm(ct, slice) : ct;
}

// Where the head ends, in tokens: the server's cut (head_cut > 0) rounded down to the grid, the role end when
// sharing is off (head_cut < 0), none (0) when the server found no cut. 0 when under one unit.
inline int kvflash_head_boundary(int head_cut, int e_role, int ct, int slice) {
    const int raw = head_cut > 0 ? head_cut : head_cut < 0 ? e_role : 0;
    const int u = kvflash_head_unit(ct, slice);
    return raw <= 0 || ct <= 0 ? 0 : (raw / u) * u;
}

// Chunks of the head: whole cut units, at most k_max, at least two chunks and one unit; 0 when there is no head.
inline int kvflash_head_k(int boundary, int ct, int unit, int k_max) {
    const int uc = std::max(1, unit / std::max(1, ct));
    const int n = boundary / std::max(1, unit) * unit / std::max(1, ct);
    const int k = std::min(n, k_max) / uc * uc;
    return n >= std::max(2, uc) && k >= std::max(2, uc) ? k : 0;
}

// How many chunks of a slot's pool the heads may pin: a quarter of it (they come off the slot's history budget).
inline int kvflash_head_k_max(int n_pool_blocks) { return std::max(0, n_pool_blocks) / 4; }

// H_k = fnv1a continued from H_{k-1} over ids[ct (k-1), ct k) for k = 1 .. n_chunks (clamped to the whole chunks
// `ids` holds). Entry i is H_{i+1}.
inline std::vector<uint64_t> kvflash_head_hashes(const std::vector<int32_t> & ids, int n_chunks, int ct) {
    std::vector<uint64_t> h;
    if (ct <= 0 || n_chunks <= 0) return h;
    const int n = std::min(n_chunks, (int)(ids.size() / (size_t)ct));
    uint64_t acc = 1469598103934665603ULL;
    for (int k = 0; k < n; k++) {
        for (int i = k * ct; i < (k + 1) * ct; i++) {
            const uint32_t t = (uint32_t)ids[(size_t)i];
            for (int b = 0; b < 4; b++) { acc ^= (t >> (8 * b)) & 0xff; acc *= 1099511628211ULL; }
        }
        h.push_back(acc);
    }
    return h;
}

// Attach what `has` (hash, k) says is held (host or disk), or cold-admit and capture; plain cold when there is no head. `k` is a
// function of the boundary alone: what is known only chooses attach versus capture, never k.
inline KvFlashPreambleDecision kvflash_head_decide(const std::vector<int32_t> & ids, int boundary, int ct, int unit,
                                                   int k_max, const std::function<bool(uint64_t, int)> & has) {
    KvFlashPreambleDecision d;
    d.k = kvflash_head_k(boundary, ct, unit, k_max);
    if (d.k == 0) return d;
    const std::vector<uint64_t> path = kvflash_head_hashes(ids, d.k, ct);
    if ((int)path.size() < d.k) { d.k = 0; return d; }
    d.kind = has(path.back(), d.k) ? KvFlashPreambleDecision::Attach : KvFlashPreambleDecision::Capture;
    return d;
}

// ---- re-pin ---------------------------------------------------------------------------------------------------
//
// A slot's binding {k chunks pinned from chunk 0} is lost by reset() and by deserialize() of a blob without the layout (they clear every pin) and
// cut by truncate() below k. The slot holds the head while it has k chunks. This re-applies the pins, reads back how
// many the pager accepted (pin_range refuses a span that would deadlock eviction), and releases the binding when the
// slot's length fell below the head or fewer than k pins were accepted: the request then runs as if cold.
// ponytail: unpin_all() on release is right because hard pins are only ever the head's; a second user of pin_range
// on a slot pager must move to a per-owner pin before it relies on this.
struct KvFlashRepin {
    int accepted = 0;
    bool released = false;
};
template <class Pager>
inline KvFlashRepin kvflash_repin_head(Pager & pager, int & bound_k) {
    KvFlashRepin r;
    if (bound_k <= 0) return r;
    const int ct = pager.chunk_tokens();
    if (ct > 0 && pager.n_chunks() >= bound_k) {
        pager.pin_range(0, (int64_t)bound_k * ct);
        for (int c = 0; c < bound_k; c++) r.accepted += pager.is_pinned(c) ? 1 : 0;
        if (r.accepted == bound_k) return r;
    }
    pager.unpin_all();
    bound_k = 0;
    r.accepted = 0;
    r.released = true;
    return r;
}

// ---- the binder -----------------------------------------------------------------------------------------------

struct KvFlashHeadBinding {
    uint64_t hash = 0;
    int k = 0;   // 0: not bound
};

// slot -> head, and the pin count recall reads. KvFlashSlotKv calls ensure() after every operation that resets,
// truncates or replaces a slot's pager, so no engine path can forget it.
class KvFlashHeadBinder {
public:
    explicit KvFlashHeadBinder(int n_slots) : b_((size_t)std::max(0, n_slots)), pins_((size_t)std::max(0, n_slots), 0) {}

    KvFlashHeadBinding binding(int slot) const { return ok_(slot) ? b_[(size_t)slot] : KvFlashHeadBinding{}; }
    // The slot's accepted pins: what kvflash_prefill_apply_on gets as slot_pins.
    int pins(int slot) const { return ok_(slot) ? pins_[(size_t)slot] : 0; }

    bool bind(int slot, uint64_t hash, int k) {
        if (!ok_(slot) || k <= 0) return false;
        b_[(size_t)slot] = {hash, k};
        return true;
    }
    void unbind(int slot) {
        if (!ok_(slot)) return;
        b_[(size_t)slot] = {};
        pins_[(size_t)slot] = 0;
    }
    // Re-applies the slot's pins and records the accepted count; a binding the pager cannot hold is released.
    template <class Pager>
    int ensure(int slot, Pager & pager) {
        if (!ok_(slot)) return 0;
        KvFlashHeadBinding & b = b_[(size_t)slot];
        if (b.k <= 0) { pins_[(size_t)slot] = 0; return 0; }
        const KvFlashRepin r = kvflash_repin_head(pager, b.k);
        pins_[(size_t)slot] = r.accepted;
        if (r.released) b = {};
        return r.accepted;
    }

private:
    bool ok_(int slot) const { return slot >= 0 && (size_t)slot < b_.size(); }
    std::vector<KvFlashHeadBinding> b_;
    std::vector<int> pins_;
};

// ---- the head, and the bytes an attach adopts -----------------------------------------------------------------

// What a capture holds: the pager header, the k chunk payloads and the recurrent slab at k chunks. k = 0: none.
struct KvFlashHead {
    int k = 0;
    uint64_t hash = 0;
    std::vector<uint8_t> hdr, chunks, slab;
};

// KvFlashPager::serialize(k) of a pager whose chunks 0..k-1 are all resident with score 0: header, the k chunk
// payloads, a ledger of {resident, materialized, score 0}. The deserialize of it places chunk c in block c.
inline std::vector<uint8_t> kvflash_head_blob(const KvFlashHead & h) {
    std::vector<uint8_t> out;
    out.reserve(h.hdr.size() + h.chunks.size() + (size_t)h.k * 8);
    out.insert(out.end(), h.hdr.begin(), h.hdr.end());
    out.insert(out.end(), h.chunks.begin(), h.chunks.end());
    for (int c = 0; c < h.k; c++) { const uint8_t led[8] = {1, 1, 0, 0, 0, 0, 0, 0}; out.insert(out.end(), led, led + 8); }
    return out;
}

// The capture cut ran: the slot's pager holds chunks 0..k-1 of `ids` and the caller has the slab at k chunks. Refused
// (k = 0 returned, nothing bound) past k_max, for k < 2, ids shorter than k chunks, or a pager that does not hold k
// chunks. Otherwise the slot is bound to the head and pinned.
template <class Pager>
inline KvFlashHead kvflash_head_capture(KvFlashHeadBinder & bd, int slot, Pager & pager, const std::vector<int32_t> & ids,
                                        int k, int k_max, std::vector<uint8_t> slab) {
    KvFlashHead h;
    const int ct = pager.chunk_tokens();
    if (k < 2 || k > k_max || ct <= 0 || pager.n_chunks() < k || (int)(ids.size() / (size_t)ct) < k) return h;
    const std::vector<uint64_t> path = kvflash_head_hashes(ids, k, ct);
    if ((int)path.size() < k) return h;
    pager.synchronize_paging();
    const size_t cb = pager.chunk_bytes();
    h.chunks.resize((size_t)k * cb);
    for (int i = 0; i < k; i++) pager.read_chunk(i, h.chunks.data() + (size_t)i * cb);
    h.hdr = pager.blob_header(k);
    h.slab = std::move(slab);
    h.hash = path.back();
    h.k = k;
    if (bd.bind(slot, h.hash, k)) bd.ensure(slot, pager);
    return h;
}

// Attach the head to the free slot: bind, adopt the blob as a parked session of k chunks, and check that the pager
// kept the pins. False leaves the slot free and unbound (the request goes cold). The caller then resumes the slot at
// k chunks and restores `head.slab`.
template <class Kv>
inline bool kvflash_head_attach(Kv & kv, KvFlashHeadBinder & bd, int slot, uint64_t ledger_id, const KvFlashHead & head) {
    if (head.k <= 0 || head.hdr.empty() || head.slab.empty()) return false;
    if (!bd.bind(slot, head.hash, head.k)) return false;
    const std::vector<uint8_t> blob = kvflash_head_blob(head);
    const uint32_t len = (uint32_t)head.k * kv.block_size();
    const auto adopted = kv.adopt_parked(slot, ledger_id, len, blob.data(), blob.size());
    if (adopted != std::decay_t<decltype(adopted)>::Ok || bd.binding(slot).k != head.k) {
        if (kv.parked(slot)) kv.drop_parked(slot);   // adopted, but the pins did not hold
        bd.unbind(slot);
        return false;
    }
    return true;
}

// A restored session (restart, evicted-to-disk): honour its persisted binding when the history still starts with the
// head's chunks.
template <class Kv>
inline bool kvflash_head_rebind(Kv & kv, KvFlashHeadBinder & bd, int slot, const std::vector<int32_t> & history,
                                uint64_t hash, int k) {
    // The pager comes back with the hard pins it parked with (KVFLASH3 layout): a head that does not hold runs unpinned.
    const std::vector<uint64_t> path = k < 2 ? std::vector<uint64_t>() : kvflash_head_hashes(history, k, (int)kv.block_size());
    if ((int)path.size() < k || path.empty() || path.back() != hash || !bd.bind(slot, hash, k)) { kv.pager(slot).unpin_all(); return false; }
    bd.ensure(slot, kv.pager(slot));
    return bd.binding(slot).k == k;
}

// A session a start-up adopt put back into `slot`, with the binding its meta.json holds.
struct KvFlashRestoredSession {
    int slot = -1;
    const std::vector<int32_t> * history = nullptr;
    std::string preamble;   // hex16 of H_k, "" none
    int k = 0;
};

// Start-up order: the adopts run in init_bundles_ before init_heads_ installs the binder and head_k_max, so no adopt can
// rebind. This is the one pass after init_heads_: every restored session whose meta binds a head is re-bound; one whose
// head does not hold is unbound and unpinned (cold-startable). Returns the slots left bound.
template <class Kv>
inline int kvflash_rebind_restored(Kv & kv, KvFlashHeadBinder & bd, int head_k_max, const std::vector<KvFlashRestoredSession> & sessions) {
    if (head_k_max <= 0) return 0;
    int bound = 0;
    for (const KvFlashRestoredSession & r : sessions) {
        if (r.preamble.empty() || r.k <= 0 || !r.history) continue;
        if (kvflash_head_rebind(kv, bd, r.slot, *r.history, std::strtoull(r.preamble.c_str(), nullptr, 16), r.k)) ++bound;
        else bd.unbind(r.slot);
    }
    return bound;
}

// The layout id a head file is saved under: the execution identity (model, KV types, geometry, backend, ggml build),
// so a file written by another execution is never read back. Two FNV-1a lanes, 16 bytes.
inline std::array<uint8_t, 16> kvflash_head_layout_id(const std::string & identity) {
    std::array<uint8_t, 16> out{};
    for (int lane = 0; lane < 2; lane++) {
        uint64_t h = 1469598103934665603ULL ^ (lane ? 0x9e3779b97f4a7c15ULL : 0);
        for (unsigned char c : std::string("kvflash-head/1|") + identity) { h ^= c; h *= 1099511628211ULL; }
        for (int b = 0; b < 8; b++) out[(size_t)(lane * 8 + b)] = (uint8_t)(h >> (8 * b));
    }
    return out;
}

}  // namespace luce::common
