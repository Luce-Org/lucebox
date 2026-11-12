// KVFlash C1.5 F3: one slot's parked session as a directory of append-only files.
//
//   <dir>/slot<i>/head                 the commit point: KvFlashSlotHead (history, the pager
//                                      header, the chunk map, the ledger, the rung positions
//                                      and the generation of every file the park consists of)
//   <dir>/slot<i>/chunks.<g>.log       16 B header, then records: 16 B tag + the chunk's KV
//   <dir>/slot<i>/slab.<id>            a recurrent slab: the history end, a rung (ids are per slot)
//
// A park appends only the chunks the turn changed, [floor(L / chunk_tokens), nc), to the log,
// writes its slab files, then renames a new head over the old one: a reader sees the previous
// park or this one, never a torn one, and a record past head.log_bytes is a torn tail that the
// next park truncates. A log whose dead records outnumber the live ones is compacted into a new
// generation. Pure: the format, the dirty range, the head, assembly and the commit executor are
// testable on a directory with no engine; kvflash_bundle_commit takes a crash point for that.
#pragma once

#include "kvflash_uq.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <cerrno>
#include <sys/stat.h>
#if defined(_WIN32)
#include <climits>
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace luce::common {

constexpr size_t kKvFlashLogHeader = 16;   // "KVFSLOG3", u32 log generation, u32 zero
constexpr size_t kKvFlashRecordTag = 16;   // u32 chunk, u32 park generation, u64 crc64 of the payload
constexpr size_t kKvFlashPagerHdr = 44;    // KvFlashPager's blob header, kept whole in the head
constexpr size_t kKvFlashLedgerEntry = 8;  // its per-chunk ledger entry: u8 was_resident, u8 materialized, 2 pad, f32 score
constexpr size_t kKvFlashPoolHdr = 16;     // its pool layout (KvFlashPager::pool_layout_blob): u32 n_blocks, u32 n_free, u64 use clock,
constexpr size_t kKvFlashPoolEntry = 16;   // then per chunk: i32 block, u32 hard-pin flag, u64 LRU rank; then n_free x i32 free blocks in order
constexpr char kKvFlashPagerMagic3[9] = "3HSALFVK";   // "KVFLASH3" as the pager writes its u64: the blob with the pool layout after the ledger

inline uint64_t kvflash_fnv1a(std::string_view s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

// What a bundle's identity text is made of: the model file (path, size, mtime), the KV types and
// rotation, the pager geometry, the context, and the paged-attention kernel route its KV was
// computed under (true: the WMMA route; false: V_DOT2). A bundle from any other server is skipped.
struct KvFlashIdentityParts {
    std::string model;
    long long   model_size = -1;
    long long   model_mtime = -1;
    int         kv_k_type = 0;
    int         kv_v_type = 0;
    bool        kv_k_rotated = false;
    int         chunk_tokens = 0;
    int         n_pool_blocks = 0;
    int         max_context = 0;
    bool        paged_wmma_route = false;
};

inline std::string kvflash_identity_key(const KvFlashIdentityParts & p) {
    return p.model + "|" + std::to_string(p.model_size) + "|" + std::to_string(p.model_mtime) + "|" +
        std::to_string(p.kv_k_type) + "|" + std::to_string(p.kv_v_type) + "|" + (p.kv_k_rotated ? "rot" : "flat") + "|" +
        std::to_string(p.chunk_tokens) + "|" + std::to_string(p.n_pool_blocks) + "|" + std::to_string(p.max_context) + "|" +
        (p.paged_wmma_route ? "wmma" : "dot2");
}

// CRC-64/XZ, slicing by 8: a chunk is ~1 MiB, so the byte-at-a-time table would cost a second
// per cold park on the writer thread.
inline const std::array<std::array<uint64_t, 256>, 8> & kvflash_crc64_tables() {
    static const std::array<std::array<uint64_t, 256>, 8> tables = [] {
        std::array<std::array<uint64_t, 256>, 8> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint64_t c = i;
            for (int k = 0; k < 8; ++k) c = (c >> 1) ^ ((c & 1) ? 0xC96C5795D7870F42ULL : 0);
            t[0][i] = c;
        }
        for (int s = 1; s < 8; ++s) for (uint32_t i = 0; i < 256; ++i) t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xff];
        return t;
    }();
    return tables;
}
inline uint64_t kvflash_crc64(const uint8_t * p, size_t n) {
    const auto & t = kvflash_crc64_tables();
    uint64_t crc = ~0ULL;
    for (; n >= 8; n -= 8, p += 8) {
        uint64_t w;
        std::memcpy(&w, p, 8);
        crc ^= w;
        crc = t[7][crc & 0xff] ^ t[6][(crc >> 8) & 0xff] ^ t[5][(crc >> 16) & 0xff] ^ t[4][(crc >> 24) & 0xff] ^
              t[3][(crc >> 32) & 0xff] ^ t[2][(crc >> 40) & 0xff] ^ t[1][(crc >> 48) & 0xff] ^ t[0][crc >> 56];
    }
    for (; n; --n, ++p) crc = t[0][(crc ^ *p) & 0xff] ^ (crc >> 8);
    return ~crc;
}

// ---- the record and log format ---------------------------------------------------------------

struct KvFlashRecordTag { uint32_t chunk = 0, gen = 0; uint64_t crc = 0; };

namespace detail {
constexpr char kHeadMagic[9] = "KVFSLOT3";
constexpr char kLogMagic[9] = "KVFSLOG3";
template <class T> void put(std::vector<uint8_t> & out, const T & v) {
    const uint8_t * p = reinterpret_cast<const uint8_t *>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}
template <class T> void put_array(std::vector<uint8_t> & out, const std::vector<T> & v) {   // u32 count, the elements
    put(out, (uint32_t)v.size());
    const uint8_t * p = reinterpret_cast<const uint8_t *>(v.data());
    out.insert(out.end(), p, p + v.size() * sizeof(T));
}
inline void put_list(std::vector<uint8_t> & out, const std::vector<std::vector<uint8_t>> & l) {
    put(out, (uint32_t)l.size());
    for (const auto & v : l) put_array(out, v);
}
struct Cursor {
    const uint8_t * p; size_t n, at = 0;
    template <class T> bool get(T & v) {
        if (n - at < sizeof(T)) return false;
        std::memcpy(&v, p + at, sizeof(T)); at += sizeof(T);
        return true;
    }
    template <class T> bool get_array(std::vector<T> & v) {
        uint32_t cnt = 0;
        if (!get(cnt) || cnt > (n - at) / sizeof(T)) return false;
        v.resize(cnt);
        if (cnt) std::memcpy(v.data(), p + at, cnt * sizeof(T));
        at += cnt * sizeof(T);
        return true;
    }
    bool get_list(std::vector<std::vector<uint8_t>> & l) {
        uint32_t cnt = 0;
        if (!get(cnt) || cnt > 4096) return false;
        l.resize(cnt);
        for (auto & v : l) if (!get_array(v)) return false;
        return true;
    }
};
}  // namespace detail

inline std::vector<uint8_t> kvflash_log_header(uint32_t gen) {
    std::vector<uint8_t> out(detail::kLogMagic, detail::kLogMagic + 8);
    detail::put(out, gen);
    detail::put(out, (uint32_t)0);
    return out;
}
inline bool kvflash_log_header_parse(const uint8_t * data, size_t n, uint32_t & gen) {
    if (!data || n < kKvFlashLogHeader || std::memcmp(data, detail::kLogMagic, 8) != 0) return false;
    std::memcpy(&gen, data + 8, 4);
    return true;
}

inline std::array<uint8_t, kKvFlashRecordTag> kvflash_record_tag(uint32_t chunk, uint32_t gen, const uint8_t * payload, size_t n) {
    std::array<uint8_t, kKvFlashRecordTag> t;
    const uint64_t crc = kvflash_crc64(payload, n);
    std::memcpy(t.data(), &chunk, 4); std::memcpy(t.data() + 4, &gen, 4); std::memcpy(t.data() + 8, &crc, 8);
    return t;
}
inline bool kvflash_record_tag_parse(const uint8_t * data, size_t n, KvFlashRecordTag & t) {
    if (!data || n < kKvFlashRecordTag) return false;
    std::memcpy(&t.chunk, data, 4); std::memcpy(&t.gen, data + 4, 4); std::memcpy(&t.crc, data + 8, 8);
    return true;
}
inline size_t kvflash_record_bytes(size_t chunk_bytes) { return kKvFlashRecordTag + chunk_bytes; }

// ---- naming and the dirty range --------------------------------------------------------------

inline std::string kvflash_bundle_log_name(uint32_t gen) { return "chunks." + std::to_string(gen) + ".log"; }
// A slab file is named by a per-slot id, not by its rung: a checkpoint that becomes the trunk keeps its file.
inline std::string kvflash_bundle_slab_name(uint32_t id) { return "slab." + std::to_string(id); }

// The chunks a park must rewrite when the turn resumed at token L: every chunk from the one that
// holds L on. A chunk below floor(L / chunk_tokens) was sealed before the turn and its record is
// still right; a cold admit (L <= 0) rewrites everything.
inline std::pair<int, int> kvflash_bundle_dirty_chunks(int L, int n_chunks, int chunk_tokens) {
    return {std::clamp(std::max(L, 0) / std::max(chunk_tokens, 1), 0, n_chunks), n_chunks};
}

// ---- the head ---------------------------------------------------------------------------------

struct KvFlashSlotHead {
    uint64_t identity = 0;                 // kvflash_fnv1a of the server's model, KV types and geometry
    uint32_t gen = 0;                      // this park's generation
    uint32_t log_gen = 0;                  // the log the map points into
    uint64_t log_bytes = 0;                // the committed length of that log: bytes past it are a torn tail
    uint32_t ckpt_pos = 0, trunk_pos = 0;  // rung positions in history, 0: no rung
    uint32_t end_gen = 0, ckpt_gen = 0, trunk_gen = 0;   // slab file ids (slab.<id>), 0: no file
    uint32_t pin_pos = 0;                                 // the pinned branch point (v5): 0 none, else the position of the ckpt, trunk or third
    uint32_t extra_pos = 0, extra_gen = 0;                // the third rung (any position but the ckpt's and the trunk's), 0: none
    // The slot's user-question query rows (LUCE_KVFLASH_UQ_QUERIES, KvFlashUqStore), an optional section after the pin: the n rows
    // of positions [uq_hi - uq_n, uq_hi) of the question window [uq_lo, uq_hi) live in file slab.<uq_gen>, keyed by the hash of the
    // prompt ids [0, uq_hi) as the live store is. uq_gen 0: none (a head from before this section, or a slot that held none).
    uint32_t uq_gen = 0, uq_lo = 0, uq_hi = 0, uq_n = 0;
    uint64_t uq_hash = 0;
    // The pager's pool layout (KvFlashPager::pool_layout_blob), an optional section after the UQ one: the chunk -> block table, the
    // LRU ranks, the use clock and the free list in order. Without it a restored pager gets the identity layout and a chunk-order LRU
    // and so evicts other chunks than the live slot would. Empty: none (a head from before this section).
    std::vector<uint8_t> pool;
    std::vector<int32_t> history;
    std::vector<uint8_t> pager_hdr;        // kKvFlashPagerHdr bytes
    std::vector<uint64_t> map;             // per chunk: the record's offset in the log, 0: no record (zeros)
    std::vector<uint8_t> ledger;           // map.size() * 8 bytes, the pager's per-chunk ledger
};

// Every slab file id a head names (0: none): the one place that lists them, so a new rung cannot be missed by a save, a
// leftover sweep, a validate, a storage listing or a next-gen pick.
inline std::array<uint32_t, 4> kvflash_head_gens(const KvFlashSlotHead & h) { return {h.end_gen, h.ckpt_gen, h.trunk_gen, h.extra_gen}; }
// Those plus the UQ rows' file: every file id a head owns. The slabs above are required (a validate refuses a head whose one is
// missing); the rows are derived data, optional at load, so a save, a leftover sweep, a cleanup and the next-gen pick use this list
// and a validate does not.
inline std::array<uint32_t, 5> kvflash_head_owned_gens(const KvFlashSlotHead & h) { return {h.end_gen, h.ckpt_gen, h.trunk_gen, h.extra_gen, h.uq_gen}; }
// The bytes of every file a head owns, the ones a park kept from the last park as well as the ones it wrote: one definition for
// the session's cache-budget figure at a park and at a start-up scan. size_of(id) is a file's size.
template <class F> inline uint64_t kvflash_head_slab_bytes(const KvFlashSlotHead & h, F size_of) {
    uint64_t b = 0;
    for (uint32_t id : kvflash_head_owned_gens(h)) if (id) b += size_of(id);
    return b;
}
inline uint32_t kvflash_head_max_gen(const KvFlashSlotHead & h) { const auto g = kvflash_head_owned_gens(h); return std::max(h.gen, *std::max_element(g.begin(), g.end())); }

inline size_t kvflash_head_bytes(size_t nh, size_t nc) { return 8 + 8 + 4 + 4 + 8 + 5 * 4 + 4 + nh * 4 + 4 + kKvFlashPagerHdr + 4 + nc * 8 + 4 + nc * 8 + 12; }
// + the UQ section (kUqTail) and the pool section: its length, the layout of nc_max chunks and a free list of up to blocks_max blocks
inline size_t kvflash_head_max_bytes(size_t history_max, size_t nc_max, size_t blocks_max = 0) {
    return kvflash_head_bytes(history_max, nc_max) + 24 + 4 + kKvFlashPoolHdr + nc_max * kKvFlashPoolEntry + blocks_max * 4;
}

inline std::vector<uint8_t> kvflash_head_pack(const KvFlashSlotHead & h) {
    std::vector<uint8_t> out(detail::kHeadMagic, detail::kHeadMagic + 8);   // constructed, not inserted: gcc 11 -Wstringop-overflow
    out.reserve(kvflash_head_bytes(h.history.size(), h.map.size()));
    detail::put(out, h.identity); detail::put(out, h.gen); detail::put(out, h.log_gen); detail::put(out, h.log_bytes);
    for (uint32_t v : {h.ckpt_pos, h.trunk_pos, h.end_gen, h.ckpt_gen, h.trunk_gen}) detail::put(out, v);
    detail::put_array(out, h.history); detail::put_array(out, h.pager_hdr); detail::put_array(out, h.map); detail::put_array(out, h.ledger);
    detail::put(out, h.extra_pos); detail::put(out, h.extra_gen); detail::put(out, h.pin_pos);   // an older head ends before them
    if (h.uq_gen != 0 || !h.pool.empty()) {   // the UQ section, only when there are rows or a pool section follows it (24 B, all zero for none)
        for (uint32_t v : {h.uq_gen, h.uq_lo, h.uq_hi, h.uq_n}) detail::put(out, v);
        detail::put(out, h.uq_hash);
    }
    if (!h.pool.empty()) detail::put_array(out, h.pool);   // last, and only when the pager gave one: a head with none is the bytes it was before it
    return out;
}

// What a park takes from the slot's pager for its head: the header, the ledger and the pool layout (the layout and the LRU order a
// restore puts back, so the restored slot continues like the live one). The map and the records are the caller's.
template <class Pager> inline void kvflash_head_take_pager(KvFlashSlotHead & h, const Pager & p) {
    h.pager_hdr = p.blob_header();
    h.ledger = p.ledger_blob();
    h.pool = p.pool_layout_blob();
}

// False (out untouched) on a bad magic, another identity, a truncated or oversized section, an
// empty history, a rung past the history or with only one of position and generation, no end
// slab, a pager header or ledger of the wrong size.
inline bool kvflash_head_unpack(const uint8_t * data, size_t n, uint64_t identity, KvFlashSlotHead & out) {
    if (!data || n < 8 || std::memcmp(data, detail::kHeadMagic, 8) != 0) return false;
    detail::Cursor c{data, n, 8};
    KvFlashSlotHead h;
    if (!c.get(h.identity) || h.identity != identity || !c.get(h.gen) || !c.get(h.log_gen) || !c.get(h.log_bytes) ||
        !c.get(h.ckpt_pos) || !c.get(h.trunk_pos) || !c.get(h.end_gen) || !c.get(h.ckpt_gen) || !c.get(h.trunk_gen) ||
        !c.get_array(h.history) || !c.get_array(h.pager_hdr) || !c.get_array(h.map) || !c.get_array(h.ledger)) return false;
    // the tail: nothing (the base head), 4 bytes (the first B4 try's branch_pos, ignored), the third rung, or the third and its pin.
    // A new head cut by exactly 4 or 8 bytes reads as an older one (the pin or the third is lost): accepted, a head is written by an atomic rename.
    // 36 bytes: the third, its pin and the UQ rows' section (kUqTail); an older head has none, and a head cut inside it is refused.
    // More than 36: the pager's pool layout after them (a length and its bytes), which must end the head.
    constexpr size_t kUqTail = 24;
    const size_t tail = n - c.at;
    if (tail == 8 || tail == 12 || tail >= 12 + kUqTail) { if (!c.get(h.extra_pos) || !c.get(h.extra_gen)) return false; }
    if ((tail == 12 || tail >= 12 + kUqTail) && !c.get(h.pin_pos)) return false;
    if (tail >= 12 + kUqTail && (!c.get(h.uq_gen) || !c.get(h.uq_lo) || !c.get(h.uq_hi) || !c.get(h.uq_n) || !c.get(h.uq_hash))) return false;
    if (tail > 12 + kUqTail && (!c.get_array(h.pool) || h.pool.empty() || c.at != n)) return false;
    if (tail != 0 && tail != 4 && tail != 8 && tail != 12 && tail < 12 + kUqTail) return false;
    const size_t nh = h.history.size();
    if (nh == 0 || h.ckpt_pos > nh || h.trunk_pos > nh || h.end_gen == 0 || (h.ckpt_pos == 0) != (h.ckpt_gen == 0) ||
        (h.trunk_pos == 0) != (h.trunk_gen == 0) || h.pager_hdr.size() != kKvFlashPagerHdr || h.ledger.size() != h.map.size() * kKvFlashLedgerEntry) return false;
    // the ladder: a trunk is below the checkpoint, and each slab is its own file (a head naming one file for two positions
    // would restore that state at the wrong one)
    if ((h.trunk_pos != 0 && h.trunk_pos >= h.ckpt_pos) || (h.ckpt_gen != 0 && h.ckpt_gen == h.end_gen) ||
        (h.trunk_gen != 0 && (h.trunk_gen == h.end_gen || h.trunk_gen == h.ckpt_gen))) return false;
    // the third rung: any position but a rung's (it would add nothing), its own file
    if (h.extra_pos > nh || (h.extra_pos == 0) != (h.extra_gen == 0) ||
        (h.extra_gen != 0 && (h.extra_gen == h.end_gen || h.extra_gen == h.ckpt_gen || h.extra_gen == h.trunk_gen)) ||
        (h.extra_pos != 0 && (h.extra_pos == h.ckpt_pos || h.extra_pos == h.trunk_pos))) return false;
    // the pin: none, or the position of the third (when there is one) or else of the checkpoint or the trunk
    if (h.pin_pos != 0 && h.pin_pos != (h.extra_pos != 0 ? h.extra_pos : h.ckpt_pos) && !(h.extra_pos == 0 && h.pin_pos == h.trunk_pos)) return false;
    // the pool layout: a header and an entry per chunk, then the free list its header counts
    if (!h.pool.empty()) {
        uint32_t n_free = 0;
        if (h.pool.size() < kKvFlashPoolHdr + h.map.size() * kKvFlashPoolEntry) return false;
        std::memcpy(&n_free, h.pool.data() + 4, 4);
        if (h.pool.size() != kKvFlashPoolHdr + h.map.size() * kKvFlashPoolEntry + (size_t)n_free * 4) return false;
    }
    // the UQ rows: a window inside the history holding n rows, in a file of their own
    if (h.uq_gen != 0 && (h.uq_n == 0 || h.uq_lo >= h.uq_hi || h.uq_hi > nh || h.uq_n > h.uq_hi - h.uq_lo ||
                          h.uq_gen == h.end_gen || h.uq_gen == h.ckpt_gen || h.uq_gen == h.trunk_gen || h.uq_gen == h.extra_gen)) return false;
    out = std::move(h);
    return true;
}


// The UQ rows' file is a slab file (one "ssm" tensor holding the rows' float bytes, no conv), so its length checks, size cap and
// cleanup are the slab's. The window, the key and n are the head's; the rows come back as the live store held them.
inline std::vector<uint8_t> kvflash_uq_file_pack(const KvFlashUqStore & u) {
    std::vector<uint8_t> rows(u.rows.size() * sizeof(float));
    if (!rows.empty()) std::memcpy(rows.data(), u.rows.data(), rows.size());
    std::vector<uint8_t> out;
    out.reserve(8 + 4 + rows.size());
    detail::put_list(out, std::vector<std::vector<uint8_t>>{std::move(rows)}); detail::put_list(out, std::vector<std::vector<uint8_t>>{});
    return out;
}
// False when the head names no rows, or the file is not one tensor of n equal rows of floats (the caller then has no rows).
inline bool kvflash_uq_file_unpack(const KvFlashSlotHead & h, const std::vector<uint8_t> & file, KvFlashUqStore & out) {
    if (h.uq_gen == 0 || h.uq_n == 0) return false;
    detail::Cursor c{file.data(), file.size()};
    std::vector<std::vector<uint8_t>> rows, none;
    if (!c.get_list(rows) || !c.get_list(none) || c.at != file.size() || rows.size() != 1 || !none.empty()) return false;
    const size_t bytes = rows[0].size();
    if (bytes == 0 || bytes % (sizeof(float) * h.uq_n) != 0) return false;
    KvFlashUqStore u;
    u.lo = (int)h.uq_lo; u.hi = (int)h.uq_hi; u.n = (int)h.uq_n; u.hash = h.uq_hash;
    u.rows.resize(bytes / sizeof(float));
    std::memcpy(u.rows.data(), rows[0].data(), bytes);
    out = std::move(u);
    return true;
}

// A slab file: the ssm and conv tensors of one recurrent state.
inline size_t kvflash_slab_bytes(const std::vector<std::vector<uint8_t>> & ssm, const std::vector<std::vector<uint8_t>> & conv) {
    size_t n = 8;
    for (const auto & v : ssm) n += 4 + v.size();
    for (const auto & v : conv) n += 4 + v.size();
    return n;
}
// Reserved at its exact size: a pack is a slab of host memory in the park's staging charge, not 1.5x of one.
inline std::vector<uint8_t> kvflash_slab_pack(const std::vector<std::vector<uint8_t>> & ssm, const std::vector<std::vector<uint8_t>> & conv) {
    std::vector<uint8_t> out;
    out.reserve(kvflash_slab_bytes(ssm, conv));
    detail::put_list(out, ssm); detail::put_list(out, conv);
    return out;
}
inline bool kvflash_slab_unpack(const uint8_t * data, size_t n, std::vector<std::vector<uint8_t>> & ssm, std::vector<std::vector<uint8_t>> & conv) {
    detail::Cursor c{data, n};
    std::vector<std::vector<uint8_t>> s, v;
    if (!data || !c.get_list(s) || !c.get_list(v) || c.at != n) return false;
    ssm = std::move(s); conv = std::move(v);
    return true;
}
inline size_t kvflash_slab_max_bytes(size_t slab_bytes) { return slab_bytes + 2 * (4 + 4096 * 4); }

// ---- assembly: the head and a log back into a KvFlashPager::serialize() blob -----------------

// read_at(offset, dst, n) -> bool. Refuses a log of another generation, a record past the
// committed length, a short read, a record that names another chunk, a crc mismatch, and a map
// that disagrees with the ledger: a materialized chunk must have a record (a hole there would
// restore as zero KV the pager trusts) and a chunk never materialized must have none. A chunk the
// map does not place is zeros, as the pager serializes a chunk it does not hold.
template <class ReadAt, class Sink>
inline bool kvflash_bundle_walk(const KvFlashSlotHead & h, size_t chunk_bytes, ReadAt && read_at, Sink && sink, std::string & why) {
    const size_t nc = h.map.size(), rb = kvflash_record_bytes(chunk_bytes);
    if (h.ledger.size() != nc * kKvFlashLedgerEntry) { why = "ledger of " + std::to_string(h.ledger.size()) + " bytes for " + std::to_string(nc) + " chunks"; return false; }
    uint8_t lh[kKvFlashLogHeader];
    uint32_t gen = 0;
    if (!read_at(0, lh, sizeof lh) || !kvflash_log_header_parse(lh, sizeof lh, gen) || gen != h.log_gen) {
        why = "log header is not generation " + std::to_string(h.log_gen);
        return false;
    }
    std::vector<uint8_t> rec(rb);
    for (size_t c = 0; c < nc; ++c) {
        const uint64_t off = h.map[c];
        const bool materialized = h.ledger[c * kKvFlashLedgerEntry + 1] != 0;
        if ((off != 0) != materialized) { why = "chunk " + std::to_string(c) + (materialized ? ": materialized with no record" : ": a record for a chunk never materialized"); return false; }
        if (off == 0) { sink(c, nullptr); continue; }
        const std::string at = "chunk " + std::to_string(c) + " at " + std::to_string(off);
        KvFlashRecordTag t;
        if (off < kKvFlashLogHeader || off + rb > h.log_bytes) { why = at + ": past the committed " + std::to_string(h.log_bytes) + " bytes"; return false; }
        if (!read_at(off, rec.data(), rb)) { why = at + ": short read"; return false; }
        kvflash_record_tag_parse(rec.data(), rb, t);
        if (t.chunk != c) { why = at + ": the record names chunk " + std::to_string(t.chunk); return false; }
        if (t.crc != kvflash_crc64(rec.data() + kKvFlashRecordTag, chunk_bytes)) { why = at + ": crc mismatch"; return false; }
        sink(c, rec.data() + kKvFlashRecordTag);
    }
    return true;
}

template <class ReadAt>
inline bool kvflash_bundle_assemble(const KvFlashSlotHead & h, size_t chunk_bytes, ReadAt && read_at, std::vector<uint8_t> & out, std::string & why) {
    const size_t nc = h.map.size();
    std::vector<uint8_t> blob;
    const size_t total = h.pager_hdr.size() + nc * chunk_bytes + h.ledger.size() + h.pool.size();
    try { blob.reserve(total); } catch (const std::bad_alloc &) { why = "cannot hold " + std::to_string(total) + " bytes"; return false; }
    blob.insert(blob.end(), h.pager_hdr.begin(), h.pager_hdr.end());
    const bool ok = kvflash_bundle_walk(h, chunk_bytes, read_at, [&](size_t, const uint8_t * payload) {
        if (payload) blob.insert(blob.end(), payload, payload + chunk_bytes); else blob.insert(blob.end(), chunk_bytes, 0);
    }, why);
    if (!ok) return false;
    blob.insert(blob.end(), h.ledger.begin(), h.ledger.end());
    if (!h.pool.empty()) {   // the pool layout makes it the pager's KVFLASH3 blob: it restores the park's layout, not an identity one
        std::memcpy(blob.data(), kKvFlashPagerMagic3, 8);
        blob.insert(blob.end(), h.pool.begin(), h.pool.end());
    }
    out = std::move(blob);
    return true;
}

// ---- files ------------------------------------------------------------------------------------

// The POSIX file calls of the store, and their MSVC CRT equivalents (the style of disk_store.cpp).
#if defined(_WIN32)
inline long kvflash_pid() { return (long)_getpid(); }
inline bool kvflash_fsync(FILE * f) { return _commit(_fileno(f)) == 0; }
inline bool kvflash_fseek(FILE * f, uint64_t off) { return _fseeki64(f, (__int64)off, SEEK_SET) == 0; }
inline bool kvflash_ftruncate(FILE * f, uint64_t bytes) { return _chsize_s(_fileno(f), (__int64)bytes) == 0; }
inline bool kvflash_mkdir(const std::string & dir) { return _mkdir(dir.c_str()) == 0; }
// rename() does not replace an existing file there; std::filesystem::rename does (MoveFileEx).
inline bool kvflash_rename(const std::string & from, const std::string & to) { std::error_code ec; std::filesystem::rename(from, to, ec); return !ec; }
inline int kvflash_open_ro(const std::string & path) { return _open(path.c_str(), _O_RDONLY | _O_BINARY | _O_NOINHERIT); }
// No pread: a seek and a read on the caller's own descriptor (every reader here opens its own).
inline bool kvflash_pread(int fd, uint8_t * dst, size_t n, uint64_t off) {
    return n <= (size_t)INT_MAX && _lseeki64(fd, (__int64)off, SEEK_SET) == (__int64)off && _read(fd, dst, (unsigned)n) == (int)n;
}
inline void kvflash_close(int fd) { _close(fd); }
#else
inline long kvflash_pid() { return (long)::getpid(); }
inline bool kvflash_fsync(FILE * f) { return ::fsync(fileno(f)) == 0; }
inline bool kvflash_fseek(FILE * f, uint64_t off) { return ::fseeko(f, (off_t)off, SEEK_SET) == 0; }
inline bool kvflash_ftruncate(FILE * f, uint64_t bytes) { return ::ftruncate(fileno(f), (off_t)bytes) == 0; }
inline bool kvflash_mkdir(const std::string & dir) { return ::mkdir(dir.c_str(), 0755) == 0; }
inline bool kvflash_rename(const std::string & from, const std::string & to) { return std::rename(from.c_str(), to.c_str()) == 0; }
inline int kvflash_open_ro(const std::string & path) { return ::open(path.c_str(), O_RDONLY | O_CLOEXEC); }
inline bool kvflash_pread(int fd, uint8_t * dst, size_t n, uint64_t off) { return ::pread(fd, dst, n, (off_t)off) == (ssize_t)n; }
inline void kvflash_close(int fd) { ::close(fd); }
#endif

// Test seam: a hook that makes one I/O step fail (op "write": an atomic file replace, "fsync": a
// directory fsync, "stat": an existence query of the session store), given the step's path. Unset in
// production (one empty std::function test per step).
using KvFlashIoFault = std::function<bool(std::string_view op, const std::string & path)>;
inline KvFlashIoFault & kvflash_io_fault() { static KvFlashIoFault f; return f; }
inline bool kvflash_io_fails(std::string_view op, const std::string & path) { const KvFlashIoFault & f = kvflash_io_fault(); return f && f(op, path); }

// `head` then whatever `more(put)` writes, to `path` through path.tmp.<pid> + rename: a reader
// sees the old file or the new one, never a torn one, and two servers never share a tmp inode.
template <class More>
inline bool kvflash_bundle_write_stream(const std::string & path, const std::vector<uint8_t> & head, More && more) {
    if (kvflash_io_fails("write", path)) return false;
    const std::string tmp = path + ".tmp." + std::to_string(kvflash_pid());
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    auto put = [f](const uint8_t * d, size_t n) { return std::fwrite(d, 1, n, f) == n; };
    const bool ok = put(head.data(), head.size()) && more(put) && std::fflush(f) == 0 && kvflash_fsync(f);
    const bool closed = std::fclose(f) == 0;
    if (!ok || !closed || !kvflash_rename(tmp, path)) { std::remove(tmp.c_str()); return false; }
    return true;
}

inline bool kvflash_bundle_write(const std::string & path, const std::vector<uint8_t> & bytes) {
    return kvflash_bundle_write_stream(path, bytes, [](auto &) { return true; });
}

// False (out untouched) when the file is missing, unreadable, over max_bytes or too big to hold.
inline bool kvflash_bundle_read(const std::string & path, std::vector<uint8_t> & out, size_t max_bytes = SIZE_MAX) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = std::fseek(f, 0, SEEK_END) == 0;
    const long size = ok ? std::ftell(f) : -1;
    ok = ok && size >= 0 && (size_t)size <= max_bytes && std::fseek(f, 0, SEEK_SET) == 0;
    std::vector<uint8_t> buf;
    if (ok) {
        try { buf.resize((size_t)size); } catch (const std::bad_alloc &) { ok = false; }
        ok = ok && std::fread(buf.data(), 1, buf.size(), f) == buf.size();
    }
    std::fclose(f);
    if (ok) out = std::move(buf);
    return ok;
}

// The directory entries (a renamed file, an unlink) are durable. A filesystem that cannot fsync a
// directory (EINVAL) is taken as is.
inline bool kvflash_fsync_dir(const std::string & dir) {
    if (kvflash_io_fails("fsync", dir)) return false;
#if defined(_WIN32)
    return true;   // a directory cannot be fsynced there: the rename is as durable as NTFS makes it (disk_store.cpp)
#else
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return false;
    const bool ok = ::fsync(fd) == 0 || errno == EINVAL;
    ::close(fd);
    return ok;
#endif
}

// ---- a park on disk: commit, load, cleanup -----------------------------------------------------

// Everything one park writes, built by the engine, executed by the writer thread.
struct KvFlashBundleCommit {
    std::string dir;                       // the slot directory
    bool new_dir = false;                  // create it (the slot's first park): its parent is fsynced too
    uint32_t gen = 0;                      // the park generation, stamped on every record
    uint32_t log_gen = 0;                  // the log the records go to
    bool new_log = false;                  // true: write chunks.<log_gen>.log afresh (header, copies, records)
    uint64_t log_bytes = 0;                // append (new_log false): the committed length to append at
    uint32_t copy_from = 0;                // compaction: the old log the copies are read from
    std::vector<uint64_t> copies;          // offsets of live records in copy_from, written in order after the header
    size_t record_bytes = 0;               // kKvFlashRecordTag + chunk_bytes
    std::vector<std::pair<uint32_t, std::vector<uint8_t>>> records;         // (chunk, KV bytes), written in order after the copies
    std::vector<std::pair<std::string, std::vector<uint8_t>>> files;        // slab files: name, bytes
    std::vector<uint8_t> head;             // kvflash_head_pack of the park's head
    std::vector<std::string> unlink;       // files the new head no longer names
};

// 1 the log (fsync), 2 the slab files (tmp + fsync + rename), 3 the directory (and its parent for
// a new one), 4 the head (tmp + fsync + rename + directory): the commit point, 5 the unlinks.
// Returns false right after step crash_after_step (tests, LUCE_KVFLASH_BUNDLE_CRASH_AT): a
// process killed there leaves the previous park (steps 1-3) or this one (4, 5) loadable, and
// kvflash_bundle_clean removes what the surviving head does not name.
inline bool kvflash_bundle_commit(const KvFlashBundleCommit & c, int crash_after_step = 0) {
    for (const auto & r : c.records) if (r.second.size() + kKvFlashRecordTag != c.record_bytes) return false;
    if (c.new_dir && !kvflash_mkdir(c.dir) && errno != EEXIST) return false;
    const std::string log = c.dir + "/" + kvflash_bundle_log_name(c.log_gen);
    FILE * f = std::fopen(log.c_str(), c.new_log ? "wb" : "r+b");
    if (!f) return false;
    auto put = [f](const uint8_t * d, size_t n) { return std::fwrite(d, 1, n, f) == n; };
    bool ok = true;
    if (c.new_log) {
        const auto hdr = kvflash_log_header(c.log_gen);
        ok = put(hdr.data(), hdr.size());
        if (ok && !c.copies.empty()) {
            FILE * src = std::fopen((c.dir + "/" + kvflash_bundle_log_name(c.copy_from)).c_str(), "rb");
            std::vector<uint8_t> buf(c.record_bytes);
            ok = src != nullptr;
            for (size_t i = 0; ok && i < c.copies.size(); ++i)
                ok = kvflash_fseek(src, c.copies[i]) && std::fread(buf.data(), 1, buf.size(), src) == buf.size() && put(buf.data(), buf.size());
            if (src) std::fclose(src);
        }
    } else {
        ok = kvflash_ftruncate(f, c.log_bytes) && kvflash_fseek(f, c.log_bytes);
    }
    for (size_t i = 0; ok && i < c.records.size(); ++i) {
        const auto & [chunk, payload] = c.records[i];
        const auto tag = kvflash_record_tag(chunk, c.gen, payload.data(), payload.size());
        ok = put(tag.data(), tag.size()) && put(payload.data(), payload.size());
    }
    ok = ok && std::fflush(f) == 0 && kvflash_fsync(f);
    if (std::fclose(f) != 0 || !ok) return false;
    if (crash_after_step == 1) return false;
    for (const auto & [name, bytes] : c.files) if (!kvflash_bundle_write(c.dir + "/" + name, bytes)) return false;
    if (crash_after_step == 2) return false;
    if (!kvflash_fsync_dir(c.dir)) return false;
    if (c.new_dir) {
        const std::string parent = std::filesystem::path(c.dir).parent_path().string();
        if (!kvflash_fsync_dir(parent.empty() ? "." : parent)) return false;
    }
    if (crash_after_step == 3) return false;
    if (!kvflash_bundle_write(c.dir + "/head", c.head) || !kvflash_fsync_dir(c.dir)) return false;
    if (crash_after_step == 4) return false;
    for (const auto & name : c.unlink) std::remove((c.dir + "/" + name).c_str());
    return crash_after_step != 5;
}

// A log is compacted when its dead records outnumber the live ones: the disk and the restart
// read stay within 2x of the session.
inline bool kvflash_bundle_needs_compaction(uint64_t log_bytes, size_t live, size_t record_bytes) {
    const uint64_t records = log_bytes > kKvFlashLogHeader ? (log_bytes - kKvFlashLogHeader) / record_bytes : 0;
    return records > live && records - live > live;
}
// A park's verdict: the log as its append would leave it (the kept records' dead siblings plus the appended ones' predecessors).
inline bool kvflash_bundle_compacts(uint64_t log_bytes, size_t kept, size_t written, size_t record_bytes) {
    return kvflash_bundle_needs_compaction(log_bytes + written * record_bytes, kept + written, record_bytes);
}

// The files a park that failed to commit may have left that the durable head does not name: its
// log when it was a new one, its new slab files. The next park unlinks them.
inline std::vector<std::string> kvflash_bundle_leftovers(const KvFlashSlotHead & durable, const KvFlashSlotHead & failed) {
    std::vector<std::string> out;
    if (failed.log_gen != durable.log_gen) out.push_back(kvflash_bundle_log_name(failed.log_gen));
    const auto named = kvflash_head_owned_gens(durable);
    for (uint32_t id : kvflash_head_owned_gens(failed))
        if (id && std::find(named.begin(), named.end(), id) == named.end()) out.push_back(kvflash_bundle_slab_name(id));
    return out;
}

struct KvFlashBundleLoaded {
    KvFlashSlotHead head;
    KvFlashUqStore uq;
    std::string uq_why;
    std::vector<uint8_t> pager;            // a KvFlashPager::serialize() blob
    std::vector<std::vector<uint8_t>> end_ssm, end_conv, ckpt_ssm, ckpt_conv, trunk_ssm, trunk_conv, extra_ssm, extra_conv;
};

// What this server can hold: the head is read under the bytes of a head at these limits, and a
// head past them is refused before its pager blob (nc x chunk_bytes) is allocated.
struct KvFlashBundleLimits {
    size_t history_max = 0;                // tokens (the context)
    size_t nc_max = 0;                     // chunks (the context in chunks)
    size_t slab_max = 0;                   // bytes of one slab file (kvflash_slab_max_bytes)
    size_t blocks_max = 0;                 // blocks of a slot's pool: the longest free list of the pool layout in a head
};
inline size_t kvflash_bundle_head_max(const KvFlashBundleLimits & lim) { return kvflash_head_max_bytes(lim.history_max, lim.nc_max, lim.blocks_max); }

// The park a slot directory holds: its head (within the limits, of this identity), the pager
// blob assembled from its log, its slab files. False with the reason.
inline bool kvflash_bundle_load(const std::string & dir, uint64_t identity, size_t chunk_bytes, const KvFlashBundleLimits & lim,
                                KvFlashBundleLoaded & out, std::string & why) {
    std::vector<uint8_t> bytes;
    KvFlashBundleLoaded l;
    const size_t head_max = kvflash_bundle_head_max(lim);
    if (!kvflash_bundle_read(dir + "/head", bytes, head_max)) { why = "no head under " + std::to_string(head_max) + " bytes"; return false; }
    if (!kvflash_head_unpack(bytes.data(), bytes.size(), identity, l.head)) { why = "head refused: another server or a bad format"; return false; }
    if (l.head.history.size() > lim.history_max || l.head.map.size() > lim.nc_max) {
        why = "head of " + std::to_string(l.head.history.size()) + " tokens, " + std::to_string(l.head.map.size()) + " chunks is over this server's " +
              std::to_string(lim.history_max) + ", " + std::to_string(lim.nc_max);
        return false;
    }
    const std::string log = kvflash_bundle_log_name(l.head.log_gen);
    const int fd = kvflash_open_ro(dir + "/" + log);
    if (fd < 0) { why = "no " + log; return false; }
    const bool ok = kvflash_bundle_assemble(l.head, chunk_bytes, [fd](uint64_t off, uint8_t * dst, size_t n) { return kvflash_pread(fd, dst, n, off); }, l.pager, why);
    kvflash_close(fd);
    if (!ok) { why = log + ": " + why; return false; }
    auto slab = [&](const char * kind, uint32_t id, std::vector<std::vector<uint8_t>> & ssm, std::vector<std::vector<uint8_t>> & conv) {
        if (id == 0) return true;
        const std::string name = kvflash_bundle_slab_name(id);
        std::vector<uint8_t> b;
        if (kvflash_bundle_read(dir + "/" + name, b, lim.slab_max) && kvflash_slab_unpack(b.data(), b.size(), ssm, conv)) return true;
        why = std::string(kind) + " slab " + name + ": missing, over " + std::to_string(lim.slab_max) + " bytes or a bad format";
        return false;
    };
    if (!slab("end", l.head.end_gen, l.end_ssm, l.end_conv) || !slab("ckpt", l.head.ckpt_gen, l.ckpt_ssm, l.ckpt_conv) ||
        !slab("trunk", l.head.trunk_gen, l.trunk_ssm, l.trunk_conv) || !slab("third", l.head.extra_gen, l.extra_ssm, l.extra_conv)) return false;
    if (l.head.uq_gen != 0) {   // derived data: a lost or damaged file costs the rows (a new question recaptures them), not the session
        const std::string name = kvflash_bundle_slab_name(l.head.uq_gen);
        std::vector<uint8_t> b;
        if (!kvflash_bundle_read(dir + "/" + name, b, lim.slab_max) || !kvflash_uq_file_unpack(l.head, b, l.uq)) {
            l.uq = KvFlashUqStore{};
            l.uq_why = "uq rows " + name + ": missing, over " + std::to_string(lim.slab_max) + " bytes or a bad format; the slot runs without them";
        }
    }
    out = std::move(l);
    return true;
}

// Is a slab file well-formed (kvflash_slab_unpack would take it)? Walks the length prefixes with
// seeks, so a slab of 150 MiB costs a few reads.
inline bool kvflash_slab_file_valid(const std::string & path, size_t max_bytes) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = std::fseek(f, 0, SEEK_END) == 0;
    const long size = ok ? std::ftell(f) : -1;
    ok = ok && size >= 0 && (size_t)size <= max_bytes && std::fseek(f, 0, SEEK_SET) == 0;
    uint64_t at = 0;
    auto u32 = [&](uint32_t & v) { if (std::fread(&v, 4, 1, f) != 1) return false; at += 4; return true; };
    for (int list = 0; ok && list < 2; ++list) {
        uint32_t cnt = 0;
        ok = u32(cnt) && cnt <= 4096;
        for (uint32_t i = 0; ok && i < cnt; ++i) {
            uint32_t n = 0;
            ok = u32(n) && at + n <= (uint64_t)size && std::fseek(f, (long)n, SEEK_CUR) == 0;
            at += n;
        }
    }
    std::fclose(f);
    return ok && at == (uint64_t)size;
}

// Can the park a directory holds be restored here? The same checks as kvflash_bundle_load without its
// memory: the log is streamed one record at a time, each slab is walked, no pager blob is built.
inline bool kvflash_bundle_validate(const std::string & dir, const KvFlashSlotHead & h, size_t chunk_bytes, size_t slab_max, std::string & why) {
    const std::string log = kvflash_bundle_log_name(h.log_gen);
    const int fd = kvflash_open_ro(dir + "/" + log);
    if (fd < 0) { why = "no " + log; return false; }
    const bool ok = kvflash_bundle_walk(h, chunk_bytes, [fd](uint64_t off, uint8_t * dst, size_t n) { return kvflash_pread(fd, dst, n, off); },
                                        [](size_t, const uint8_t *) {}, why);
    kvflash_close(fd);
    if (!ok) { why = log + ": " + why; return false; }
    for (uint32_t id : kvflash_head_gens(h))
        if (id && !kvflash_slab_file_valid(dir + "/" + kvflash_bundle_slab_name(id), slab_max)) { why = "slab " + kvflash_bundle_slab_name(id) + ": missing, over " + std::to_string(slab_max) + " bytes or a bad format"; return false; }
    return true;
}

// After a restart: unlink every file the head does not name (a park cut short, a stale tmp) and
// cut the log back to the committed length. With no head, empty the directory.
inline void kvflash_bundle_clean(const std::string & dir, const KvFlashSlotHead * head) {
    std::vector<std::string> keep;
    if (head) {
        keep = {"head", "meta.json", kvflash_bundle_log_name(head->log_gen), kvflash_bundle_slab_name(head->end_gen)};   // meta.json belongs to the session store, not to a park
        if (head->ckpt_gen) keep.push_back(kvflash_bundle_slab_name(head->ckpt_gen));
        if (head->trunk_gen) keep.push_back(kvflash_bundle_slab_name(head->trunk_gen));
        if (head->extra_gen) keep.push_back(kvflash_bundle_slab_name(head->extra_gen));
        if (head->uq_gen) keep.push_back(kvflash_bundle_slab_name(head->uq_gen));
    }
    std::error_code ec;
    for (const auto & e : std::filesystem::directory_iterator(dir, ec))
        if (std::find(keep.begin(), keep.end(), e.path().filename().string()) == keep.end()) std::filesystem::remove_all(e.path(), ec);
    if (head) {
        const auto log = std::filesystem::path(dir) / kvflash_bundle_log_name(head->log_gen);
        const auto size = std::filesystem::file_size(log, ec);
        if (!ec && size > head->log_bytes) std::filesystem::resize_file(log, head->log_bytes, ec);
    }
}
inline void kvflash_bundle_remove(const std::string & dir) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}
// LUCE_KVFLASH_BUNDLE_CRASH_AT=<step>[:<commit>]: the writer exits the process right after that
// commit step (1-5) of its n-th commit (default the first). {0, 0}: off.
inline std::pair<int, int> kvflash_bundle_crash_at(const char * env) {
    if (!env || !*env) return {0, 0};
    const int step = std::atoi(env);
    const char * colon = std::strchr(env, ':');
    const int commit = colon ? std::atoi(colon + 1) : 1;
    if (step < 1 || step > 5 || commit < 1) return {0, 0};
    return {step, commit};
}

// ---- the writer thread ------------------------------------------------------------------------

// The engine thread hands over a commit and goes on; one thread executes them in order, so a later
// park of a slot lands after an earlier one. drain() waits for the queue; the destructor drains.
class KvFlashBundleWriter {
public:
    struct Job { int slot = -1; KvFlashBundleCommit commit; };
    using Done = std::function<void(const Job &, bool ok, double ms)>;   // on the writer thread

    explicit KvFlashBundleWriter(Done done = {})
        : done_(std::move(done)), crash_(kvflash_bundle_crash_at(std::getenv("LUCE_KVFLASH_BUNDLE_CRASH_AT"))), th_([this] { run_(); }) {}
    ~KvFlashBundleWriter() {
        drain();
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
        cv_.notify_all();
        th_.join();
    }
    KvFlashBundleWriter(const KvFlashBundleWriter &) = delete;
    KvFlashBundleWriter & operator=(const KvFlashBundleWriter &) = delete;

    void submit(Job j) {
        { std::lock_guard<std::mutex> lk(m_); q_.push_back(std::move(j)); }
        cv_.notify_all();
    }
    void drain() {
        std::unique_lock<std::mutex> lk(m_);
        idle_.wait(lk, [&] { return q_.empty() && !busy_; });
    }
    // Until no job of this slot is queued or being written: the slot's next park is staged only
    // when the earlier one is on disk (one staged park per slot, the host ladder's charge), and
    // a removal of the slot's directory cannot be undone by a commit still queued.
    void wait(int slot) {
        std::unique_lock<std::mutex> lk(m_);
        idle_.wait(lk, [&] {
            return !(busy_ && busy_slot_ == slot) &&
                   std::none_of(q_.begin(), q_.end(), [&](const Job & j) { return j.slot == slot; });
        });
    }
    // A job of this slot is queued or being written (its Done callback included): the staging it holds is still allocated.
    bool in_flight(int slot) const {
        std::lock_guard<std::mutex> lk(m_);
        return (busy_ && busy_slot_ == slot) || std::any_of(q_.begin(), q_.end(), [&](const Job & j) { return j.slot == slot; });
    }
    size_t pending() const { std::lock_guard<std::mutex> lk(m_); return q_.size() + (busy_ ? 1 : 0); }

private:
    void run_() {
        std::unique_lock<std::mutex> lk(m_);
        for (;;) {
            cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
            if (q_.empty()) return;
            Job j = std::move(q_.front());
            q_.pop_front();
            busy_ = true;
            busy_slot_ = j.slot;
            lk.unlock();
            const auto t0 = std::chrono::steady_clock::now();
            const int crash_step = crash_.first && ++n_commits_ == crash_.second ? crash_.first : 0;
            const bool ok = kvflash_bundle_commit(j.commit, crash_step);
            if (crash_step) {   // the crash gate: the process dies here, as a kill would, and the restart adopts what the directory holds
                std::fprintf(stderr, "[kvflash] LUCE_KVFLASH_BUNDLE_CRASH_AT: exiting after step %d of commit %d (slot %d gen %u)\n", crash_step, n_commits_, j.slot, j.commit.gen);
                std::fflush(stderr);
                std::_Exit(70);
            }
            if (done_) done_(j, ok, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            j = Job{};   // the staging goes before the slot is reported idle
            lk.lock();
            busy_ = false;
            busy_slot_ = -1;
            idle_.notify_all();
        }
    }

    Done done_;
    std::pair<int, int> crash_;            // LUCE_KVFLASH_BUNDLE_CRASH_AT
    int n_commits_ = 0;
    mutable std::mutex m_;
    std::condition_variable cv_, idle_;
    std::deque<Job> q_;
    bool busy_ = false, stop_ = false;
    int busy_slot_ = -1;
    std::thread th_;
};

}  // namespace luce::common
