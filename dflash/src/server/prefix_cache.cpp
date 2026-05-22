// Prefix cache implementation.

#include "prefix_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <chrono>

namespace dflash::common {

// ─── Chat marker resolution ────────────────────────────────────────────

bool resolve_chat_markers(const Tokenizer & tok, ChatMarkers & out) {
    // Try Qwen family: <|im_end|> and <|im_start|> should be single tokens.
    auto im_end = tok.encode("<|im_end|>");
    auto im_start = tok.encode("<|im_start|>");
    if (im_end.size() == 1 && im_start.size() == 1) {
        auto sys = tok.encode("system");
        out.family = "qwen";
        out.sys_role_prefix = {im_start[0]};
        if (sys.size() == 1) out.sys_role_prefix.push_back(sys[0]);
        out.end_msg_seqs = {{im_end[0]}};
        out.next_role_starts = {{im_start[0]}};
        return true;
    }

    // Try Gemma family: <|turn> (start) and <turn|> (end) are single tokens.
    auto turn_start = tok.encode("<|turn>");
    auto turn_end   = tok.encode("<turn|>");
    if (turn_start.size() == 1 && turn_end.size() == 1) {
        out.family = "gemma";
        out.sys_role_prefix = {turn_start[0]};
        out.end_msg_seqs = {{turn_end[0]}};
        out.next_role_starts = {{turn_start[0]}};
        return true;
    }

    // Try Laguna family: XML-style markers.
    auto start_sys = tok.encode("<system>");
    auto end_sys   = tok.encode("</system>");
    auto start_usr = tok.encode("<user>");
    auto end_usr   = tok.encode("</user>");
    auto start_ast = tok.encode("<assistant>");
    auto end_ast   = tok.encode("</assistant>");
    if (!start_sys.empty() && !end_sys.empty() && !start_usr.empty() &&
        !end_usr.empty() && !start_ast.empty() && !end_ast.empty()) {
        out.family = "laguna";
        out.sys_role_prefix = start_sys;
        out.end_msg_seqs = {end_sys, end_usr, end_ast};
        out.next_role_starts = {start_usr, start_ast, start_sys};
        return true;
    }

    return false;
}

// ─── Boundary detection ─────────────────────────────────────────────────

static bool seq_at(const std::vector<int32_t> & ids, int idx,
                   const std::vector<int32_t> & seq) {
    if (idx < 0 || idx + (int)seq.size() > (int)ids.size()) return false;
    for (int k = 0; k < (int)seq.size(); k++) {
        if (ids[idx + k] != seq[k]) return false;
    }
    return true;
}

static int find_first_seq(const std::vector<int32_t> & ids,
                          const std::vector<int32_t> & seq, int start = 0) {
    if (seq.empty()) return -1;
    int n = (int)ids.size(), m = (int)seq.size();
    for (int i = start; i + m <= n; i++) {
        if (ids[i] == seq[0] && seq_at(ids, i, seq)) return i;
    }
    return -1;
}

static std::pair<int, int> find_first_seq_any(
        const std::vector<int32_t> & ids,
        const std::vector<std::vector<int32_t>> & seqs, int start = 0) {
    int best = -1, best_len = 0;
    for (const auto & s : seqs) {
        int idx = find_first_seq(ids, s, start);
        if (idx >= 0 && (best < 0 || idx < best)) {
            best = idx;
            best_len = (int)s.size();
        }
    }
    return {best, best_len};
}

std::vector<int> find_all_boundaries(const std::vector<int32_t> & ids,
                                     const ChatMarkers & markers) {
    std::vector<int> out;
    int sys_idx = find_first_seq(ids, markers.sys_role_prefix);
    if (sys_idx < 0) return out;

    int cursor = sys_idx + (int)markers.sys_role_prefix.size();
    while (true) {
        auto [end_idx, end_len] = find_first_seq_any(ids, markers.end_msg_seqs, cursor);
        if (end_idx < 0) break;
        int after_end = end_idx + end_len;

        int next_match = -1, next_len = 0;
        for (int skip = 0; skip < 5; skip++) {
            int probe = after_end + skip;
            for (const auto & s : markers.next_role_starts) {
                if (seq_at(ids, probe, s)) {
                    next_match = probe;
                    next_len = (int)s.size();
                    goto found;
                }
            }
        }
        found:
        if (next_match < 0) break;
        int boundary = next_match + next_len;
        out.push_back(boundary);
        cursor = boundary;
    }
    return out;
}

// ─── Hashing ────────────────────────────────────────────────────────────

// Simple SHA-1 using a minimal inline implementation (no OpenSSL dependency).
// We only need a 16-byte hash for cache keys.

// Minimal SHA-1 — just enough for cache keys. Using a simple portable impl.
static void sha1_hash(const void * data, size_t len, uint8_t out[20]) {
    // Rotate left
    auto rotl = [](uint32_t x, int n) -> uint32_t {
        return (x << n) | (x >> (32 - n));
    };

    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE,
             h3 = 0x10325476, h4 = 0xC3D2E1F0;

    // Pad message
    size_t new_len = len + 1;
    while (new_len % 64 != 56) new_len++;
    std::vector<uint8_t> msg(new_len + 8, 0);
    std::memcpy(msg.data(), data, len);
    msg[len] = 0x80;
    uint64_t bit_len = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) {
        msg[new_len + i] = (uint8_t)(bit_len >> (56 - 8 * i));
    }

    // Process blocks
    for (size_t offset = 0; offset < msg.size(); offset += 64) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++) {
            w[i] = ((uint32_t)msg[offset + 4*i] << 24) |
                    ((uint32_t)msg[offset + 4*i+1] << 16) |
                    ((uint32_t)msg[offset + 4*i+2] << 8) |
                    ((uint32_t)msg[offset + 4*i+3]);
        }
        for (int i = 16; i < 80; i++) {
            w[i] = rotl(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
        }

        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4;
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20)      { f = (b & c) | (~b & d); k = 0x5A827999; }
            else if (i < 40) { f = b ^ c ^ d;          k = 0x6ED9EBA1; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else              { f = b ^ c ^ d;          k = 0xCA62C1D6; }
            uint32_t temp = rotl(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rotl(b, 30); b = a; a = temp;
        }
        h0 += a; h1 += b; h2 += c; h3 += d; h4 += e;
    }

    // Output
    auto store32 = [](uint8_t * p, uint32_t v) {
        p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
        p[2] = (uint8_t)(v >> 8);  p[3] = (uint8_t)v;
    };
    store32(out,     h0);
    store32(out + 4, h1);
    store32(out + 8, h2);
    store32(out + 12, h3);
    store32(out + 16, h4);
}

PrefixHash hash_prefix(const int32_t * ids, int count) {
    // Build hash input: [count as LE u32] + [ids as LE i32 array]
    std::vector<uint8_t> buf(4 + count * 4);
    uint32_t n = (uint32_t)count;
    std::memcpy(buf.data(), &n, 4);
    std::memcpy(buf.data() + 4, ids, count * 4);

    uint8_t sha[20];
    sha1_hash(buf.data(), buf.size(), sha);

    PrefixHash h{};
    std::memcpy(h.data(), sha, 16);
    return h;
}

// ─── PrefixCache ────────────────────────────────────────────────────────

PrefixCache::PrefixCache(int cap, const Tokenizer & tokenizer)
    : cap_(std::min(cap, MAX_SLOTS))
{
    if (cap_ <= 0) {
        disabled_ = true;
        return;
    }
    if (!resolve_chat_markers(tokenizer, markers_)) {
        std::fprintf(stderr, "[pc] could not resolve chat markers; prefix cache disabled\n");
        disabled_ = true;
        cap_ = 0;
        return;
    }
    disabled_ = false;
    std::fprintf(stderr, "[pc] enabled: cap=%d family=%s\n", cap_, markers_.family.c_str());
}

// ── LRU helpers ─────────────────────────────────────────────────────────

int PrefixCache::find_entry(const PrefixHash & h) const {
    for (int i = 0; i < (int)entries_.size(); i++) {
        if (entries_[i].hash == h) return i;
    }
    return -1;
}

void PrefixCache::move_to_end(int idx) {
    if (idx < 0 || idx >= (int)entries_.size()) return;
    auto e = entries_[idx];
    entries_.erase(entries_.begin() + idx);
    entries_.push_back(e);
}

int PrefixCache::find_full_entry(const PrefixHash & h) const {
    for (int i = 0; i < (int)full_entries_.size(); i++) {
        if (full_entries_[i].hash == h) return i;
    }
    return -1;
}

void PrefixCache::move_full_to_end(int idx) {
    if (idx < 0 || idx >= (int)full_entries_.size()) return;
    auto e = std::move(full_entries_[idx]);
    full_entries_.erase(full_entries_.begin() + idx);
    full_entries_.push_back(std::move(e));
}

// ── Inline prefix cache ─────────────────────────────────────────────────

std::pair<int, int> PrefixCache::lookup(const std::vector<int32_t> & prompt_ids) {
    if (disabled_) return {-1, 0};

    auto boundaries = find_all_boundaries(prompt_ids, markers_);
    int best_slot = -1, best_len = 0;

    for (int cut : boundaries) {
        auto key = hash_prefix(prompt_ids.data(), cut);
        int idx = find_entry(key);
        if (idx >= 0) {
            if (cut > best_len) {
                best_slot = entries_[idx].slot;
                best_len = cut;
            }
            move_to_end(idx);
        }
    }

    if (best_slot >= 0) {
        lifetime_hits_++;
        std::fprintf(stderr, "[pc] lookup hit slot=%d prefix_len=%d (of %zu total)\n",
                     best_slot, best_len, prompt_ids.size());
    }
    return {best_slot, best_len};
}

std::pair<int, int> PrefixCache::prepare_inline_snap(
        const std::vector<int32_t> & prompt_ids) {
    if (disabled_) return {-1, 0};

    auto candidates = find_all_boundaries(prompt_ids, markers_);
    if (candidates.empty()) return {-1, 0};

    // Best cache point: second-to-last boundary (last completed assistant turn).
    int target_cut = candidates.size() >= 2
        ? candidates[candidates.size() - 2]
        : candidates.back();

    auto key = hash_prefix(prompt_ids.data(), target_cut);
    if (find_entry(key) >= 0) return {-1, 0};  // already cached

    int slot;
    if ((int)entries_.size() >= cap_) {
        // At capacity — reserve the LRU slot without evicting yet.
        pending_evict_key_ = entries_.front().hash;
        has_pending_evict_ = true;
        slot = entries_.front().slot;
    } else {
        slot = next_slot_;
        next_slot_ = (next_slot_ + 1) % cap_;
        has_pending_evict_ = false;
    }

    return {slot, target_cut};
}

void PrefixCache::confirm_inline_snap(int slot, int target_cut,
                                      const std::vector<int32_t> & prompt_ids) {
    if (disabled_) return;

    // Evict the reserved entry (if any).
    if (has_pending_evict_) {
        int idx = find_entry(pending_evict_key_);
        if (idx >= 0) entries_.erase(entries_.begin() + idx);
        has_pending_evict_ = false;
    }

    auto key = hash_prefix(prompt_ids.data(), target_cut);
    entries_.push_back({key, slot});
    std::fprintf(stderr, "[pc] inline-snap committed slot=%d prefix_len=%d\n",
                 slot, target_cut);
}

void PrefixCache::abort_inline_snap(int /*slot*/) {
    if (disabled_) return;
    if (has_pending_evict_) {
        int idx = find_entry(pending_evict_key_);
        if (idx >= 0) entries_.erase(entries_.begin() + idx);
        has_pending_evict_ = false;
    }
}

void PrefixCache::mark_all_cleared() {
    if (disabled_) return;
    int n = (int)entries_.size();
    entries_.clear();
    next_slot_ = 0;
    has_pending_evict_ = false;
    std::fprintf(stderr, "[pc] all-cleared — dropped %d LRU entries\n", n);
}

// ── Full-compress cache ─────────────────────────────────────────────────

void PrefixCache::init_full_cache(int full_cap) {
    if (disabled_ || full_cap <= 0) {
        full_disabled_ = true;
        full_cap_ = 0;
        return;
    }
    int remaining = MAX_SLOTS - cap_;
    if (full_cap > remaining) full_cap = remaining;
    if (full_cap <= 0) {
        full_disabled_ = true;
        return;
    }
    full_cap_ = full_cap;
    full_slot_base_ = cap_;
    full_next_slot_ = 0;
    full_disabled_ = false;
    std::fprintf(stderr, "[pc] full-cache enabled: cap=%d slots=[%d,%d)\n",
                 full_cap_, full_slot_base_, full_slot_base_ + full_cap_);
}

std::pair<int, int> PrefixCache::lookup_full(const std::vector<int32_t> & prompt_ids) {
    if (full_disabled_) return {-1, 0};

    auto key = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());
    int idx = find_full_entry(key);
    if (idx < 0) return {-1, 0};

    auto & e = full_entries_[idx].entry;
    e.hits++;
    e.last_used_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    int slot = e.slot;
    int cur_ids_len = e.cur_ids_len;
    move_full_to_end(idx);
    full_lifetime_hits_++;

    std::fprintf(stderr, "[pc] full-cache hit slot=%d cur_ids_len=%d\n",
                 slot, cur_ids_len);
    return {slot, cur_ids_len};
}

int PrefixCache::prepare_full_snap(const std::vector<int32_t> & prompt_ids) {
    if (full_disabled_) return -1;

    auto key = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());
    if (find_full_entry(key) >= 0) return -1;  // already cached

    int abs_slot;
    if ((int)full_entries_.size() >= full_cap_) {
        // Evict LRU
        full_pending_evict_key_ = full_entries_.front().hash;
        full_has_pending_evict_ = true;
        abs_slot = full_entries_.front().entry.slot;
    } else {
        abs_slot = full_slot_base_ + full_next_slot_;
        full_next_slot_ = (full_next_slot_ + 1) % full_cap_;
        full_has_pending_evict_ = false;
    }

    return abs_slot;
}

void PrefixCache::confirm_full_snap(int slot,
                                    const std::vector<int32_t> & prompt_ids,
                                    int cur_ids_len) {
    if (full_disabled_) return;

    if (full_has_pending_evict_) {
        int idx = find_full_entry(full_pending_evict_key_);
        if (idx >= 0) full_entries_.erase(full_entries_.begin() + idx);
        full_has_pending_evict_ = false;
    }

    auto key = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());
    FullCacheEntry entry;
    entry.slot = slot;
    entry.cur_ids_len = cur_ids_len;
    entry.raw_prompt_len = (int)prompt_ids.size();
    entry.last_used_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    entry.hits = 0;
    full_entries_.push_back({key, std::move(entry)});

    std::fprintf(stderr, "[pc] full-cache committed slot=%d cur_ids_len=%d\n",
                 slot, cur_ids_len);
}

void PrefixCache::abort_full_snap(int /*slot*/) {
    if (full_disabled_) return;
    full_has_pending_evict_ = false;
}

PrefixCache::InlineStats PrefixCache::stats() const {
    if (disabled_) return {0, 0, 0};
    return {cap_, (int)entries_.size(), lifetime_hits_};
}

PrefixCache::FullStats PrefixCache::full_stats() const {
    if (full_disabled_) return {false, 0, 0, 0, 0};
    return {true, full_cap_, (int)full_entries_.size(),
            full_disk_bytes_, full_lifetime_hits_};
}

}  // namespace dflash::common
