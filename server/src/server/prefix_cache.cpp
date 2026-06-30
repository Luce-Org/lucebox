// Prefix cache implementation.

#include "prefix_cache.h"
#include "common/sha1.h"

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

// ─── Prefix-aware eviction ──────────────────────────────────────────────

static bool is_strict_prefix(const std::vector<int32_t> & a,
                             const std::vector<int32_t> & b) {
    // True iff `a` is a strict (shorter) prefix of `b`.
    if (a.size() >= b.size()) return false;
    return std::equal(a.begin(), a.end(), b.begin());
}

int select_inline_evict_victim(const std::vector<const std::vector<int32_t> *> & ids_lru) {
    const int n = (int)ids_lru.size();
    if (n <= 0) return 0;
    // Oldest-first scan: evict the first entry that is not a strict prefix of any
    // other entry (a leaf). Shared ancestor prefixes are thereby kept resident.
    for (int i = 0; i < n; i++) {
        bool is_ancestor = false;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            if (is_strict_prefix(*ids_lru[i], *ids_lru[j])) { is_ancestor = true; break; }
        }
        if (!is_ancestor) return i;  // oldest leaf
    }
    return 0;  // unreachable (the longest entry is always a leaf); pure-LRU fallback
}

int select_inline_evict_victim(const std::vector<std::vector<int32_t>> & ids_lru) {
    std::vector<const std::vector<int32_t> *> ptrs;
    ptrs.reserve(ids_lru.size());
    for (const auto & v : ids_lru) ptrs.push_back(&v);
    return select_inline_evict_victim(ptrs);
}

// ─── PrefixCache ────────────────────────────────────────────────────────

PrefixCache::PrefixCache(int cap, const Tokenizer & tokenizer)
    : cap_(std::min(cap, MAX_SLOTS))
{
    if (cap_ <= 0) {
        disabled_ = true;
        cap_ = 0;
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
    auto e = std::move(entries_[idx]);
    entries_.erase(entries_.begin() + idx);
    entries_.push_back(std::move(e));
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
        lifetime_hits_.fetch_add(1, std::memory_order_relaxed);
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
        // At capacity — reserve a slot without evicting yet. Prefix-aware: prefer
        // the oldest leaf so shared ancestor prefixes (reused by later branches)
        // stay resident. entries_ is already in LRU order (front = oldest).
        std::vector<const std::vector<int32_t> *> ids_lru;
        ids_lru.reserve(entries_.size());
        for (const auto & e : entries_) ids_lru.push_back(&e.ids);
        int victim = select_inline_evict_victim(ids_lru);
        pending_evict_key_ = entries_[victim].hash;
        has_pending_evict_ = true;
        slot = entries_[victim].slot;
        if (victim != 0) {
            std::fprintf(stderr,
                "[pc] prefix-aware evict: victim idx=%d (len=%zu) kept oldest "
                "ancestor (len=%zu)\n",
                victim, entries_[victim].ids.size(), entries_.front().ids.size());
        }
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
        if (idx >= 0) {
            entries_.erase(entries_.begin() + idx);
            entries_size_count_.fetch_sub(1, std::memory_order_relaxed);
        }
        has_pending_evict_ = false;
    }

    // The new snapshot replaces whatever this slot previously held. Drop any
    // other entries still pointing at the slot: their hashes describe a
    // different (or shorter) token stream than the new snapshot, and a later
    // restore through them would attach mismatched KV. Stale entries arise
    // when an aborted snap burns a round-robin next_slot_ step and a later
    // confirm wraps onto a slot with a live entry (PR #370 repro).
    for (int i = (int)entries_.size() - 1; i >= 0; --i) {
        if (entries_[(size_t)i].slot == slot) {
            std::fprintf(stderr,
                "[pc] dropping stale entry for reused slot=%d\n", slot);
            entries_.erase(entries_.begin() + i);
            entries_size_count_.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    auto key = hash_prefix(prompt_ids.data(), target_cut);
    std::vector<int32_t> ids(prompt_ids.begin(), prompt_ids.begin() + target_cut);
    entries_.push_back({key, slot, std::move(ids)});
    entries_size_count_.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "[pc] inline-snap committed slot=%d prefix_len=%d\n",
                 slot, target_cut);
}

void PrefixCache::abort_inline_snap(int /*slot*/) {
    if (disabled_) return;
    if (has_pending_evict_) {
        int idx = find_entry(pending_evict_key_);
        if (idx >= 0) {
            entries_.erase(entries_.begin() + idx);
            entries_size_count_.fetch_sub(1, std::memory_order_relaxed);
        }
        has_pending_evict_ = false;
    }
}

void PrefixCache::mark_all_cleared() {
    if (disabled_) return;
    int n = (int)entries_.size();
    entries_.clear();
    entries_size_count_.store(0, std::memory_order_relaxed);
    next_slot_ = 0;
    has_pending_evict_ = false;
    std::fprintf(stderr, "[pc] all-cleared — dropped %d LRU entries\n", n);
}

// ── Full-compress cache ─────────────────────────────────────────────────

void PrefixCache::init_full_cache(int full_cap) {
    if (full_cap <= 0) {
        full_disabled_ = true;
        full_cap_ = 0;
        return;
    }
    // Reserve the last slot (MAX_SLOTS-1) for the disk-prefix-cache staging
    // slot (http_server DISK_STAGING_SLOT = kMaxSlots-1). Without this the full
    // cache can claim slot 63 and disk-cache traffic silently clobbers a
    // committed full-cache snapshot -> empty/corrupt responses on a later hit.
    int remaining = MAX_SLOTS - cap_ - 1;
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
    full_lifetime_hits_.fetch_add(1, std::memory_order_relaxed);

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
        if (idx >= 0) {
            full_entries_.erase(full_entries_.begin() + idx);
            full_entries_size_count_.fetch_sub(1, std::memory_order_relaxed);
        }
        full_has_pending_evict_ = false;
    }

    for (int i = (int)full_entries_.size() - 1; i >= 0; --i) {
        if (full_entries_[(size_t)i].entry.slot == slot) {
            std::fprintf(stderr,
                "[pc] dropping stale full-cache entry for reused slot=%d\n", slot);
            full_entries_.erase(full_entries_.begin() + i);
            full_entries_size_count_.fetch_sub(1, std::memory_order_relaxed);
        }
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
    full_entries_size_count_.fetch_add(1, std::memory_order_relaxed);

    std::fprintf(stderr, "[pc] full-cache committed slot=%d cur_ids_len=%d\n",
                 slot, cur_ids_len);
}

void PrefixCache::abort_full_snap(int /*slot*/) {
    if (full_disabled_) return;
    full_has_pending_evict_ = false;
}

PrefixCache::InlineStats PrefixCache::stats() const {
    if (disabled_) return {0, 0, 0};
    return {cap_,
            (int)entries_size_count_.load(std::memory_order_relaxed),
            lifetime_hits_.load(std::memory_order_relaxed)};
}

PrefixCache::FullStats PrefixCache::full_stats() const {
    if (full_disabled_) return {false, 0, 0, 0, 0};
    return {true, full_cap_,
            (int)full_entries_size_count_.load(std::memory_order_relaxed),
            full_disk_bytes_.load(std::memory_order_relaxed),
            full_lifetime_hits_.load(std::memory_order_relaxed)};
}

}  // namespace dflash::common
