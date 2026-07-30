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

int select_inline_snapshot_boundary(const std::vector<int> & boundaries,
                                    int restored_prefix_len) {
    if (boundaries.empty()) return 0;
    const int target = boundaries.size() >= 2
        ? boundaries[boundaries.size() - 2]
        : boundaries.back();
    return target > restored_prefix_len ? target : 0;
}

// ─── PrefixCache ────────────────────────────────────────────────────────

PrefixCache::PrefixCache(int cap, const Tokenizer & tokenizer)
    : cap_(std::min(cap, MAX_SLOTS)),
      inline_state_(std::max(0, std::min(cap, MAX_SLOTS)))
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

void PrefixCache::sync_inline_size() {
    entries_size_count_.store(
        inline_state_.size(), std::memory_order_relaxed);
}

void PrefixCache::sync_full_size() {
    full_entries_size_count_.store(
        full_state_.size(), std::memory_order_relaxed);
}

// ── Inline prefix cache ─────────────────────────────────────────────────

std::pair<int, int> PrefixCache::lookup(const std::vector<int32_t> & prompt_ids) {
    if (disabled_) return {-1, 0};

    auto boundaries = find_all_boundaries(prompt_ids, markers_);
    int best_slot = -1, best_len = 0;

    for (int cut : boundaries) {
        auto key = hash_prefix(prompt_ids.data(), cut);
        const auto result = inline_state_.lookup_candidate(key, cut);
        if (result.stale_removed) {
            // Slot was refreshed in-place at a deeper boundary; a shallow
            // hash→slot entry would restore the wrong cur_pos.
            std::fprintf(stderr,
                "[pc] lookup stale slot=%d key_cut=%d committed=%d — evicting\n",
                result.stale_slot, cut, result.stale_committed_len);
            continue;
        }
        if (result.slot >= 0 && cut > best_len) {
            best_slot = result.slot;
            best_len = result.prefix_len;
        }
    }
    sync_inline_size();

    if (best_slot >= 0) {
        lifetime_hits_.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr, "[pc] lookup hit slot=%d prefix_len=%d (of %zu total)\n",
                     best_slot, best_len, prompt_ids.size());
    }
    return {best_slot, best_len};
}

std::pair<int, int> PrefixCache::prepare_inline_snap(
        const std::vector<int32_t> & prompt_ids,
        int restored_prefix_len) {
    if (disabled_) return {-1, 0};

    auto candidates = find_all_boundaries(prompt_ids, markers_);
    const int target_cut =
        select_inline_snapshot_boundary(candidates, restored_prefix_len);
    if (target_cut <= 0) return {-1, 0};

    auto key = hash_prefix(prompt_ids.data(), target_cut);
    const auto reservation = inline_state_.prepare(key, target_cut);
    if (reservation.slot < 0) return {-1, 0};
    if (reservation.victim_index > 0) {
        std::fprintf(stderr,
            "[pc] prefix-aware evict: victim idx=%d (len=%d) kept oldest "
            "ancestor (len=%d)\n",
            reservation.victim_index, reservation.victim_len,
            reservation.oldest_len);
    }

    return {reservation.slot, reservation.target_cut};
}

void PrefixCache::confirm_inline_snap(int slot, int target_cut,
                                      const std::vector<int32_t> & prompt_ids) {
    if (disabled_) return;
    if (slot < 0 || slot >= cap_ || target_cut <= 0 ||
        target_cut > (int)prompt_ids.size()) {
        std::fprintf(stderr,
            "[pc] rejected inline-snap slot=%d prefix_len=%d prompt_len=%zu\n",
            slot, target_cut, prompt_ids.size());
        return;
    }

    const auto key = hash_prefix(prompt_ids.data(), target_cut);
    const auto result =
        inline_state_.confirm(slot, key, target_cut, prompt_ids);
    if (!result.accepted) {
        std::fprintf(stderr,
            "[pc] rejected inline-snap slot=%d prefix_len=%d prompt_len=%zu\n",
            slot, target_cut, prompt_ids.size());
        return;
    }
    for (int i = 0; i < result.stale_slot_entries_removed; ++i) {
        std::fprintf(stderr,
            "[pc] dropping stale entry for reused slot=%d\n", slot);
    }
    sync_inline_size();
    std::fprintf(stderr, "[pc] inline-snap committed slot=%d prefix_len=%d\n",
                 slot, target_cut);
}

void PrefixCache::abort_inline_snap(int slot) {
    if (disabled_) return;
    // The HTTP layer clears the reserved backend slot before generation. Any
    // metadata still pointing at it is therefore invalid, whether the slot was
    // selected through the explicit eviction path or through a round-robin
    // hole left by an earlier aborted reservation.
    inline_state_.abort(slot);
    sync_inline_size();
}

void PrefixCache::cancel_inline_snap(int slot) {
    if (disabled_) return;
    inline_state_.cancel(slot);
}

void PrefixCache::mark_all_cleared() {
    if (disabled_) return;
    const int n = inline_state_.size();
    inline_state_.clear();
    sync_inline_size();
    std::fprintf(stderr, "[pc] all-cleared — dropped %d LRU entries\n", n);
}

// ── Full-compress cache ─────────────────────────────────────────────────

void PrefixCache::init_full_cache(int full_cap) {
    constexpr int DISK_STAGING_SLOT = MAX_SLOTS - 1;
    if (full_cap <= 0) {
        full_disabled_ = true;
        full_cap_ = 0;
        full_state_.configure(0, 0, DISK_STAGING_SLOT);
        sync_full_size();
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
        full_cap_ = 0;
        full_state_.configure(0, 0, DISK_STAGING_SLOT);
        sync_full_size();
        return;
    }
    full_cap_ = full_cap;
    full_disabled_ =
        !full_state_.configure(cap_, full_cap_, DISK_STAGING_SLOT);
    sync_full_size();
    if (full_disabled_) {
        full_cap_ = 0;
        std::fprintf(stderr,
            "[pc] full-cache disabled: invalid slot range base=%d cap=%d "
            "staging=%d\n",
            cap_, full_cap, DISK_STAGING_SLOT);
        return;
    }
    std::fprintf(stderr, "[pc] full-cache enabled: cap=%d slots=[%d,%d)\n",
                 full_cap_, full_state_.slot_base(),
                 full_state_.slot_base() + full_cap_);
}

std::pair<int, int> PrefixCache::lookup_full(const std::vector<int32_t> & prompt_ids) {
    if (full_disabled_) return {-1, 0};

    auto key = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());
    const int64_t now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    const auto result = full_state_.lookup(key, now_ns);
    if (result.slot < 0) return {-1, 0};

    full_lifetime_hits_.fetch_add(1, std::memory_order_relaxed);

    std::fprintf(stderr, "[pc] full-cache hit slot=%d cur_ids_len=%d\n",
                 result.slot, result.cur_ids_len);
    return {result.slot, result.cur_ids_len};
}

int PrefixCache::prepare_full_snap(const std::vector<int32_t> & prompt_ids,
                                   int expected_snapshot_len) {
    if (full_disabled_) return -1;

    auto key = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());
    const auto reservation =
        full_state_.prepare(key, expected_snapshot_len);
    return reservation.accepted ? reservation.slot : -1;
}

bool PrefixCache::confirm_full_snap(int slot,
                                    const std::vector<int32_t> & prompt_ids,
                                    int saved_snapshot_len) {
    if (full_disabled_) return false;

    auto key = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());
    const int64_t now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    const auto result = full_state_.confirm(
        slot, key, (int)prompt_ids.size(), saved_snapshot_len, now_ns);
    if (!result.accepted) {
        std::fprintf(stderr,
            "[pc] rejected full-cache confirm slot=%d raw_len=%zu "
            "saved_pos=%d expected_pos=%d\n",
            slot, prompt_ids.size(), saved_snapshot_len,
            full_state_.pending_expected_snapshot_len());
        return false;
    }
    sync_full_size();

    std::fprintf(stderr, "[pc] full-cache committed slot=%d cur_ids_len=%d\n",
                 slot, saved_snapshot_len);
    return true;
}

void PrefixCache::abort_full_snap(int slot) {
    if (full_disabled_) return;

    if (full_state_.has_pending_reservation()) {
        const auto result = full_state_.abort(slot);
        if (!result.accepted) {
            std::fprintf(stderr,
                "[pc] rejected full-cache abort slot=%d pending_slot=%d\n",
                slot, full_state_.pending_slot());
            return;
        }
    } else {
        // This is invalidation of a committed backend snapshot, not an
        // in-flight reservation abort (e.g. snapshot loss after recovery).
        full_state_.invalidate_slot(slot);
    }
    sync_full_size();
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
