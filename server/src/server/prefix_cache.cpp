// Prefix cache implementation.

#include "prefix_cache.h"
#include "common/sha1.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <chrono>

namespace luce::common {

// ─── Chat marker resolution ────────────────────────────────────────────

bool resolve_chat_markers(const Tokenizer & tok, ChatMarkers & out) {
    out.role_starts_delimit = false;
    // DeepSeek V4 uses full-width punctuation in its control tokens. Require
    // each marker to encode as its exact vocabulary token so an unrelated BPE
    // tokenizer cannot be misclassified merely because it can spell the text.
    const auto exact_control_token = [&tok](const char * marker) -> int32_t {
        const int32_t id = tok.token_to_id(marker);
        if (id < 0) return -1;
        const auto encoded = tok.encode(marker);
        return encoded.size() == 1 && encoded[0] == id ? id : -1;
    };
    const int32_t ds_bos = exact_control_token("<｜begin▁of▁sentence｜>");
    const int32_t ds_eos = exact_control_token("<｜end▁of▁sentence｜>");
    const int32_t ds_user = exact_control_token("<｜User｜>");
    const int32_t ds_assistant = exact_control_token("<｜Assistant｜>");
    if (ds_bos >= 0 && ds_eos >= 0 && ds_user >= 0 && ds_assistant >= 0) {
        out.family = "deepseek";
        out.sys_role_prefix = {ds_bos};
        out.end_msg_seqs = {{ds_eos}};
        out.next_role_starts = {{ds_user}, {ds_assistant}};
        out.role_starts_delimit = true;
        return true;
    }

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
    // Boundaries normally start after the system role prefix. A prompt with
    // no system message (common for API clients) still has role boundaries,
    // so start from the first role marker instead of caching nothing. Take
    // whichever comes first: a system prefix quoted inside a later message
    // (literal ChatML in content) must not displace the real leading role.
    const auto first_start = find_first_seq_any(ids, markers.next_role_starts);
    int start_idx = find_first_seq(ids, markers.sys_role_prefix);
    int start_len = (int)markers.sys_role_prefix.size();
    if (first_start.first >= 0 &&
        (start_idx < 0 || first_start.first < start_idx)) {
        start_idx = first_start.first;
        start_len = first_start.second;
    }
    if (start_idx < 0 || start_len <= 0) return out;

    int cursor = start_idx + start_len;
    if (markers.role_starts_delimit) {
        // Each role marker opens a message and closes the previous one, so
        // the cut after every marker is a reusable prefix: the system text
        // before the first user marker, each completed turn, and the
        // generation prompt itself. A marker quoted inside message content
        // adds a candidate cut too; cuts are reuse hints and exact token
        // matching decides every restore, so that costs at most a snapshot
        // taken at a less useful place.
        while (true) {
            const auto [idx, len] =
                find_first_seq_any(ids, markers.next_role_starts, cursor);
            if (idx < 0) break;
            cursor = idx + len;
            out.push_back(cursor);
        }
        return out;
    }
    int stray_skips = 0;
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
        if (next_match < 0) {
            // Stray end-of-message marker with no following role start — chatml
            // tokens embedded in message content (file dumps, terminal output,
            // model-echoed markers). Skip this marker and keep scanning instead
            // of truncating the whole boundary list. A lone stray previously cut
            // the walk off here, hiding every real boundary after it; that pinned
            // the inline-snapshot deepen target (second-to-last boundary) at the
            // already-restored prefix length, so no snapshot was ever deepened and
            // every turn re-prefilled the entire tail. Guard against pathological
            // input (or a marker-family mismatch) by capping consecutive strays.
            if (++stray_skips > 8192) break;
            cursor = after_end;
            continue;
        }
        int boundary = next_match + next_len;
        out.push_back(boundary);
        cursor = boundary;
        stray_skips = 0;
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

// True iff `a` is a strict (shorter) prefix of b[0, b_len).
static bool is_strict_prefix(const std::vector<int32_t> & a,
                             const int32_t * b, size_t b_len) {
    return a.size() < b_len && std::equal(a.begin(), a.end(), b);
}

static bool is_strict_prefix(const std::vector<int32_t> & a,
                             const std::vector<int32_t> & b) {
    return is_strict_prefix(a, b.data(), b.size());
}

int select_inline_evict_victim(const std::vector<const std::vector<int32_t> *> & ids_lru,
                               const std::vector<bool> * protected_lru,
                               int skip_index) {
    const int n = (int)ids_lru.size();
    if (n <= 0) return 0;
    auto is_protected = [&](int i) {
        return protected_lru && i >= 0 && i < (int)protected_lru->size() &&
               (*protected_lru)[(size_t)i];
    };
    auto is_ancestor = [&](int i) {
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            if (is_strict_prefix(*ids_lru[i], *ids_lru[j])) return true;
        }
        return false;
    };
    auto has_ancestor = [&](int i) {
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            if (is_strict_prefix(*ids_lru[j], *ids_lru[i])) return true;
        }
        return false;
    };
    // A turn kept between a conversation's head and its newest snapshot only
    // serves an edit of the last message: it goes first.
    for (int i = 0; i < n; i++) {
        if (i == skip_index || is_protected(i)) continue;
        if (is_ancestor(i) && has_ancestor(i)) return i;
    }
    // Oldest-first scan: prefer an unprotected leaf so sticky tools pins
    // survive. skip_index (the in-flight restore source) is never a victim.
    int oldest_protected_leaf = -1;
    for (int i = 0; i < n; i++) {
        if (i == skip_index) continue;
        if (is_ancestor(i)) continue;
        if (!is_protected(i)) return i;  // oldest unprotected leaf
        if (oldest_protected_leaf < 0) oldest_protected_leaf = i;
    }
    if (skip_index >= 0) {
        // No unprotected leaf outside the restore source — e.g. a linearly
        // growing conversation whose only leaf is the restore source itself.
        // Evict the shallowest non-protected ancestor (its KV is subsumed by
        // every deeper entry) so the new snapshot lands in a different slot
        // and the restore point can slide forward. Never the protected tools
        // pin, never the restore source.
        int shallowest_ancestor = -1;
        for (int i = 0; i < n; i++) {
            if (i == skip_index || is_protected(i)) continue;
            if (!is_ancestor(i)) continue;
            if (shallowest_ancestor < 0 ||
                ids_lru[i]->size() < ids_lru[(size_t)shallowest_ancestor]->size()) {
                shallowest_ancestor = i;
            }
        }
        if (shallowest_ancestor >= 0) return shallowest_ancestor;
    }
    // Every other leaf is a pin. Give up the least recently used one: its
    // head costs one re-prefill, while refusing leaves the restored
    // conversation unable to save any later turn. A leaf is never an
    // ancestor of the restore source.
    if (oldest_protected_leaf >= 0) return oldest_protected_leaf;
    // Only the restore source and its ancestors remain. Without a restore
    // source this is unreachable: the longest entry is always a leaf.
    return skip_index >= 0 ? -1 : 0;
}

int select_inline_evict_victim(const std::vector<std::vector<int32_t>> & ids_lru,
                               const std::vector<bool> * protected_lru,
                               int skip_index) {
    std::vector<const std::vector<int32_t> *> ptrs;
    ptrs.reserve(ids_lru.size());
    for (const auto & v : ids_lru) ptrs.push_back(&v);
    return select_inline_evict_victim(ptrs, protected_lru, skip_index);
}

int select_inline_snapshot_boundary(const std::vector<int> & boundaries,
                                    int restored_prefix_len,
                                    bool prefer_head,
                                    int reachable_from) {
    if (boundaries.empty()) return 0;
    const auto usable = [&](int cut) {
        return cut > restored_prefix_len && cut >= reachable_from;
    };
    // Save the system/tools head (first marker) before deepening into
    // conversation turns. A head the backend cannot save is skipped: it is
    // cheap to prefill, and waiting for it would keep the cache from ever
    // saving a deeper cut.
    if (prefer_head) {
        const int head = boundaries.front();
        if (usable(head)) return head;
    }
    return usable(boundaries.back()) ? boundaries.back() : 0;
}

std::vector<int> spaced_restore_points(const std::vector<int> & boundaries, int min_gap) {
    if (min_gap <= 0) return boundaries;
    std::vector<int> kept;
    for (int b : boundaries) {
        if (kept.empty() || b - kept.back() >= min_gap) kept.push_back(b);
    }
    return kept;
}

std::vector<int> prefix_restore_points(const std::vector<int32_t> & prompt,
                                       const ChatMarkers & markers,
                                       std::initializer_list<int> cuts,
                                       bool drop_last_boundary,
                                       int spacing) {
    std::vector<int> points = find_all_boundaries(prompt, markers);
    if (drop_last_boundary && !points.empty()) points.pop_back();
    points = spaced_restore_points(points, spacing);
    for (int cut : cuts) {
        if (cut > 0 && cut < (int) prompt.size()) points.push_back(cut);
    }
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    return points;
}

bool should_force_inline_snapshot_boundary(
        const std::vector<int> & boundaries,
        int prompt_len,
        int restored_prefix_len,
        bool prefer_tools_boundary,
        int forced_cut,
        int reachable_from) {
    const bool tools_pin_restored =
        prefer_tools_boundary && !boundaries.empty() &&
        restored_prefix_len >= boundaries.front();
    return !tools_pin_restored &&
           forced_cut > restored_prefix_len &&
           forced_cut >= reachable_from &&
           forced_cut <= prompt_len;
}

// ─── PrefixCache ────────────────────────────────────────────────────────

PrefixCache::PrefixCache(int cap, const Tokenizer & tokenizer,
                         size_t max_resident_bytes)
    : cap_(std::min(cap, MAX_CACHE_SLOTS))
    , max_resident_bytes_(max_resident_bytes)
{
    max_resident_bytes_published_.store(max_resident_bytes_, std::memory_order_relaxed);
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
    if (max_resident_bytes_ > 0) {
        std::fprintf(stderr, "[pc] enabled: cap=%d family=%s resident_budget=%zu MiB\n",
                     cap_, markers_.family.c_str(), max_resident_bytes_ / (1024 * 1024));
    } else {
        std::fprintf(stderr, "[pc] enabled: cap=%d family=%s\n",
                     cap_, markers_.family.c_str());
    }
}

// ── LRU helpers ─────────────────────────────────────────────────────────

int PrefixCache::find_entry(const PrefixHash & h) const {
    for (int i = 0; i < (int)entries_.size(); i++) {
        if (entries_[i].hash == h) return i;
    }
    return -1;
}

int PrefixCache::find_slot_entry(int slot) const {
    for (int i = 0; i < (int)entries_.size(); ++i) {
        if (entries_[(size_t)i].slot == slot) return i;
    }
    return -1;
}

void PrefixCache::erase_inline_entry(int idx) {
    if (idx < 0 || idx >= (int)entries_.size()) return;
    const size_t bytes = entries_[(size_t)idx].resident_bytes;
    resident_bytes_ = bytes <= resident_bytes_ ? resident_bytes_ - bytes : 0;
    resident_bytes_count_.store(
        (uint64_t)resident_bytes_, std::memory_order_relaxed);
    entries_.erase(entries_.begin() + idx);
    entries_size_count_.fetch_sub(1, std::memory_order_relaxed);
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

std::pair<int, int> PrefixCache::lookup(
        const std::vector<int32_t> & prompt_ids) {
    return lookup_impl(
        prompt_ids, (int)prompt_ids.size(), /*record_hit=*/true);
}

std::pair<int, int> PrefixCache::lookup_candidate(
        const std::vector<int32_t> & prompt_ids,
        int max_prefix_tokens) {
    return lookup_impl(prompt_ids, max_prefix_tokens, /*record_hit=*/false);
}

std::pair<int, int> PrefixCache::lookup_impl(
        const std::vector<int32_t> & prompt_ids,
        int max_prefix_tokens,
        bool record_hit) {
    if (disabled_ || max_prefix_tokens <= 0) return {-1, 0};

    auto boundaries = find_all_boundaries(prompt_ids, markers_);
    int best_slot = -1, best_len = 0;
    int best_idx = -1;

    for (int cut : boundaries) {
        if (cut > max_prefix_tokens) continue;
        auto key = hash_prefix(prompt_ids.data(), cut);
        int idx = find_entry(key);
        if (idx >= 0) {
            const int committed = (int)entries_[idx].ids.size();
            if (committed != cut) {
                // Slot was refreshed in-place at a deeper boundary; a shallow
                // hash→slot entry would restore the wrong cur_pos.
                std::fprintf(stderr,
                    "[pc] lookup stale slot=%d key_cut=%d committed=%d — evicting\n",
                    entries_[idx].slot, cut, committed);
                erase_inline_entry(idx);
                continue;
            }
            if (cut > best_len) {
                best_slot = entries_[idx].slot;
                best_len = cut;
                best_idx = idx;
            }
        }
    }

    // Match committed entry prefixes directly. Required for PPP mid-message
    // pin_end cuts that are not chat-template boundaries.
    for (int i = 0; i < (int)entries_.size(); ++i) {
        const auto & e = entries_[(size_t)i];
        const int len = (int)e.ids.size();
        if (len <= best_len || len > (int)prompt_ids.size() ||
            len > max_prefix_tokens) continue;
        if (!std::equal(e.ids.begin(), e.ids.end(), prompt_ids.begin())) {
            continue;
        }
        best_slot = e.slot;
        best_len = len;
        best_idx = i;
    }

    if (best_idx >= 0 && record_hit)
        record_inline_hit(best_slot, best_len, prompt_ids.size());
    return {best_slot, best_len};
}

void PrefixCache::record_inline_hit(
        int slot, int prefix_len, size_t prompt_len) {
    for (int i = 0; i < (int)entries_.size(); ++i) {
        if (entries_[(size_t)i].slot != slot ||
            (int)entries_[(size_t)i].ids.size() != prefix_len) continue;
        move_to_end(i);
        lifetime_hits_.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr,
            "[pc] lookup hit slot=%d prefix_len=%d (of %zu total)\n",
            slot, prefix_len, prompt_len);
        return;
    }
}

PrefixCache::InlineReservation::InlineReservation(
        PrefixCache * cache, uint64_t id, int slot, int target_cut,
        PrefixHash victim, bool has_victim, bool protect)
    : cache_(cache), id_(id), slot_(slot), target_cut_(target_cut),
      victim_(victim), has_victim_(has_victim), protect_(protect) {}

PrefixCache::InlineReservation::~InlineReservation() { cancel(); }

PrefixCache::InlineReservation::InlineReservation(
        InlineReservation && other) noexcept {
    take(std::move(other));
}

PrefixCache::InlineReservation &
PrefixCache::InlineReservation::operator=(
        InlineReservation && other) noexcept {
    if (this != &other) {
        cancel();
        take(std::move(other));
    }
    return *this;
}

bool PrefixCache::InlineReservation::active() const {
    return cache_ && cache_->inline_reservation_active(id_);
}

std::vector<int> PrefixCache::InlineReservation::extra_victim_slots() const {
    std::vector<int> out;
    const ActiveReservation * active =
        cache_ ? cache_->find_active_reservation(id_) : nullptr;
    if (!active) return out;
    for (size_t i = 0; i < active->extra_victims.size(); ++i) {
        const int idx = cache_->find_entry(active->extra_victims[i]);
        if (idx >= 0 && cache_->entries_[(size_t)idx].slot == active->extra_slots[i]) {
            out.push_back(active->extra_slots[i]);
        }
    }
    return out;
}

bool PrefixCache::InlineReservation::commit(
        const std::vector<int32_t> & prompt_ids,
        size_t resident_bytes, bool protect) {
    if (!active()) {
        clear();
        return false;
    }
    return cache_->commit_inline_reservation(
        *this, prompt_ids, target_cut_, resident_bytes, protect);
}

bool PrefixCache::InlineReservation::commit_at(
        const std::vector<int32_t> & prompt_ids, int committed_cut,
        size_t resident_bytes, bool protect) {
    if (!active()) {
        clear();
        return false;
    }
    return cache_->commit_inline_reservation(
        *this, prompt_ids, committed_cut, resident_bytes, protect);
}

void PrefixCache::InlineReservation::cancel() {
    if (cache_) cache_->release_inline_reservation(id_);
    clear();
}

void PrefixCache::InlineReservation::abort() {
    if (active()) {
        cache_->abort_inline_reservation(*this);
    } else {
        clear();
    }
}

void PrefixCache::InlineReservation::clear() {
    cache_ = nullptr;
    id_ = 0;
    slot_ = -1;
    target_cut_ = 0;
    victim_ = {};
    has_victim_ = false;
    protect_ = false;
}

void PrefixCache::InlineReservation::take(InlineReservation && other) {
    cache_ = other.cache_;
    id_ = other.id_;
    slot_ = other.slot_;
    target_cut_ = other.target_cut_;
    victim_ = other.victim_;
    has_victim_ = other.has_victim_;
    protect_ = other.protect_;
    other.clear();
}

bool PrefixCache::inline_reservation_active(uint64_t id) const {
    return find_active_reservation(id) != nullptr;
}

const PrefixCache::ActiveReservation *
PrefixCache::find_active_reservation(uint64_t id) const {
    if (id == 0) return nullptr;
    for (const auto & active : active_reservations_) {
        if (active.id == id) return &active;
    }
    return nullptr;
}

void PrefixCache::release_inline_reservation(uint64_t id) {
    for (size_t i = 0; i < active_reservations_.size(); ++i) {
        if (active_reservations_[i].id == id) {
            active_reservations_.erase(active_reservations_.begin() + (std::ptrdiff_t)i);
            return;
        }
    }
}

bool PrefixCache::slot_reserved(int slot) const {
    for (const auto & active : active_reservations_) {
        if (active.slot == slot) return true;
        for (const int extra : active.extra_slots) {
            if (extra == slot) return true;
        }
    }
    return false;
}

bool PrefixCache::slot_credited(int slot) const {
    for (const auto & active : active_reservations_) {
        for (const int credited : active.credited_slots) {
            if (credited == slot) return true;
        }
    }
    return false;
}

namespace {
// The whole value as a non-negative int, else `fallback` (also when unset).
int env_nonneg_int(const char * name, int fallback) {
    const char * v = std::getenv(name);
    if (!v || !*v) return fallback;
    char * end = nullptr;
    errno = 0;
    const long n = std::strtol(v, &end, 10);
    if (errno != 0 || *end != '\0' || n < 0 || n > INT_MAX) {
        std::fprintf(stderr, "[pc] ignoring %s=%s (want a non-negative integer); using %d\n",
                     name, v, fallback);
        return fallback;
    }
    return (int) n;
}
}  // namespace

PrefixCache::InlineReservation PrefixCache::reserve_inline_snap(
        const std::vector<int32_t> & prompt_ids,
        int restored_prefix_len,
        bool prefer_tools_boundary,
        int forced_cut,
        int restore_source_slot,
        InlineSnapshotSize estimate_bytes,
        int reachable_from,
        int restore_point_spacing) {
    if (disabled_) return {};

    const auto candidates = find_all_boundaries(prompt_ids, markers_);
    // A conversation's first turn (one message past a head it did not
    // restore) saves its system head, which every new conversation on that
    // head restores; each later turn saves its end. Tool requests keep their
    // system/tools head pinned until it is restored (prefer_tools_boundary).
    bool prefer_head = prefer_tools_boundary ||
        (candidates.size() == 2 && restored_prefix_len < candidates.front());
    // A long tail past a short head (a first turn, cold or with only the
    // shared head restored): snapshot at the last message boundary instead of
    // the head, or the first follow-up re-prefills the whole turn. Only a
    // short head gives up its own pin this way, so a new conversation that
    // shares it re-prefills at most LUCE_PC_DEEP_FIRST_MAX_HEAD tokens
    // (default 2048). LUCE_PC_DEEP_FIRST_MIN is the tail length that triggers
    // it (default 4096 tokens; 0 keeps the head pin). Values that are not a
    // whole non-negative integer keep the defaults.
    // A forced pin still ahead of the restored prefix (an explicit pin_end, or
    // PPP's pin of a head seen before) wins and keeps its protection, and a
    // prompt the resident budget could never hold keeps the head pin instead
    // of saving nothing.
    static const int deep_first_min = env_nonneg_int("LUCE_PC_DEEP_FIRST_MIN", 4096);
    static const int deep_first_max_head = env_nonneg_int("LUCE_PC_DEEP_FIRST_MAX_HEAD", 2048);
    const auto whole_prompt_fits = [&]() {
        if (max_resident_bytes_ == 0) return true;
        const size_t bytes = estimate_bytes ? estimate_bytes((int) prompt_ids.size()) : 0;
        return bytes != 0 && bytes <= max_resident_bytes_;
    };
    if (deep_first_min > 0 && prefer_head && !candidates.empty() &&
        restored_prefix_len <= candidates.front() &&
        forced_cut <= restored_prefix_len &&
        candidates.front() <= deep_first_max_head &&
        (int) prompt_ids.size() - candidates.front() >= deep_first_min &&
        whole_prompt_fits()) {
        prefer_head = false;
        prefer_tools_boundary = false;
    }
    int target_cut = 0;
    bool forced = false;
    if (should_force_inline_snapshot_boundary(
            candidates, (int)prompt_ids.size(), restored_prefix_len,
            prefer_tools_boundary, forced_cut, reachable_from)) {
        target_cut = forced_cut;
        forced = true;
    } else {
        target_cut = select_inline_snapshot_boundary(
            candidates, restored_prefix_len, prefer_head, reachable_from);
    }
    if (restore_point_spacing > 0 && target_cut > 0 &&
        target_cut != candidates.back()) {
        int point = 0;
        for (int p : spaced_restore_points(candidates, restore_point_spacing))
            if (p <= target_cut) point = p;
        if (point != target_cut) forced = false;
        target_cut = point > restored_prefix_len && point >= reachable_from ? point : 0;
    }
    if (target_cut <= 0) {
        // An expected no-op when the restored prefix already covers the next
        // boundary (single-turn prompts, or a cache-primed conversation): log
        // once only, so the diagnostic — a truncated boundary list from stray
        // chatml in content — stays visible without flooding long runs. The
        // HTTP layer may retry within a request, so this can otherwise fire
        // twice per turn.
        static std::atomic<bool> s_snap_blocked_logged{false};
        if (!s_snap_blocked_logged.exchange(true)) {
            std::fprintf(stderr,
                "[pc] inline snap blocked: boundaries=%zu restored=%d target<=0 "
                "(deepen target not past restored prefix; logged once)\n",
                candidates.size(), restored_prefix_len);
        }
        return {};
    }

    const auto key = hash_prefix(prompt_ids.data(), target_cut);
    if (find_entry(key) >= 0) return {};
    for (const auto & active : active_reservations_) {
        if (active.key == key) return {};
    }

    const bool protect = prefer_tools_boundary &&
        (forced || (!candidates.empty() && target_cut == candidates.front()));
    PrefixHash victim_key{};
    bool has_victim = false;
    int slot = -1;
    // Captures in flight hold their slots and budget until they resolve.
    int reserved_free_slots = 0;
    size_t pending_bytes = 0;
    for (const auto & active : active_reservations_) {
        if (!active.has_victim) ++reserved_free_slots;
        pending_bytes += active.bytes;
    }
    const auto log_no_slot = [&](const char * reason) {
        int pins = 0;
        for (const auto & entry : entries_) pins += entry.protect ? 1 : 0;
        std::fprintf(stderr,
            "[pc] inline snap skipped, %s: entries=%zu protected=%d "
            "in_flight=%zu cap=%d restore_slot=%d cut=%d\n",
            reason, entries_.size(), pins, active_reservations_.size(), cap_,
            restore_source_slot, target_cut);
    };
    if ((int)entries_.size() + reserved_free_slots >= cap_) {
        // At capacity — reserve a slot without evicting yet. Prefix-aware: prefer
        // the oldest leaf so shared ancestor prefixes (reused by later branches)
        // stay resident. Skip protected tools pins when an unprotected leaf
        // exists. The in-flight restore source is never a victim, so the new
        // snapshot lands in a different slot and the restore point can slide
        // forward past the deepest slot.
        const int victim = pick_evict_victim(
            restore_source_slot >= 0 ? find_slot_entry(restore_source_slot) : -1);
        if (victim < 0) {
            // Only the restore source and its pinned ancestors remain, or a
            // capture in flight holds the victim. Skip this snapshot; the
            // restore point stays put rather than being destroyed.
            log_no_slot("no evictable entry");
            return {};
        }
        victim_key = entries_[(size_t)victim].hash;
        has_victim = true;
        slot = entries_[(size_t)victim].slot;
        if (victim != 0 || entries_[(size_t)victim].protect) {
            std::fprintf(stderr,
                "[pc] prefix-aware evict: victim idx=%d protect=%d (len=%zu) "
                "kept oldest ancestor (len=%zu)\n",
                victim, (int)entries_[(size_t)victim].protect,
                entries_[(size_t)victim].ids.size(),
                entries_.front().ids.size());
        }
    } else {
        // Below capacity, take a slot no committed entry owns. Round-robin
        // alone wraps onto live snapshots once invalidation or pruning has
        // freed slots elsewhere, evicting by slot position instead of age.
        // Skip the in-flight restore source too, so the new snapshot lands
        // in a different slot (the http_server/agent-replay guards would
        // cancel an unlucky collision, leaving the restore point pinned).
        for (int step = 0; step < cap_; ++step) {
            const int candidate = (next_slot_ + step) % cap_;
            if (candidate == restore_source_slot && cap_ > 1) continue;
            if (find_slot_entry(candidate) >= 0 || slot_reserved(candidate)) continue;
            slot = candidate;
            break;
        }
        if (slot < 0) {
            log_no_slot("no free slot");
            return {};
        }
        next_slot_ = (slot + 1) % cap_;
    }

    size_t estimated_bytes = 0;
    std::vector<PrefixHash> extra_victims;
    std::vector<int> extra_slots;
    std::vector<int> credited_slots;
    if (max_resident_bytes_ > 0) {
        estimated_bytes = estimate_bytes ? estimate_bytes(target_cut) : 0;
        // When the caller prunes after every commit, the entries this
        // capture supersedes (including its restore source) are freed as it
        // lands, so they do not compete with it for the budget. A failed
        // capture prunes nothing and keeps the restore point. A capture that
        // commits shorter than it reserved can leave some credited entries
        // resident; enforce_resident_budget() evicts them after the commit.
        std::vector<bool> reclaimed(entries_.size(), false);
        size_t reclaimed_bytes = 0;
        if (prunes_superseded_) {
            for (const int i : superseded_entries(
                     prompt_ids.data(), (size_t)target_cut)) {
                if (slot_reserved(entries_[(size_t)i].slot)) continue;
                reclaimed[(size_t)i] = true;
                reclaimed_bytes += entries_[(size_t)i].resident_bytes;
            }
        }
        const auto fits = [&](int victim_idx) {
            if (estimated_bytes == 0 ||
                estimated_bytes > max_resident_bytes_) return false;
            size_t freed = reclaimed_bytes;
            if (victim_idx >= 0 && !reclaimed[(size_t)victim_idx]) {
                freed += entries_[(size_t)victim_idx].resident_bytes;
            }
            const size_t held = resident_bytes_ + pending_bytes;
            const size_t after_free = freed <= held ? held - freed : 0;
            return after_free <= max_resident_bytes_ &&
                estimated_bytes <= max_resident_bytes_ - after_free;
        };

        const int planned_victim = has_victim
            ? find_entry(victim_key) : find_slot_entry(slot);
        if (!fits(planned_victim)) {
            const auto is_leaf = [&](int candidate) {
                for (int i = 0; i < (int)entries_.size(); ++i) {
                    if (i != candidate &&
                        is_strict_prefix(entries_[(size_t)candidate].ids,
                                         entries_[(size_t)i].ids)) {
                        return false;
                    }
                }
                return true;
            };
            const auto is_middle = [&](int candidate) {
                if (is_leaf(candidate)) return false;
                for (int i = 0; i < (int)entries_.size(); ++i) {
                    if (i != candidate &&
                        is_strict_prefix(entries_[(size_t)i].ids,
                                         entries_[(size_t)candidate].ids)) {
                        return true;
                    }
                }
                return false;
            };
            // A turn kept behind a conversation's newest snapshot first (see
            // select_inline_evict_victim), then leaves, then the rest;
            // unprotected entries before protected ones.
            const auto pick_single = [&](int first_pass, int last_pass) {
                for (int pass = first_pass; pass <= last_pass; ++pass) {
                    const bool require_middle = pass == 0;
                    const bool require_leaf = pass == 1 || pass == 2;
                    const bool allow_protected = pass == 2 || pass == 4;
                    for (int i = 0; i < (int)entries_.size(); ++i) {
                        if (restore_source_slot >= 0 &&
                            entries_[(size_t)i].slot == restore_source_slot)
                            continue;
                        if (slot_reserved(entries_[(size_t)i].slot)) continue;
                        if (!fits(i)) continue;
                        if (require_middle && !is_middle(i)) continue;
                        if (require_leaf && !is_leaf(i)) continue;
                        if (!allow_protected && entries_[(size_t)i].protect)
                            continue;
                        return i;
                    }
                }
                return -1;
            };
            // No single unprotected turn or leaf makes room: plan several,
            // least recently used first, before falling back to a protected
            // pin or a head. Each one goes to disk as it leaves RAM, so a
            // long session that arrives after many short ones is still
            // captured. Upstream skipped this capture instead (one victim
            // per reservation) so a capture never disturbs the committed
            // cache until it lands; the plan keeps that: it is all or
            // nothing, decided before anything is evicted, and its entries
            // are evicted only when the capture commits. A capture that
            // fails or is cancelled evicts none of them, and one larger
            // than the whole budget plans none. What a plan never takes:
            // protected pins; heads (entries that other resident entries
            // extend, unless kept only as an edit backup); this capture's
            // own ancestors and restore source, which it builds on; and
            // entries a capture in flight writes, evicts or already counts
            // as freed, so two captures never count the same bytes.
            const auto plan_several = [&]() {
                std::vector<int> plan;
                if (estimated_bytes == 0 ||
                    estimated_bytes > max_resident_bytes_) return plan;
                size_t freed = reclaimed_bytes;
                if (planned_victim >= 0 && !reclaimed[(size_t)planned_victim]) {
                    freed += entries_[(size_t)planned_victim].resident_bytes;
                }
                const size_t held = resident_bytes_ + pending_bytes;
                const auto room = [&]() {
                    const size_t after_free = freed <= held ? held - freed : 0;
                    return after_free <= max_resident_bytes_ &&
                        estimated_bytes <= max_resident_bytes_ - after_free;
                };
                // Edit backups first (they are not written to disk while
                // their deeper snapshot is resident), then leaves.
                for (int pass = 0; pass < 2; ++pass) {
                    for (int i = 0; i < (int)entries_.size(); ++i) {
                        const auto & entry = entries_[(size_t)i];
                        if (i == planned_victim || reclaimed[(size_t)i] ||
                            entry.protect) continue;
                        if (restore_source_slot >= 0 &&
                            entry.slot == restore_source_slot) continue;
                        if (slot_reserved(entry.slot) ||
                            slot_credited(entry.slot)) continue;
                        if (is_strict_prefix(entry.ids, prompt_ids.data(),
                                             (size_t)target_cut)) continue;
                        const bool leaf = is_leaf(i);
                        if (pass == 0 ? !(entry.edit_backup && !leaf) : !leaf)
                            continue;
                        plan.push_back(i);
                        freed += entry.resident_bytes;
                        if (room()) return plan;
                    }
                }
                plan.clear();
                return plan;
            };
            int victim = pick_single(0, 1);
            std::vector<int> several;
            if (victim < 0) several = plan_several();
            if (victim < 0 && several.empty()) victim = pick_single(2, 4);
            if (!several.empty()) {
                size_t bytes = 0;
                for (const int i : several) {
                    extra_victims.push_back(entries_[(size_t)i].hash);
                    extra_slots.push_back(entries_[(size_t)i].slot);
                    bytes += entries_[(size_t)i].resident_bytes;
                }
                std::fprintf(stderr,
                    "[pc] resident budget plans %zu evictions (%zu bytes) for "
                    "capture slot=%d estimate=%zu resident=%zu budget=%zu\n",
                    several.size(), bytes, slot, estimated_bytes,
                    resident_bytes_, max_resident_bytes_);
            } else if (victim >= 0) {
                victim_key = entries_[(size_t)victim].hash;
                has_victim = true;
                const int redirected_slot = entries_[(size_t)victim].slot;
                std::fprintf(stderr,
                    "[pc] resident budget redirects capture slot=%d -> %d "
                    "estimate=%zu resident=%zu budget=%zu\n",
                    slot, redirected_slot, estimated_bytes, resident_bytes_,
                    max_resident_bytes_);
                slot = redirected_slot;
            } else {
                const uint64_t skipped =
                    budget_skips_.fetch_add(1, std::memory_order_relaxed) + 1;
                if (skipped == 1 || skipped % 64 == 0) {
                    std::fprintf(stderr,
                        "[pc] resident budget skips capture estimate=%zu "
                        "resident=%zu budget=%zu (skips=%llu)\n",
                        estimated_bytes, resident_bytes_, max_resident_bytes_,
                        (unsigned long long)skipped);
                }
                return {};
            }
        }
        for (size_t i = 0; i < reclaimed.size(); ++i) {
            if (reclaimed[i]) credited_slots.push_back(entries_[i].slot);
        }
    }

    uint64_t id = next_inline_reservation_++;
    if (id == 0) id = next_inline_reservation_++;
    if (has_victim) credited_slots.push_back(slot);
    credited_slots.insert(credited_slots.end(), extra_slots.begin(), extra_slots.end());
    active_reservations_.push_back(
        {id, slot, key, victim_key, has_victim, estimated_bytes,
         std::move(extra_victims), std::move(extra_slots),
         std::move(credited_slots)});
    return InlineReservation(
        this, id, slot, target_cut, victim_key, has_victim, protect);
}

void PrefixCache::replace_inline_entry(
        int slot, int target_cut,
        const std::vector<int32_t> & prompt_ids,
        bool protect, size_t resident_bytes) {
    for (int i = (int)entries_.size() - 1; i >= 0; --i) {
        if (entries_[(size_t)i].slot == slot) {
            std::fprintf(stderr,
                "[pc] dropping stale entry for reused slot=%d\n", slot);
            record_eviction(i);
            erase_inline_entry(i);
        }
    }

    const auto key = hash_prefix(prompt_ids.data(), target_cut);
    std::vector<int32_t> ids(
        prompt_ids.begin(), prompt_ids.begin() + target_cut);
    entries_.push_back(
        {key, slot, std::move(ids), protect, resident_bytes});
    entries_size_count_.fetch_add(1, std::memory_order_relaxed);
    resident_bytes_ += resident_bytes;
    resident_bytes_count_.store(
        (uint64_t)resident_bytes_, std::memory_order_relaxed);
    std::fprintf(stderr,
        "[pc] inline-snap committed slot=%d prefix_len=%d protect=%d "
        "bytes=%zu resident=%zu\n",
        slot, target_cut, (int)protect, resident_bytes, resident_bytes_);
}

bool PrefixCache::commit_inline_reservation(
        InlineReservation & reservation,
        const std::vector<int32_t> & prompt_ids,
        int committed_cut, size_t resident_bytes, bool protect) {
    if (!reservation.active() || committed_cut <= 0 ||
        committed_cut > reservation.target_cut_ ||
        committed_cut > (int)prompt_ids.size()) {
        reservation.abort();
        return false;
    }
    // The further entries its plan evicts, before the slot's own victim:
    // an edit backup leaves while its deeper snapshot is still resident, so
    // it is not written to disk (persist_on_evict).
    if (const ActiveReservation * active = find_active_reservation(reservation.id_)) {
        size_t evicted = 0, freed = 0;
        for (size_t i = 0; i < active->extra_victims.size(); ++i) {
            const int idx = find_entry(active->extra_victims[i]);
            if (idx < 0 || entries_[(size_t)idx].slot != active->extra_slots[i]) continue;
            ++evicted;
            freed += entries_[(size_t)idx].resident_bytes;
            record_eviction(idx);
            erase_inline_entry(idx);
        }
        if (evicted > 0) {
            std::fprintf(stderr,
                "[pc] resident budget evicted %zu snapshot(s), %zu MiB, for "
                "capture slot=%d\n",
                evicted, freed / (1024 * 1024), reservation.slot_);
        }
    }
    if (reservation.has_victim_) {
        const int victim = find_entry(reservation.victim_);
        if (victim >= 0) {
            record_eviction(victim);
            erase_inline_entry(victim);
        }
    }
    replace_inline_entry(
        reservation.slot_, committed_cut, prompt_ids,
        protect || reservation.protect_, resident_bytes);
    release_inline_reservation(reservation.id_);
    reservation.clear();
    return true;
}

void PrefixCache::abort_inline_reservation(
        InlineReservation & reservation) {
    if (!reservation.active()) {
        reservation.clear();
        return;
    }
    for (int i = (int)entries_.size() - 1; i >= 0; --i) {
        if (entries_[(size_t)i].slot == reservation.slot_) {
            erase_inline_entry(i);
        }
    }
    release_inline_reservation(reservation.id_);
    reservation.clear();
}

void PrefixCache::confirm_inline_snap(
        int slot, int target_cut,
        const std::vector<int32_t> & prompt_ids,
        bool protect, size_t resident_bytes) {
    if (disabled_ || slot < 0 || target_cut <= 0 ||
        target_cut > (int)prompt_ids.size()) return;
    if (!active_reservations_.empty()) {
        std::fprintf(stderr,
            "[pc] direct commit refused while a reservation is active\n");
        return;
    }
    replace_inline_entry(
        slot, target_cut, prompt_ids, protect, resident_bytes);
}

void PrefixCache::invalidate_inline_snap(int slot) {
    if (disabled_) return;
    for (int i = (int)entries_.size() - 1; i >= 0; --i) {
        if (entries_[(size_t)i].slot == slot) erase_inline_entry(i);
    }
}

int PrefixCache::pick_evict_victim(int skip_index) const {
    // Every slot can be held by captures in flight, with nothing committed yet.
    if (entries_.empty()) return -1;
    std::vector<const std::vector<int32_t> *> ids_lru;
    std::vector<bool> protected_lru;
    ids_lru.reserve(entries_.size());
    protected_lru.reserve(entries_.size());
    for (const auto & entry : entries_) {
        ids_lru.push_back(&entry.ids);
        // A capture in flight writes over its victim's slot: never a second victim.
        protected_lru.push_back(entry.protect || slot_reserved(entry.slot));
    }
    const int victim = select_inline_evict_victim(ids_lru, &protected_lru, skip_index);
    return victim >= 0 && victim < (int)entries_.size() &&
        slot_reserved(entries_[(size_t)victim].slot) ? -1 : victim;
}

std::vector<int> PrefixCache::superseded_entries(
        const int32_t * ids, size_t len) const {
    // A conversation saves one snapshot per turn, each a strict prefix of the
    // next. Restores use the deepest one, or the one before it when a person
    // edits their last message; the turns before those are dead weight. Keep
    // the shallowest ancestor (the system/tools head that a compacted or
    // sibling conversation reuses) and any protected pin. The previous turn
    // is in the list: prune_superseded_ancestors keeps it while the budget
    // holds it, and a capture's budget counts it as freed.
    std::vector<int> out;
    int shallowest = -1;
    for (int i = 0; i < (int)entries_.size(); ++i) {
        const auto & entry = entries_[(size_t)i];
        if (!is_strict_prefix(entry.ids, ids, len)) continue;
        if (shallowest < 0 ||
            entry.ids.size() < entries_[(size_t)shallowest].ids.size()) {
            shallowest = i;
        }
        if (!entry.protect) out.push_back(i);
    }
    out.erase(std::remove(out.begin(), out.end(), shallowest), out.end());
    return out;
}

std::vector<int> PrefixCache::prune_superseded_ancestors(
        int slot, std::vector<int> * pruned_tokens) {
    std::vector<int> pruned;
    if (disabled_) return pruned;
    const int newest = find_slot_entry(slot);
    if (newest < 0) return pruned;
    const auto & newest_ids = entries_[(size_t)newest].ids;
    const size_t newest_len = newest_ids.size();
    auto superseded = superseded_entries(newest_ids.data(), newest_len);
    // Keep the previous turn, the deepest of them, if the budget holds it
    // once the others are gone.
    int previous = -1;
    size_t others = 0;
    for (const int i : superseded) {
        if (previous < 0 ||
            entries_[(size_t)i].ids.size() > entries_[(size_t)previous].ids.size()) {
            previous = i;
        }
        others += entries_[(size_t)i].resident_bytes;
    }
    if (previous >= 0) others -= entries_[(size_t)previous].resident_bytes;
    if (previous >= 0 &&
        (max_resident_bytes_ == 0 || resident_bytes_ - others <= max_resident_bytes_)) {
        superseded.erase(std::find(superseded.begin(), superseded.end(), previous));
        entries_[(size_t)previous].edit_backup = true;
    }
    size_t freed = 0;
    // Erase from the back so earlier indices stay valid.
    for (auto it = superseded.rbegin(); it != superseded.rend(); ++it) {
        const auto & entry = entries_[(size_t)*it];
        freed += entry.resident_bytes;
        pruned.push_back(entry.slot);
        if (pruned_tokens) pruned_tokens->push_back((int)entry.ids.size());
        erase_inline_entry(*it);
    }
    if (!pruned.empty()) {
        std::fprintf(stderr,
            "[pc] pruned %zu superseded snapshot(s), %zu MiB, behind slot=%d "
            "prefix_len=%zu\n",
            pruned.size(), freed / (1024 * 1024), slot, newest_len);
    }
    return pruned;
}

std::vector<int> PrefixCache::enforce_resident_budget(
        int keep_slot, std::vector<bool> * persist) {
    std::vector<int> evicted;
    if (disabled_ || max_resident_bytes_ == 0) return evicted;
    const size_t before = resident_bytes_;
    while (resident_bytes_ > max_resident_bytes_) {
        const int keep = find_slot_entry(keep_slot);
        if (keep < 0) break;
        // Oldest unprotected leaf first, then the shallowest unprotected
        // ancestor, then the oldest pin that is a leaf; never the new entry
        // or a pinned ancestor of it.
        const int victim = pick_evict_victim(keep);
        if (victim < 0) break;
        const int slot = entries_[(size_t)victim].slot;
        evicted.push_back(slot);
        if (persist) persist->push_back(persist_on_evict(slot));
        record_eviction(victim);
        erase_inline_entry(victim);
    }
    if (!evicted.empty()) {
        std::fprintf(stderr,
            "[pc] resident budget evicted %zu snapshot(s) after commit "
            "resident=%zu->%zu budget=%zu\n",
            evicted.size(), before, resident_bytes_, max_resident_bytes_);
    }
    return evicted;
}

bool PrefixCache::persist_on_evict(int slot) const {
    const int idx = find_slot_entry(slot);
    if (idx < 0) return false;
    const auto & entry = entries_[(size_t)idx];
    if (!entry.edit_backup || entry.protect) return true;
    for (int i = 0; i < (int)entries_.size(); ++i) {
        if (i != idx && is_strict_prefix(entry.ids, entries_[(size_t)i].ids)) {
            return false;  // its conversation's deeper snapshot is resident
        }
    }
    return true;
}

std::vector<int32_t> PrefixCache::entry_tokens(int slot) const {
    const int idx = find_slot_entry(slot);
    return idx < 0 ? std::vector<int32_t>{} : entries_[(size_t)idx].ids;
}

void PrefixCache::record_eviction(int idx) {
    if (!record_evictions_ || idx < 0 || idx >= (int)entries_.size()) return;
    const auto & entry = entries_[(size_t)idx];
    evicted_log_.push_back({entry.slot, entry.ids, persist_on_evict(entry.slot)});
}

std::vector<PrefixCache::Evicted> PrefixCache::entries_for_shutdown() const {
    std::vector<Evicted> out;
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
        out.push_back({it->slot, it->ids, persist_on_evict(it->slot)});
    }
    return out;
}

static void update_atomic_max(std::atomic<uint64_t> & value,
                              uint64_t candidate) {
    uint64_t current = value.load(std::memory_order_relaxed);
    while (current < candidate &&
           !value.compare_exchange_weak(
               current, candidate, std::memory_order_relaxed)) {
    }
}

void PrefixCache::record_capture_attempt(uint64_t elapsed_us, bool success) {
    capture_attempts_.fetch_add(1, std::memory_order_relaxed);
    if (!success) {
        capture_failures_.fetch_add(1, std::memory_order_relaxed);
    }
    capture_stall_us_total_.fetch_add(elapsed_us, std::memory_order_relaxed);
    update_atomic_max(capture_stall_us_max_, elapsed_us);
}

void PrefixCache::record_restore_attempt(uint64_t elapsed_us, bool restored) {
    restore_attempts_.fetch_add(1, std::memory_order_relaxed);
    if (!restored) {
        restore_invalidations_.fetch_add(1, std::memory_order_relaxed);
    }
    restore_stall_us_total_.fetch_add(elapsed_us, std::memory_order_relaxed);
    update_atomic_max(restore_stall_us_max_, elapsed_us);
}

void PrefixCache::mark_all_cleared() {
    if (disabled_) return;
    int n = (int)entries_.size();
    entries_.clear();
    entries_size_count_.store(0, std::memory_order_relaxed);
    resident_bytes_ = 0;
    resident_bytes_count_.store(0, std::memory_order_relaxed);
    next_slot_ = 0;
    active_reservations_.clear();
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
    int remaining = MAX_CACHE_SLOTS - cap_;
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

void PrefixCache::abort_full_snap(int slot) {
    if (full_disabled_) return;
    // The reserved backend slot was cleared before generation. Purge every
    // stale key that still names it, including round-robin reuse through a
    // sparse pool where no LRU eviction key was recorded.
    for (int i = (int)full_entries_.size() - 1; i >= 0; --i) {
        if (full_entries_[(size_t)i].entry.slot == slot) {
            full_entries_.erase(full_entries_.begin() + i);
            full_entries_size_count_.fetch_sub(1, std::memory_order_relaxed);
        }
    }
    full_has_pending_evict_ = false;
}

PrefixCache::InlineStats PrefixCache::stats() const {
    InlineStats out{};
    if (disabled_) return out;
    out.capacity = cap_;
    out.in_use =
        (int)entries_size_count_.load(std::memory_order_relaxed);
    out.lifetime_hits = lifetime_hits_.load(std::memory_order_relaxed);
    out.max_resident_bytes = max_resident_bytes_published_.load(std::memory_order_relaxed);
    out.resident_bytes =
        resident_bytes_count_.load(std::memory_order_relaxed);
    out.budget_skips = budget_skips_.load(std::memory_order_relaxed);
    out.capture_attempts = capture_attempts_.load(std::memory_order_relaxed);
    out.capture_failures = capture_failures_.load(std::memory_order_relaxed);
    out.capture_stall_us_total =
        capture_stall_us_total_.load(std::memory_order_relaxed);
    out.capture_stall_us_max =
        capture_stall_us_max_.load(std::memory_order_relaxed);
    out.restore_attempts = restore_attempts_.load(std::memory_order_relaxed);
    out.restore_invalidations =
        restore_invalidations_.load(std::memory_order_relaxed);
    out.restore_stall_us_total =
        restore_stall_us_total_.load(std::memory_order_relaxed);
    out.restore_stall_us_max =
        restore_stall_us_max_.load(std::memory_order_relaxed);
    return out;
}

PrefixCache::FullStats PrefixCache::full_stats() const {
    if (full_disabled_) return {false, 0, 0, 0, 0};
    return {true, full_cap_,
            (int)full_entries_size_count_.load(std::memory_order_relaxed),
            full_disk_bytes_.load(std::memory_order_relaxed),
            full_lifetime_hits_.load(std::memory_order_relaxed)};
}

}  // namespace luce::common
