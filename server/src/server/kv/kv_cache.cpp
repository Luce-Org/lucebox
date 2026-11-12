#include "server/kv/kv_cache.h"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <utility>

namespace luce::common {

// ─── match ──────────────────────────────────────────────────────────────

KvLease KvCache::match(const KvMatchQuery & query) {
    std::vector<int32_t> & prompt = *query.prompt;
    KvLease lease;
    lease.disk_policy = query.disk_policy;
    lease.full_snap_key_effective = query.key_by_served;
    lease.restore = query.exact_hit;
    lease.prefix_len = query.exact_hit_len;
    lease.using_restore = lease.restore.valid();

    // Exact full-prompt entries. After a DiffPin rewrite, key by the tokens
    // actually served, not the client wire form.
    if (!lease.using_restore) {
        const auto & full_key = query.key_by_served ? prompt : *query.client_prompt;
        auto [full_slot, full_len] = index_.lookup_full(full_key);
        if (full_slot >= 0) {
            lease.restore = Checkpoint::of_cache_slot(full_slot);
            lease.prefix_len = full_len;
            lease.using_restore = true;
        }
    }
    // An entry whose backend snapshot is gone is dropped and the next
    // deepest one tried, instead of prefilling the whole prompt.
    while (!lease.using_restore) {
        const auto [inline_slot, inline_len] = index_.lookup_candidate(
            prompt, (int) prompt.size());
        if (inline_slot < 0) break;
        if (!checkpoints_.used(Checkpoint::of_cache_slot(inline_slot))) {
            index_.invalidate_inline_snap(inline_slot);
            continue;
        }
        // A state-only backup restores with the rows of a deeper snapshot,
        // which must hold at least the backup's positions (a chunk-aligned
        // save can sit a little before its key).
        Checkpoint restore_kv;
        if (index_.state_only(inline_slot)) {
            restore_kv = Checkpoint::of_cache_slot(index_.kv_source_slot(inline_slot));
            if (!checkpoints_.used(restore_kv) || checkpoints_.state_only(restore_kv) ||
                checkpoints_.position(restore_kv) <
                    checkpoints_.position(Checkpoint::of_cache_slot(inline_slot))) {
                checkpoints_.free(Checkpoint::of_cache_slot(inline_slot));
                index_.invalidate_inline_snap(inline_slot);
                continue;
            }
            std::fprintf(stderr,
                "[pc] state backup slot=%d prefix_len=%d restores with the rows of slot=%d\n",
                inline_slot, inline_len, restore_kv.cache_slot());
        }
        lease.restore_kv = restore_kv;
        index_.record_inline_hit(inline_slot, inline_len, prompt.size());
        consider_pin(inline_slot, prompt, query.client ? *query.client : std::string());
        lease.restore = Checkpoint::of_cache_slot(inline_slot);
        lease.prefix_len = inline_len;
        lease.using_restore = true;
    }

    match_disk(lease, query);

    // Edited or summarized histories can be shorter than the stored KV.
    // Such snapshots cannot be diff-prefilled safely.
    // The logical entry length (what the PrefixCache believes is cached, e.g.
    // the tools-head pin at 3620) can exceed the chunk-aligned KV position the
    // backend actually restored (e.g. 3584). Deepen decisions must compare
    // against the logical length; otherwise that chunk gap keeps
    // `forced_cut > restored` true on every tool-heavy turn, the pin branch
    // keeps targeting the already-cached head, and the cache never deepens
    // past the head pin (full conversation re-prefilled each turn).
    lease.logical_prefix_len = lease.using_restore ? lease.prefix_len : 0;
    if (lease.using_restore) {
        const int snapshot_length = checkpoints_.position(lease.restore);
        if (snapshot_length <= 0 || snapshot_length > (int) prompt.size()) {
            std::fprintf(stderr,
                "[pc] slot=%d invalid snapshot pos=%d prompt=%zu — treating as miss\n",
                lease.restore.cache_slot(), snapshot_length, prompt.size());
            if (lease.disk_hit) {
                checkpoints_.free(checkpoints_.staging());
            } else {
                // A memory-cache key whose backend snapshot disappeared
                // must not remain discoverable on every later request.
                // Inline and full caches use disjoint slot ranges, so
                // invalidating both ownership tables is unambiguous.
                checkpoints_.free(lease.restore);
                index_.invalidate_inline_snap(lease.restore.cache_slot());
                index_.abort_full_snap(lease.restore.cache_slot());
            }
            lease.restore = {};
            lease.restore_kv = {};
            lease.prefix_len = 0;
            lease.using_restore = false;
            lease.disk_hit = false;
        } else {
            // PrefixCache keys use safe chat boundaries. Qwen may save at an
            // earlier chunk-aligned position to keep prefill numerically
            // identical, so backend state is the authoritative amount of
            // work actually reused.
            lease.prefix_len = snapshot_length;
        }
    }
    return lease;
}

void KvCache::match_disk(KvLease & lease, const KvMatchQuery & query) {
    const std::vector<int32_t> & prompt = *query.prompt;
    lease.scoped_cut = disk_prefix_cache_fixed_boundary(
        lease.disk_policy, (int) prompt.size(), disk_min_tokens_);

    std::vector<int> lookup_lengths;
    DiskPrefixCache::PrefixProbe disk_probe;
    if (lease.disk_policy.mode == DiskPrefixCacheMode::Full && !prompt.empty()) {
        // Probes are index lookups; only a hit reads a file. A restart then
        // recovers any inline snapshot the previous process persisted, not
        // just exact prompts.
        lookup_lengths = disk_prefix_cache_full_lookup_lengths(
            (int) prompt.size(), find_all_boundaries(prompt, index_.chat_markers()),
            disk_min_tokens_);
        // Session states (agent turns with their generated text) are keyed
        // past the prompt's chat boundaries; the index knows their lengths.
        disk_probe = disk_.longest_prefix(prompt);
        if (disk_probe.len > 0 &&
            std::find(lookup_lengths.begin(), lookup_lengths.end(), disk_probe.len) ==
                lookup_lengths.end()) {
            lookup_lengths.insert(lookup_lengths.begin(), disk_probe.len);
            std::sort(lookup_lengths.begin(), lookup_lengths.end(),
                      std::greater<int>());
        }
    } else if (lease.scoped_cut > 0) {
        lookup_lengths.push_back(lease.scoped_cut);
    }

    // A RAM hit does not end the search: a session whose deep snapshot was
    // evicted to disk often still matches a shallow resident entry (the
    // shared system/tools head). The deeper disk state wins; the index
    // answers without a read, and a failed read keeps the RAM hit.
    bool disk_lookup = !lease.using_restore;
    const int ram_prefix_len = lease.using_restore ? lease.prefix_len : 0;
    if (lease.using_restore && !disk_.disabled() &&
        (!query.compressed || query.unified_view) && disk_probe.len > 0 &&
        disk_probe.cur_pos > lease.prefix_len &&
        disk_probe.len < (int) prompt.size()) {
        lookup_lengths.assign(1, disk_probe.len);
        disk_lookup = true;
    }
    if (disk_lookup && !disk_.disabled()) {
        const Checkpoint staging = checkpoints_.staging();
        for (int lookup_length : lookup_lengths) {
            const std::vector<int32_t> prefix_tokens(
                prompt.begin(), prompt.begin() + lookup_length);
            if (!disk_.lookup(prefix_tokens, staging)) continue;

            const int hit_pos = checkpoints_.position(staging);
            if (hit_pos <= 0 || hit_pos > (int) prompt.size()) {
                std::fprintf(stderr,
                    "[disk-cache] ignoring invalid hit pos=%d prompt=%zu\n",
                    hit_pos, prompt.size());
                checkpoints_.free(staging);
                continue;
            }
            // A pinned prefix's file: the hit keeps the pin from expiring.
            if (shared_ && shared_->learns()) {
                if (const auto pin = pinned_file(prefix_tokens)) {
                    pins_->hit(pin->id, disk_.layout_name());
                }
            }
            lease.restore = staging;
            lease.restore_kv = {};
            lease.prefix_len = hit_pos;
            lease.using_restore = true;
            lease.disk_hit = true;
            std::fprintf(stderr,
                "[disk-cache] hit policy=%s len=%d slot=%d pos=%d (RAM had %d)\n",
                disk_prefix_cache_policy_name(lease.disk_policy).c_str(),
                lookup_length, staging.cache_slot(), lease.prefix_len,
                ram_prefix_len);
            break;
        }
    }

    // Scoped entries must prefill exactly to their hashed boundary before
    // the staging snapshot can be used for the remaining prompt.
    if (!lease.using_restore && !disk_.disabled() && lease.scoped_cut > 0) {
        prefill_scoped(lease, prompt, query.forced_cut);
    }
}

void KvCache::prefill_scoped(KvLease & lease, const std::vector<int32_t> & prompt,
                             int forced_cut) {
    const int boundary = lease.scoped_cut;
    std::fprintf(stderr, "[disk-cache] scoped prefix: policy=%s boundary=%d\n",
                 disk_prefix_cache_policy_name(lease.disk_policy).c_str(), boundary);
    const Checkpoint staging = checkpoints_.staging();
    const std::vector<int32_t> prefix(prompt.begin(), prompt.begin() + boundary);
    const auto result = checkpoints_.prefill(
        staging, prefix,
        prefix_restore_points(prefix, index_.chat_markers(), {forced_cut},
                              false, checkpoints_.restore_point_spacing()));
    if (!result.ok() || !checkpoints_.used(staging)) {
        checkpoints_.free(staging);
        return;
    }
    disk_.learn_layout(staging);
    const bool saved = disk_.save(staging, prefix);
    lease.restore = staging;
    lease.restore_kv = {};
    lease.prefix_len = boundary;
    lease.using_restore = true;
    lease.disk_hit = true;
    std::fprintf(stderr, "[disk-cache] scoped prefix %s, restoring from %d\n",
                 saved ? "saved" : "staged", boundary);
}

// ─── admit ──────────────────────────────────────────────────────────────

void KvCache::admit(KvLease & lease, const std::vector<int32_t> & prompt,
                    const std::vector<int32_t> & client_prompt,
                    const KvAdmitQuery & query, GenerateRequest & request) {
    // A generation can save only one snapshot during prefill. Tool-heavy
    // requests prefer the reusable system/tool boundary; otherwise an
    // enabled exact full-prompt cache retains its existing priority.
    auto prepare_inline = [&]() {
        // Never let the new snapshot land in the slot this request restores
        // from: the guard below would cancel it, pinning the restore point at
        // the deepest slot on linearly-growing conversations.
        const int restore_source_slot =
            lease.using_restore ? lease.restore.cache_slot() : -1;
        // Prefill resumes at the backend's restored position, and a save
        // lands at least one snapshot step past it.
        const int reachable_from =
            (lease.using_restore ? lease.prefix_len : 0) + checkpoints_.granularity();
        lease.snap_reservation = index_.reserve_inline_snap(
            prompt,
            lease.using_restore ? lease.logical_prefix_len : 0,
            query.prefer_head,
            query.forced_cut,
            restore_source_slot,
            [this](int target_cut) {
                return checkpoints_.estimate_bytes(target_cut);
            },
            reachable_from, checkpoints_.restore_point_spacing(), query.learned_cut,
            lease.using_restore ? lease.restore_kv.cache_slot() : -1);
        lease.snap = Checkpoint::of_cache_slot(lease.snap_reservation.slot());
        lease.snap_cut = lease.snap_reservation.target_cut();
    };
    auto prepare_full = [&]() {
        const auto & full_key = lease.full_snap_key_effective ? prompt : client_prompt;
        lease.full_snap = Checkpoint::of_cache_slot(index_.prepare_full_snap(full_key));
        if (lease.full_snap.valid()) {
            lease.full_snap_pos = (int) prompt.size();
            checkpoints_.bind_capture(request, lease.full_snap, lease.full_snap_pos);
            lease.full_snap_prepared = true;
        }
    };

    // An agent turn that continues its generated-turn checkpoint is itself
    // continued the same way by the next request, so its own snapshot only
    // backs an interrupted or edited turn. Take one when the deepest other
    // checkpoint lies kAgentFallbackStride tokens behind; each copy costs
    // the whole prefix. Under tool checks a note pass may replace the reply
    // (a replan), and the client then continues from this prompt's end:
    // always take it there. A note pass's own prompt never comes back, so it
    // takes none: its checkpoints would only push the client's out.
    bool skip_inline = !query.keep_checkpoints;
    if (query.keep_checkpoints && query.agent_step &&
        lease.using_restore && !lease.disk_hit &&
        index_.agent_turn(lease.restore.cache_slot())) {
        constexpr int kAgentFallbackStride = 2048;
        const auto boundaries = find_all_boundaries(prompt, index_.chat_markers());
        const int cut = boundaries.empty() ? 0 : boundaries.back();
        const int fallback = index_.lookup_candidate(
            prompt, lease.logical_prefix_len - 1).second;
        skip_inline = cut - fallback < kAgentFallbackStride;
    }
    if (!skip_inline && (query.capture_inline || lease.using_restore)) {
        prepare_inline();
    }
    if (query.keep_checkpoints && !lease.using_restore && !lease.snap.valid()) {
        prepare_full();
    }

    // Full cache may be disabled or already contain this exact key. Fall
    // back to an inline snapshot when no target has been selected yet.
    if (!skip_inline && !lease.full_snap_prepared && !lease.snap.valid()) {
        prepare_inline();
    }

    // Never destroy the snapshot needed by this request's restore. With a
    // one-slot cache there may be no independent destination for a deeper
    // checkpoint; preserving the current hit is better than invalidating
    // it before restore starts.
    if (lease.using_restore &&
        (lease.snap == lease.restore ||
         (lease.restore_kv.valid() && lease.snap == lease.restore_kv))) {
        lease.snap_reservation.cancel();
        lease.snap = {};
        lease.snap_cut = 0;
    }
    lease.snap_prepared = lease.snap.valid();
    if (lease.snap_prepared) {
        clear_for_capture(lease.snap_reservation);
        checkpoints_.bind_capture(request, lease.snap, lease.snap_cut);
    }
    if (lease.full_snap_prepared) {
        checkpoints_.free(lease.full_snap);
    }
}

std::vector<int> KvCache::restore_points(const std::vector<int32_t> & prompt,
                                         std::initializer_list<int> cuts) const {
    return prefix_restore_points(prompt, index_.chat_markers(), cuts,
        /*drop_last_boundary=*/!checkpoints_.prefill_cuts_at_restore_points(),
        checkpoints_.restore_point_spacing());
}

// ─── release ────────────────────────────────────────────────────────────

Checkpoint KvCache::release(KvLease & lease, const std::vector<int32_t> & prompt,
                            const std::vector<int32_t> & client_prompt,
                            bool produced_output, bool disk_keyable) {
    Checkpoint keyed;
    // A deferred restore can be consumed, or a deeper snapshot can replace
    // its source under memory pressure. Neither entry may stay discoverable.
    if (lease.using_restore && !lease.disk_hit && !checkpoints_.used(lease.restore)) {
        index_.invalidate_inline_snap(lease.restore.cache_slot());
    }

    if (lease.full_snap_prepared) {
        const int saved_position = produced_output && checkpoints_.used(lease.full_snap)
            ? checkpoints_.position(lease.full_snap) : 0;
        if (saved_position > 0 && saved_position <= lease.full_snap_pos) {
            const auto & full_key = lease.full_snap_key_effective ? prompt : client_prompt;
            index_.confirm_full_snap(lease.full_snap.cache_slot(), full_key, saved_position);
        } else {
            checkpoints_.free(lease.full_snap);
            index_.abort_full_snap(lease.full_snap.cache_slot());
        }
    }

    if (lease.snap_prepared) {
        const int saved_position = produced_output && checkpoints_.used(lease.snap)
            ? checkpoints_.position(lease.snap) : 0;
        if (saved_position > 0 && saved_position <= lease.snap_cut) {
            std::fprintf(stderr,
                "[pc] inline snapshot requested=%d saved=%d slot=%d\n",
                lease.snap_cut, saved_position, lease.snap.cache_slot());
            // Key the entry at the requested cut even when Qwen saved at an
            // earlier chunk boundary. Matching the longer prefix still
            // guarantees the saved rows, the restore resumes at the
            // backend's own position, and the next request's deepen decision
            // sees the requested cut as covered. Keying at the saved
            // position re-targets the same cut on every request and the
            // restore point never moves past it.
            commit_capture(lease.snap_reservation, prompt, saved_position,
                           /*key_at_saved=*/false, /*prune=*/true);
            // The disk key is exact: it must not claim rows the snapshot
            // lacks. A compressed prompt gets no key, except one served from
            // a unified view: that view is kept on disk too, so the
            // conversation's next request serves the same tokens again. The
            // write happens only if this snapshot is evicted from RAM
            // (evict) or at shutdown.
            if (disk_keyable && lease.disk_policy.mode == DiskPrefixCacheMode::Full) {
                index_.set_disk_key(lease.snap.cache_slot(), saved_position);
                keyed = lease.snap;
            }
        } else {
            checkpoints_.free(lease.snap);
            lease.snap_reservation.abort();
        }
    }

    if (lease.disk_hit) checkpoints_.free(checkpoints_.staging());
    return keyed;
}

// ─── The concurrent path ────────────────────────────────────────────────

void KvCache::attach_engine(EngineCheckpoints & engine) {
    engine_ = &engine;
    // Disk session cache. Checkpoints that leave RAM (a capture replacing
    // its reservation's victim) are written by the disk writer thread; a
    // prompt whose deepest state is on disk waits at the head of the line
    // while a reader thread loads it, so live streams keep decoding. Only
    // engines whose checkpoints live in the backend's snapshot slots
    // (Qwen paged) take part.
    concurrent_disk_ = !disk_.disabled() && engine.supported() &&
        !index_.disabled() && engine.in_backend_slots();
    if (!disk_.disabled() && !concurrent_disk_) {
        std::fprintf(stderr,
            "[disk-cache] off for concurrent serving: this engine keeps no "
            "checkpoints in backend snapshot slots\n");
    }
    // Recorded with the disk off too: a capture whose resident-budget plan
    // evicts further checkpoints (PrefixCache::reserve_inline_snap) frees
    // their payloads in captured().
    index_.set_record_evictions(true);
}

bool KvCache::plans(bool images) const {
    return !images && engine_ && engine_->supported() && !index_.disabled();
}

KvCache::PlanStatus KvCache::plan(const KvPlanQuery & query, KvPlan & out) {
    const std::vector<int32_t> & prompt = *query.prompt;
    EngineCheckpoints & engine = *engine_;
    PrefixStorePlan & plan = out.plan;
    const auto hit = index_.lookup_candidate(prompt, (int)prompt.size() - 1);
    int restore_kv_slot = -1;
    if (hit.first >= 0 && hit.second > 0 && hit.second < (int)prompt.size()) {
        out.restore_slot = hit.first;
        plan.restore = EngineCheckpoints::ref(Checkpoint::of_cache_slot(hit.first), hit.second);
        // A state-only backup restores with the rows of a deeper checkpoint
        // (lookup_candidate returns one only while such a checkpoint is resident).
        if (index_.state_only(hit.first)) {
            restore_kv_slot = index_.kv_source_slot(hit.first);
            plan.restore_kv = EngineCheckpoints::ref(
                Checkpoint::of_cache_slot(restore_kv_slot), index_.tokens(restore_kv_slot));
            std::fprintf(stderr,
                "[pc] state backup slot=%d prefix_len=%d restores with the rows of slot=%d\n",
                hit.first, hit.second, restore_kv_slot);
        }
    }
    // A deeper state on disk (the index answers without a read): load it
    // off this thread, then restore from the staging slot.
    std::vector<int> scoped_lengths;
    const auto mode = query.disk_policy.mode;
    if (mode == DiskPrefixCacheMode::Fixed) {
        scoped_lengths.push_back(disk_prefix_cache_fixed_boundary(
            query.disk_policy, (int)prompt.size(), disk_min_tokens_));
    }
    const auto probe = concurrent_disk_ && mode != DiskPrefixCacheMode::Off
        ? disk_.longest_prefix(prompt,
            mode == DiskPrefixCacheMode::Full ? nullptr : &scoped_lengths)
        : DiskPrefixCache::PrefixProbe{};
    if (read_ && read_->owner != query.owner) drop_read();
    if (probe.cur_pos > plan.restore.tokens && probe.cur_pos == probe.len &&
        probe.cur_pos < (int)prompt.size()) {
        if (!read_) {
            read_.emplace();
            read_->owner = query.owner;
            read_->result = std::make_shared<DiskRead::Result>();
            read_->reader = std::thread(
                [this, key = std::vector<int32_t>(prompt.begin(), prompt.begin() + probe.len),
                 result = read_->result]() {
                    result->snap = disk_.load(key);
                    result->done.store(true, std::memory_order_release);
                });
            std::fprintf(stderr,
                "[disk-cache] %s: reading %d-token state from disk "
                "(RAM has %d)\n", query.request_id, probe.len, plan.restore.tokens);
        }
        if (!read_->result->done.load(std::memory_order_acquire)) {
            return PlanStatus::loading;
        }
        read_->reader.join();
        OwnedSnapshot snap = std::move(read_->result->snap);
        read_.reset();
        const int pos = snap.cur_pos();
        if (!snap.empty() && disk_.adopt(std::move(snap), engine.staging())) {
            // A pinned prefix's file: the hit keeps the pin from expiring.
            if (shared_ && shared_->learns()) {
                const std::vector<int32_t> key(prompt.begin(), prompt.begin() + probe.len);
                if (const auto pin = pinned_file(key)) pins_->hit(pin->id, disk_.layout_name());
            }
            out.disk_restore = true;
            out.restore_slot = -1;
            restore_kv_slot = -1;
            plan.restore_kv = {};
            plan.restore = EngineCheckpoints::ref(engine.staging(), pos);
        } else {
            std::fprintf(stderr, "[disk-cache] %s: disk state unusable, %s\n",
                query.request_id,
                plan.restore.valid() ? "restoring from RAM" : "prefilling");
        }
    } else if (read_ && read_->owner == query.owner) {
        drop_read();
    }

    if (plan.restore.valid() && disk_cache_digest_enabled()) {
        int n_tensors = 0;
        uint64_t n_bytes = 0;
        const uint64_t digest = snapshot_state_digest(
            checkpoints_.ref(EngineCheckpoints::of(plan.restore)), &n_tensors, &n_bytes);
        std::fprintf(stderr,
            "[disk-cache] digest %s source=%s pos=%d tensors=%d "
            "bytes=%llu xxh64=%016llx\n",
            query.request_id, out.disk_restore ? "ssd" : "ram",
            plan.restore.tokens, n_tensors,
            (unsigned long long)n_bytes, (unsigned long long)digest);
    }

    // The restore source is never the capture destination: the new
    // checkpoint lands in a different slot so the restore point can slide
    // forward past the deepest slot (same rule as the single-sequence
    // path's restore_source_slot).
    const int restored_len = plan.restore.valid() ? plan.restore.tokens : 0;
    const auto reserve_capture = [&](int forced_cut, bool learned) {
        return index_.reserve_inline_snap(
            prompt, restored_len, query.prefer_head, forced_cut, out.restore_slot,
            [&engine](int target_cut) { return engine.estimate_bytes(target_cut); },
            /*reachable_from=*/0, checkpoints_.restore_point_spacing(), learned,
            restore_kv_slot);
    };
    PrefixCache::InlineReservation capture =
        reserve_capture(query.forced_cut, query.learned_cut);
    if (capture.active() && query.spare_pins && takes_a_pin(capture)) capture.cancel();
    // Subagents launched together: when the requests still queued share
    // more of this prompt than its own capture keeps whole (system messages
    // that differ only after the tools), it captures the shared prefix
    // instead, for them to restore.
    const int wait_min = query.shared_prefix_min_tokens;
    if (capture.active() && wait_min > 0 && query.forced_cut <= restored_len &&
        query.queued_shared_prefix) {
        const int own_cut = capture.target_cut();
        const int shared = query.queued_shared_prefix();
        if (shared < own_cut && shared - restored_len >= wait_min) {
            capture.cancel();
            capture = reserve_capture(shared, false);
            if (capture.active() && capture.target_cut() <= shared &&
                capture.target_cut() - restored_len >= wait_min) {
                std::fprintf(stderr,
                    "[pc] %s captures the %d-token prefix it shares with "
                    "queued requests, not %d\n", query.request_id,
                    capture.target_cut(), own_cut);
            } else {
                capture.cancel();
                capture = reserve_capture(query.forced_cut, query.learned_cut);
            }
        }
    }
    if (capture.active()) {
        plan.capture.id = next_capture_id_++;
        if (next_capture_id_ == 0) next_capture_id_ = 1;
        plan.capture.checkpoint = EngineCheckpoints::ref(
            Checkpoint::of_cache_slot(capture.slot()), capture.target_cut());
        if ((plan.restore.valid() && plan.capture.checkpoint == plan.restore) ||
            capture.slot() == restore_kv_slot) {
            capture.cancel();
            plan.capture = {};
        }
    }
    // The same cuts on hits and misses, whatever the cache holds (as the
    // single-sequence path passes GenerateRequest::restore_points).
    plan.restore_points = restore_points(
        prompt, {plan.restore.tokens, plan.capture.checkpoint.tokens});
    if (plan.capture.valid()) {
        out.capture = PrefixCaptureTxn(std::move(capture), engine.engine(), plan.capture);
    }
    return PlanStatus::ready;
}

std::string KvCache::admitted(KvPlan & plan, const PrefixStoreAdmission & prefix,
                              bool admission_ok, const char * request_id) {
    std::string error;
    const PrefixStorePlan & requested = plan.plan;
    if (prefix.restore_attempted) {
        index_.record_restore_attempt(prefix.restore_elapsed_us, prefix.restored.valid());
    }
    if (prefix.malformed_restore_state()) {
        error = "engine returned malformed prefix restore state";
        // A malformed outcome does not tell us whether the engine-owned
        // payload is still usable. Drop both sides of the requested
        // checkpoint so a later lookup cannot retry stale metadata.
        plan.capture.cancel();
        if (requested.restore.valid() && plan.restore_slot >= 0) {
            engine_->discard(requested.restore);
            index_.invalidate_inline_snap(plan.restore_slot);
        }
    }
    if (plan.disk_restore) {
        // The engine copied the staging checkpoint into the sequence's
        // pages during admission (or rejected it); it is not kept.
        checkpoints_.free(engine_->staging());
        if (prefix.restored.valid() && prefix.restored == requested.restore) {
            std::fprintf(stderr, "[disk-cache] hit %s: restored %d tokens from disk\n",
                         request_id, prefix.restored.tokens);
        }
    }
    if (plan.disk_restore && prefix.invalidated.valid() &&
        prefix.invalidated == requested.restore) {
        // A disk state that does not fit this cache layout: the engine
        // already admitted the prompt cold.
        std::fprintf(stderr,
            "[disk-cache] %s: disk state rejected by the engine, prefilling\n",
            request_id);
    } else if (prefix.invalidated.valid()) {
        if (prefix.invalidated != requested.restore || plan.restore_slot < 0) {
            error = "engine invalidated an unrequested prefix checkpoint";
        } else {
            // Cancel this admission's reservation before removing the stale
            // restore metadata. The invalidation itself must not clear a
            // capture reservation owned by another live request. Qwen
            // already discarded the engine-owned payload.
            plan.capture.cancel();
            index_.invalidate_inline_snap(plan.restore_slot);
        }
    }
    if (prefix.restored.valid() && prefix.restored != requested.restore) {
        engine_->discard(prefix.restored);
        error = "engine restored an unrequested prefix checkpoint";
    }
    if (prefix.restored.valid() && prefix.invalidated.valid()) {
        error = "engine both restored and invalidated one checkpoint";
    }
    if (prefix.capture.valid() && prefix.capture != requested.capture) {
        error = "engine accepted an unrequested prefix capture";
    }
    if (!admission_ok && (prefix.restored.valid() || prefix.capture.valid())) {
        error = "failed admission returned accepted prefix state";
    }
    if (!error.empty()) {
        plan.capture.cancel();
    } else if (prefix.capture != requested.capture) {
        // Rejected capture: no payload was touched, so preserve any
        // incumbent selected by the cache's transactional eviction.
        plan.capture.cancel();
    }
    return error;
}

void KvCache::restored(const KvPlan & plan, const PrefixStoreAdmission & prefix,
                       const std::vector<int32_t> & prompt, const std::string & client) {
    if (prefix.restored.valid() && plan.restore_slot >= 0) {
        index_.record_inline_hit(plan.restore_slot, prefix.restored.tokens, prompt.size());
        consider_pin(plan.restore_slot, prompt, client);
    }
}

void KvCache::captured(PrefixCaptureTxn & capture, const PrefixStoreEvent & event,
                       const std::vector<int32_t> & prompt, DiskPrefixCacheMode mode,
                       const std::function<std::string()> & aux) {
    index_.record_capture_attempt(
        event.elapsed_us, event.status == PrefixStoreEvent::Status::saved);
    using Resolution = PrefixCaptureTxn::Resolution;
    const Resolution resolution = capture.resolve(event, prompt);
    if (resolution == Resolution::mismatched || resolution == Resolution::inactive) {
        // The transaction has already aborted only its own destination.
        // Never act on an event-supplied checkpoint: a payload the engine
        // displaced there is freed by that slot's next capture.
        std::fprintf(stderr,
            "[parallel-pc] capture ticket mismatch id=%llu checkpoint=%llu\n",
            (unsigned long long)event.ticket.id,
            (unsigned long long)event.ticket.checkpoint.id);
        return;
    }
    const int captured_slot = EngineCheckpoints::of(event.ticket.checkpoint).cache_slot();
    // The checkpoint this capture replaced in its slot (the reservation's
    // victim) left RAM: write it to disk unless it was only an edit backup.
    // Claimed on every outcome, and freed here when there is no disk to
    // write it to.
    OwnedSnapshot displaced =
        checkpoints_.take_displaced(Checkpoint::of_cache_slot(captured_slot));
    for (auto & evicted : index_.take_evicted()) {
        if (evicted.slot != captured_slot) {
            // A further checkpoint the capture's resident-budget plan
            // evicted as it committed: its payload is still in its own slot.
            // Written like the displaced one, then freed.
            const Checkpoint victim = Checkpoint::of_cache_slot(evicted.slot);
            if (concurrent_disk_ && worth_writing(evicted)) {
                write_out(evicted, checkpoints_.detach(victim), "ram eviction");
            }
            engine_->discard(victim, (int)evicted.ids.size());
            continue;
        }
        write_out(evicted, std::move(displaced), "ram eviction");
        displaced.reset();
    }
    if (resolution == Resolution::saved) {
        // The slot holds the new checkpoint now; its entry starts with no
        // disk key, aux or agent-turn mark.
        if (concurrent_disk_ && mode == DiskPrefixCacheMode::Full) {
            index_.set_disk_key(captured_slot, event.ticket.checkpoint.tokens);
            index_.set_aux(captured_slot, aux ? aux() : std::string());
        }
        trim_superseded(captured_slot);
    } else if (resolution == Resolution::failed) {
        std::fprintf(stderr, "[parallel-pc] capture failed checkpoint=%llu: %s\n",
                     (unsigned long long)event.ticket.checkpoint.id, event.error.c_str());
    }
    // Whatever left RAM above may have been the rows a backup restored with.
    free_orphaned_state_backups();
}

void KvCache::drop_read(const void * owner) {
    if (!read_ || (owner && read_->owner != owner)) return;
    if (read_->reader.joinable()) finished_reads_.push_back(std::move(*read_));
    read_.reset();
}

void KvCache::reap_reads() {
    for (auto it = finished_reads_.begin(); it != finished_reads_.end();) {
        if (it->result->done.load(std::memory_order_acquire)) {
            it->reader.join();
            it = finished_reads_.erase(it);
        } else {
            ++it;
        }
    }
}

void KvCache::join_reads() {
    drop_read();
    for (auto & read : finished_reads_) read.reader.join();
    finished_reads_.clear();
}

// ─── Shared prefixes ────────────────────────────────────────────────────

void KvCache::attach_shared_prefixes(SharedPrefixes & shared, PrefixPins & pins) {
    shared_ = &shared;
    pins_ = &pins;
}

void KvCache::remember_prefix(const std::vector<int32_t> & prompt,
                              const std::vector<int> & boundaries) {
    if (!shared_) return;
    shared_->remember(prompt, boundaries);
    if (pins_ && shared_->learns()) pins_->set_ring(shared_->ring());
}

void KvCache::consider_pin(int slot, const std::vector<int32_t> & prompt,
                           const std::string & client) {
    if (!shared_ || !pins_ || !shared_->on() || slot < 0) return;
    if (index_.pinned(slot)) {
        const std::string id = PrefixPins::id_of(index_.entry_tokens(slot));
        if (!pins_->has(id)) {
            index_.set_pinned(slot, false);  // deleted (HTTP) or dropped at the caps
        } else if (shared_->learns()) {
            pins_->hit(id, disk_.layout_name());
        }
        return;
    }
    // A pooled payload is never written (DiskPrefixCache::save), so its pin
    // would have no file and be warmed again after every restart.
    if (checkpoints_.kind(Checkpoint::of_cache_slot(slot)) == CheckpointKind::pooled) return;
    // Without a disk tier for this engine a pin keeps only its token ids
    // (warmed again after a restart), so it needs a pins file that lasts.
    const bool files = pins_have_files();
    if (!shared_->learns() || !index_.shared_prefix(slot) ||
        (!files && pins_->path().empty()) ||
        !index_.restored_by_another_conversation(slot, prompt)) {
        return;
    }
    PrefixCache::Evicted entry = index_.entry(slot);
    if (!files) {
        entry.disk_key.clear();
    } else if (entry.disk_key.empty()) {
        return;  // a compressed view or a scoped disk policy
    }
    const int cut = (int)entry.ids.size();
    const auto boundaries = find_all_boundaries(prompt, index_.chat_markers());
    const bool at_boundary =
        std::find(boundaries.begin(), boundaries.end(), cut) != boundaries.end();
    if (!at_boundary && shared_->recurrences(prompt, cut) < SharedPrefixes::kPinRecurrences) {
        return;
    }
    index_.set_pinned(slot, true);
    pending_pins_.push_back({slot, entry.ids, entry.disk_key, entry.aux, client});
}

void KvCache::unpin_files(const SharedPrefixPin & pin) {
    for (const auto & file : pin.files) disk_.unpin_file(file.path);
}

void KvCache::drop_pinned(const SharedPrefixPin & dropped) {
    unpin_files(dropped);
    if (pins_->has(dropped.id)) return;  // only some of its files went
    const int slot = index_.slot_of(dropped.tokens);
    if (slot >= 0) index_.set_pinned(slot, false);
    std::fprintf(stderr, "[shared-prefix] unpinned %s (%zu tokens)\n",
                 dropped.id.c_str(), dropped.tokens.size());
}

SharedPrefixPin::File KvCache::pin_file(const std::vector<int32_t> & key) {
    SharedPrefixPin::File file;
    file.layout = disk_.layout_name();
    file.disk_tokens = (int)key.size();
    file.path = disk_.file_path(key);
    file.bytes = estimate_bytes(file.disk_tokens);
    return file;
}

void KvCache::flush_pins(bool force) {
    if (!pins_ || !shared_) return;
    for (auto & pending : pending_pins_) {
        SharedPrefixPin::File file;
        if (!pending.disk_key.empty()) {
            // Written now unless the entry already left RAM, which wrote it.
            const Checkpoint checkpoint = Checkpoint::of_cache_slot(pending.slot);
            const bool resident = index_.entry_tokens(pending.slot) == pending.ids &&
                checkpoints_.used(checkpoint);
            const bool on_disk = resident
                ? disk_.save(checkpoint, pending.disk_key, pending.aux)
                : disk_.contains(pending.disk_key);
            if (on_disk) file = pin_file(pending.disk_key);
            if (file.path.empty()) {
                if (resident) index_.set_pinned(pending.slot, false);
                continue;
            }
        }
        SharedPrefixPin pin;
        pin.tokens = pending.ids;
        pin.source = "learned";
        pin.client = pending.client;
        pin.hits = 1;
        pin.id = PrefixPins::id_of(pin.tokens);
        if (!file.path.empty()) pin.files.push_back(file);
        std::fprintf(stderr,
            "[shared-prefix] pinned %s: %zu tokens, client %s -> %s\n",
            pin.id.c_str(), pin.tokens.size(),
            pin.client.empty() ? "-" : pin.client.c_str(),
            file.path.empty() ? "token ids only" : file.path.c_str());
        const std::string id = pin.id;
        for (const auto & d : pins_->add(std::move(pin))) drop_pinned(d);
        if (!file.path.empty() && pins_->has(id)) disk_.pin(pending.disk_key, true);
    }
    pending_pins_.clear();
    // RAM entries of pins gone from the catalog (an HTTP DELETE runs on its
    // own thread and leaves the index to this one) are no longer pinned.
    for (const int slot : index_.pinned_slots()) {
        if (!pins_->has(PrefixPins::id_of(index_.entry_tokens(slot)))) {
            index_.set_pinned(slot, false);
        }
    }
    // `freeze` keeps the pin set as it is.
    if (shared_->learns()) {
        for (const auto & expired : pins_->expire()) drop_pinned(expired);
    }
    if (force || pins_->due(60)) pins_->save(force);
}

bool KvCache::unpin(const std::string & id) {
    if (!pins_) return false;
    const auto pin = pins_->remove(id);
    if (!pin) return false;
    unpin_files(*pin);
    pins_->save(true);
    std::fprintf(stderr, "[shared-prefix] unpinned %s on request\n", id.c_str());
    return true;
}

size_t KvCache::estimate_bytes(int tokens) const {
    return engine_ ? engine_->estimate_bytes(tokens) : checkpoints_.estimate_bytes(tokens);
}

bool KvCache::takes_a_pin(const PrefixCache::InlineReservation & reservation) const {
    if (index_.pinned(reservation.slot())) return true;
    for (const int slot : reservation.extra_victim_slots()) {
        if (index_.pinned(slot)) return true;
    }
    return false;
}

std::optional<SharedPrefixPin> KvCache::pinned_file(const std::vector<int32_t> & key) const {
    if (!pins_ || !shared_ || !shared_->on() || pins_->size() == 0) return std::nullopt;
    return pins_->find_by_disk_key(key, disk_.layout_name());
}

void KvCache::load_pins(size_t ram_cap, int max_ctx) {
    if (!pins_ || !shared_ || !shared_->on() || index_.disabled()) return;
    if (engine_ && !engine_->supported()) return;
    pin_ram_cap_ = ram_cap;
    const bool files = pins_have_files();
    const std::string layout = disk_.layout_name();
    const auto pins = pins_->list();
    int cleared = 0;
    if (files) {
        // A pinned mark no pin lists: a crash between the mark and the
        // catalog's save. The file falls under the TTL and the LRU again.
        for (const auto & path : disk_.pinned_files()) {
            const bool listed = std::any_of(pins.begin(), pins.end(), [&](const SharedPrefixPin & p) {
                const auto * f = p.file_for(layout);
                return f && f->path == path;
            });
            if (!listed) {
                disk_.unpin_file(path);
                ++cleared;
            }
        }
    }
    int too_long = 0;
    for (const auto & pin : pins) {
        if (max_ctx > 0 && (int)pin.tokens.size() + 1 >= max_ctx) {
            ++too_long;  // kept for settings with a longer context
            continue;
        }
        // A pin with a file for these settings is served by disk hits; the
        // concurrent path warms the others (an engine without a disk tier,
        // or after a settings change).
        const auto * file = files ? pin.file_for(layout) : nullptr;
        const bool on_disk = file && disk_.contains(std::vector<int32_t>(
            pin.tokens.begin(), pin.tokens.begin() + file->disk_tokens));
        if (!on_disk && engine_) to_warm_.push_back(pin.id);
    }
    if (!pins.empty()) {
        std::fprintf(stderr,
            "[shared-prefix] startup: %zu pins, %zu to warm, %d longer than the context, "
            "%d stray pinned marks cleared\n",
            pins.size(), to_warm_.size(), too_long, cleared);
    }
}

bool KvCache::record_warm(const SharedPrefixPin & pin, Checkpoint checkpoint, int saved,
                          double seconds, const char * where) {
    SharedPrefixPin::File file;
    std::vector<int32_t> key;
    if (pins_have_files()) {
        key.assign(pin.tokens.begin(), pin.tokens.begin() + saved);
        disk_.learn_layout(checkpoint);
        if (disk_.save(checkpoint, key)) file = pin_file(key);
    }
    bool kept = false;
    if (!file.path.empty()) {
        // Only into the pin it warmed: a DELETE during the warm stands.
        const auto dropped = pins_->attach(pin.id, file);
        kept = dropped.has_value();
        if (kept) {
            for (const auto & d : *dropped) drop_pinned(d);
            disk_.pin(key, true);
            pins_->save();
        }
    } else {
        kept = pins_->has(pin.id);
    }
    // Not on disk (no tier, full, or too large for the budget): it serves
    // from RAM until it leaves, and warms again at the next start.
    std::fprintf(stderr, "[shared-prefix] warmed %s: %d tokens in %.2fs, %s -> %s%s\n",
                 pin.id.c_str(), saved, seconds, where,
                 file.path.empty() ? "not written" : file.path.c_str(),
                 kept ? "" : " (unpinned meanwhile: not kept)");
    return kept;
}

bool KvCache::warm_next_on_engine(const CancellationProbe & cancel) {
    if (!engine_ || !engine_->supported()) return false;
    SeqEngine & engine = engine_->engine();
    while (pins_ && !to_warm_.empty()) {
        const auto pin = pins_->get(to_warm_.front());
        if (!pin || pin->tokens.empty() || index_.slot_of(pin->tokens) >= 0) {
            to_warm_.erase(to_warm_.begin());
            continue;
        }
        const int cut = (int)pin->tokens.size();
        // Into RAM only while the pins there fit their allowance (the engine
        // has no staging capture to write it to disk only).
        if (index_.pinned_bytes() + estimate_bytes(cut) > pin_ram_cap_) {
            std::fprintf(stderr, "[shared-prefix] warm of %s skipped: the pins in RAM "
                         "fill their allowance\n", pin->id.c_str());
            to_warm_.erase(to_warm_.begin());
            continue;
        }
        // One token past the cut: any token, the state at the cut does not
        // depend on it, and the sequence retires before it is computed.
        std::vector<int32_t> prompt(pin->tokens);
        prompt.push_back(pin->tokens.back());
        KvPlanQuery query;
        query.owner = &to_warm_;
        query.request_id = "warm";
        query.prompt = &prompt;
        query.disk_policy.mode = DiskPrefixCacheMode::Off;  // the file is what is missing
        query.prefer_head = true;  // a shared prefix (protect), as a learned cut is
        query.forced_cut = cut;
        query.spare_pins = true;
        KvPlan plan_out;
        if (plan(query, plan_out) != PlanStatus::ready) return false;
        if (!plan_out.capture.active()) {
            std::fprintf(stderr, "[shared-prefix] warm of %s skipped: no room\n", pin->id.c_str());
            to_warm_.erase(to_warm_.begin());
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        auto admitted_result = engine.admit_with_prefix(
            (1ull << 62) + next_warm_id_++, prompt, SamplerCfg{}, plan_out.plan);
        const bool ok = admitted_result.status == SeqEngine::AdmitResult::Status::admitted;
        const std::string error = admitted(plan_out, admitted_result.prefix_store, ok, "warm");
        const int slot = admitted_result.slot;
        if (!ok || !error.empty() || !admitted_result.prefix_store.capture.valid()) {
            if (ok) engine.retire(slot);
            plan_out.capture.cancel();
            std::fprintf(stderr, "[shared-prefix] warm of %s not admitted: %s\n", pin->id.c_str(),
                         !error.empty() ? error.c_str()
                         : !ok ? admitted_result.error.c_str() : "no capture");
            if (admitted_result.status == SeqEngine::AdmitResult::Status::busy) return false;
            to_warm_.erase(to_warm_.begin());
            continue;
        }
        const int step_tokens = engine.step_plan_limits(0).max_prefill_tokens_per_sequence;
        int saved = 0;
        bool done = false;
        bool preempted = false;
        while (!done) {
            if (cancel && cancel()) {
                preempted = true;
                break;
            }
            SeqEngine::StepPlan step;
            step.prefills.push_back({slot, step_tokens});
            const auto result = engine.step(step);
            if (!result.ok()) {
                std::fprintf(stderr, "[shared-prefix] warm of %s failed: %s\n",
                             pin->id.c_str(), result.error.c_str());
                break;
            }
            for (const auto & out : result.prefills) {
                if (out.slot != slot) continue;
                if (out.prefix_store.attempted()) {
                    captured(plan_out.capture, out.prefix_store, prompt,
                             DiskPrefixCacheMode::Off, {});
                    if (out.prefix_store.status == PrefixStoreEvent::Status::saved) {
                        saved = out.prefix_store.ticket.checkpoint.tokens;
                    }
                    done = true;
                }
                if (out.status != SeqEngine::PrefillOutput::Status::advanced) done = true;
            }
        }
        engine.retire(slot);
        plan_out.capture.cancel();  // a no-op once captured() resolved it
        to_warm_.erase(to_warm_.begin());
        if (preempted) {
            // One attempt per process: a preempted warm keeps nothing, and on
            // a head longer than one engine step retrying at every idle could
            // block requests again and again. The head is captured by the
            // requests that need it, or warmed at the next start.
            std::fprintf(stderr, "[shared-prefix] warm of %s preempted by a request; "
                         "not retried until the next start\n", pin->id.c_str());
            return false;
        }
        const int entry = index_.slot_of(pin->tokens);
        if (saved != cut || entry < 0) {
            std::fprintf(stderr, "[shared-prefix] warm of %s failed: no capture at %d\n",
                         pin->id.c_str(), cut);
            continue;
        }
        index_.set_pinned(entry, true);
        if (pins_have_files()) index_.set_disk_key(entry, saved);
        if (!record_warm(*pin, Checkpoint::of_cache_slot(entry), saved,
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t0).count(),
                         "in RAM")) {
            index_.set_pinned(entry, false);
        }
        return true;
    }
    return false;
}

// ─── Captures outside a request's prefill ──────────────────────────────

Checkpoint KvCache::clear_for_capture(const PrefixCache::InlineReservation & reservation) {
    const Checkpoint checkpoint = Checkpoint::of_cache_slot(reservation.slot());
    evict(reservation.slot(), "ram eviction");
    checkpoints_.free(checkpoint);
    return checkpoint;
}

void KvCache::commit_capture(PrefixCache::InlineReservation & reservation,
                             const std::vector<int32_t> & ids, int saved_pos,
                             bool key_at_saved, bool prune) {
    const int slot = reservation.slot();
    evict_extra_victims(reservation);
    const size_t bytes = checkpoints_.estimate_bytes(saved_pos);
    if (key_at_saved) {
        reservation.commit_at(ids, saved_pos, bytes);
    } else {
        reservation.commit(ids, bytes);
    }
    if (prune) {
        trim_after_commit(slot);
    } else {
        enforce_budget(slot);
    }
}

// ─── Entries leaving RAM ────────────────────────────────────────────────

bool KvCache::worth_writing(const PrefixCache::Evicted & entry) const {
    return entry.slot >= 0 && entry.persist && !entry.disk_key.empty() &&
           !disk_.disabled();
}

bool KvCache::write_out(const PrefixCache::Evicted & entry, OwnedSnapshot payload,
                        const char * reason) {
    if (!worth_writing(entry) || payload.empty()) return false;
    const int pos = payload.cur_pos();
    const bool queued = disk_.save_async(std::move(payload), entry.disk_key, entry.aux);
    const int pruned = queued
        ? disk_.prune_ancestors(entry.disk_key, disk_keep_max_tokens_)
        : 0;
    std::fprintf(stderr,
        "[disk-cache] %s slot=%d tokens=%zu pos=%d -> %s (pruned %d older)\n",
        reason, entry.slot, entry.disk_key.size(), pos,
        queued ? "write queued" : "not written", pruned);
    return queued;
}

bool KvCache::write_out_slot(const PrefixCache::Evicted & entry, const char * reason) {
    // Detaching copies a device-resident payload to the host: only for a write.
    if (!worth_writing(entry)) return false;
    const Checkpoint checkpoint = Checkpoint::of_cache_slot(entry.slot);
    if (!checkpoints_.used(checkpoint)) return false;
    return write_out(entry, checkpoints_.detach(checkpoint), reason);
}

void KvCache::evict(int slot, const char * reason) {
    // The cheap checks first: the record copies the entry's tokens.
    if (slot < 0 || disk_.disabled() || !index_.has_disk_key(slot) ||
        !index_.persist_on_evict(slot)) {
        return;
    }
    write_out_slot(index_.entry(slot), reason);
}

void KvCache::evict_extra_victims(const PrefixCache::InlineReservation & reservation) {
    for (const int slot : reservation.extra_victim_slots()) {
        evict(slot, "ram eviction");
        checkpoints_.free(Checkpoint::of_cache_slot(slot));
    }
}

void KvCache::trim_after_commit(int slot) {
    trim_superseded(slot);
    enforce_budget(slot);
}

void KvCache::free_payload(int slot, int tokens) {
    const Checkpoint checkpoint = Checkpoint::of_cache_slot(slot);
    if (engine_) {
        engine_->discard(checkpoint, tokens);
    } else {
        checkpoints_.free(checkpoint);
    }
}

bool KvCache::drop_payload_kv(int slot, int tokens) {
    const Checkpoint checkpoint = Checkpoint::of_cache_slot(slot);
    return engine_ ? engine_->drop_kv(checkpoint, tokens) : checkpoints_.drop_kv(checkpoint);
}

size_t KvCache::state_payload_bytes(int tokens) const {
    return engine_ ? engine_->estimate_state_bytes(tokens)
                   : checkpoints_.estimate_state_bytes(tokens);
}

void KvCache::trim_superseded(int slot) {
    // A conversation's checkpoints are prefixes of each other, and the
    // superseded ones repeat the newest one's attention rows.
    std::vector<int> tokens;
    std::vector<int> keep_state;
    const auto released = index_.prune_superseded_ancestors(
        slot, &tokens, index_.state_backups() ? &keep_state : nullptr);
    for (size_t i = 0; i < released.size(); ++i) free_payload(released[i], tokens[i]);
    for (const int backup : keep_state) {
        const int length = index_.tokens(backup);
        if (drop_payload_kv(backup, length)) {
            index_.mark_state_only(backup, state_payload_bytes(length));
        } else {
            free_payload(backup, length);
            index_.invalidate_inline_snap(backup);
        }
    }
}

void KvCache::enforce_budget(int keep_slot) {
    for (const auto & entry : index_.enforce_resident_budget(keep_slot)) {
        write_out_slot(entry, "ram eviction");
        checkpoints_.free(Checkpoint::of_cache_slot(entry.slot));
    }
    free_orphaned_state_backups();
}

void KvCache::free_orphaned_state_backups() {
    std::vector<int> tokens;
    const auto slots = index_.take_orphaned_state_backups(&tokens);
    for (size_t i = 0; i < slots.size(); ++i) free_payload(slots[i], tokens[i]);
}

void KvCache::flush(const char * reason, double limit_s,
                    std::chrono::steady_clock::time_point started) {
    flush_pins(/*force=*/true);
    const auto elapsed = [&] {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    };
    size_t queued = 0, backups = 0, skipped = 0;
    const auto before = disk_.stats();
    // Graceful shutdown evicts everything: each conversation's newest
    // snapshot and the heads, most recently used first, so the time limit
    // drops the least valuable ones. Queuing can wait for the writer
    // (bounded queued bytes), so the limit is checked between snapshots.
    for (const auto & entry : index_.entries_for_shutdown()) {
        if (!entry.persist) {
            ++backups;
            continue;
        }
        if (limit_s >= 0 && elapsed() >= limit_s) {
            ++skipped;
            continue;
        }
        if (write_out_slot(entry, reason)) ++queued;
    }
    const auto drained = disk_.drain(
        limit_s < 0 ? -1.0 : (std::max)(0.0, limit_s - elapsed()));
    std::fprintf(stderr,
        "[disk-cache] %s flush: %zu snapshots queued (%zu edit backups not "
        "written), %llu files %.1f MB written in %.2fs (limit %.0fs; %llu "
        "writes and %zu snapshots dropped by it)\n",
        reason, queued, backups,
        (unsigned long long)(disk_.stats().writes - before.writes),
        (double)(disk_.stats().write_bytes - before.write_bytes) / (1024.0 * 1024.0),
        elapsed(), limit_s,
        (unsigned long long)drained.cancelled, skipped);
}

}  // namespace luce::common
