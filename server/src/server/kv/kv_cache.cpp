#include "server/kv/kv_cache.h"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace luce::common {

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
    if (slot >= 0) write_out_slot(index_.entry(slot), reason);
}

void KvCache::evict_extra_victims(const PrefixCache::InlineReservation & reservation) {
    for (const int slot : reservation.extra_victim_slots()) {
        evict(slot, "ram eviction");
        checkpoints_.free(Checkpoint::of_cache_slot(slot));
    }
}

void KvCache::trim_after_commit(int slot) {
    for (const int released : index_.prune_superseded_ancestors(slot)) {
        checkpoints_.free(Checkpoint::of_cache_slot(released));
    }
    enforce_budget(slot);
}

void KvCache::enforce_budget(int keep_slot) {
    for (const auto & entry : index_.enforce_resident_budget(keep_slot)) {
        write_out_slot(entry, "ram eviction");
        checkpoints_.free(Checkpoint::of_cache_slot(entry.slot));
    }
}

void KvCache::flush(const char * reason, double limit_s,
                    std::chrono::steady_clock::time_point started) {
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
