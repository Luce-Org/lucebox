// The cache operations both serving paths share.
//
// For now: what happens when a checkpoint leaves RAM. An entry the index
// says to keep (PrefixCache::persist_on_evict) and that has a disk key is
// handed to the disk writer, then its payload is freed; the shutdown flush
// is the same thing for every entry. The request protocol (match, admit,
// checkpoint, release) joins it later.

#pragma once

#include "server/disk_prefix_cache.h"
#include "server/kv/checkpoint_store.h"
#include "server/prefix_cache.h"

#include <chrono>

namespace luce::common {

class KvCache {
public:
    KvCache(PrefixCache & index, DiskPrefixCache & disk,
            BackendCheckpoints & checkpoints, int disk_keep_max_tokens)
        : index_(index), disk_(disk), checkpoints_(checkpoints),
          disk_keep_max_tokens_(disk_keep_max_tokens) {}

    // Whether `entry` goes to disk when it leaves RAM: the index keeps it
    // and it has a disk key, and the disk cache is on.
    bool worth_writing(const PrefixCache::Evicted & entry) const;

    // Hand `payload`, the state of `entry`, to the disk writer when it is
    // worth writing, and drop the older files of its session
    // (DiskPrefixCache::prune_ancestors). Returns true when queued.
    bool write_out(const PrefixCache::Evicted & entry, OwnedSnapshot payload,
                   const char * reason);

    // Single-sequence path, where payloads stay in their backend slots.
    //
    // The entry in `slot` is about to be replaced or dropped: write it out
    // first. The slot is left empty when it was written; the caller frees it.
    void evict(int slot, const char * reason);
    // Just before `reservation` commits: write out and free the further
    // entries its resident-budget plan evicts (extra_victim_slots). The
    // commit then drops them from the index.
    void evict_extra_victims(const PrefixCache::InlineReservation & reservation);
    // After a commit in `slot`: free what it supersedes, then evict to the
    // resident budget (enforce_budget).
    void trim_after_commit(int slot);
    // Evict, writing out what is worth keeping, until the resident bytes
    // fit the budget again; never `keep_slot`.
    void enforce_budget(int keep_slot);

    // Graceful shutdown: write out every entry worth keeping, most recently
    // used first, while `limit_s` (counted from `started`) lasts; < 0: no
    // limit. Waits for the writer within the same limit.
    void flush(const char * reason, double limit_s,
               std::chrono::steady_clock::time_point started);

private:
    bool write_out_slot(const PrefixCache::Evicted & entry, const char * reason);

    PrefixCache & index_;
    DiskPrefixCache & disk_;
    BackendCheckpoints & checkpoints_;
    int disk_keep_max_tokens_;
};

}  // namespace luce::common
