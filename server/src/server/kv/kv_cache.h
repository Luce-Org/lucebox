// One cache manager per model: the prefix index (PrefixCache), the
// checkpoints holding the payloads (BackendCheckpoints) and the disk cache.
//
// A request goes through the same three steps:
//   match   the deepest prefix of its prompt the cache holds: an exact
//           entry, a RAM entry or a disk file read into the staging
//           checkpoint (a fixed-scope prefix is prefilled there first);
//   admit   the one checkpoint its prefill saves (its capture), and the
//           restore points its prefill is cut at (restore_points);
//   release the capture committed when the request produced output,
//           dropped when it did not.
// An entry leaving RAM is written to disk on the way out when it is worth
// keeping (evict, enforce_budget, flush).
//
// The single-sequence path runs match, admit and release around one backend
// generate, holding a KvLease in between. The concurrent scheduler runs
// match and admit as one plan for its SeqEngine (KvPlan), reads a disk
// state off its thread first (plan answers loading meanwhile), and
// releases as the engine reports the capture (captured).

#pragma once

#include "common/concurrency/seq_engine.h"
#include "server/kv/disk_prefix_cache.h"
#include "server/kv/checkpoint_store.h"
#include "server/kv/parallel_prefix_txn.h"
#include "server/kv/prefix_cache.h"
#include "server/kv/prefix_pins.h"
#include "server/kv/shared_prefixes.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace luce::common {

// A single-sequence request's hold on the cache, from match to release.
struct KvLease {
    DiskPrefixCachePolicy disk_policy;
    // What the request restores from: an inline or exact-prompt
    // checkpoint, or the disk-staging one after a disk hit.
    Checkpoint restore;
    // A state-only `restore` (PrefixCache::mark_state_only): the deeper
    // snapshot whose attention rows it restores with. Invalid otherwise.
    Checkpoint restore_kv;
    // Positions the backend restores: its own position, which may sit a
    // little before the entry's key (a chunk-aligned save).
    int prefix_len = 0;
    // The entry's key length; capture decisions compare against it.
    int logical_prefix_len = 0;
    bool using_restore = false;
    bool disk_hit = false;
    // A fixed disk scope's boundary (disk_prefix_cache_fixed_boundary).
    int scoped_cut = 0;
    // Exact-prompt entries are keyed by the served tokens after a token
    // rewrite (DiffPin), else by the client's.
    bool full_snap_key_effective = false;
    // The capture: an exact-prompt checkpoint at the end of the prompt...
    Checkpoint full_snap;
    int full_snap_pos = 0;
    bool full_snap_prepared = false;
    // ...or an inline one at snap_cut.
    PrefixCache::InlineReservation snap_reservation;
    Checkpoint snap;
    int snap_cut = 0;
    bool snap_prepared = false;
};

// What match needs to know about a request.
struct KvMatchQuery {
    // The tokens served.
    std::vector<int32_t> * prompt = nullptr;
    // The tokens the client sent (the exact-entry key without a rewrite).
    const std::vector<int32_t> * client_prompt = nullptr;
    bool key_by_served = false;
    // The disk scope after the prompt transforms (PFlash).
    DiskPrefixCachePolicy disk_policy;
    // A compressed view of the prompt (PFlash), and whether it is
    // served from a unified view, which is kept on disk too.
    bool compressed = false;
    bool unified_view = false;
    // An exact entry the prompt transforms already found (PFlash).
    Checkpoint exact_hit;
    int exact_hit_len = 0;
    // The client (User-Agent product) recorded with a pin.
    const std::string * client = nullptr;
    // A forced capture cut (a PPP pin): also a restore point of a scoped
    // prefill.
    int forced_cut = 0;
};

// What admit needs to know about a request.
struct KvAdmitQuery {
    // Capture inline even without a restore (a request with tools).
    bool capture_inline = false;
    // Capture the system/tools head until it is restored
    // (select_inline_snapshot_boundary's prefer_head).
    bool prefer_head = false;
    int forced_cut = 0;
    // The forced cut is the learned shared-prefix cut (it yields to the
    // deep-first rule; an explicit pin does not).
    bool learned_cut = false;
    // False for a prompt that never comes back (a tool-checks note pass):
    // no capture at all.
    bool keep_checkpoints = true;
    // An agent step outside a tool-checks pass: a request that ends with
    // tool results. It continues a generated turn, which the next request
    // continues the same way.
    bool agent_step = false;
};

using PrefixCaptureTxn = BasicPrefixCaptureTxn<PrefixCache::InlineReservation, SeqEngine>;

// A concurrent request's plan, from match to its engine admission.
struct KvPlan {
    // What the engine is asked to restore and capture.
    PrefixStorePlan plan;
    // The capture, held until the engine reports it (KvCache::captured).
    PrefixCaptureTxn capture;
    // The RAM entry it restores; -1 without one, or for a disk read.
    int restore_slot = -1;
    // It restores a disk read from the staging checkpoint.
    bool disk_restore = false;
};

// What plan needs to know about a concurrent request.
struct KvPlanQuery {
    // The request: it owns the one disk read in flight.
    const void * owner = nullptr;
    const char * request_id = "";
    const std::vector<int32_t> * prompt = nullptr;
    DiskPrefixCachePolicy disk_policy;
    // Capture the system/tools head until it is restored (a request with tools).
    bool prefer_head = false;
    int forced_cut = 0;
    bool learned_cut = false;  // see KvAdmitQuery::learned_cut
    // Never capture in place of a pinned prefix (a warm: warm_next_on_engine).
    bool spare_pins = false;
    // The prefix the queued requests share with this prompt
    // (HttpServer::queued_shared_prefix), asked only when it matters, and
    // the least it must add to be captured instead of the request's own cut.
    std::function<int()> queued_shared_prefix;
    int shared_prefix_min_tokens = 0;
};

class KvCache {
public:
    KvCache(PrefixCache & index, DiskPrefixCache & disk,
            BackendCheckpoints & checkpoints, int disk_keep_max_tokens,
            int disk_min_tokens)
        : index_(index), disk_(disk), checkpoints_(checkpoints),
          disk_keep_max_tokens_(disk_keep_max_tokens),
          disk_min_tokens_(disk_min_tokens) {}

    // ── The request protocol (single-sequence path) ─────────────────

    // The deepest prefix of the prompt the cache holds. RAM entries whose
    // payload is gone are dropped on the way, and a RAM hit is counted. A
    // disk hit, or a fixed-scope prefix prefilled for it, lands in the
    // staging checkpoint (disk_hit).
    KvLease match(const KvMatchQuery & query);
    // Choose the request's capture and bind it into `request`: the payload
    // its reservation replaces is written out and freed first. Nothing is
    // captured into the checkpoint the request restores from.
    void admit(KvLease & lease, const std::vector<int32_t> & prompt,
               const std::vector<int32_t> & client_prompt,
               const KvAdmitQuery & query, GenerateRequest & request);
    // Commit the capture when the request produced output and the backend
    // saved it within its cut, else drop it. A committed inline capture gets
    // a disk key when `disk_keyable` and the scope is full; returns it then,
    // else an invalid checkpoint.
    Checkpoint release(KvLease & lease, const std::vector<int32_t> & prompt,
                       const std::vector<int32_t> & client_prompt,
                       bool produced_output, bool disk_keyable);

    // A request's restore points (prefix_restore_points), on both serving
    // paths: its chat boundaries spaced for the backend, plus `cuts`. The
    // generation prompt's own boundary, where the request's capture goes,
    // stays only on a backend that cuts prefill at every restore point.
    std::vector<int> restore_points(const std::vector<int32_t> & prompt,
                                    std::initializer_list<int> cuts) const;

    // ── The request protocol (concurrent path) ──────────────────────

    // At the start of concurrent serving: the engine whose checkpoints the
    // cache plans. The disk cache takes part only when the engine keeps its
    // checkpoints in backend snapshot slots (logged when it does not).
    void attach_engine(EngineCheckpoints & engine);
    bool concurrent_disk() const { return concurrent_disk_; }
    // Whether a request is planned at all: the engine takes prefix plans,
    // RAM slots are on, and it has no images (tokens alone do not identify
    // an image).
    bool plans(bool images) const;
    enum class PlanStatus { ready, loading };
    // match and admit for the engine: the deepest RAM entry, a deeper disk
    // state (read off this thread: loading until it is in, then adopted
    // into staging), the capture and the restore points. A request that got
    // loading is planned again from scratch.
    PlanStatus plan(const KvPlanQuery & query, KvPlan & out);
    // After the engine's admission (planned or not): restore statistics, a
    // disk read's staging freed, the entries the engine invalidated dropped.
    // Returns a protocol error when the engine answered what the plan did
    // not ask; the capture is cancelled then, and when the engine refused it.
    std::string admitted(KvPlan & plan, const PrefixStoreAdmission & admission,
                         bool admission_ok, const char * request_id);
    // A RAM restore the engine carried out promotes its entry, and may pin
    // it (consider_pin; written by flush_pins).
    void restored(const KvPlan & plan, const PrefixStoreAdmission & admission,
                  const std::vector<int32_t> & prompt, const std::string & client);
    // The engine's capture event: commit or drop the capture; the payloads
    // that left RAM with it (the one it displaced, those its budget plan
    // evicted) are written out when worth keeping and freed; a saved entry
    // gets its disk key and `aux` (asked only then) under a full scope, and
    // the entries it supersedes are dropped.
    void captured(PrefixCaptureTxn & capture, const PrefixStoreEvent & event,
                  const std::vector<int32_t> & prompt, DiskPrefixCacheMode mode,
                  const std::function<std::string()> & aux);
    // The disk read in flight: dropped (its owner left, or another request
    // got the head of the line), reaped once done, joined at shutdown.
    void drop_read(const void * owner = nullptr);
    void reap_reads();
    void join_reads();
    // Dropped reads still running: an idle scheduler polls to reap them.
    bool reads_to_reap() const { return !finished_reads_.empty(); }
    ~KvCache() { join_reads(); }

    // ── Shared prefixes ─────────────────────────────────────────────

    // The shared prefixes and the pins this cache keeps (owned by the server).
    void attach_shared_prefixes(SharedPrefixes & shared, PrefixPins & pins);
    // After `prompt` restored the RAM entry in `slot`: a pinned prefix
    // counts the hit; a shared prefix restored by another conversation is
    // pinned, at a chat boundary at once, inside a message once its cut
    // recurs (SharedPrefixes::recurrences). Only while the shared prefixes
    // learn, and only an entry with a disk key. Its KV goes to disk at the
    // next flush_pins.
    void consider_pin(int slot, const std::vector<int32_t> & prompt, const std::string & client);
    // Write the KV of the prefixes pinned since the last call (after the
    // response, or when the scheduler is idle), drop pins idle for too
    // long, and save the pins file when due (`force`: now).
    void flush_pins(bool force = false);
    // Drop pin `id` (the HTTP DELETE; any thread): its disk file falls under
    // the TTL again, and its RAM entry is unpinned at the next flush_pins.
    bool unpin(const std::string & id);
    // Remember a request's prefix region in the ring (SharedPrefixes), and
    // in the pins file with it.
    void remember_prefix(const std::vector<int32_t> & prompt, const std::vector<int> & boundaries);

    // At startup. A pin with a disk file for these settings is served by
    // disk hits. On the concurrent path the others (an engine without a
    // disk tier, or after a settings change) are queued to warm, within
    // `ram_cap` bytes of pins in RAM. A pin that does not fit in `max_ctx`
    // (> 0) is neither (it stays for settings that can serve it). This
    // layout's pinned files no pin lists are unpinned (a crash between the
    // mark and the catalog's save).
    void load_pins(size_t ram_cap, int max_ctx);
    // Idle work on the concurrent path, while no sequence is live: the next
    // queued pin's prompt goes through plan/admitted/captured on the engine
    // alone, which captures it at its cut as a request's prefill would (the
    // prompt runs one token past the cut, so the capture lies inside it),
    // and the sequence retires at the capture. `cancel` is asked between
    // engine steps: one step can be a whole head (DeepSeek's staged prefill
    // runs up to 2048 rows), so a request arriving meanwhile waits for it.
    // Each pin is tried once per process; a preempted warm is dropped. False
    // when nothing is queued, or the warm was preempted.
    bool warm_next_on_engine(const CancellationProbe & cancel);
    bool warm_pending() const { return !to_warm_.empty(); }

    // ── Captures outside a request's prefill (a generated turn, a replay) ──

    // The checkpoint `reservation` captures into, emptied for the backend
    // to save into: the entry it replaces is written out first.
    Checkpoint clear_for_capture(const PrefixCache::InlineReservation & reservation);
    // Commit what the backend saved for `reservation` at `saved_pos`: the
    // further entries its budget plan evicts are written out and freed,
    // then the entry is keyed at `saved_pos` (key_at_saved) or at its
    // target cut. Then the resident budget is enforced, after dropping the
    // entries the new one supersedes when `prune`.
    void commit_capture(PrefixCache::InlineReservation & reservation,
                        const std::vector<int32_t> & ids, int saved_pos,
                        bool key_at_saved, bool prune);

    // ── Entries leaving RAM ─────────────────────────────────────────

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
    // Free the state-only backups whose deeper snapshot left RAM.
    void free_orphaned_state_backups();

    // Graceful shutdown: write out every entry worth keeping, most recently
    // used first, while `limit_s` (counted from `started`) lasts; < 0: no
    // limit. Waits for the writer within the same limit.
    void flush(const char * reason, double limit_s,
               std::chrono::steady_clock::time_point started);

private:
    bool write_out_slot(const PrefixCache::Evicted & entry, const char * reason);
    // The disk part of match: a deeper state on disk read into staging, or
    // a fixed-scope prefix prefilled there.
    void match_disk(KvLease & lease, const KvMatchQuery & query);
    void prefill_scoped(KvLease & lease, const std::vector<int32_t> & prompt,
                        int forced_cut);

    PrefixCache & index_;
    DiskPrefixCache & disk_;
    BackendCheckpoints & checkpoints_;
    int disk_keep_max_tokens_;
    int disk_min_tokens_;

    // Shared prefixes.
    SharedPrefixes * shared_ = nullptr;
    PrefixPins * pins_ = nullptr;
    struct PendingPin {
        int slot = -1;
        std::vector<int32_t> ids;
        std::vector<int32_t> disk_key;
        std::string aux;
        std::string client;
    };
    std::vector<PendingPin> pending_pins_;
    std::vector<std::string> to_warm_;  // pin ids, most recently hit first
    size_t pin_ram_cap_ = 0;            // load_pins' allowance for warmed pins
    void unpin_files(const SharedPrefixPin & pin);
    // A pin (or some of its files) dropped at the caps or by age.
    void drop_pinned(const SharedPrefixPin & dropped);
    // This layout's file keyed by `key`, for a pin; its path is empty when
    // the file is not there. The file is marked pinned on disk only once
    // the catalog lists it (disk_.pin), so a concurrent unpin of another
    // pin sharing it (PrefixPins::remove) cannot clear a mark it still needs.
    SharedPrefixPin::File pin_file(const std::vector<int32_t> & key);
    // The disk tier serves this model's checkpoints (pins have files).
    bool pins_have_files() const { return !disk_.disabled() && (!engine_ || concurrent_disk_); }
    // A warmed prefix's file, added to its pin (written to disk when the
    // engine has a tier). False when the pin is gone meanwhile (deleted, or
    // dropped at another model's caps): the caller unpins its RAM entry.
    bool record_warm(const SharedPrefixPin & pin, Checkpoint checkpoint, int saved,
                     double seconds, const char * where);
    uint64_t next_warm_id_ = 1;
    // The pin whose file (this layout) is keyed by `key`: a disk hit on it
    // counts as the pin's hit.
    std::optional<SharedPrefixPin> pinned_file(const std::vector<int32_t> & key) const;
    // The reservation would replace a pinned prefix.
    bool takes_a_pin(const PrefixCache::InlineReservation & reservation) const;
    // A checkpoint's bytes as the index counts them: the engine's estimate
    // on the concurrent path, the backend's on the single one.
    size_t estimate_bytes(int tokens) const;

    // Concurrent path.
    EngineCheckpoints * engine_ = nullptr;
    bool concurrent_disk_ = false;
    uint64_t next_capture_id_ = 1;
    struct DiskRead {
        const void * owner = nullptr;
        struct Result {
            std::atomic<bool> done{false};
            OwnedSnapshot snap;
        };
        std::shared_ptr<Result> result;
        std::thread reader;
    };
    std::optional<DiskRead> read_;
    std::vector<DiskRead> finished_reads_;
};

}  // namespace luce::common
