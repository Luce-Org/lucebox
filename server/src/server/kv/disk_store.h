// The files of one KV cache directory, shared by every model of the process
// that uses it (DiskStore), and their format. DiskPrefixCache
// (disk_prefix_cache.h) is one model's view of them.
//
// File format (v5): 96-byte header + tensor table + the exact key token ids
// + an auxiliary blob (u32 length + bytes; the server stores the session's
// tool-call replay text there) + raw tensor data + an 8-byte checksum of
// everything before it (the header with last_used zeroed, since a hit
// rewrites that field in place). Tensor bytes are copied verbatim in the
// cache's own type (e.g. q4_0 KV stays q4_0): no requantization on save or
// restore. Files are keyed by SHA-1 of prompt token IDs (same as in-memory
// cache); a hit also compares the stored token ids with the prompt before
// adopting anything, and verifies the checksum over the whole file: a
// corrupt or partial file is a logged miss and is deleted. A layout
// fingerprint (SHA-1 of an identity salt plus the tensor names/types/shapes)
// and the identity salt itself prevent loading snapshots from another
// model, KV type or config.
//
// Writes are asynchronous: DiskPrefixCache::save_async() takes ownership of
// a snapshot that left RAM, indexes it at once (it counts against the budget
// and a lookup serves it from memory until it is on disk), and a writer
// thread writes it (temporary file, fdatasync, rename, directory sync).
//
// Directory structure:
//   <cache_dir>/<layout_fingerprint_hex>/<token_hash_hex>.dkv
//
// Every layout directory is indexed for the disk budget and the TTL; only
// the live layout's files are ever restored.

#pragma once

#include "common/model_backend.h"
#include "server/kv/prefix_hash.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace luce::common {

// ─── Configuration ──────────────────────────────────────────────────────

struct DiskCacheConfig {
    std::string cache_dir;          // base directory (empty = disabled)
    size_t      budget_bytes = (size_t)128 * 1024 * 1024 * 1024;  // 128 GiB default
    int         min_tokens   = 512; // only persist snapshots >= this many tokens
    // Files unused for longer than this are deleted (at startup, before each
    // save, and when a lookup meets one). 0 = never expire.
    uint64_t    ttl_seconds = 72ull * 3600;
    // Snapshots handed to save_async() but not yet on disk hold host memory.
    // Past this many bytes, save_async() waits for the writer (logged).
    uint64_t    max_pending_bytes = 4ull * 1024 * 1024 * 1024;
    // The budget never exceeds what fits on the filesystem: the cache's own
    // bytes plus the free space, minus this reserve or 5% of the disk (at
    // most 16 GiB), whichever is larger.
    uint64_t    free_space_reserve_bytes = 1ull * 1024 * 1024 * 1024;
};

// Debug aid for the SSD-vs-RAM numerics check: XXH64 over every tensor of a
// snapshot (name, type, shape, bytes) plus cur_pos and last_tok. Enabled at
// restore time by LUCE_DISK_CACHE_DIGEST=1; costs a full device-to-host copy.
bool     disk_cache_digest_enabled();
uint64_t snapshot_state_digest(const ModelBackend::SnapshotRef & ref,
                               int * n_tensors = nullptr,
                               uint64_t * n_bytes = nullptr);

// ─── File header (96 bytes, little-endian) ──────────────────────────────

struct DiskCacheHeader {
    char     magic[4];          // "DKVC"
    uint32_t version;           // DISK_CACHE_VERSION
    uint8_t  layout_id[16];    // SHA-1 truncated: tensor structure fingerprint
    uint32_t cur_pos;
    uint32_t n_tensors;
    uint32_t token_count;      // number of prompt tokens
    uint8_t  token_hash[16];   // SHA-1 of prompt token IDs (same as PrefixHash)
    uint64_t payload_bytes;    // total tensor data bytes
    uint64_t created_at;       // unix seconds
    uint64_t last_used;        // unix seconds (updated on hit) | DISK_CACHE_PINNED_BIT
    int32_t  last_tok;         // last prefill token (needed for decode seeding)
    uint8_t  identity_salt[16]; // model/KV-type/config identity (v4)
};
// NOTE: The header is serialized field-by-field, not as a raw struct,
// to avoid alignment/packing issues. The on-disk size is exactly 96 bytes.

static constexpr size_t DISK_CACHE_HEADER_SIZE = 96;
// Byte offset of last_used, rewritten in place when a hit touches a file.
static constexpr long DISK_CACHE_LAST_USED_OFFSET = 68;
// The top bit of the stored last_used marks a pinned file (a pinned shared
// prefix, kv/prefix_pins.h): exempt from the TTL, the LRU and ancestor
// pruning. Rewritten in place like last_used, outside the checksum; a
// reader without it sees a file used in the far future.
static constexpr uint64_t DISK_CACHE_PINNED_BIT = 1ull << 63;
// v3: ggml_type id 42 changed meaning (TQ3_0 -> Q2_0).
// v4: identity salt in the header and the key token ids after the tensor
// table.
// v5: an 8-byte checksum trailer over the whole file. Older files are
// deleted at startup.
static constexpr uint32_t DISK_CACHE_VERSION = 5;
static constexpr size_t DISK_CACHE_CHECKSUM_SIZE = 8;

// ─── Tensor table entry (on-disk) ──────────────────────────────────────

struct DiskTensorEntry {
    std::string name;
    uint32_t    type;          // ggml_type enum
    int64_t     ne[4];
    size_t      nbytes;
};

struct DiskCacheStats {
    uint64_t writes = 0;          // files written
    uint64_t write_bytes = 0;
    uint64_t write_failures = 0;
    uint64_t write_cancelled = 0; // removed (budget, TTL, superseded) before written
    uint64_t pending_files = 0;
    uint64_t pending_bytes = 0;
    uint64_t backpressure_waits = 0;
    uint64_t loads = 0;           // hits served (file or queued write)
    uint64_t load_failures = 0;   // corrupt, partial or mismatched files
};

// Wait until every queued write is on disk, at most `max_seconds`
// (< 0: no limit). Writes still queued then are cancelled and the one
// in progress is abandoned (its temporary file is deleted).
struct DiskDrainResult {
    uint64_t files = 0;       // written during the drain
    uint64_t bytes = 0;
    uint64_t cancelled = 0;   // dropped by the time limit
    double   seconds = 0.0;
};

// ─── DiskStore ──────────────────────────────────────────────────────────

// The files of one cache directory, shared by every model of this process
// that uses it (a DiskPrefixCache is one model's view of it): one lock on
// <dir>/.lock against other processes, one index of every layout's files,
// one writer thread, one budget and TTL. Startup deletes temporaries and the
// budget evicts files of every layout, which is why one process owns the
// directory. Everything below open() is the DiskPrefixCache's to use, under
// mu_.
class DiskStore {
public:
    // The store of `config.cache_dir`: the one this process already has
    // open, or a new one, which takes the directory's lock, indexes every
    // file (deleting corrupt, partial and outdated ones and leftover
    // temporaries), applies the TTL and the budget and starts the writer.
    // Null when another process holds the lock or, with `failed` set, when
    // the directory cannot be created. The clock and free-space probe
    // (tests) are a new store's.
    static std::shared_ptr<DiskStore> open(const DiskCacheConfig & config,
                                           std::function<uint64_t()> clock,
                                           std::function<uint64_t()> free_space,
                                           bool & failed);
    // Finishes the queued writes, stops the writer, releases the lock.
    ~DiskStore();
    DiskStore(const DiskStore &) = delete;
    DiskStore & operator=(const DiskStore &) = delete;

    // One model's files: its layout and identity salt.
    struct Owner {
        std::array<uint8_t, 16> layout{};
        std::array<uint8_t, 16> salt{};
    };
    // With `only`, waits for (and on the time limit cancels) that model's
    // writes only; the other models' stay queued.
    DiskDrainResult drain(double max_seconds = -1.0, const Owner * only = nullptr);
    uint64_t budget_bytes() const;
    size_t total_bytes() const;
    size_t entry_count() const;
    DiskCacheStats stats() const;

    // A snapshot waiting for the writer. Shared with the index entry so a
    // lookup can copy it while the writer is busy with it.
    struct PendingWrite {
        OwnedSnapshot snap;
        std::vector<int32_t> key;
        std::string aux;
        std::string path;
        uint64_t created_at = 0;
        uint64_t last_used = 0;  // the entry's, when the write started
        bool pinned = false;     // likewise
        std::array<uint8_t, 16> layout_id{};
        std::array<uint8_t, 16> salt{};  // the writing model's identity salt
    };

    // In-memory index of on-disk files (every layout directory).
    struct DiskEntry {
        uint64_t    id = 0;
        std::string path;
        PrefixHash  token_hash;
        std::array<uint8_t, 16> layout_id{};
        std::array<uint8_t, 16> salt{};  // the identity salt it was written with
        uint32_t    token_count = 0;
        uint32_t    cur_pos     = 0;
        uint64_t    file_size   = 0;
        uint64_t    last_used   = 0;
        bool        pinned      = false;  // DISK_CACHE_PINNED_BIT
        std::shared_ptr<PendingWrite> pending;  // non-null until on disk
    };

    const DiskCacheConfig config_;  // the first opener's; its dir, budget, TTL
    std::function<uint64_t()> clock_;
    std::function<uint64_t()> free_space_;

    std::vector<DiskEntry> entries_;
    size_t total_bytes_ = 0;
    uint64_t next_entry_id_ = 1;
    uint64_t effective_budget_ = 0;
    mutable std::mutex mu_;
    // Writer thread and its queue (entry ids, oldest first).
    std::deque<uint64_t> queue_;
    uint64_t writing_id_ = 0;       // entry being written, 0 = none
    std::atomic<bool> abort_write_{false};  // drain() time limit hit
    bool     stop_writer_ = false;
    uint64_t pending_bytes_ = 0;
    std::condition_variable writer_cv_;   // work for the writer
    std::condition_variable done_cv_;     // a write finished
    DiskCacheStats stats_;
    bool pinned_locked(const DiskEntry & e) const { return e.pinned; }
    // Pin or unpin `e`, rewriting its header in place when it is on disk.
    void set_pinned_locked(DiskEntry & e, bool on);

    // Helpers (the _locked ones need mu_).
    uint64_t now() const;
    bool expired(const DiskEntry & e, uint64_t t) const;
    void remove_entry(size_t idx, const char * why);
    void expire_locked();
    uint64_t budget_locked();
    // Evict least-recently-used files until `incoming` more bytes fit.
    // Pinned files are never evicted: false when they leave no room.
    bool make_room_locked(uint64_t incoming);
    void release_pending_locked(DiskEntry & e);
    void touch_locked(DiskEntry & e);
    int find_entry_id(uint64_t id) const;
    // The snapshot in `path`, checked against its key, `layout_id` and the
    // reading model's identity `salt`; a bad file is rejected (logged).
    OwnedSnapshot read_file(const std::string & path,
                            const std::vector<int32_t> & prompt_ids,
                            const std::array<uint8_t, 16> & layout_id,
                            const std::array<uint8_t, 16> & salt);
    static bool write_header(FILE * f, const DiskCacheHeader & hdr);
    static bool read_header(FILE * f, DiskCacheHeader & hdr);
    // The auxiliary blob of the file at `path` (reads only its metadata).
    static bool read_aux(const std::string & path, std::string & aux);

private:
    explicit DiskStore(const DiskCacheConfig & config);
    bool lock_dir();
    void scan_all();
    void writer_loop();
    bool write_file(const std::string & path, const PendingWrite & job);

    std::thread writer_;
#if defined(_WIN32)
    void * lock_handle_ = nullptr;
#else
    int lock_fd_ = -1;
#endif
};

}  // namespace luce::common
