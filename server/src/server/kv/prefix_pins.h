// Pinned shared prefixes: the shared-prefix entries the cache keeps across
// restarts.
//
// A shared-prefix entry (PrefixCache's protect kind) is pinned the first time
// another conversation restores it: one cut at a chat boundary at once, one
// cut inside a message once the cut recurs in SharedPrefixes' ring. Its KV is
// written to disk, the file is marked pinned in its header (exempt from the
// disk cache's TTL, LRU and pruning), and the pin is listed here as token
// ids, so a restart (or a model swap back) restores it at once, and a
// settings change warms it again from the ids.
//
// One catalog per token identity (tokenizer and chat template) and cache
// directory, shared by every model of the process with that identity
// (PrefixPins::open): <kv-cache-dir>/pins/<identity>.json, written
// atomically and fsynced, holding the pins and the ring. A pin keeps one
// disk file per model layout. Caps: max_pins and max_bytes, the least
// recently hit pin dropped first; a pin with no hit for unpin_after seconds
// is dropped too, and its files fall under the normal TTL again.
// Thread-safe (HTTP threads list and delete pins).

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace luce::common {

struct SharedPrefixPin {
    std::string id;                // hex of the prefix hash (PrefixPins::id_of)
    std::vector<int32_t> tokens;   // the prefix, up to its cut
    std::string source;            // learned | prewarm
    std::string client;            // the User-Agent product that pinned it
    uint64_t hits = 0;
    uint64_t created = 0;          // unix seconds
    uint64_t last_hit = 0;
    // Its KV on disk, one file per model layout (several models of one
    // process share a catalog; a settings change adds a layout).
    struct File {
        std::string layout;        // DiskPrefixCache::layout_name()
        int disk_tokens = 0;       // the file's key: the first disk_tokens ids
        std::string path;          // relative to the cache dir
        uint64_t bytes = 0;        // snapshot bytes (estimate)
        uint64_t last_hit = 0;
    };
    std::vector<File> files;
    const File * file_for(const std::string & layout) const;
    uint64_t bytes() const;
};

class PrefixPins {
public:
    struct Limits {
        size_t max_pins = 16;
        uint64_t max_bytes = 16ull << 30;
        uint64_t unpin_after_s = 14ull * 24 * 3600;
        size_t max_files = 3;      // layouts kept per pin, most recently hit
    };

    // The process's catalog at `path`, opened (and read) once, shared by
    // every model with that identity. `path` empty: one in memory only.
    static std::shared_ptr<PrefixPins> open(const std::string & path, Limits limits,
                                            std::function<uint64_t()> clock = {});
    const std::string & path() const { return path_; }
    // The ring saved with the pins (SharedPrefixes').
    std::vector<std::vector<int32_t>> ring() const;
    void set_ring(std::vector<std::vector<int32_t>> ring);

    // Write the catalog atomically (fsync, rename, fsync of the directory)
    // when anything changed since the last save, or `force`. False on a
    // write error (logged).
    bool save(bool force = false);
    // Something changed, at least `seconds` ago (hits are written lazily).
    bool due(uint64_t seconds) const;

    // Add `pin`, or merge it into the pin with its id (its files by layout).
    // Returns the pins and files dropped: at the caps, or a layout's file
    // replaced by another. A returned pin lists only the files to unpin on
    // disk: never one a pin still listed references (two cuts can share a
    // file, Qwen saving both at one chunk start).
    std::vector<SharedPrefixPin> add(SharedPrefixPin pin);
    // Add `file` to the pin `id`, as add() merges it. Nullopt when the pin
    // is gone (deleted, or dropped by another model's caps): nothing added.
    std::optional<std::vector<SharedPrefixPin>> attach(const std::string & id,
                                                       SharedPrefixPin::File file);
    bool has(const std::string & id) const;
    std::optional<SharedPrefixPin> get(const std::string & id) const;
    // The pin whose `layout` file is keyed by exactly `key`.
    std::optional<SharedPrefixPin> find_by_disk_key(const std::vector<int32_t> & key,
                                                    const std::string & layout) const;
    // A restore of the pin (from RAM or from its `layout` file).
    void hit(const std::string & id, const std::string & layout = {});
    // The removed pin, listing only the files to unpin (as add()).
    std::optional<SharedPrefixPin> remove(const std::string & id);
    // Drop the pins without a hit for unpin_after seconds (as remove()).
    std::vector<SharedPrefixPin> expire();
    // Most recently hit first.
    std::vector<SharedPrefixPin> list() const;
    size_t size() const;

    static std::string id_of(const std::vector<int32_t> & tokens);

private:
    PrefixPins(std::string path, Limits limits, std::function<uint64_t()> clock);
    void load_locked();
    uint64_t now() const;
    void mark_dirty_locked();
    std::vector<SharedPrefixPin> enforce_caps_locked();
    // Merge `files` into `into` by layout; a replaced file goes to `dropped`.
    void merge_files_locked(SharedPrefixPin & into, std::vector<SharedPrefixPin::File> files,
                            std::vector<SharedPrefixPin> & dropped);
    // Keep in `dropped` only the files no pin listed references.
    void keep_unreferenced_locked(std::vector<SharedPrefixPin> & dropped) const;
    mutable std::mutex mu_;
    // One save at a time, from its snapshot to its rename: a later save
    // never publishes an older catalog.
    std::mutex save_mu_;
    const std::string path_;
    const Limits limits_;
    const std::function<uint64_t()> clock_;
    std::vector<SharedPrefixPin> pins_;
    std::vector<std::vector<int32_t>> ring_;
    bool dirty_ = false;
    uint64_t dirty_since_ = 0;
};

}  // namespace luce::common
