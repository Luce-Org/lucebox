// One model's view of its disk cache directory, and the disk scope
// policies (disk_prefix_cache.h).

#include "server/kv/disk_prefix_cache.h"
#include "common/sha1.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>

namespace luce::common {

namespace fs = std::filesystem;

const char * disk_prefix_cache_mode_name(DiskPrefixCacheMode mode) {
    switch (mode) {
        case DiskPrefixCacheMode::Off:   return "off";
        case DiskPrefixCacheMode::Full:  return "full";
        case DiskPrefixCacheMode::Fixed: return "fixed";
    }
    return "full";
}

std::string disk_prefix_cache_policy_name(const DiskPrefixCachePolicy & policy) {
    std::string base;
    if (policy.mode == DiskPrefixCacheMode::Fixed) {
        base = "fixed:" + std::to_string(policy.fixed_tokens);
    } else {
        base = disk_prefix_cache_mode_name(policy.mode);
    }
    return base;
}

bool parse_disk_prefix_cache_policy(const std::string & value,
                                    DiskPrefixCachePolicy & out) {
    std::string v;
    v.reserve(value.size());
    for (char c : value) v.push_back((char)std::tolower((unsigned char)c));

    if (v == "off" || v == "none" || v == "disabled") {
        out = {};
        out.mode = DiskPrefixCacheMode::Off;
        return true;
    }
    if (v == "full" || v == "full-prefix") {
        out = {};
        out.mode = DiskPrefixCacheMode::Full;
        return true;
    }
    // The `auto` scope (the prefix shared with recent prompts) is retired:
    // `full` keeps every boundary of every conversation, shared prefixes
    // included. Still accepted, as `full`, so old flags and requests work.
    const std::string auto_prefix = "auto:";
    bool retired_auto = v == "auto";
    if (!retired_auto && v.rfind(auto_prefix, 0) == 0) {
        char * end = nullptr;
        long n = std::strtol(v.c_str() + auto_prefix.size(), &end, 10);
        if (!end || *end != '\0' || n <= 0 || n > 1000000) return false;
        retired_auto = true;
    }
    if (retired_auto) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            std::fprintf(stderr,
                "[disk-cache] scope \"%s\" is retired; using \"full\"\n",
                value.c_str());
        }
        out = {};
        out.mode = DiskPrefixCacheMode::Full;
        return true;
    }

    char * end = nullptr;
    long n = std::strtol(v.c_str(), &end, 10);
    if (end && *end == '\0' && n > 0 && n <= 1000000) {
        out = {};
        out.mode = DiskPrefixCacheMode::Fixed;
        out.fixed_tokens = (int)n;
        return true;
    }
    return false;
}

bool apply_request_scope_override(DiskPrefixCachePolicy & server_policy,
                                  const std::string & scope_str) {
    DiskPrefixCachePolicy parsed;
    if (!parse_disk_prefix_cache_policy(scope_str, parsed)) {
        return false;
    }
    server_policy = parsed;
    return true;
}

static bool valid_boundary(int n, int full_len) {
    return n > 0 && n <= full_len;
}

std::vector<int> disk_prefix_cache_full_lookup_lengths(
        int prompt_len,
        const std::vector<int> & boundaries,
        int min_tokens) {
    std::vector<int> out;
    // Nothing shorter than the persistence minimum is ever written, so such
    // lengths are not worth a probe.
    if (prompt_len >= min_tokens) out.push_back(prompt_len);
    for (auto it = boundaries.rbegin(); it != boundaries.rend(); ++it) {
        if (*it >= min_tokens && *it < prompt_len) out.push_back(*it);
    }
    return out;
}

int disk_prefix_cache_fixed_boundary(const DiskPrefixCachePolicy & policy,
                                     int full_len,
                                     int min_tokens) {
    if (policy.mode != DiskPrefixCacheMode::Fixed) return 0;
    if (policy.fixed_tokens < min_tokens) return 0;
    return valid_boundary(policy.fixed_tokens, full_len) ? policy.fixed_tokens : 0;
}

static double mib(uint64_t bytes) { return (double)bytes / (1024.0 * 1024.0); }

// ─── Construction ───────────────────────────────────────────────────────

DiskPrefixCache::DiskPrefixCache(const DiskCacheConfig & cfg, CheckpointStore & store)
    : config_(cfg), store_(store) {}

DiskPrefixCache::DiskPrefixCache(const DiskCacheConfig & cfg, ModelBackend & backend)
    : config_(cfg),
      owned_store_(std::make_unique<BackendCheckpoints>(backend)),
      store_(*owned_store_) {}

DiskPrefixCache::~DiskPrefixCache() {
    // The store outlives this model while another one uses it; this
    // model's queued writes finish either way.
    drain();
}

bool DiskPrefixCache::init() {
    if (disabled()) return true;
    bool failed = false;
    disk_ = DiskStore::open(config_, clock_, free_space_, failed);
    if (!disk_) {
        // No writer: leaving the cache enabled would queue payloads forever
        // and block serving once backpressure is reached.
        config_.cache_dir.clear();
        return !failed;
    }
    std::lock_guard<std::mutex> lock(disk_->mu_);
    // Restore into the layout this identity used most recently; learn_layout
    // confirms it against the live model before anything is written.
    const DiskStore::DiskEntry * best = nullptr;
    for (const auto & e : disk_->entries_) {
        if (e.salt == identity_salt_ && (!best || e.last_used > best->last_used)) best = &e;
    }
    if (best) {
        layout_id_ = best->layout_id;
        layout_known_ = true;
        layout_from_disk_ = true;
        layout_dir_ = config_.cache_dir + "/" + hex_string(layout_id_.data(), 16);
    }
    std::fprintf(stderr,
                 "[disk-cache] initialized dir=%s budget=%.1f GB (configured %.1f GB) "
                 "ttl=%s files=%zu used=%.1f MB layout=%s\n",
                 config_.cache_dir.c_str(),
                 (double)disk_->effective_budget_ / (1024.0 * 1024.0 * 1024.0),
                 (double)disk_->config_.budget_bytes / (1024.0 * 1024.0 * 1024.0),
                 disk_->config_.ttl_seconds
                     ? (std::to_string(disk_->config_.ttl_seconds) + "s").c_str()
                     : "never",
                 disk_->entries_.size(), mib(disk_->total_bytes_),
                 layout_known_ ? hex_string(layout_id_.data(), 16).c_str() : "pending");
    return true;
}

DiskDrainResult DiskPrefixCache::drain(double max_seconds) {
    if (!disk_) return DiskDrainResult{};
    DiskStore::Owner owner;
    bool known = false;
    {
        std::lock_guard<std::mutex> lock(disk_->mu_);
        owner.layout = layout_id_;
        owner.salt = identity_salt_;
        known = layout_known_;
    }
    // Only this model's writes: another model sharing the directory keeps
    // its queue through this one's shutdown time limit.
    return known ? disk_->drain(max_seconds, &owner) : DiskDrainResult{};
}

uint64_t DiskPrefixCache::budget_bytes() const { return disk_ ? disk_->budget_bytes() : 0; }
size_t DiskPrefixCache::total_bytes() const { return disk_ ? disk_->total_bytes() : 0; }
size_t DiskPrefixCache::entry_count() const { return disk_ ? disk_->entry_count() : 0; }
DiskCacheStats DiskPrefixCache::stats() const { return disk_ ? disk_->stats() : DiskCacheStats{}; }

void DiskPrefixCache::set_clock(std::function<uint64_t()> clock) {
    clock_ = clock;
    if (disk_) disk_->clock_ = std::move(clock);
}

void DiskPrefixCache::set_free_space_probe(std::function<uint64_t()> probe) {
    free_space_ = probe;
    if (disk_) disk_->free_space_ = std::move(probe);
}


// ─── Layout fingerprint ─────────────────────────────────────────────────

std::array<uint8_t, 16> DiskPrefixCache::layout_of(ggml_context * ctx) const {
    // Collect tensor metadata sorted by name for deterministic fingerprint.
    struct TInfo { std::string name; uint32_t type; int64_t ne[4]; };
    std::vector<TInfo> tensors;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        TInfo ti;
        ti.name = t->name;
        ti.type = (uint32_t)t->type;
        ti.ne[0] = t->ne[0];
        ti.ne[1] = 1;  // normalize sequence-length dimension
        ti.ne[2] = t->ne[2];
        ti.ne[3] = t->ne[3];
        tensors.push_back(std::move(ti));
    }
    std::sort(tensors.begin(), tensors.end(), [](const TInfo & a, const TInfo & b) {
        return a.name < b.name;
    });
    // Build a single buffer and hash it.
    // Prepend identity_salt_ so that config/model differences (model file,
    // KV types, max_ctx, chat_template) rotate the layout_id independently of
    // tensor structure. All-zero salt (the default) is back-compatible.
    std::vector<uint8_t> buf;
    buf.insert(buf.end(), identity_salt_.begin(), identity_salt_.end());
    for (const auto & ti : tensors) {
        buf.insert(buf.end(), ti.name.begin(), ti.name.end());
        buf.insert(buf.end(), (uint8_t *)&ti.type, (uint8_t *)&ti.type + 4);
        buf.insert(buf.end(), (uint8_t *)ti.ne, (uint8_t *)ti.ne + 32);
    }
    uint8_t digest[20];
    sha1_hash(buf.data(), buf.size(), digest);
    std::array<uint8_t, 16> out{};
    std::memcpy(out.data(), digest, 16);
    return out;
}

void DiskPrefixCache::learn_layout_from(ggml_context * ctx) {
    const auto id = layout_of(ctx);
    std::lock_guard<std::mutex> lock(disk_->mu_);
    if (layout_known_ && !layout_from_disk_) return;  // verified from the live model
    if (layout_from_disk_ && id != layout_id_) {
        // The files stay indexed (budget/TTL) but are never restored.
        std::fprintf(stderr, "[disk-cache] layout mismatch: disk=%s model=%s — switching\n",
                     hex_string(layout_id_.data(), 16).c_str(), hex_string(id.data(), 16).c_str());
    }
    layout_id_ = id;
    layout_known_ = true;
    layout_from_disk_ = false;
    layout_dir_ = config_.cache_dir + "/" + hex_string(layout_id_.data(), 16);
    std::error_code ec;
    fs::create_directories(layout_dir_, ec);
    std::fprintf(stderr, "[disk-cache] layout learned: %s\n",
                 hex_string(layout_id_.data(), 16).c_str());
}

void DiskPrefixCache::learn_layout(Checkpoint checkpoint) {
    if (disabled()) return;
    if (disk_) {
        std::lock_guard<std::mutex> lock(disk_->mu_);
        if (layout_known_ && !layout_from_disk_) return;
    }
    // A memory-only backend is detected even before init().
    auto ref = store_.ref(checkpoint);
    if (!ref.ctx) {
        if (store_.used(checkpoint)) {
            backend_snapshot_unsupported_ = true;
            std::fprintf(stderr,
                         "[disk-cache] disabled: backend snapshots are "
                         "memory-only\n");
        }
        return;
    }
    if (ref.pooled) return;  // no dense KV tensors: would fingerprint a wrong layout
    if (disk_) learn_layout_from(ref.ctx);
}

// ─── Lookup ─────────────────────────────────────────────────────────────

OwnedSnapshot DiskPrefixCache::load(const std::vector<int32_t> & prompt_ids) {
    if (off()) return {};
    const PrefixHash hash = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());
    std::shared_ptr<DiskStore::PendingWrite> pending;
    std::array<uint8_t, 16> layout{};
    std::string path;
    uint64_t id = 0;
    uint32_t cur_pos = 0;
    {
        std::lock_guard<std::mutex> lock(disk_->mu_);
        if (!layout_known_) return {};
        layout = layout_id_;
        const int idx = find_entry(hash);
        if (idx < 0) return {};
        if (disk_->expired(disk_->entries_[(size_t)idx], disk_->now()) &&
            !disk_->pinned_locked(disk_->entries_[(size_t)idx])) {
            disk_->remove_entry((size_t)idx, "ttl expired");
            return {};
        }
        const DiskStore::DiskEntry & e = disk_->entries_[(size_t)idx];
        id = e.id;
        pending = e.pending;
        path = e.path;
        cur_pos = e.cur_pos;
    }
    OwnedSnapshot out;
    const auto t_read = std::chrono::steady_clock::now();
    if (pending) {
        // Still queued for the writer: copy it from memory. The shared
        // pointer keeps it alive even if the write finishes meanwhile.
        if (pending->key != prompt_ids) return {};
        out = OwnedSnapshot::clone_to_host(pending->snap.ctx(),
                                           pending->snap.cur_pos(),
                                           pending->snap.last_tok());
        if (out.empty()) return {};
    } else {
        out = disk_->read_file(path, prompt_ids, layout, identity_salt_);
    }
    std::lock_guard<std::mutex> lock(disk_->mu_);
    const int idx = disk_->find_entry_id(id);
    if (out.empty()) {
        ++disk_->stats_.load_failures;
        // A corrupt, partial or mismatched file is never retried.
        if (idx >= 0 && !disk_->entries_[(size_t)idx].pending) {
            disk_->remove_entry((size_t)idx, "lookup failed");
        }
        return {};
    }
    std::fprintf(stderr, "[disk-cache] read %s (%u pos, %.1f MB, %.2fs)\n",
                 pending ? "pending write" : fs::path(path).filename().string().c_str(),
                 cur_pos, mib(out.bytes()),
                 std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t_read).count());
    ++disk_->stats_.loads;
    if (idx >= 0) {
        // Persist last_used in the file so LRU order and the TTL survive a
        // restart. Not fsynced: losing it in a crash only ages the entry.
        disk_->touch_locked(disk_->entries_[(size_t)idx]);
    }
    return out;
}

bool DiskPrefixCache::adopt(OwnedSnapshot snap, Checkpoint checkpoint) {
    if (snap.empty()) return false;
    {
        // The first hit after startup confirms the layout read from disk.
        const auto id = layout_of(snap.ctx());
        std::lock_guard<std::mutex> lock(disk_->mu_);
        if (layout_from_disk_) {
            if (id != layout_id_) {
                std::fprintf(stderr,
                             "[disk-cache] adopted layout mismatch: disk=%s model=%s\n",
                             hex_string(layout_id_.data(), 16).c_str(),
                             hex_string(id.data(), 16).c_str());
                layout_id_ = id;
                layout_from_disk_ = false;
                layout_dir_ = config_.cache_dir + "/" + hex_string(layout_id_.data(), 16);
                std::error_code ec;
                fs::create_directories(layout_dir_, ec);
                return false;
            }
        }
    }
    if (!store_.adopt(checkpoint, snap)) {
        return false;  // `snap` still owns and frees the payload
    }
    std::lock_guard<std::mutex> lock(disk_->mu_);
    layout_from_disk_ = false;
    return true;
}

bool DiskPrefixCache::lookup(const std::vector<int32_t> & prompt_ids,
                             Checkpoint checkpoint) {
    return adopt(load(prompt_ids), checkpoint);
}

void DiskPrefixCache::for_each_aux(
        const std::function<void(const std::string &)> & visit) {
    if (off()) return;
    std::vector<std::pair<uint64_t, std::string>> files;
    {
        std::lock_guard<std::mutex> lock(disk_->mu_);
        for (const auto & e : disk_->entries_) {
            if (e.salt == identity_salt_ && !e.pending) files.push_back({e.last_used, e.path});
        }
    }
    std::sort(files.begin(), files.end());
    for (const auto & file : files) {
        std::string aux;
        if (DiskStore::read_aux(file.second, aux) && !aux.empty()) visit(aux);
    }
}

DiskPrefixCache::PrefixProbe DiskPrefixCache::longest_prefix(
        const std::vector<int32_t> & prompt_ids,
        const std::vector<int> * allowed_lengths) {
    PrefixProbe out;
    if (off()) return out;
    std::lock_guard<std::mutex> lock(disk_->mu_);
    if (!layout_known_) return out;
    const uint64_t t = disk_->now();
    std::vector<uint32_t> lengths;
    for (const auto & e : disk_->entries_) {
        if (e.layout_id != layout_id_ ||
            (disk_->expired(e, t) && !disk_->pinned_locked(e))) continue;
        if (e.token_count > prompt_ids.size()) continue;
        if ((int)e.token_count < config_.min_tokens) continue;
        if (allowed_lengths &&
            std::find(allowed_lengths->begin(), allowed_lengths->end(),
                      (int)e.token_count) == allowed_lengths->end()) continue;
        lengths.push_back(e.token_count);
    }
    std::sort(lengths.begin(), lengths.end(), std::greater<uint32_t>());
    lengths.erase(std::unique(lengths.begin(), lengths.end()), lengths.end());
    for (uint32_t len : lengths) {
        const int idx = find_entry(hash_prefix(prompt_ids.data(), (int)len));
        if (idx >= 0) {
            out.len = (int)len;
            out.cur_pos = (int)disk_->entries_[(size_t)idx].cur_pos;
            return out;
        }
    }
    return out;
}

int DiskPrefixCache::prune_ancestors(const std::vector<int32_t> & tokens,
                                     int keep_max) {
    if (off()) return 0;
    std::lock_guard<std::mutex> lock(disk_->mu_);
    if (!layout_known_) return 0;
    // The session's older states on disk: strict prefixes of `tokens`.
    std::vector<size_t> ancestors;
    bool head_kept = false;  // one at or below keep_max: the shared head
    size_t shallowest = SIZE_MAX;
    for (size_t i = 0; i < disk_->entries_.size(); ++i) {
        const DiskStore::DiskEntry & e = disk_->entries_[i];
        if (e.layout_id != layout_id_ || e.token_count >= tokens.size()) continue;
        if (e.pinned) continue;  // a pinned shared prefix stays
        if (hash_prefix(tokens.data(), (int)e.token_count) != e.token_hash) continue;
        if ((int)e.token_count <= keep_max) {
            head_kept = true;
            continue;
        }
        ancestors.push_back(i);
        if (shallowest == SIZE_MAX ||
            e.token_count < disk_->entries_[shallowest].token_count) shallowest = i;
    }
    // Without a short head on disk, the shallowest ancestor stands in for
    // it (a long system/tools head other sessions restore), as the RAM
    // cache keeps its shallowest entry.
    int removed = 0;
    std::sort(ancestors.begin(), ancestors.end(), std::greater<size_t>());
    for (const size_t i : ancestors) {
        if (!head_kept && i == shallowest) continue;
        disk_->remove_entry(i, "superseded by deeper session state");
        ++removed;
    }
    return removed;
}

std::string DiskPrefixCache::pin(const std::vector<int32_t> & tokens, bool on) {
    if (off() || tokens.empty()) return {};
    std::lock_guard<std::mutex> lock(disk_->mu_);
    if (!layout_known_) return {};
    const PrefixHash hash = hash_prefix(tokens.data(), (int)tokens.size());
    const int idx = find_entry(hash);
    if (idx < 0) return {};
    disk_->set_pinned_locked(disk_->entries_[(size_t)idx], on);
    return hex_string(layout_id_.data(), 16) + "/" + hex_string(hash.data(), 16) + ".dkv";
}

std::string DiskPrefixCache::file_path(const std::vector<int32_t> & tokens) {
    if (off() || tokens.empty()) return {};
    std::lock_guard<std::mutex> lock(disk_->mu_);
    if (!layout_known_) return {};
    const PrefixHash hash = hash_prefix(tokens.data(), (int)tokens.size());
    if (find_entry(hash) < 0) return {};
    return hex_string(layout_id_.data(), 16) + "/" + hex_string(hash.data(), 16) + ".dkv";
}

std::vector<std::string> DiskPrefixCache::pinned_files() {
    std::vector<std::string> out;
    if (off()) return out;
    std::lock_guard<std::mutex> lock(disk_->mu_);
    if (!layout_known_) return out;
    const std::string layout = hex_string(layout_id_.data(), 16);
    for (const auto & e : disk_->entries_) {
        if (e.layout_id != layout_id_ || !disk_->pinned_locked(e)) continue;
        out.push_back(layout + "/" + hex_string(e.token_hash.data(), 16) + ".dkv");
    }
    return out;
}

void DiskPrefixCache::unpin_file(const std::string & relative) {
    if (off() || relative.empty()) return;
    const fs::path rel(relative);
    const std::string layout = rel.parent_path().filename().string();
    const std::string name = rel.filename().string();
    std::lock_guard<std::mutex> lock(disk_->mu_);
    for (auto & e : disk_->entries_) {
        const fs::path p(e.path);
        if (p.filename() == name && p.parent_path().filename() == layout) {
            disk_->set_pinned_locked(e, false);
        }
    }
}

std::string DiskPrefixCache::layout_name() const {
    if (!disk_) return {};
    std::lock_guard<std::mutex> lock(disk_->mu_);
    return layout_known_ ? hex_string(layout_id_.data(), 16) : std::string();
}

bool DiskPrefixCache::contains(const std::vector<int32_t> & tokens) {
    if (off() || tokens.empty()) return false;
    std::lock_guard<std::mutex> lock(disk_->mu_);
    return layout_known_ && find_entry(hash_prefix(tokens.data(), (int)tokens.size())) >= 0;
}

// ─── Save ───────────────────────────────────────────────────────────────

bool DiskPrefixCache::save_async(OwnedSnapshot snap,
                                 const std::vector<int32_t> & prompt_ids,
                                 const std::string & aux) {
    if (off() || snap.empty()) return false;
    if ((int)prompt_ids.size() < config_.min_tokens) return false;
    // A file is keyed by the hash of `prompt_ids` only, and a hit restores
    // the whole snapshot. A snapshot that extends past its key would be
    // restored for any prompt sharing just the key prefix, with unverified
    // tokens in between, so such files are never written.
    if (snap.cur_pos() > (int)prompt_ids.size() || snap.cur_pos() <= 0) {
        std::fprintf(stderr,
                     "[disk-cache] skip save: snapshot pos=%d does not fit a key of "
                     "%zu tokens\n", snap.cur_pos(), prompt_ids.size());
        return false;
    }
    learn_layout_from(snap.ctx());

    uint64_t payload = 0;
    size_t table_bytes = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(snap.ctx()); t;
         t = ggml_get_next_tensor(snap.ctx(), t)) {
        payload += ggml_nbytes(t);
        table_bytes += 2 + std::strlen(t->name) + 4 + 32 + 8;
    }
    const uint64_t est_size = DISK_CACHE_HEADER_SIZE + table_bytes +
        4ull * prompt_ids.size() + 4 + aux.size() + payload +
        DISK_CACHE_CHECKSUM_SIZE;
    const PrefixHash hash = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());

    std::unique_lock<std::mutex> lock(disk_->mu_);
    // Snapshots waiting for the writer hold host memory: bound it.
    if (disk_->pending_bytes_ > 0 && disk_->pending_bytes_ + est_size > disk_->config_.max_pending_bytes) {
        ++disk_->stats_.backpressure_waits;
        const auto t0 = std::chrono::steady_clock::now();
        disk_->done_cv_.wait(lock, [&] {
            return disk_->pending_bytes_ == 0 ||
                disk_->pending_bytes_ + est_size <= disk_->config_.max_pending_bytes;
        });
        std::fprintf(stderr,
                     "[disk-cache] waited %.2fs for the writer (queued writes over "
                     "%.0f MB)\n",
                     std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0).count(),
                     mib(disk_->config_.max_pending_bytes));
    }
    disk_->expire_locked();
    // Already on disk (or queued): count it as a use.
    const int existing = find_entry(hash);
    if (existing >= 0) {
        disk_->touch_locked(disk_->entries_[(size_t)existing]);
        return true;
    }
    // Make room first, so the budget holds while the file is being written.
    if (est_size > disk_->budget_locked()) {
        std::fprintf(stderr, "[disk-cache] skip save: %.1f MB exceeds budget\n",
                     mib(est_size));
        return false;
    }
    if (!disk_->make_room_locked(est_size)) {
        std::fprintf(stderr, "[disk-cache] skip save: %.1f MB does not fit beside "
                     "the pinned files\n", mib(est_size));
        return false;
    }

    auto job = std::make_shared<DiskStore::PendingWrite>();
    job->snap = std::move(snap);
    job->key = prompt_ids;
    job->aux = aux;
    job->path = make_path(hash);
    job->created_at = disk_->now();
    job->layout_id = layout_id_;
    job->salt = identity_salt_;

    DiskStore::DiskEntry entry;
    entry.id          = disk_->next_entry_id_++;
    entry.path        = job->path;
    entry.token_hash  = hash;
    entry.layout_id   = layout_id_;
    entry.token_count = (uint32_t)prompt_ids.size();
    entry.cur_pos     = (uint32_t)job->snap.cur_pos();
    entry.last_used   = job->created_at;
    entry.salt        = identity_salt_;
    entry.file_size   = est_size;
    entry.pending     = job;
    disk_->total_bytes_ += est_size;
    disk_->pending_bytes_ += est_size;
    disk_->queue_.push_back(entry.id);
    disk_->entries_.push_back(std::move(entry));
    lock.unlock();
    disk_->writer_cv_.notify_one();
    return true;
}

bool DiskPrefixCache::save(Checkpoint checkpoint, const std::vector<int32_t> & prompt_ids,
                           const std::string & aux) {
    if (off()) return false;
    learn_layout(checkpoint);
    if (off()) return false;
    if ((int)prompt_ids.size() < config_.min_tokens) return false;
    const auto ref = store_.ref(checkpoint);
    if (!ref.ctx) return false;
    // Pooled snapshots keep KV in a pager blob that is not a ggml tensor in
    // ref.ctx; a file written from ref.ctx would restore with no KV.
    if (ref.pooled) {
        std::fprintf(stderr, "[disk-cache] skip save: pooled (KVFlash) snapshot not persistable\n");
        return false;
    }
    if (ref.cur_pos > (int)prompt_ids.size()) {
        std::fprintf(stderr,
                     "[disk-cache] skip save: snapshot pos=%d exceeds key of "
                     "%zu tokens\n", ref.cur_pos, prompt_ids.size());
        return false;
    }
    {
        // Already on disk: no copy needed.
        const PrefixHash hash = hash_prefix(prompt_ids.data(), (int)prompt_ids.size());
        std::lock_guard<std::mutex> lock(disk_->mu_);
        const int existing = find_entry(hash);
        if (existing >= 0) {
            disk_->touch_locked(disk_->entries_[(size_t)existing]);
            return true;
        }
    }
    return save_async(OwnedSnapshot::clone_to_host(ref.ctx, ref.cur_pos, ref.last_tok),
                      prompt_ids, aux);
}

// ─── Helpers ────────────────────────────────────────────────────────────

std::string DiskPrefixCache::make_path(const PrefixHash & hash) const {
    return layout_dir_ + "/" + hex_string(hash.data(), 16) + ".dkv";
}

int DiskPrefixCache::find_entry(const PrefixHash & hash) const {
    for (int i = 0; i < (int)disk_->entries_.size(); ++i) {
        if (disk_->entries_[i].token_hash == hash &&
            disk_->entries_[i].layout_id == layout_id_) return i;
    }
    return -1;
}

}  // namespace luce::common
