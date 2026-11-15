// The pinned heads' files in the cache directory: a DiskPrefixCache view of its own, under the execution identity's
// head layout (kvflash_pinned_head.h), next to the model's prefix checkpoints and in the same budget.

#pragma once

#include "common/kvflash_pinned_head.h"
#include "server/kv/disk_prefix_cache.h"

#include <memory>
#include <string>

namespace luce::common {

// The head files' disk view never touches a checkpoint: a head goes in and out as host bytes (OwnedSnapshot).
class KvFlashNoCheckpoints final : public CheckpointStore {
public:
    int granularity() const override { return 1; }
    int restore_point_spacing() const override { return 0; }
    bool prefill_cuts_at_restore_points() const override { return false; }
    size_t estimate_bytes(int) const override { return 0; }
    bool used(Checkpoint) const override { return false; }
    CheckpointKind kind(Checkpoint) const override { return CheckpointKind::dense; }
    int position(Checkpoint) const override { return 0; }
    void free(Checkpoint) override {}
    Checkpoint staging() const override { return {}; }
    ModelBackend::SnapshotRef ref(Checkpoint) const override { return {}; }
    OwnedSnapshot copy(Checkpoint) const override { return {}; }
    OwnedSnapshot detach(Checkpoint) override { return {}; }
    bool adopt(Checkpoint, OwnedSnapshot &) override { return false; }
};

// The head view of `identity` over the open cache directory `dir`; null when the directory is not open or another
// server on it serves the same identity. The view holds the identity's own salt, so servers of different models
// keep their heads side by side.
inline std::unique_ptr<DiskPrefixCache> kvflash_open_head_view(const std::string & dir, CheckpointStore & store,
                                                               const std::string & identity) {
    DiskCacheConfig dc;
    dc.cache_dir = dir;
    dc.min_tokens = 1;
    auto disk = std::make_unique<DiskPrefixCache>(dc, store);
    disk->set_identity_salt(kvflash_head_layout_id(identity));
    if (!disk->init() || disk->disabled()) return nullptr;
    disk->use_layout(kvflash_head_layout_id(identity));
    return disk;
}

}  // namespace luce::common
