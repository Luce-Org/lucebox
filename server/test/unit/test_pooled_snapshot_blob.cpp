// PrefixSnapshot carries a pooled-KVFlash flag plus the pager blob (patch 2).
// The snapshot struct is the storage layer reachable without a GPU/model:
// defaults must be unpooled, and copy/move must preserve the blob. The disk
// prefix cache keys its refusal of pooled snapshots off SnapshotRef::pooled.
#include "CppUnitTestFramework.hpp"
#include "../../src/internal.h"
#include "../../src/common/model_backend.h"

#include <cstdint>
#include <utility>
#include <vector>

using luce::common::PrefixSnapshot;

namespace {
struct PooledSnapshotFixture {};

std::vector<uint8_t> make_blob() {
    std::vector<uint8_t> b(257);
    for (size_t i = 0; i < b.size(); ++i) b[i] = (uint8_t)(i * 7 + 3);
    return b;
}
}  // namespace

TEST_CASE(PooledSnapshotFixture, defaults_are_unpooled_and_empty) {
    PrefixSnapshot s;
    CHECK_FALSE(s.is_pooled);
    CHECK(s.kvflash_blob.empty());
    luce::common::ModelBackend::SnapshotRef ref;
    CHECK_FALSE(ref.pooled);
}

TEST_CASE(PooledSnapshotFixture, copy_and_move_preserve_blob) {
    PrefixSnapshot s;
    s.is_pooled = true;
    s.cur_pos = 1234;
    s.kvflash_blob = make_blob();

    PrefixSnapshot c = s;
    CHECK(c.is_pooled);
    CHECK(c.kvflash_blob == make_blob());

    PrefixSnapshot m = std::move(c);
    CHECK(m.is_pooled);
    CHECK(m.cur_pos == 1234);
    CHECK(m.kvflash_blob == make_blob());
}

// #839 hazard: the RAM->SSD handoff (snapshot_detach / snapshot_take_displaced)
// turns a PrefixSnapshot into an OwnedSnapshot, which carries ggml tensors
// only. A pooled snapshot has no dense KV tensors and its pager blob cannot
// ride along, so a file written from it would restore with no KV. It must
// never be persisted, with or without a blob.
TEST_CASE(PooledSnapshotFixture, pooled_snapshot_is_never_disk_persistable) {
    using Layout = PrefixSnapshot::Layout;
    PrefixSnapshot s;
    s.layout = Layout::dense;
    CHECK(s.disk_persistable());

    s.is_pooled = true;                       // pooled, blob missing
    CHECK_FALSE(s.disk_persistable());
    s.kvflash_blob = make_blob();             // pooled, blob present: still not
    CHECK_FALSE(s.disk_persistable());

    PrefixSnapshot paged;
    paged.layout = Layout::paged;
    CHECK(paged.disk_persistable());
    for (Layout l : {Layout::empty, Layout::thin}) {
        PrefixSnapshot t;
        t.layout = l;
        CHECK_FALSE(t.disk_persistable());
    }
}
