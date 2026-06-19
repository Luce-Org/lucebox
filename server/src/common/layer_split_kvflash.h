// Shared KVFlash helpers for target layer-split adapters.
//
// Model adapters still own model-specific KV tensor discovery, scoring, and
// remote-shard synchronization. These helpers keep the common pager/history
// contracts identical across Qwen, Gemma, and Laguna layer-split backends.

#pragma once

#include "kvflash_pager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace dflash::common {

inline bool layer_split_kvflash_sync_identity(
        KvFlashPager & pager,
        int committed,
        int pool_tokens,
        const char * log_prefix) {
    const char * prefix = log_prefix ? log_prefix : "target-split";
    if (committed < 0) {
        std::fprintf(stderr,
            "[%s][kvflash] invalid committed position %d\n",
            prefix, committed);
        return false;
    }
    if (committed > pool_tokens) {
        std::fprintf(stderr,
            "[%s][kvflash] prefix (%d) exceeds resident pool %d\n",
            prefix, committed, pool_tokens);
        return false;
    }
    pager.reset();
    for (int p = 0; p < committed; ++p) {
        const int slot = pager.slot_for(p);
        if (slot != p) {
            std::fprintf(stderr,
                "[%s][kvflash] identity slot mismatch %d != %d\n",
                prefix, slot, p);
            return false;
        }
    }
    pager.zero_free_blocks();
    return pager.is_identity();
}

inline void layer_split_kvflash_sync_history(
        std::vector<int32_t> & history,
        const std::vector<int32_t> & tokens,
        int base_pos) {
    if (base_pos <= 0) {
        history.assign(tokens.begin(), tokens.end());
        return;
    }
    if ((int)history.size() > base_pos) {
        history.resize((size_t)base_pos);
    } else if ((int)history.size() < base_pos) {
        history.resize((size_t)base_pos, 0);
    }
    history.insert(history.end(), tokens.begin(), tokens.end());
}

inline void layer_split_kvflash_restore_history(
        std::vector<int32_t> & history,
        const std::vector<std::vector<int32_t>> & snapshots,
        int slot,
        int cur_pos) {
    if (slot >= 0 && slot < (int)snapshots.size() &&
        !snapshots[(size_t)slot].empty()) {
        history = snapshots[(size_t)slot];
    } else {
        history.clear();
    }
    if (cur_pos <= 0) {
        history.clear();
    } else if ((int)history.size() > cur_pos) {
        history.resize((size_t)cur_pos);
    } else if ((int)history.size() < cur_pos) {
        history.resize((size_t)cur_pos, 0);
    }
}

inline void layer_split_kvflash_save_history_snapshot(
        const std::vector<int32_t> & history,
        int cur_pos,
        std::vector<int32_t> & snapshot) {
    snapshot.clear();
    if (cur_pos <= 0) return;
    const size_t keep = (size_t)std::min(cur_pos, (int)history.size());
    snapshot.assign(history.begin(), history.begin() + keep);
    if ((int)snapshot.size() < cur_pos) {
        snapshot.resize((size_t)cur_pos, 0);
    }
}

}  // namespace dflash::common
