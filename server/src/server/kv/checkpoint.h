// A checkpoint: one saved model state the prefix cache can restore from.
//
// It holds everything a backend needs to resume at one prompt position:
// attention KV and, for hybrid models, the recurrent and conv state, plus
// the decode seed (last token, DFlash features). The store that issued the
// handle owns the payload (kv/checkpoint_store.h).

#pragma once

#include <cstdint>

namespace luce::common {

struct Checkpoint {
    // The backend snapshot slot plus one, which is also the id a
    // PrefixStoreRef carries on the concurrent path. 0: no checkpoint.
    uint32_t id = 0;

    bool valid() const { return id != 0; }

    // PrefixCache still hands out slot numbers; these two bridge it until
    // the entry index issues checkpoints itself.
    static Checkpoint of_cache_slot(int slot) {
        return slot < 0 ? Checkpoint{} : Checkpoint{(uint32_t)slot + 1};
    }
    int cache_slot() const { return valid() ? (int)id - 1 : -1; }
};

inline bool operator==(Checkpoint a, Checkpoint b) { return a.id == b.id; }
inline bool operator!=(Checkpoint a, Checkpoint b) { return a.id != b.id; }

}  // namespace luce::common
