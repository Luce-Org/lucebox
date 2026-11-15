// The KVFlash session store as a DiskStore tenant (disk_store.h): the sessions under <kv-cache-dir>/sessions/
// count in the one cache budget, go by its LRU next to the prefix checkpoints, and expire by its TTL; a park
// asks the budget for room before it writes. One lock for the directory (the DiskStore's), none of the
// store's own.

#pragma once

#include "common/kvflash_session_store.h"
#include "server/kv/disk_store.h"

#include <array>
#include <memory>
#include <string>

namespace luce::common {

// Attach `store` (rooted at the DiskStore's directory) to `disk`; the id for DiskStore::detach_tenant(). The
// budget and the TTL apply to the sessions at once, so ones that no longer fit are gone before the first request.
inline uint64_t attach_session_store(const std::shared_ptr<DiskStore> & disk, KvFlashSessionStore & store) {
    const std::shared_ptr<KvFlashSessionLedger> ledger = store.ledger();
    DiskStore::Tenant tenant;
    tenant.usage = [ledger] { return ledger->usage(); };
    tenant.oldest = [ledger](uint64_t & last_used) { return ledger->oldest(last_used); };
    tenant.evict_oldest = [ledger] { return ledger->evict_oldest(); };
    const uint64_t id = disk->attach_tenant(std::move(tenant));
    DiskStore * raw = disk.get();   // the store is detached before its DiskStore goes: the engine owns both
    store.set_budget({[raw](uint64_t n) { return raw->reserve(n); }, [raw](uint64_t n) { raw->release(n); }});
    if (disk->reserve(0)) disk->release(0);
    return id;
}

// The identity a server's sessions hold on the directory (DiskStore::hold_identity): the execution identity under
// the sessions' own namespace, so it is never a prefix layout's salt nor a head view's. Two FNV-1a lanes, 16 bytes.
inline std::array<uint8_t, 16> kvflash_session_salt(const std::string & identity) {
    std::array<uint8_t, 16> out{};
    for (int lane = 0; lane < 2; lane++) {
        uint64_t h = 1469598103934665603ULL ^ (lane ? 0x9e3779b97f4a7c15ULL : 0);
        for (unsigned char c : std::string("kvflash-sessions/1|") + identity) { h ^= c; h *= 1099511628211ULL; }
        for (int b = 0; b < 8; b++) out[(size_t)(lane * 8 + b)] = (uint8_t)(h >> (8 * b));
    }
    return out;
}

// Attach as the one server of `salt`'s sessions on the directory. 0, and nothing attached, when another server
// holds them: two servers of one identity would restore, evict and overwrite each other's sessions.
inline uint64_t attach_session_store(const std::shared_ptr<DiskStore> & disk, KvFlashSessionStore & store,
                                     const std::array<uint8_t, 16> & salt) {
    if (!disk->hold_identity(salt)) return 0;
    return attach_session_store(disk, store);
}

inline void detach_session_store(const std::shared_ptr<DiskStore> & disk, uint64_t id, const std::array<uint8_t, 16> & salt) {
    disk->detach_tenant(id);
    disk->release_identity(salt);
}

}  // namespace luce::common
