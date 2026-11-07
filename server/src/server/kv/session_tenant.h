// The KVFlash session store as a DiskStore tenant (disk_store.h): the sessions under <kv-cache-dir>/sessions/
// count in the one cache budget, go by its LRU next to the prefix checkpoints, and expire by its TTL; a park
// asks the budget for room before it writes. One lock for the directory (the DiskStore's), none of the
// store's own.

#pragma once

#include "common/kvflash_session_store.h"
#include "server/kv/disk_store.h"

#include <memory>

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

}  // namespace luce::common
