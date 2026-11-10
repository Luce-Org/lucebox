// KvFlashArena: the page stream and the completion point
// shared by every slot pager of one engine. Members only queue copies
// (batch submit, no hidden waits); complete_all() is the one host wait after
// which their paged-out blocks become allocatable again, so a block is never
// handed out with a copy or a zeroing still queued on it.
//
// The arena must outlive its members: a member's destructor synchronizes the
// shared stream.

#pragma once

#include "kvflash_pager.h"

#include <algorithm>
#include <functional>
#include <vector>

namespace luce::common {

// The arena's page stream: without one each pager makes a private stream and complete_all()
// waits on nothing, so a freed block could be reused under a running copy. Required on async-DMA builds.
inline bool kvflash_arena_stream_usable(bool async_dma, bool have_stream) { return !async_dma || have_stream; }

class KvFlashArena {
public:
    KvFlashArena() {
#ifdef KVFLASH_HAS_ASYNC_DMA
        if (cudaStreamCreate(&stream_) != cudaSuccess) { stream_ = nullptr; (void)cudaGetLastError(); }
        host_wait_ = [this] { if (stream_) cudaStreamSynchronize(stream_); };
#endif
    }
    ~KvFlashArena() {
#ifdef KVFLASH_HAS_ASYNC_DMA
        if (stream_) { cudaStreamSynchronize(stream_); cudaStreamDestroy(stream_); }
#endif
    }
    KvFlashArena(const KvFlashArena &) = delete;
    KvFlashArena & operator=(const KvFlashArena &) = delete;

    void * stream() const {
#ifdef KVFLASH_HAS_ASYNC_DMA
        return stream_;
#else
        return nullptr;
#endif
    }

    // False when an async-DMA build has no page stream (cudaStreamCreate failed): the arena must not be used.
    bool stream_usable() const {
#ifdef KVFLASH_HAS_ASYNC_DMA
        return kvflash_arena_stream_usable(true, stream_ != nullptr);
#else
        return true;
#endif
    }

    // Join before attach(): the stream is chosen there. False if already
    // attached or already a member.
    bool add(KvFlashPager & p) {
        if (p.attached() || std::find(members_.begin(), members_.end(), &p) != members_.end()) return false;
        p.set_external_page_stream(stream());
        p.set_batch_submit(true);
        members_.push_back(&p);
        return true;
    }
    void remove(KvFlashPager & p) {
        members_.erase(std::remove(members_.begin(), members_.end(), &p), members_.end());
    }

    // The completion phase: if any member has a page-out in flight, ONE host
    // wait for the shared stream (every member's copy and zeroing is on it),
    // then each member releases its settled blocks. Returns whether it waited.
    bool complete_all() {
        bool pending = false;
        for (KvFlashPager * p : members_) pending = pending || p->pending_free_blocks() > 0;
        if (!pending) return false;
        if (host_wait_) host_wait_();
        for (KvFlashPager * p : members_) p->complete(p->last_record());
        return true;
    }

protected:   // the test fixture swaps host_wait_
#ifdef KVFLASH_HAS_ASYNC_DMA
    cudaStream_t stream_ = nullptr;
#endif
    std::function<void()> host_wait_;
    std::vector<KvFlashPager *> members_;
};

}  // namespace luce::common
