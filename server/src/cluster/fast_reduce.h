// fast_reduce.h - a lean all-reduce for the decode path's tiny payloads.
//
// WHY THIS EXISTS. A qwen4exp decode step on two nodes computes in 29.6 ms of
// GPU work and takes 41.0. The difference is ninety-seven all-reduces of
// 10 KiB each -- 117 us apiece. That is not the wire: ib_write_lat puts the
// same payload at 13.65 us, and the round trip through a pinned flag between
// the GPU and a host thread measures 0.79 us. It is what a general-purpose
// collective costs per call, ninety-seven times, and it is the single reason
// adding ranks to this model buys nothing: every axis worth sharding pays for
// its bytes with reductions at about 11.9 MB each, and every axis this model
// has sits on that line.
//
// So this replaces the collective for exactly the case it is bad at: a few
// kilobytes, latency-bound, on the critical path of every layer. Each rank
// writes its partial straight into every peer's pre-registered buffer and adds
// what arrives. There is no algorithm selection, no protocol negotiation and
// no proxy handshake -- one RDMA write per peer and a flag.
//
// WHAT IT ASSUMES. Unified memory. On Strix Halo there is no separate VRAM, so
// a pinned host buffer *is* GPU memory: the NIC writes into it and a kernel
// reads it without a copy and without GPUDirect. It also assumes the ranks run
// the same graph in lockstep, so that the n-th reduction on one rank is the
// n-th on all of them -- which is what the sequence numbers check rather than
// trust.
//
// WHAT IT IS NOT. Not a general collective: sum of f32, small sizes, no
// in-place aliasing games, no groups. Anything outside that keeps using RCCL.
//
// Include convention: #include "cluster/fast_reduce.h"

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace luce::cluster {

// Opaque so verbs headers stay out of every translation unit that reduces.
struct FastReduceImpl;

class FastReduce {
public:
    FastReduce();
    ~FastReduce();

    FastReduce(const FastReduce &) = delete;
    FastReduce & operator=(const FastReduce &) = delete;

    struct Config {
        int         rank = 0;
        int         size = 1;
        std::string hca;                 // e.g. "rocep197s0f3"
        int         ib_port = 1;
        int         gid_index = 1;       // RoCE v2
        std::string bootstrap_host;      // rank 0's address
        int         bootstrap_port = 9500;
        // Decode sizes only -- 10240 floats is the largest a step reduces --
        // and a ring deep enough that the guard against lapping it almost never
        // fires. It has to be deep: the guard is a stream wait, and a stream
        // wait that fires stops the GPU running ahead, which is where this
        // path's speed comes from. Prefill's much larger reductions fall back
        // to RCCL, where they cost nothing that matters.
        int         max_elems = 16384;   // 64 KiB
        int         slots = 4096;        // a step has ~97
        uint32_t    timeout_ms = 30000;
    };

    // Brings the fabric up: registers the buffers, exchanges endpoints over a
    // plain TCP mesh of its own, and connects one queue pair per peer. The
    // bootstrap is separate from the cluster control channel on purpose --
    // this has to be able to fail without taking the run with it.
    bool init(const Config & cfg, std::string * err);

    void shutdown();

    bool ok() const;

    // Enqueue a sum-reduction of `n` floats at `data` on `stream`. `data` must
    // be device-visible and is updated in place. Returns false when the
    // payload is larger than the fabric was built for, so the caller can fall
    // back rather than corrupt the step. `exact` keeps this payload in f32 on
    // the wire and in the add when LUCE_CLUSTER_FAST_REDUCE_BF16 rounds the
    // others, for values that must arrive bit for bit.
    bool submit(float * data, size_t n, void * stream, bool exact = false);

    // Broadcast from `root` as a sum: every other rank zeroes its copy on the
    // stream first, so the sum is the root's data everywhere (with `exact`,
    // the root's data itself: x + 0 is x).
    bool submit_from_root(float * data, size_t n, void * stream, int root, bool exact = false);

    // Hybrid exchange (LUCE_CLUSTER_HYBRID_EXCHANGE=1, zero-copy two-rank mode):
    // a rank whose partial is split between a dGPU (hot experts, shared
    // expert) and the iGPU that owns its cold experts. The dGPU exports its
    // parts to pinned memory while it waits anyway; the iGPU adds its part to
    // them in the order the dGPU used to, packs and publishes the rank's row,
    // and adds the peer's when it lands; the dGPU imports the finished sum.
    // The iGPU's partial never crosses the dGPU's PCIe link, and the dGPU
    // crosses it once each way. `k` is the exchange's index in issue order;
    // its export, combine and import may be enqueued in any host order.
    bool hyb_available() const;
    bool hyb_export(const float * data, size_t n, void * stream, int part, uint64_t k);
    bool hyb_combine(const float * data, size_t n, void * stream, uint64_t k);
    bool hyb_import(float * data, size_t n, void * stream, uint64_t k);

    // Reductions issued and how long the progress thread waited for the GPU,
    // for the telemetry line.
    uint64_t submitted() const;
    uint64_t timed_out() const;

private:
    std::unique_ptr<FastReduceImpl> p_;
};

}  // namespace luce::cluster
