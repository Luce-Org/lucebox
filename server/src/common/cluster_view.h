// cluster_view.h - dependency-free snapshots of cluster state for the HTTP
// layer.
//
// server/src/server/ has no business knowing about RCCL, the control channel
// or the DeepSeek4 runtime, and it must keep compiling with
// -DLUCE_CLUSTER=OFF. So the cluster backend copies what the API should
// expose into these plain structs, exactly the way get_routing_stats() hands
// out routing telemetry, and ModelBackend answers "no cluster" by default.
//
// Threading: cluster_request_telemetry() is filled by the rank-0 decorator at
// the end of a request and must be read on the same thread that ran it (the
// request worker), which is where GenTimings is assembled. cluster_props() is
// constant after bootstrap and is safe to read from any thread.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace luce::common {

// One rank's contribution to the request that just finished.
struct ClusterRankTiming {
    int      rank              = 0;
    uint32_t steps             = 0;
    uint64_t compute_us        = 0;  // local prefill+decode wall time
    uint64_t allreduce_calls   = 0;
    uint64_t allreduce_bytes   = 0;
    uint64_t allreduce_wait_us = 0;  // host time blocked on collectives; 0 with
                                     // the in-graph all-reduce
    uint64_t ctrl_wait_us      = 0;  // host time blocked on the control channel
    uint64_t peak_device_bytes = 0;  // high-water mark, sampled at request ends
};

// Per-request cluster telemetry. `ranks[0]` is the head.
struct ClusterTelemetryView {
    bool        active              = false;  // false: not a cluster run
    int         size                = 0;
    uint64_t    request_id          = 0;
    uint32_t    steps               = 0;
    bool        complete            = false;  // every worker reported
    uint64_t    head_ctrl_wait_us   = 0;
    uint64_t    hash_probes         = 0;
    uint64_t    hash_mismatches     = 0;
    int         first_mismatch_rank = -1;
    uint32_t    first_mismatch_step = 0;
    std::string error;                        // non-empty when the gather failed
    std::vector<ClusterRankTiming> ranks;
};

// Constant after bootstrap: what this cluster is, for /props.
struct ClusterPropsView {
    bool        active         = false;
    int         size           = 0;
    int         rank           = 0;
    std::string ifname;
    std::string ib_hca;
    int         gid_index      = 0;
    std::string placement;                    // uniform | balanced | <file>
    uint64_t    placement_hash = 0;
    int         replicate_hot  = 0;
    std::string shared_expert;                // replicate | shard | rank0
    std::string allreduce_dtype;              // f32 | bf16 | auto
    bool        ingraph_allreduce = false;    // collectives run inside the graph
    uint64_t    resident_expert_bytes = 0;    // this rank's expert shard
    uint32_t    timeout_ms     = 0;
};

}  // namespace luce::common
