// common/cluster_participant.h - what a backend must offer to run in a cluster.
//
// The head decorator and the worker loop were written against
// DeepSeek4Backend because it was the only architecture that sharded. Nothing
// they do actually needs that type: they broadcast requests, replay them, and
// hand the backend a config and a communicator. This interface is the whole of
// what they need, so a second architecture can join without either of them
// learning its name.
//
// Attachment happens in two steps, in this order, because the placement must
// exist before the ranks can agree on it and the communicator does not exist
// until they have:
//
//   cluster_attach(&cfg, nullptr)  before init() -- build the rank's placement
//   cluster_attach(&cfg, comm)     after RCCL is up -- attach the transport
//
// Include convention: #include "common/cluster_participant.h"

#pragma once

#include <cstdint>

namespace luce::cluster {
struct ClusterConfig;
class IClusterComm;
struct Ds4ClusterHooks;
}  // namespace luce::cluster

namespace luce::common {

class IClusterParticipant {
public:
    virtual ~IClusterParticipant() = default;

    // Returns false when the config is one this backend cannot honour.
    virtual bool cluster_attach(const cluster::ClusterConfig * cfg,
                                cluster::IClusterComm *        comm) = 0;

    // A digest of this rank's weight placement, compared across ranks during
    // the handshake so a mismatched shard set fails at startup rather than as
    // wrong output. Zero means "no placement built yet", which is an error at
    // handshake time.
    virtual uint64_t cluster_placement_hash() const = 0;

    // Where the rank-0-decides hooks are installed, and where they are removed
    // again at shutdown (nullptr).
    virtual void cluster_set_hooks(cluster::Ds4ClusterHooks * hooks) = 0;

    // Whether this rank can speculate. The head must not speculate while a
    // worker cannot, so a mismatch is rejected rather than silently downgraded.
    virtual bool cluster_spec_decode_ready() const { return false; }

    // For /props: bytes of routed experts resident on this rank, and whether
    // the collective runs inside the graph rather than between per-layer
    // computes. Defaults describe a rank that shards nothing and reduces
    // between computes.
    virtual uint64_t cluster_resident_expert_bytes() const { return 0; }
    virtual bool cluster_ingraph_allreduce() const { return false; }
};

}  // namespace luce::common
