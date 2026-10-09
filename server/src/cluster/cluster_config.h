// cluster_config.h - caller-requested multi-node cluster configuration.
//
// lucebox-halo-cluster: SPMD expert-parallel execution of one DeepSeek4 model
// across N machines (rank 0 = head that serves HTTP, ranks 1..N-1 = workers),
// joined per MoE layer by one RCCL all-reduce over RoCE v2.
//
// This header carries only what the operator asked for on the command line.
// It has no ggml/HIP/RCCL dependency so the feature gate, the CLI parser and
// the GPU-free unit tests can include it unconditionally. Runtime facts
// (communicator handle, resolved placement, peer identities) live in the
// cluster runtime objects, not here.
//
// Policy (variables.md): what the operator chooses ships as CLI flags. The
// measured tuning defaults of cluster mode are environment variables that
// cluster_launch_env() (server/launch_profiles.h) installs, each with a 0
// kill switch; debug tracing stays in LUCE_CLUSTER_TRACE.

#pragma once

#include <cstdint>
#include <string>

namespace luce::cluster {

inline constexpr int kClusterMinSize = 2;
inline constexpr int kClusterMaxSize = 8;
inline constexpr int kClusterDefaultPort = 9400;
inline constexpr uint32_t kClusterDefaultTimeoutMs = 30000;

// How the shared expert (present on every DeepSeek4 layer) is evaluated.
enum class SharedExpertMode {
    Replicate,  // every rank computes it; all-reduce covers routed experts only
    Shard,      // n_ff/N slice per rank, summed inside the same all-reduce
    Rank0,      // only rank 0 computes it and adds it after the all-reduce
};

// Wire precision of the per-layer partial-sum reduction.
enum class AllreduceDType {
    F32,
    BF16,
    Auto,  // F32 below kAutoBf16ThresholdBytes, BF16 above (prefill chunks)
};

inline constexpr uint64_t kAutoBf16ThresholdBytes = 256ull * 1024ull;

// Expert-to-rank ownership source (--cluster-expert-placement).
enum class PlacementSource {
    Uniform,   // expert e of every layer owned by rank e % N
    Balanced,  // greedy LPT on routing hotness (needs LUCE_DS4_HOTNESS_CSV)
    File,      // explicit JSON produced by scripts/cluster/build_expert_placement.py
};

struct ClusterConfig {
    // --cluster-rank / --cluster-size. size == 0 means "not a cluster run".
    int rank = -1;
    int size = 0;

    // --cluster-head <host:port>: rank 0 binds here, workers connect here.
    std::string head_host;
    int         head_port = kClusterDefaultPort;

    // RCCL transport hints (--cluster-ifname / --cluster-ib-hca /
    // --cluster-gid-index). Exported to NCCL_SOCKET_IFNAME, NCCL_IB_HCA and
    // NCCL_IB_GID_INDEX unless the environment already sets them.
    std::string ifname;
    std::string ib_hca;
    int         gid_index = -1;

    // Expert placement.
    PlacementSource placement_source = PlacementSource::Uniform;
    std::string     placement_file;      // when placement_source == File
    int             replicate_hot = 0;   // --cluster-replicate-hot <k>
    SharedExpertMode shared_expert = SharedExpertMode::Replicate;

    // Reduction and robustness knobs.
    AllreduceDType allreduce_dtype = AllreduceDType::Auto;
    uint32_t       timeout_ms      = kClusterDefaultTimeoutMs;  // --cluster-timeout-ms

    // Debug: cross-rank hidden-state hash every n steps (0 = off).
    int verify_hash_every = 0;

    // --cluster-selftest: bootstrap, run the collective self-test, exit.
    bool selftest = false;

    bool enabled() const { return size > 0; }
    bool is_head() const { return enabled() && rank == 0; }
    bool is_worker() const { return enabled() && rank > 0; }

    // Structural validation independent of model/backend (rank range, size
    // range, head endpoint present, replicate_hot >= 0, file present when
    // placement_source == File). Returns an empty string when valid.
    std::string validate() const;
};

// Parsing helpers shared by server_main.cpp and the unit tests. Each returns
// false and fills *err on malformed input; none touches the process
// environment.
bool parse_host_port(const std::string & text, std::string & host, int & port, std::string * err);
bool parse_shared_expert_mode(const std::string & text, SharedExpertMode & out, std::string * err);
bool parse_allreduce_dtype(const std::string & text, AllreduceDType & out, std::string * err);
// "uniform" | "balanced" | <path ending in .json>
bool parse_placement_source(const std::string & text, PlacementSource & out,
                            std::string & file, std::string * err);

const char * shared_expert_mode_name(SharedExpertMode mode);
const char * allreduce_dtype_name(AllreduceDType dtype);
const char * placement_source_name(PlacementSource source);

// Exports the RCCL/NCCL environment for this node. Existing variables win;
// NCCL_NET_GDR_LEVEL=0 is always set because gfx1151 has no GPUDirect RDMA.
// Also sets the RCCL hardening defaults measured on the 4-node Strix Halo
// cluster (NCCL_IB_DISABLE=0, NCCL_ASYNC_ERROR_HANDLING=1,
// NCCL_IB_QPS_PER_CONNECTION=2, NCCL_IB_TIMEOUT=22, NCCL_IB_RETRY_CNT=7) and
// NCCL_PROTO=Simple unless already present. Returns the number of variables it set.
int export_rccl_environment(const ClusterConfig & cfg);

// Kill switches / debug (environment, documented in variables.md).
bool cluster_env_no_ingraph_allreduce();  // LUCE_CLUSTER_NO_INGRAPH_ALLREDUCE
bool cluster_env_no_graph_capture();      // LUCE_CLUSTER_NO_GRAPH_CAPTURE
bool cluster_env_trace();                 // LUCE_CLUSTER_TRACE

}  // namespace luce::cluster
