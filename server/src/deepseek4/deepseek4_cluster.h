// deepseek4_cluster.h - DeepSeek4 expert-parallel cluster runtime.
//
// Every rank loads the full dense model and only its shard of routed experts
// (owned + replicated, see cluster_expert_placement.h). Per MoE layer the
// rank evaluates its resident experts as an owner partial sum through the
// existing hybrid machinery with cold owner MoeHybridColdBackend::None, then
// ONE all-reduce (IClusterComm::allreduce_sum_f32 on the backend stream)
// makes the routed partial identical on every rank before the shared expert
// is added and HC-post runs.
//
// By default the all-reduce is a node of the fused whole-model verify/decode
// graph (ggml_cluster_allreduce, one per MoE layer). With the fused graph
// unavailable (see ds4_cluster_fused_graph_available) it is a host-driven
// call between per-layer graph computes inside
// deepseek4_step_layer_range / eval_ds4_layer_range_hybrid_ffn.
//
// Shared expert: SharedExpertMode::Replicate only. Every rank computes it
// locally from the same normalized input and adds it after the all-reduce;
// the routed partial is evaluated with the shared-expert tensors removed
// from the MoeLayerDesc (the per-layer equivalent of include_shared=false).
//
// Replicated experts: expert e at batch slot t is evaluated only by rank
// (e + t) % N. Slot = token index inside the current batch (0..n_tokens-1),
// identical on every rank because routing is replicated; for decode
// (n_tokens == 1) this pins replicated expert e to rank e % N.

#pragma once

#include "deepseek4_internal.h"

#include "cluster/cluster_comm.h"
#include "cluster/fast_reduce.h"
#include "cluster/cluster_config.h"
#include "cluster/cluster_expert_placement.h"
#include "cluster/cluster_telemetry.h"
#include "common/moe_hybrid_placement.h"
#include "common/moe_hybrid_storage.h"
#include "common/moe_hybrid_types.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace luce::common {

struct Ds4ClusterRuntime {
    // cfg points at cfg_storage (a copy taken by DeepSeek4Backend::set_cluster)
    // so the caller's ClusterConfig need not outlive the backend. comm is
    // borrowed and may be attached after the model was loaded; it is required
    // by the first forward.
    cluster::ClusterConfig         cfg_storage;
    const cluster::ClusterConfig * cfg  = nullptr;
    cluster::IClusterComm *        comm = nullptr;
    // Decode-path all-reduces over raw RDMA (LUCE_CLUSTER_FAST_REDUCE=1); RCCL keeps
    // whatever it declines (prefill-sized payloads).
    std::unique_ptr<cluster::FastReduce> fast;
    // Hybrid exchange call counters: the k-th export/combine/import call of each kind belongs to exchange k.
    uint64_t hyb_export0 = 0, hyb_export1 = 0, hyb_combine = 0, hyb_import = 0;
    cluster::ClusterExpertPlacement placement;
    // Rank-local hot set (owned + replicated experts) for the hybrid storage.
    MoeHybridPlacement rank_placement;
    // [n_layer][slot_validity_n_slots][n_expert] 0/1 from
    // ClusterExpertPlacement::slot_validity for this rank; consulted for
    // slots below slot_validity_n_slots, larger slots fall back to
    // placement.slot_owner().
    std::vector<uint8_t> slot_validity_cache;
    int slot_validity_n_slots = 0;
    cluster::ClusterStepTelemetry telemetry;
    bool trace = false;
    // Path 3b: a graph node cannot return an error, so the in-graph
    // all-reduce records the first failure here and the forward fails after
    // the compute. Cleared before every graph that contains such a node.
    std::string node_error;
    // Attention head parallelism: this rank builds only heads
    // [attn_head_begin, attn_head_begin + attn_head_count) and the ranks
    // all-reduce the attention output, which is a partial sum over the
    // grouped output projection. count 0 = attention stays replicated.
    int attn_head_begin = 0;
    int attn_head_count = 0;
    // Shared-expert sharding (--cluster-shared-expert shard): this rank's
    // slice of the intermediate axis. count 0 = the shared expert is
    // replicated and added once after the reduction.
    int shexp_ff_begin = 0;
    int shexp_ff_count = 0;

    // Device staging for host-resident partials (per-layer path): one F32 [n] and
    // one bf16 [n] (stored as I16) tensor in a single backend buffer, grown
    // on demand. Owned here, freed in the destructor / free_scratch().
    ggml_backend_t        scratch_backend = nullptr;
    ggml_context *        scratch_ctx = nullptr;
    ggml_backend_buffer_t scratch_buf = nullptr;
    ggml_tensor *         scratch_f32 = nullptr;
    ggml_tensor *         scratch_bf16 = nullptr;
    size_t                scratch_elems = 0;

    // Resident routed-expert bytes on this rank (sum of hot buffers), for logs.
    uint64_t resident_expert_bytes = 0;

    // Set by a forward for the duration of one graph: called from the
    // in-graph broadcast's callback with its buffer and stream before the
    // broadcast is queued, so data a host thread is still producing can be
    // uploaded right before its consumer (the verify's Engram keys). False
    // fails the forward.
    std::function<bool(void * data, size_t n, void * stream)> before_broadcast;

    Ds4ClusterRuntime() = default;
    ~Ds4ClusterRuntime();
    Ds4ClusterRuntime(const Ds4ClusterRuntime &) = delete;
    Ds4ClusterRuntime & operator=(const Ds4ClusterRuntime &) = delete;

    int rank() const { return cfg ? cfg->rank : 0; }
    int size() const { return cfg ? cfg->size : 1; }

    // True when this rank evaluates (layer, expert) for batch slot `slot`.
    bool evaluates(int layer, int expert, int slot) const;

    bool ensure_scratch(ggml_backend_t backend, size_t n_elems, std::string * err);
    void free_scratch();
};

// MoeHybridConfig for cluster expert storage: cold owner None, cold experts
// never materialized, hot experts materialized on the local GPU.
MoeHybridConfig ds4_cluster_moe_config(const DeepSeek4Weights & w);

// Build the N-rank placement from cfg (uniform | balanced | file) and derive
// this rank's MoeHybridPlacement and slot-validity cache. hotness_csv_path
// (from LUCE_DS4_HOTNESS_CSV) is required for `balanced` and used by
// `uniform` when replicate_hot > 0; may be nullptr otherwise.
bool ds4_cluster_build_placement(const cluster::ClusterConfig & cfg,
                                 const DeepSeek4Weights & w,
                                 const std::string * hotness_csv_path,
                                 Ds4ClusterRuntime & rt,
                                 std::string * err);

// Build this rank's expert storage: hot set = rt.rank_placement, cold owner
// None. `w` must already be loaded with TargetLoadPlan.skip_expert_tensors
// (expert tensors present as metadata only). Logs resident expert bytes.
bool ds4_cluster_init_experts(const std::string & model_path,
                              ggml_backend_t backend,
                              const DeepSeek4Weights & w,
                              Ds4ClusterRuntime & rt,
                              MoeHybridStorage & storage,
                              std::string * err);

// Zero out every route this rank does not evaluate: selected id -> -1,
// weight -> 0. selected/weights are [route_width, n_tokens] row-major
// (token-major), slot = token index. Routing statistics must be observed
// before this call.
void ds4_cluster_mask_routes(const Ds4ClusterRuntime & rt,
                             int layer,
                             int32_t * selected,
                             float * weights,
                             int route_width,
                             int n_tokens);

// All-reduce a device-resident F32 partial [n_embd, n_tokens] in place on the
// backend stream. Stays stream-ordered (no host sync) unless rt.trace is set,
// in which case it waits with cfg->timeout_ms and logs the layer. Updates
// rt.telemetry and, when given, telemetry->cluster_allreduce_*.
bool ds4_cluster_allreduce_layer(Ds4ClusterRuntime & rt,
                                 ggml_backend_t backend,
                                 ggml_tensor * partial,
                                 int layer,
                                 DeepSeek4StepTelemetry * telemetry,
                                 std::string * err);

// Path 3b: an all-reduce as an ordinary graph node. `partial` is summed
// across the ranks on the executing backend's stream, so no host round trip
// separates the kernels that produce it from those that consume the sum.
// Returns `partial` unchanged for a single-rank runtime and nullptr when the
// tensor cannot be reduced (non-contiguous or not F32). Errors during the
// compute land in rt.node_error. `inplace` sums into `partial` itself (no
// device copy first); only for a partial that no other node reads.
ggml_tensor * ds4_cluster_allreduce_node(ggml_context * ctx,
                                         ggml_tensor * partial,
                                         Ds4ClusterRuntime & rt,
                                         bool inplace = false);

// The same sum-all-reduce as a raw callback, for graphs built outside the
// deepseek4 layer. server/src/common/dspark_head.cpp needs it to combine the
// vocabulary-sliced logits of the DSpark head, and must not gain a cluster
// dependency to do so, so it takes an ggml_cluster_allreduce_fn and a void*
// rather than the runtime itself. Pass `&rt` as the user pointer: errors land
// in rt.node_error and byte counts in the usual telemetry.
ggml_cluster_allreduce_fn ds4_cluster_allreduce_fn();

// LUCE_CLUSTER_HYBRID_EXCHANGE=1: the decode FFN reduction of a rank whose
// experts are split between its dGPU (hot + shared) and iGPU (cold) runs as
// export (dGPU) -> combine + exchange (iGPU) -> import (dGPU); see
// cluster::FastReduce::hyb_export. ds4_cluster_hybrid_exchange_nodes returns
// the dGPU tensor holding the reduced sum (null when unavailable) and the iGPU
// combine node, which the caller must keep in the graph and on the iGPU.
bool ds4_cluster_hybrid_exchange_enabled(const Ds4ClusterRuntime & rt);
ggml_tensor * ds4_cluster_hybrid_exchange_nodes(ggml_context * ctx,
                                                ggml_tensor * hot,
                                                ggml_tensor * shared,
                                                ggml_tensor * cold,
                                                Ds4ClusterRuntime & rt,
                                                ggml_tensor ** combine_node,
                                                ggml_tensor ** export_hot_node,
                                                ggml_tensor ** export_shared_node);

// LUCE_CLUSTER_ATTENTION_PARALLEL: split the attention heads across the ranks
// (on in cluster mode; =0 keeps attention replicated). Cached after the first
// call.
bool ds4_cluster_attention_parallel_enabled();
// LUCE_CLUSTER_PREFILL_DEVICE_JOIN (default on): cluster prefill keeps the
// device-resident hot+cold join and reduces in the HC-post graph.
bool ds4_cluster_prefill_device_join_enabled();
// Band-pipelined prefill on a cluster rank: one part's cold FFN runs beside
// the next part's attention. On when every band of the pass has at least
// kDs4ClusterPipelineMinBand rows (two boxes, 2K prompt in one band: 649-669
// -> 710-727 tok/s); smaller bands split into parts too short to pay for it
// (512-row bands measured slower). LUCE_CLUSTER_PREFILL_PIPELINE=1 forces it
// on, =0 off. A pure function of the bands, which every rank agrees on.
constexpr int kDs4ClusterPipelineMinBand = 1536;
bool ds4_cluster_prefill_pipeline_enabled(const std::vector<int> & bands);

// True when this runtime may use the fused whole-model graph: the
// opt-in is set, a real multi-rank communicator is attached, the shared
// expert is replicated, and expert ownership is a function of the expert id
// alone. Replicated experts are owned per token slot, which the fused graph's
// per-expert owner LUT cannot express, so they fall back to the per-layer path.
bool ds4_cluster_fused_graph_available(const Ds4ClusterRuntime * rt);

// In-graph all-reduce (default). LUCE_CLUSTER_NO_INGRAPH_ALLREDUCE=1 falls back
// to the host-enqueued per-layer all-reduce, which also gives up the
// fused whole-model graph. Cached after the first call.
bool ds4_cluster_ingraph_allreduce_enabled();

// Host-resident variant used by the per-layer path: uploads to the runtime's
// device scratch, all-reduces on the backend stream, waits (deadline
// cfg->timeout_ms) and downloads the sum back into partial_host.
bool ds4_cluster_allreduce_layer_host(Ds4ClusterRuntime & rt,
                                      ggml_backend_t backend,
                                      float * partial_host,
                                      int n_embd,
                                      int n_tokens,
                                      int layer,
                                      DeepSeek4StepTelemetry * telemetry,
                                      std::string * err);

// ─── Lucebox additions ────────────────────────────────────────────────
// Engram tables live on rank 0 only (a rank's sparse model copy leaves them
// out). Rank 0's decoded rows reach the other ranks by broadcast: as a graph
// node in the fused verify/decode graph, host-staged in the per-layer path.
ggml_tensor * ds4_cluster_broadcast_node(ggml_context * ctx,
                                         ggml_tensor * value,
                                         Ds4ClusterRuntime & rt);
// The same broadcast, exact on a bf16 fast-reduce wire (f32 payload).
ggml_tensor * ds4_cluster_broadcast_exact_node(ggml_context * ctx,
                                               ggml_tensor * value,
                                               Ds4ClusterRuntime & rt);
bool ds4_cluster_broadcast_host(Ds4ClusterRuntime & rt,
                                ggml_backend_t backend,
                                float * data,
                                size_t n,
                                std::string * err);
// Every rank's value in rank order: an all-gather built on the all-reduce
// (one writer per slot). Used to agree on load-time sizes that would
// otherwise depend on each rank's free memory, such as the prefill chunk.
bool ds4_cluster_allgather_i32(Ds4ClusterRuntime & rt,
                               ggml_backend_t backend,
                               int32_t value,
                               std::vector<int32_t> & out,
                               std::string * err);
// True when this rank evaluates `expert` of `layer` for some token slot
// (owned or replicated): the experts its tiers hold.
bool ds4_cluster_resident(const Ds4ClusterRuntime & rt, int layer, int expert);

}  // namespace luce::common
