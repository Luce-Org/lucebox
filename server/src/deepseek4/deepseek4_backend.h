// DeepSeek4Backend — ModelBackend for DeepSeek V4 Flash MLA+MoE models.
//
// Architecture: Multi-head Latent Attention (MLA), KV compression with
// learned compressors, Hierarchical Controller (HC), MoE with hash routing
// (first 3 layers) + top-k routing + shared expert.

#pragma once

#include "common/model_backend.h"
#include "common/cluster_participant.h"
#include "common/moe_expert_compute.h"
#include "common/sampler.h"
#include "../common/moe_hybrid_expert_cache.h"
#include "../common/moe_hybrid_placement.h"
#include "../common/moe_hybrid_routing_stats.h"
#include "../common/moe_hybrid_storage.h"
#include "../common/moe_hybrid_stream.h"
#include "deepseek4_internal.h"
#include "deepseek4_dspark.h"
#include "deepseek4_vision.h"
#include "deepseek4_image_prompt.h"
#include "deepseek4_image_assembly.h"
#include "deepseek4_image_admission.h"
#include "qwen3/qwen3_drafter.h"
#include "deepseek4_seq_engine.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace luce::common {

class DeepSeek4ImagePrompt;
struct Ds4ClusterRuntime;  // deepseek4_cluster.h

// Bounds the sparse heterogeneous prefill arena once accumulated attention
// context dominates its memory footprint. Decode batching is unaffected.
// Long-context prefill chunk caps: kDs4QualifiedLongContextChunk on the
// qualified R9700 + Strix Halo placement, kDs4DefaultLongContextChunk
// elsewhere. LUCE_DS4_LONG_CONTEXT_CHUNK overrides either.
inline constexpr int kDs4DefaultLongContextChunk = 1024;
inline constexpr int kDs4QualifiedLongContextChunk = 2048;

int deepseek4_hybrid_prefill_chunk_tokens(
    int requested_chunk,
    int context_end,
    int current_cap = 0,
    int long_context_default = kDs4DefaultLongContextChunk);

// Selects the next sparse heterogeneous prefill batch. Large batches retain
// their throughput through the memory-light part of the prompt, then shrink
// at the late-context boundary where one attention arena would otherwise
// exhaust a tightly packed discrete GPU.
int deepseek4_hybrid_prefill_step_tokens(
    int configured_chunk,
    int position,
    int remaining_tokens);

// Mixed ROCmFP MMQ changes the reduction/quantization topology, so only the
// already-approximate prefill modes may select it automatically. The policy is
// kept separate from the qtype kernels so future model backends can reuse the
// same generic MMQ path after device-level qualification.
bool deepseek4_mix_mmq_prefill_default(
    PrefillAttentionMode mode,
    const char * gcn_arch);
ggml_mixed_mmq_policy deepseek4_mix_mmq_prefill_policy(
    PrefillAttentionMode mode, const char * gcn_arch, const char * explicit_value);

class DeepSeek4Backend : public ModelBackend, public IClusterParticipant {
public:
    explicit DeepSeek4Backend(DeepSeek4BackendConfig cfg);
    ~DeepSeek4Backend() override;

    DeepSeek4Backend(const DeepSeek4Backend &) = delete;
    DeepSeek4Backend & operator=(const DeepSeek4Backend &) = delete;

    bool init();

    // ModelBackend interface
    void print_ready_banner() const override;
    bool supports_images() const override { return image_capable_ && vision_ != nullptr; }
    std::string image_placeholder() const override { return vision::DS4V_IMAGE_PLACEHOLDER; }
    ImagePrepareStatus prepare_images(std::vector<int32_t> & tokens,
                                      std::vector<EncodedImage> images,
                                      uint64_t context_capacity,
                                      uint64_t output_reserve,
                                      ImagePromptHandle & payload,
                                      std::string & error) const override;

    bool park(ParkTarget target) override;
    bool unpark(ParkTarget target) override;
    bool is_target_parked() const override { return parked_; }

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;
    size_t snapshot_bytes_estimate(int tokens) const override;
    MemoryReport memory_report() const override;
    // Ondisk prefix cache: DeepSeek snapshots are CPU ggml contexts whose
    // tensors carry stable names plus a meta/logits/feature sidecar, so they
    // serialize and rebind like the Qwen snapshots do.
    SnapshotRef snapshot_ref(int slot) const override;
    bool snapshot_adopt(int slot, ggml_context * ctx,
                        ggml_backend_buffer_t buf, int cur_pos,
                        int32_t last_tok = -1) override;

    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    CompressResult compress(const CompressRequest & req) override;
    std::vector<CompressResult> compress_batch(
        const std::vector<CompressRequest> & requests) override;
    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override;

    void shutdown() override;
    SeqEngine * seq_engine() override { return seq_engine_.get(); }

    const MoeHybridRoutingStats * get_routing_stats() const override {
        return routing_stats_.get();
    }

    // ── Expert-parallel cluster (after maikzz32/lucebox-halo-cluster) ──
    // set_cluster(&cfg, nullptr) before init() builds this rank's shard
    // (init() does it from cfg_.cluster); set_cluster(&cfg, comm) once RCCL
    // is up only attaches the communicator.
    bool set_cluster(const cluster::ClusterConfig * cfg, cluster::IClusterComm * comm);
    bool cluster_attach(const cluster::ClusterConfig * cfg,
                        cluster::IClusterComm * comm) override {
        return set_cluster(cfg, comm);
    }
    uint64_t cluster_placement_hash() const override;
    void cluster_set_hooks(cluster::Ds4ClusterHooks * hooks) override { hooks_ = hooks; }
    bool cluster_spec_decode_ready() const override {
        return spec_enabled_ && spec_drafter_ != nullptr;
    }
    uint64_t cluster_resident_expert_bytes() const override;
    bool cluster_ingraph_allreduce() const override;

private:
    // Rank-0-decides hooks; nullptr outside a cluster.
    cluster::Ds4ClusterHooks * hooks_ = nullptr;
    bool cluster_active() const;
    bool cluster_worker() const;
    // Load-time agreements every rank must share: the prefill chunk size and
    // the drafter (none, whole or split).
    bool cluster_agree_prefill_chunk();
    bool cluster_agree_drafter();
    std::unique_ptr<Ds4ClusterRuntime> cluster_;

    DeepSeek4BackendConfig cfg_;
    ggml_backend_t         backend_      = nullptr;
    ggml_backend_t         snap_backend_ = nullptr;
    ggml_backend_t         expert_backend_ = nullptr;
    DeepSeek4Weights       w_;
    DeepSeek4Cache         cache_;
    DeepSeek4PagedCache    paged_cache_;
    std::unique_ptr<DeepSeek4SeqEngine> seq_engine_;
    bool                   parked_       = false;
    bool                   image_capable_ = false;
    bool                   cache_has_images_ = false;
    std::unique_ptr<vision::VisionRuntime> vision_;
    // Owned backend for the vision encoder when --mmproj-device names a GPU
    // other than the target's; null when the encoder shares backend_.
    ggml_backend_t         vision_backend_ = nullptr;
    // Encoder worker on vision_backend_: encodes queued image requests in
    // order and publishes each image as it lands, so neither the scheduler
    // nor prefill waits for a whole request. Started on first use.
    std::thread            encode_worker_;
    std::mutex             encode_mutex_;
    std::condition_variable encode_ready_;
    std::deque<std::shared_ptr<const DeepSeek4ImagePrompt>> encode_queue_;
    std::atomic<bool>      encode_stop_{false};  // also read by the worker's cancel check
    vision::ImageSentinels image_sentinels_;
    // Batched image serving: one single-request staging cache per slot
    // (slot 0 uses cache_), allocated at startup.
    std::vector<std::unique_ptr<DeepSeek4Cache>> image_staging_caches_;
    bool                   text_staging_ = false;
    vision::ImageAdmissionReserves image_reserves_;

    // Sampler
    SamplerCfg             sampler_;
    std::mt19937_64        sampler_rng_{std::random_device{}()};

    // Snapshots
    static constexpr int PREFIX_SLOTS = 64;
    struct SnapshotAux {
        std::vector<float> last_logits;
        std::vector<float> spec_feat_window;
        bool used = false;
    };
    DeepSeek4Snapshot      snapshots_[PREFIX_SLOTS];
    SnapshotAux            snapshot_aux_[PREFIX_SLOTS];
    std::vector<float>     last_logits_;
    // Absolute cache position represented by last_logits_. A snapshot is
    // safe only when this matches cache_.cur_pos.
    int                    last_logits_pos_ = -1;

    // DSpark speculative decode (opt-in: --draft <gguf>, or
    // LUCE_DS4_SPEC=1 + LUCE_DS4_DRAFT=<gguf>).
    bool                           spec_requested_ = false;
    bool                           spec_enabled_ = false;
    bool                           spec_drafter_parked_ = false;
    // LUCE_DS4_DRAFT_SWAP: a pinned host mirror of the drafter's core
    // weights, so long prompts can prefill with the drafter's device memory;
    // the chunk caps the prefill uses with the drafter in and out.
    struct DraftSwap {
        ggml_backend_buffer_type_t buft = nullptr;
        ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        size_t size = 0;
        ggml_backend_buffer_t host = nullptr;
        std::vector<std::pair<ggml_tensor *, size_t>> tensors;  // tensor, offset
        std::vector<ggml_tensor *> views;
        bool out = false;
        int cap_with = 0;
        int cap_without = 0;
    };
    DraftSwap                      draft_swap_;
    // Device bytes size_hybrid_prefill_chunk() counts as free on the target.
    size_t                         prefill_sizing_extra_free_ = 0;
    bool setup_draft_swap();
    bool draft_swap_prepare();
    bool draft_swap_out();
    bool draft_swap_in();
    void draft_swap_release();
    // Decoder SWA bounded replay (LUCE_DS41_DECODER_BOUNDED_REPLAY): the last
    // layer every prompt row runs, or -1 when replay is off or the model has
    // no such cut; and whether a prefill of this kind can replay at all.
    int bounded_replay_cut() const;
    bool bounded_replay_runs(bool images) const;
    std::string                    spec_draft_path_;
    ggml_backend_t                 spec_backend_ = nullptr;
    std::unique_ptr<DSparkDrafter> spec_drafter_;
    std::vector<float>             spec_feat_window_;
    DrafterContext                 pflash_drafter_ctx_;
    bool                           pflash_drafter_loaded_ = false;
    std::string                    pflash_drafter_path_;
    int                            pflash_drafter_gpu_ = -1;
    // Once a long prompt selects the fragmentation-safe prefill shape, retain
    // it for later requests so the HIP arenas never switch back under load.
    int                            hybrid_prefill_chunk_cap_ = 0;
    // Snapshot slot a restored request started from (-1: fresh prompt);
    // a failed prefill restores it before its one retry.
    int                            prefill_retry_slot_ = -1;
    // Whether a retried prefill already turned the pipeline back on.
    bool                           pipeline_retry_recovered_ = false;
    int                            hybrid_long_context_chunk_ = kDs4DefaultLongContextChunk;

    bool load_spec_drafter();
    void release_spec_drafter(bool mark_parked);
    void release_pflash_drafter();
    void keep_spec_feature_tail(std::vector<float> & features,
                                size_t max_rows) const;
    // True when a wide prefill path returns per-token DSpark features and the
    // caller can retain only the requested capture window without splitting.
    static bool supports_batched_spec_feature_capture(
        bool hybrid,
        PrefillAttentionMode mode,
        int n_tokens);
    // Limit a prefill batch to a region with a uniform DSpark capture policy.
    // Wide GPU paths can capture a subrange without splitting the final
    // feature window; other paths still stop exactly at capture boundaries.
    static int capture_safe_prefill_tokens(int token_offset,
                                           int requested_tokens,
                                           int final_capture_from,
                                           bool batch_final_capture,
                                           bool snapshot_pending,
                                           int snapshot_capture_from,
                                           int snapshot_capture_to);

    // Batched mixed-owner prefill: per-token scratch on the target and on the
    // second owner's GPU, the chunk that fits a device, and the chunk length at
    // `pos` that stops at the next restore point.
    struct HybridPrefillScratch {
        size_t target = 0;
        size_t second = 0;
        size_t target_fixed = 0;   // per chunk, whatever its size
    };
    static HybridPrefillScratch hybrid_prefill_scratch_per_token(
        const DeepSeek4Weights & w, int max_ctx, int chunk);
    static int hybrid_prefill_fit_tokens(size_t free_bytes, size_t keep_bytes,
                                         size_t per_token_bytes);
    static int restore_safe_prefill_tokens(int pos, int requested_tokens,
                                           const std::vector<int> & restore_points);

    // Prefill prompt tokens in chunks, return absolute committed position.
    // prefix_tokens > 0 prefills only that many leading tokens (the batched
    // image admission leaves the last prompt token to the paged engine). A
    // batched prefill starts a chunk at every absolute `restore_points`
    // position (see GenerateRequest::restore_points).
    int do_prefill(const std::vector<int32_t> & tokens, const DaemonIO & io,
                   int kv_offset = 0, int snap_slot = -1, int snap_pos = -1,
                   const DeepSeek4ImagePrompt * images = nullptr,
                   int prefix_tokens = 0,
                   const std::vector<int> & restore_points = {});
    bool load_vision();
    bool init_single_gpu_vision();
    // Encodes one image with the vision runtime (caller serialises use).
    bool encode_one_image(const vision::PromptImage & image, vision::ImageRaster & raster,
                          std::string & error);
    // Queues an image request on the encoder worker (--mmproj-device only).
    void enqueue_image_encode(std::shared_ptr<const DeepSeek4ImagePrompt> images);
    void encode_worker_loop();
    void cancel_image_encode(const ImagePromptPayload & images) const;
    // Stops the encoder worker (failing queued requests), then frees vision_.
    void release_vision();
    // Batched serving. A staged prefill fills one request's first `prefix`
    // tokens into its slot's staging cache over several steps, in shared
    // layer-major passes (expert weights read once per pass for all of them)
    // that the engine advances a few layers per step.
    using StagedPrefill = DeepSeek4StagedPrefill;
    DeepSeek4Cache * image_staging_cache(int slot);
    // --ds4-prefill sparse with paged serving: text prompts prefill through
    // the staging caches (layer-major sparse) and are copied into their slot.
    // A heterogeneous placement stages text only through the in-process
    // expert path that run_staged_text_chunk() drives.
    bool text_staging_available() const {
        return text_staging_ &&
               (!w_.moe_hybrid ||
                (moe_hybrid_ && (expert_runtime_.compute || expert_backend_)));
    }
    // The shared staged pass needs the whole model on one GPU. With
    // heterogeneous experts, a text prompt stages one bounded chunk per call
    // through the regular sparse layer-major prefill instead.
    bool staged_text_uses_chunks() const { return text_staging_available() && w_.moe_hybrid; }
    bool run_staged_text_chunk(StagedPrefill & item, int max_rows, std::string & error);
    // Starts encoding an admitted image request: queued on the encoder
    // worker with --mmproj-device, otherwise encoded here.
    bool encode_image_request(const std::vector<int32_t> & prompt, const ImagePromptHandle & images,
                              std::string & error);
    bool begin_staged_prefill(StagedPrefill & item);
    // Starts one shared pass over the ready, unfinished items, about
    // `row_budget` rows in total (a whole image block may exceed it); `rows`
    // gets each item's share. False when no item is ready (or all failed).
    bool begin_staged_pass(const std::vector<StagedPrefill *> & items, int row_budget,
                           DeepSeek4PrefillPass & pass, std::vector<int> & rows);
    // Waits up to `timeout_ms` for the next rows of an item to have their
    // images encoded, so an otherwise idle scheduler does not spin.
    void wait_staged_ready(const StagedPrefill & item, int row_budget, int timeout_ms) const;
    bool materialize_images(const std::shared_ptr<const DeepSeek4ImagePrompt> & images,
                            const DaemonIO & io, std::string & error);

    // Generate after either a fresh prefill or a restored prefix. kv_offset is
    // the number of prompt tokens already represented by cache_ and the
    // auxiliary logits/speculative state.
    GenerateResult generate_from_state(const GenerateRequest & req,
                                       const DaemonIO & io,
                                       int kv_offset);
    bool snapshot_restore(int slot);

    // Autoregressive decode loop.
    bool do_decode(int committed, int n_gen,
                   const std::vector<int32_t> & history_prefix,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io,
                   const BudgetHook & budget_hook = {},
                   bool * forced_close_out = nullptr);

    bool load_model();
    bool init_hybrid_model();
    bool init_streamed_expert_tier();
    bool check_device_headroom() const;
    bool log_device_memory(const char * when) const;
    bool size_hybrid_prefill_chunk();
    bool requires_monolithic_model() const;
    bool validate_prefill_mode() const;
    bool validate_model_features() const;
    bool init_engram();
    bool load_routing_adjustments();
    bool apply_routing_adjustments();
    bool upload_protected_routing();
    bool apply_expert_ownership(bool secondary_owner, int secondary_gpu, MoeHybridConfig & hybrid_cfg);
    void log_route_counts(const char * phase);
    // Zeroes the per-phase route and streamed-cache counters.
    void reset_route_counts();
    bool init_moe_tensor_parallel();
    bool compute_uniform_hybrid_placement(const DeepSeek4Weights & w,
                                          int max_ctx,
                                          MoeHybridPlacement & out,
                                          MoeHybridPlacement * decode_out,
                                          std::string * err) const;
    void maybe_save_routing_stats();

    std::shared_ptr<MoeHybridStorage> moe_hybrid_;
    MoeHybridPlacement                moe_placement_;
    MoeHybridPlacement                moe_decode_placement_;
    MoeHybridStreamEngine             stream_engine_;
    MoeStreamedExpertCache            expert_cache_;
    int                               stream_cache_device_ = -1;
    MoeExpertComputeRuntime            expert_runtime_;
    std::shared_ptr<MoeHybridRoutingStats> routing_stats_;
    std::string                       routing_stats_out_path_;
    friend class DeepSeek4SeqEngine;
};

}  // namespace luce::common
