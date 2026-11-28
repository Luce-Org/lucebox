// Qwen35SeqEngine — SeqEngine implementation for the paged Qwen3.5/3.6
// backend (--max-concurrency N).
//
// Three layers, each with one job:
//   Qwen35SlotManager          host bookkeeping — pool-handle lifecycle,
//                           admission arithmetic, per-slot sampler/RNG/
//                           penalty history, the position counters
//   Qwen35SeqEngine         the device half — chunked slot prefill, the
//                           batched decode forward, sampling, and the
//                           block-table / kv-length uploads
//   Qwen35Backend           the model — weights, cache, step graph, the
//                           paged pool, park/unpark, generate()
//
// The engine borrows the backend's GPU state rather than copying accessors
// for it: it is a friend of Qwen35Backend so that concurrent serving can be
// its own subsystem without widening the backend's public surface.
//
// Single-threaded by the SeqEngine contract — the HTTP scheduler thread is
// the only caller of the engine, the pool, and the device uploads, so there
// is no locking anywhere below here.

#pragma once

#include "common/adaptive_spec_width.h"
#include "common/concurrency/seq_engine.h"
#include "common/concurrency/paged_kv_offload.h"
#include "common/dflash/dflash2_head.h"
#include "common/dflash/dflash_draft_kv.h"
#include "common/dflash/dflash_feature_ring.h"
#include "common/pinned_stage.h"
#include "common/step_graph.h"
#include "qwen35_slot_manager.h"
#include "../qwen35_image_request.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace luce::common {

class BlockingRowPool;

class Qwen35Backend;

struct FixedChainConfig {
    bool enabled = false;
    int width = 0;
    int scratch_base = 0;
    int scratch_stride = 0;
};

class Qwen35SeqEngine final : public SeqEngine {
public:
    // `pool` and `backend` must outlive the engine. `scratch_row` is outside
    // the pool and any per-slot tree slabs; it is the K/V destination of
    // graph-bucket padding rows.
    // `max_prefills` bounds scheduler-selected prompt slices per traversal.
    Qwen35SeqEngine(Qwen35Backend & backend, PagedKvPool & pool,
                    int max_ctx, int64_t scratch_row,
                    FixedChainConfig fixed_chain,
                    int max_prefills = 8,
                    int mixed_prefill_tokens = 2048,
                    int long_mixed_prefill_tokens = 4096,
                    int long_prefill_threshold = 768,
                    int idle_prefill_tokens = 4096,
                    int prefill_quantum = 512,
                    bool target_is_uma = false);
    ~Qwen35SeqEngine() override;

    // Destroy every graph that captures draft-weight tensors. Qwen35Backend
    // calls this before freeing draft weights during park; states rebuild
    // lazily after unpark.
    void release_draft_graphs();
    // Destroy the cached paged-tree target graph (called before freeing target
    // weights/scratch during park, compress_batch or release_scratch).
    void release_target_graphs();

    int slot_count() const override { return slots_.slot_count(); }
    int max_context() const override { return slots_.max_context(); }
    KvPoolTokens kv_pool_tokens() const override { return slots_.kv_pool_tokens(); }
    bool supports_prefix_store() const override { return true; }
    bool prefix_store_in_backend_snapshots() const override { return true; }
    size_t estimate_prefix_store_bytes(int tokens) const override;

    void discard_prefix_store(PrefixStoreRef checkpoint) override;
    bool arm_followup_capture(int slot, PrefixCaptureTicket ticket, int restored_tokens) override;
    size_t estimate_prefix_state_bytes(int tokens) const override;
    bool drop_prefix_kv(PrefixStoreRef checkpoint) override;

    AdmitResult admit(uint64_t request_id,
                      const std::vector<int32_t> & prompt,
                      const SamplerCfg & sampler) override;

    AdmitResult admit_with_prefix(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler,
        const PrefixStorePlan & plan) override;

    StepResult step(const StepPlan & plan) override;
    // A step that carries a prompt chunk decodes without the drafter, one
    // token per request. Beside prompts of more than 8192 new tokens
    // together (a document, or subagents starting at once; not an agent's
    // next step), the decoding requests get half the time back in steps of
    // their own, with them.
    StepPlanLimits step_plan_limits(int decode_rows) const override {
        const bool mixed = decode_rows > 0;
        const int per_sequence = mixed ? 512 : kIdlePrefillChunk;
        int total_cap = idle_prefill_tokens_;
        if (mixed) {
            total_cap = mixed_prefill_tokens_;
            if (slots_.has_prefill_prompt_at_least(long_prefill_threshold_)) {
                total_cap = std::max(total_cap, long_mixed_prefill_tokens_);
            }
        }
        return {
            max_prefills_,
            per_sequence,
            std::min(max_prefills_ * per_sequence, total_cap),
            prefill_quantum_,
            /*prefill_time_share=*/mixed ? 0.5 : 1.0,
            /*time_share_prompt_tokens=*/2048,
        };
    }

    bool reserve_decode(const StepPlan & plan) override;
    size_t kv_offload_capacity() const override { return offload_.capacity(); }
    KvOffloadState kv_offload_state(int slot) const override { return offload_.state(slot); }
    KvOffloadFootprint kv_offload_footprint(int slot) const override {
        return slots_.kv_offload_footprint(slot);
    }
    bool offload_kv(int slot, size_t bytes, std::string & error) override {
        return offload_.suspend(slot, bytes, error);
    }
    bool restore_kv(int slot, std::string & error) override;
    bool evict_kv(int slot, int32_t pending_token, std::string & error) override;
    bool kv_restore_feasible(int slot) const override {
        return slots_.kv_restore_feasible(slot);
    }
    void retire(int slot) override;
    bool supports_images() const override;
    AdmitResult admit_images(uint64_t request_id,
                             const std::vector<int32_t> & prompt,
                             const SamplerCfg & sampler,
                             const ImagePromptHandle & images) override;

    bool token_is_eos(int32_t token) const override;
    int max_decode_tokens() const override {
        return fixed_chain_ready_ ? fixed_chain_.width : 1;
    }


private:
    struct PrefillStage {
        bool ready = false;
        int kv_pos = 0;
        int chunk = 0;
        bool commit = false;
        std::vector<int64_t> rows;
        // Axis-major [4 x chunk] rotary positions when the slot holds images;
        // empty means the plain kv_pos + i positions.
        std::vector<int32_t> positions;
    };

    // Per-slot image state: the payload (kept alive for re-prefill after
    // eviction), its encoded rows, and how far rotary positions run ahead of
    // KV positions after the images.
    struct SlotImages {
        ImagePromptHandle payload;
        Qwen35ImageRows rows;
        int rope_delta = 0;
    };
    std::vector<SlotImages> slot_images_;
    int rope_delta(int slot) const {
        return slot >= 0 && slot < static_cast<int>(slot_images_.size())
            ? slot_images_[static_cast<size_t>(slot)].rope_delta : 0;
    }
    void clear_slot_images(int slot) {
        if (slot >= 0 && slot < static_cast<int>(slot_images_.size())) {
            slot_images_[static_cast<size_t>(slot)] = SlotImages{};
        }
    }

    struct PreparedChainDraft {
        std::vector<int32_t> tokens;
    };
    struct PreparedChainRound {
        std::vector<PreparedChainDraft> drafts;
        int width = 0;  // verify width, chosen once before drafting
    };

    // The widest prompt chunk of one sequence in a step (idle steps; mixed
    // steps take 512).
    static constexpr int kIdlePrefillChunk = 2048;
    int max_prefills_;
    int mixed_prefill_tokens_;
    int long_mixed_prefill_tokens_;
    int long_prefill_threshold_;
    int idle_prefill_tokens_;
    int prefill_quantum_;
    bool target_is_uma_ = false;

    bool upload_block_table_delta(int slot, int first_block,
                                  const int32_t * blocks, size_t count);
    void fail_prefill(int slot, std::vector<PrefillOutput> & outputs,
                              const char * log_message,
                              const char * client_message);
    // Stages one prompt chunk; its input embeddings go to `embeddings`.
    PrefillStage stage_prefill_chunk(int slot, int max_tokens, float * embeddings,
                                     std::vector<PrefillOutput> & outputs);
    // Samples logits row `logits_row` of `src_logits` for `slot`: the graph's
    // argmax when greedy, else the row's top-N candidates when they suffice,
    // else the full row read into `logits`.
    int32_t sample_graph_row(int slot, int logits_row, int32_t argmax,
                             std::vector<float> & logits,
                             const ggml_tensor * src_logits,
                             const int32_t * top_ids, const float * top_vals,
                             int top_n);
    std::vector<uint8_t> select_chain_lanes(
        const StepPlan & plan) const;
    // width: the block the round drafts (0 = the configured block).
    bool chain_spec_input_capable(const StepInput & input, int width = 0) const;
    DraftFeatureMirror * slot_feature_mirror(int slot);
    DraftKvState * ensure_slot_draft_kv(int slot);
    void reset_slot_draft_kv(int slot);
    StepGraph * acquire_paged_tree_graph(
        int tree_width, int tree_bucket, int max_prefix,
        int ar_count, int top_n, bool & rebuilt);
    // The longest block a round with these selected lanes drafts.
    int chain_draft_width(const std::vector<uint8_t> & selected) const;
    std::optional<PreparedChainRound> prepare_chain_drafts(
        const std::vector<StepInput> & inputs,
        const std::vector<uint8_t> & selected);
    StepResult step_chain_spec(
        const StepPlan & plan, const std::vector<uint8_t> & selected,
        PreparedChainRound && prepared);
    // Verify width for one batched chain round. Each verify graph bucket
    // keeps its own AdaptiveSpecWidth: a round's cost depends on the graph's
    // rows (bucket lanes x width), while the width that maximizes committed
    // tokens per unit cost does not depend on how many lanes share it.
    // cap: the longest chain the round may draft (0 = the configured block).
    int choose_chain_width(int bucket, int cap = 0);
    // sampled[lane]: the lane verified by the sampled walk; readback_ms and
    // walk_ms: the round's sampled-lane logits copy and walk times.
    void observe_chain_width(int bucket, int width,
                             const std::vector<size_t> & accepted,
                             const std::vector<uint8_t> & sampled,
                             double step_ms, double readback_ms,
                             double walk_ms);
    PrefixStoreEvent capture_prefix(
        int slot, PrefixCaptureTicket ticket);
    int checkpoint_index(PrefixStoreRef checkpoint) const;
    void unpin_gpu_prefix(int index);
    void unpin_all_gpu_prefixes();
    // Unpins the least recently used pin that is not `skip_index`; false when
    // no block is held by pins alone, since unpinning would then free nothing.
    bool evict_one_gpu_prefix(int skip_index = -1);
    // An admission that is busy only for want of KV blocks: unpinning frees
    // blocks, which cannot help when a request is parked or every slot is taken.
    bool short_of_blocks(const AdmitResult & result) const;

    // Per-checkpoint GPU-resident shared physical blocks (whole blocks only).
    // Pinned blocks are shared zero-copy across concurrent requests matching
    // the prefix and evicted LRU under pool pressure so they never reduce
    // usable KV capacity.
    struct GpuPinnedPrefix {
        int tokens = 0;
        std::vector<uint32_t> blocks;
        uint64_t last_use = 0;
        // PrefixSnapshot::generation of the checkpoint the blocks hold.
        uint64_t snapshot_generation = 0;
    };
    // The pin of checkpoint `index` while that checkpoint still holds the
    // payload it was pinned for; a pin left from an older payload is dropped.
    const GpuPinnedPrefix * live_gpu_prefix(int index);
    bool pin_gpu_prefix(int index, int slot, int tokens);
    std::vector<GpuPinnedPrefix> gpu_prefix_pins_;
    uint64_t gpu_prefix_clock_ = 0;

    PagedKvPool & pool_;
    Qwen35Backend & b_;
    Qwen35SlotManager  slots_;
    PagedKvOffload offload_;
    int64_t         scratch_row_ = 0;
    FixedChainConfig fixed_chain_;
    bool            fixed_chain_ready_ = false;
    struct ChainWidthState {
        AdaptiveSpecWidth controller;
        std::vector<int>  offers;   // rounds offered at each width
        std::vector<int>  samples;  // clean cost samples at each width
        long long rounds = 0;
        // Accepted rows by greedy and sampled lanes, and the sampled lanes'
        // cost: logits copy and walk time over the rounds that had sampled
        // lanes.
        double    greedy_accepted_sum = 0.0;
        size_t    greedy_lane_rounds = 0;
        double    sampled_accepted_sum = 0.0;
        size_t    sampled_lane_rounds = 0;
        long long sampled_rounds = 0;
        double    readback_ms_sum = 0.0;
        double    walk_ms_sum = 0.0;
        // Per-depth trials and acceptances (decayed). A depth is a trial
        // only when it was offered and every shallower candidate was
        // accepted, so a narrow width does not bias the deeper estimates.
        std::vector<double> depth_trials;
        std::vector<double> depth_accepts;
        std::vector<float>  conditional;  // reused per round
        // Consecutive clean rounds rejected as host-contention outliers, and
        // the round at which every width is timed again.
        int       slow_streak = 0;
        long long recalibrate_at = 0;
        explicit ChainWidthState(int max_width)
            : controller(max_width, 2, true),
              offers(static_cast<size_t>(max_width) + 1, 0),
              samples(static_cast<size_t>(max_width) + 1, 0),
              depth_trials(static_cast<size_t>(max_width), 0.0),
              depth_accepts(static_cast<size_t>(max_width), 0.0),
              conditional(static_cast<size_t>(std::max(1, max_width - 1)), 0.7f) {}
    };
    bool adaptive_chain_width_ = false;
    std::vector<int> chain_width_choices_;  // ascending, includes the block
    std::map<int, ChainWidthState> chain_width_by_bucket_;
    std::chrono::steady_clock::time_point chain_round_t0_{};
    std::vector<size_t> chain_accepted_scratch_;
    std::vector<uint8_t> chain_sampled_scratch_;
    // Logits rows sampled on the host: each sampled chain lane's verify rows,
    // or the full rows a plain step() samples, and the pool those walks run
    // on (made on the first round with two such rows).
    std::vector<float> chain_sampled_rows_;
    std::unique_ptr<BlockingRowPool> chain_sampled_pool_;
    ggml_context *  feature_view_ctx_ = nullptr;
    std::vector<DraftFeatureMirror> slot_feature_mirrors_;
    // Per-slot draft KV state sized for fixed_chain_.width; narrower batched
    // rounds use views of its leading noise/append rows.
    std::vector<std::unique_ptr<DraftKvState>> slot_draft_kv_;
    std::vector<std::unique_ptr<DraftKvState>> dummy_draft_kv_;
    std::vector<std::unique_ptr<DraftKvBatchGraph>> batch_draft_graphs_;
    DFlash2BatchSelectorCache dflash2_selector_cache_;
    // Rounds with two or more lanes draft a shorter block than the configured
    // one: a long block pays for itself only when one request decodes alone
    // (verify cost grows with lanes x width, acceptance stays per lane).
    // Such a round drafts exactly the verify width its bucket's controller
    // picks, at most batched_draft_width_. 0 = every round drafts the
    // configured block.
    int batched_draft_width_ = 0;
    std::vector<float> mask_embed_;
    int32_t mask_embed_token_id_ = -1;
    PinnedStage draft_io_stage_;

    // Dedicated paged-tree target graph with its own gallocr so mixed
    // prefill+decode steps on b_.sg_ never clobber the active verify graph or
    // its captured HIP graph, without aliasing gallocr buffers across graphs.
    StepGraph paged_tree_sg_;
    PinnedStage target_io_stage_;
    PinnedStage commit_io_stage_;

    // Hoisted per-step buffers (reused across step() and step_chain_spec()).
    std::vector<int>         reserve_growth_;
    std::vector<uint8_t>     step_seen_;   // step(): rows seen per slot
    std::vector<int>         output_rows_;
    std::vector<int32_t>     live_tokens_;
    std::vector<int32_t>     live_positions_;
    std::vector<int64_t>     live_physical_rows_;
    std::vector<int32_t>     live_slot_ids_;
    std::vector<int32_t>     dec_tokens_;
    std::vector<int64_t>     dec_rows_;
    std::vector<int32_t>     active_slot_ids_;
    std::vector<int32_t>     state_slot_ids_;
    std::vector<int32_t>     seq_lens_;
    std::vector<int> restored_kv_checkpoint_;
    std::vector<int> restored_kv_tokens_;
    std::vector<int32_t>     query_slot_ids_;
    std::vector<int32_t>     query_positions_;
    std::vector<int32_t>     logits_rows_;
    std::vector<int32_t>     feature_rows_;
    // The packed step's input embeddings, `embed_rows_cap_` rows of n_embd
    // floats, reserved once at construction: the most prompt rows any
    // step_plan_limits() allows in one step (idle, mixed or long mixed),
    // plus the decode bucket of every slot. Pinned host memory, so prompt
    // chunks are dequantized straight into it and copied to the device by
    // DMA; pageable memory if pinned memory is unavailable. Every step
    // synchronizes before it returns, so the next step may overwrite it.
    PinnedStage              embed_stage_;
    std::vector<float>       embed_rows_pageable_;
    float *                  embed_rows_ = nullptr;
    size_t                   embed_rows_cap_ = 0;
    std::vector<int32_t>     pos_buf_;
    std::vector<int64_t>     rows_buf_;
    std::vector<int32_t>     argmax_buf_;
    std::vector<float>       logits_buf_;
    std::vector<int32_t>     top_ids_buf_;
    std::vector<float>       top_vals_buf_;

    // Hoisted scratch buffers for step_chain_spec().
    std::vector<int32_t>     chain_tokens_;
    std::vector<int32_t>     chain_parents_;
    std::vector<int32_t>     chain_tree_sizes_;
    std::vector<int32_t>     chain_mapped_slots_;
    std::vector<int32_t>     chain_state_slots_;
    std::vector<int32_t>     chain_query_slots_;
    std::vector<int32_t>     chain_query_positions_;
    std::vector<int64_t>     chain_write_rows_;
    std::vector<int32_t>     chain_positions_;
    std::vector<float>       chain_embeddings_;
    std::vector<int32_t>     chain_posterior_;
    std::vector<int32_t>     chain_accepted_prefixes_;
    std::vector<int32_t>     chain_commit_slots_;
    std::vector<int64_t>     chain_commit_rows_;
    std::vector<int32_t>     chain_feature_commit_rows_;
    std::vector<const ggml_tensor *> chain_replay_logs_;
    std::vector<ggml_tensor *>       chain_states_;
    std::vector<const ggml_tensor *> chain_conv_inputs_;
    std::vector<ggml_tensor *>       chain_conv_states_;
    std::vector<ggml_tensor *>       chain_caches_;
};

}  // namespace luce::common
