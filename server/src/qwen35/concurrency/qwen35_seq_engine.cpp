// Concurrent slot engine for the paged Qwen3.5/3.6 backend
// (--max-concurrency N).
//
// All calls come from the HTTP scheduler thread, which is also the only
// caller of the pool, step graph, and device metadata uploads.

#include "qwen35_seq_engine.h"

#include "qwen35_backend.h"
#include "qwen35_roctx.h"
#include "graph_builders.h"
#include "attn_masks.h"
#include "prefill_helpers.h"
#include "common/blocking_row_pool.h"
#include "common/concurrency/chain_spec_shapes.h"
#include "common/dflash2_head.h"
#include "common/sampled_verify.h"
#include "common/sampler.h"
#include "internal.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace luce::common {

namespace {
std::vector<PagedKvTensor> qwen_paged_kv_planes(const TargetCache & cache,
                                               uint32_t block_size) {
    std::vector<PagedKvTensor> planes;
    for (const auto * tensors : {&cache.attn_k, &cache.attn_v}) {
        for (ggml_tensor * tensor : *tensors) {
            for (int64_t head = 0; head < tensor->ne[2]; ++head) {
                planes.push_back({tensor, static_cast<size_t>(head) * tensor->nb[2],
                                   block_size * tensor->nb[1]});
            }
        }
    }
    return planes;
}

// Sampled verify, the single-lane switch (on by default,
// LUCE_SAMPLED_VERIFY=0 off): sampled lanes join batched chain rounds,
// verified by the single-lane sampled walk.
bool sampled_verify_enabled() {
    static const bool enabled = luce::common::sampled_verify_switch(true);
    return enabled;
}
} // namespace

Qwen35SeqEngine::Qwen35SeqEngine(
        Qwen35Backend & backend, PagedKvPool & pool, int max_ctx,
        int64_t scratch_row, FixedChainConfig fixed_chain,
        int max_prefills,
        int mixed_prefill_tokens, int long_mixed_prefill_tokens,
        int long_prefill_threshold, int idle_prefill_tokens,
        int prefill_quantum)
    : max_prefills_(std::max(1, max_prefills)),
      mixed_prefill_tokens_(std::max(1, mixed_prefill_tokens)),
      long_mixed_prefill_tokens_(std::max(1, long_mixed_prefill_tokens)),
      long_prefill_threshold_(std::max(1, long_prefill_threshold)),
      idle_prefill_tokens_(std::max(1, idle_prefill_tokens)),
      prefill_quantum_(std::max(1, prefill_quantum)), pool_(pool),
      b_(backend), slots_(pool, max_ctx),
      offload_(slots_, pool, backend.target_backend_,
               qwen_paged_kv_planes(backend.cache_, pool.block_size())),
      scratch_row_(scratch_row),
      fixed_chain_(fixed_chain) {
    const int n_slots = slots_.slot_count();
    slot_draft_kv_.resize(static_cast<size_t>(n_slots));
    seq_lens_.assign(static_cast<size_t>(n_slots), 0);
    restored_kv_checkpoint_.assign(static_cast<size_t>(n_slots), -1);
    restored_kv_tokens_.assign(static_cast<size_t>(n_slots), 0);
    reserve_growth_.assign(static_cast<size_t>(n_slots), 0);
    step_seen_.assign(static_cast<size_t>(n_slots), 0);
    gpu_prefix_pins_.resize(static_cast<size_t>(Qwen35Backend::PREFIX_SLOTS));

    embed_rows_cap_ = static_cast<size_t>(
        std::min(max_prefills_ * kIdlePrefillChunk,
                 std::max({idle_prefill_tokens_, mixed_prefill_tokens_,
                           long_mixed_prefill_tokens_})) +
        chain_decode_bucket_width(n_slots));
    const size_t embed_bytes =
        embed_rows_cap_ * static_cast<size_t>(b_.w_.n_embd) * sizeof(float);
    if (embed_stage_.reserve(b_.target_backend_, embed_bytes)) {
        embed_rows_ = static_cast<float *>(embed_stage_.take(embed_bytes));
    } else {
        std::fprintf(stderr,
            "[parallel] no pinned memory for %zu MiB of input rows; "
            "using pageable memory\n", embed_bytes >> 20);
        embed_rows_pageable_.resize(embed_bytes / sizeof(float));
        embed_rows_ = embed_rows_pageable_.data();
    }

    fixed_chain_ready_ = fixed_chain_.enabled && fixed_chain_.width > 1 &&
        fixed_chain_.width <= 16 &&
        fixed_chain_.width == b_.dw_.block_size &&
        fixed_chain_.scratch_stride >= fixed_chain_.width &&
        b_.cache_.target_feat && b_.cache_.target_feat_cap > 0;
    if (!fixed_chain_ready_) return;

    // Candidate verify widths: a few shapes keep graph builds and captures
    // bounded while covering the measured optimum at 1..8 lanes. Width 6 is
    // the 8-lane optimum on an R9700 (Qwen 3.8 27B + DFlash2): 91 ms rounds
    // instead of 104 ms at width 8 for 5% fewer accepted tokens.
    // On by default for batched serving (measured: never slower than the
    // fixed block, +6-37% aggregate at 2-8 lanes); LUCE_ADAPTIVE_SPEC_WIDTH=0
    // restores the fixed block width.
    {
        const char * env = std::getenv("LUCE_ADAPTIVE_SPEC_WIDTH");
        adaptive_chain_width_ = !(env && env[0] == '0' && env[1] == '\0');
    }
    for (int width : {4, 6, 8, fixed_chain_.width}) {
        if (width >= 2 && width <= fixed_chain_.width &&
            std::find(chain_width_choices_.begin(), chain_width_choices_.end(),
                      width) == chain_width_choices_.end()) {
            chain_width_choices_.push_back(width);
        }
    }
    std::sort(chain_width_choices_.begin(), chain_width_choices_.end());
    // Batched rounds draft at most the native DFlash2 block (8) when the
    // configured block is longer. Measured on an R9700, Qwen 3.8 27B +
    // DFlash2 at block 16: 8 lanes 146 -> 189 tok/s; one lane keeps block
    // 16, unchanged. Below that they draft exactly the chosen verify width:
    // at 8 lanes a width-6 round that drafts 6 instead of 8 takes 89.0
    // instead of 91.5 ms at equal acceptance.
    // LUCE_BATCHED_DRAFT_WIDTH=<n> picks another maximum, 0 turns it off.
    {
        const char * env = std::getenv("LUCE_BATCHED_DRAFT_WIDTH");
        const int want = env ? std::atoi(env) : 8;
        const bool supported = want >= 2 && want < fixed_chain_.width &&
            std::find(chain_width_choices_.begin(), chain_width_choices_.end(),
                      want) != chain_width_choices_.end();
        if (n_slots > 1 && supported) {
            batched_draft_width_ = want;
            std::fprintf(stderr,
                "[parallel] batched rounds draft the verify width, at most "
                "%d (one lane: %d)\n",
                want, fixed_chain_.width);
        } else if (env && want != 0 && !supported) {
            // only the verify widths below the block have graphs
            std::fprintf(stderr,
                "[parallel] LUCE_BATCHED_DRAFT_WIDTH=%s ignored: use 0 or a "
                "verify width below the block (",
                env);
            for (int width : chain_width_choices_) {
                if (width < fixed_chain_.width) std::fprintf(stderr, " %d", width);
            }
            std::fprintf(stderr, " ); batched rounds draft block %d\n",
                         fixed_chain_.width);
        }
    }
    if (adaptive_chain_width_) {
        std::fprintf(stderr,
            "[parallel] adaptive DFlash2 verify width: widths");
        for (int width : chain_width_choices_) {
            std::fprintf(stderr, " %d", width);
        }
        std::fprintf(stderr, " (per batch bucket, acceptance x cost)\n");
    }

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() *
        static_cast<size_t>(n_slots + 1);
    params.no_alloc = true;
    feature_view_ctx_ = ggml_init(params);
    if (!feature_view_ctx_) {
        fixed_chain_ready_ = false;
        return;
    }

    const int cap = b_.cache_.target_feat_cap;
    const int64_t feature_width =
        static_cast<int64_t>(b_.w_.n_capture_layers) * b_.w_.n_embd;
    slot_feature_mirrors_.resize(static_cast<size_t>(n_slots));
    for (int slot = 0; slot < n_slots; ++slot) {
        DraftFeatureMirror & mirror =
            slot_feature_mirrors_[static_cast<size_t>(slot)];
        mirror.target_feat = ggml_view_2d(
            feature_view_ctx_, b_.cache_.target_feat, feature_width, cap,
            b_.cache_.target_feat->nb[1],
            static_cast<size_t>(slot) * static_cast<size_t>(cap) *
                b_.cache_.target_feat->nb[1]);
        mirror.device = b_.cfg_.draft_gpu;
        mirror.target_device = b_.cfg_.device.gpu;
        mirror.cap = cap;
        mirror.n_target_layers = b_.w_.n_capture_layers;
        mirror.hidden_size = b_.w_.n_embd;
        mirror.storage_type = b_.cache_.target_feat->type;
    }

    if (n_slots <= 6 && fixed_chain_.width == 8) {
        bool draft_states_ready = true;
        for (int slot = 0; slot < n_slots; ++slot) {
            draft_states_ready =
                ensure_slot_draft_kv(slot) && draft_states_ready;
        }
        if (draft_states_ready && n_slots >= 5) {
            auto dummy = std::make_unique<DraftKvState>();
            const int draft_cap = dflash_draft_context_cap(
                cap, b_.cfg_.draft_ctx_max);
            if (draft_kv_init_batched(
                    *dummy, b_.dw_, b_.draft_backend_, draft_cap)) {
                dummy_draft_kv_.push_back(std::move(dummy));
            } else {
                draft_kv_free(*dummy);
                draft_states_ready = false;
            }
        }

        if (draft_states_ready) {
            std::fprintf(stderr,
                "[parallel-chain] preallocated C=%d/W8 draft states\n",
                n_slots);
        } else {
            std::fprintf(stderr,
                "[parallel-chain] C=%d/W8 draft-state preallocation incomplete; "
                "falling back to lazy setup\n",
                n_slots);
        }
    }
}

Qwen35SeqEngine::~Qwen35SeqEngine() {
    if (b_.target_backend_) {
        ggml_backend_synchronize(b_.target_backend_);
    }
    unpin_all_gpu_prefixes();
    release_target_graphs();
    release_draft_graphs();
    for (DraftFeatureMirror & mirror : slot_feature_mirrors_) {
        draft_feature_mirror_free(mirror);
    }
    if (feature_view_ctx_) {
        ggml_free(feature_view_ctx_);
        feature_view_ctx_ = nullptr;
    }
}

void Qwen35SeqEngine::release_target_graphs() {
    step_graph_destroy(paged_tree_sg_);
}

void Qwen35SeqEngine::release_draft_graphs() {
    for (std::unique_ptr<DraftKvBatchGraph> & g : batch_draft_graphs_) {
        if (g) draft_kv_batch_free(*g);
    }
    batch_draft_graphs_.clear();
    for (std::unique_ptr<DraftKvState> & state : dummy_draft_kv_) {
        if (state) draft_kv_free(*state);
    }
    dummy_draft_kv_.clear();
    for (std::unique_ptr<DraftKvState> & state : slot_draft_kv_) {
        if (state) {
            draft_kv_free(*state);
            state.reset();
        }
    }
}

int Qwen35SeqEngine::chain_draft_width(
        const std::vector<uint8_t> & selected) const {
    const bool batched = batched_draft_width_ > 0 &&
        std::count(selected.begin(), selected.end(), uint8_t{1}) >= 2;
    return batched ? batched_draft_width_ : fixed_chain_.width;
}

void Qwen35SeqEngine::reset_slot_draft_kv(int slot) {
    if (slot >= 0 && slot < static_cast<int>(slot_draft_kv_.size()) &&
        slot_draft_kv_[static_cast<size_t>(slot)]) {
        draft_kv_reset(*slot_draft_kv_[static_cast<size_t>(slot)]);
    }
}

DraftFeatureMirror * Qwen35SeqEngine::slot_feature_mirror(int slot) {
    if (!fixed_chain_ready_ || slot < 0 ||
        slot >= static_cast<int>(slot_feature_mirrors_.size())) {
        return nullptr;
    }
    return &slot_feature_mirrors_[static_cast<size_t>(slot)];
}

DraftKvState * Qwen35SeqEngine::ensure_slot_draft_kv(int slot) {
    DraftFeatureMirror * mirror = slot_feature_mirror(slot);
    const DraftWeights & dw = b_.dw_;
    if (!mirror || slot < 0 ||
        slot >= static_cast<int>(slot_draft_kv_.size())) {
        return nullptr;
    }
    std::unique_ptr<DraftKvState> & state =
        slot_draft_kv_[static_cast<size_t>(slot)];
    if (state && state->mem_buf &&
        state->built_for == static_cast<const void *>(&dw)) {
        return state.get();
    }
    if (state) draft_kv_free(*state);
    state = std::make_unique<DraftKvState>();
    const int cap = dflash_draft_context_cap(
        mirror->cap, b_.cfg_.draft_ctx_max);
    if (!draft_kv_init_batched(
            *state, dw, b_.draft_backend_, cap)) {
        draft_kv_free(*state);
        state.reset();
        return nullptr;
    }
    return state.get();
}

StepGraph * Qwen35SeqEngine::acquire_paged_tree_graph(
        int tree_width, int tree_bucket, int max_prefix,
        int ar_count, int top_n, bool & rebuilt) {
    const ggml_cgraph * before = paged_tree_sg_.gf;
    if (!build_target_step_paged_tree(
            paged_tree_sg_, b_.w_, b_.cache_, b_.target_backend_,
            tree_width, tree_bucket, max_prefix,
            fixed_chain_.scratch_base, fixed_chain_.scratch_stride,
            b_.cfg_.kq_stride_pad, ar_count, top_n)) {
        rebuilt = false;
        return nullptr;
    }
    rebuilt = paged_tree_sg_.gf != before;
    return &paged_tree_sg_;
}

bool Qwen35SeqEngine::chain_spec_input_capable(
        const StepInput & input, int width) const {
    if (!fixed_chain_ready_ || !input.allow_speculation ||
        input.slot < 0 || input.slot >= slots_.slot_count()) {
        return false;
    }
    const Qwen35Slot & slot = slots_.slot(input.slot);
    return slot.decoding() &&
           chain_sampler_spec_capable(slot.sampler, sampled_verify_enabled()) &&
           slot.cur_pos >= 1 &&
           slot.cur_pos + (width > 0 ? width : fixed_chain_.width) <=
               slots_.max_context();
}

std::vector<uint8_t>
Qwen35SeqEngine::select_chain_lanes(const StepPlan & plan) const {
    std::vector<uint8_t> selected(plan.decode.size(), 0);
    if (!plan.prefills.empty()) return selected;
    // Two or more lanes draft at most the batched block, so near the context
    // limit a lane qualifies when that shorter block fits; a lane alone
    // drafts the configured block and needs room for all of it.
    if (batched_draft_width_ > 0) {
        size_t lanes = 0;
        for (size_t i = 0; i < plan.decode.size(); ++i) {
            selected[i] = chain_spec_input_capable(
                plan.decode[i], batched_draft_width_) ? 1 : 0;
            lanes += selected[i];
        }
        if (lanes >= 2) return selected;
    }
    for (size_t i = 0; i < plan.decode.size(); ++i) {
        selected[i] = chain_spec_input_capable(plan.decode[i]) ? 1 : 0;
    }
    return selected;
}

std::optional<Qwen35SeqEngine::PreparedChainRound>
Qwen35SeqEngine::prepare_chain_drafts(
        const std::vector<StepInput> & inputs,
        const std::vector<uint8_t> & selected) {
    if (selected.size() != inputs.size() || !fixed_chain_ready_ ||
        fixed_chain_.width <= 1 || fixed_chain_.width != b_.dw_.block_size) {
        return std::nullopt;
    }
    PreparedChainRound round;
    round.drafts.resize(inputs.size());
    // The round's width comes from the controller of the verify graph bucket
    // its lanes fill. Two or more lanes draft exactly that width (at most
    // batched_draft_width_) on the unified per-slot DraftKvState using leading
    // views; one lane drafts the configured block and verifies its prefix.
    const int max_width = chain_draft_width(selected);
    const int lane_count = static_cast<int>(
        std::count(selected.begin(), selected.end(), uint8_t{1}));
    round.width = choose_chain_width(
        chain_decode_bucket_width(lane_count), max_width);
    const bool batched = max_width != fixed_chain_.width;
    const int draft_width = batched ? round.width : max_width;
    // Active append width: always covers max accepted tokens from any single-lane
    // or batched round (fixed_chain_.width + 2) so 1<->N transitions never stall
    // on draft_kv_bulk_append, while keeping batched appends compact.
    const int active_a_step = batched
        ? std::min(2 * fixed_chain_.width + 2,
                   std::max(fixed_chain_.width + 2, 2 * draft_width + 2))
        : 0;
    const DraftWeights & dw = b_.dw_;
    std::vector<std::unique_ptr<DraftKvState>> & dummies = dummy_draft_kv_;

    struct Lane {
        size_t input_index = 0;
        int slot = -1;
        int32_t root = -1;
        DraftKvState * state = nullptr;
        DraftFeatureMirror * mirror = nullptr;
    };
    std::vector<Lane> lanes;
    lanes.reserve(inputs.size());
    const int hidden = b_.w_.n_embd;

    // Cache the constant mask-token embedding once so each lane only embeds
    // its single root token.
    if (mask_embed_.size() != static_cast<size_t>(hidden) ||
        mask_embed_token_id_ != b_.w_.mask_token_id) {
        mask_embed_.resize(static_cast<size_t>(hidden));
        const int32_t mask_id = b_.w_.mask_token_id;
        if (!b_.w_.embedder.embed(&mask_id, 1, mask_embed_.data())) {
            mask_embed_.clear();
            return std::nullopt;
        }
        mask_embed_token_id_ = mask_id;
    }

    auto reset_lanes = [&]() {
        for (const Lane & lane : lanes) {
            if (lane.state) draft_kv_reset(*lane.state);
        }
        for (const std::unique_ptr<DraftKvState> & state : dummies) {
            if (state) draft_kv_reset(*state);
        }
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!selected[i]) continue;
        const StepInput & input = inputs[i];
        if (!chain_spec_input_capable(input, draft_width)) {
            reset_lanes();
            return std::nullopt;
        }
        DraftKvState * state = ensure_slot_draft_kv(input.slot);
        DraftFeatureMirror * mirror = slot_feature_mirror(input.slot);
        if (!state || !mirror) {
            if (state) draft_kv_reset(*state);
            reset_lanes();
            return std::nullopt;
        }
        lanes.push_back({i, input.slot, input.token, state, mirror});
    }
    if (lanes.empty()) return round;

    const int bucket = chain_draft_bucket_width(
        static_cast<int>(lanes.size()));
    const int dummy_count = bucket - static_cast<int>(lanes.size());
    const int cap = dflash_draft_context_cap(
        lanes.front().mirror->cap, b_.cfg_.draft_ctx_max);
    while (static_cast<int>(dummies.size()) < dummy_count) {
        auto dummy = std::make_unique<DraftKvState>();
        if (!draft_kv_init_batched(
                *dummy, dw, b_.draft_backend_, cap)) {
            draft_kv_free(*dummy);
            reset_lanes();
            return std::nullopt;
        }
        dummies.push_back(std::move(dummy));
    }

    // Reserve one PinnedStage slice pool covering all bucket lanes' append
    // metadata, masks, and input embeddings so uploads issue asynchronously.
    const DraftKvState & ref_state = *lanes.front().state;
    const size_t per_lane_embed_bytes =
        sizeof(float) * static_cast<size_t>(hidden) * static_cast<size_t>(draft_width);
    const size_t per_lane_mask_bytes =
        sizeof(uint16_t) * static_cast<size_t>(ref_state.kv_total) *
        static_cast<size_t>(draft_width);
    const size_t per_lane_meta =
        per_lane_embed_bytes + 2 * per_lane_mask_bytes +
        sizeof(int32_t) * static_cast<size_t>(2 * ref_state.a_step + draft_width) + 512;
    draft_io_stage_.reserve(
        b_.draft_backend_, per_lane_meta * static_cast<size_t>(bucket));
    draft_io_stage_.reset();

    std::vector<float> embed_fallback;
    auto upload_lane_embed = [&](DraftKvState * state, int32_t root_token) -> bool {
        float * pinned = static_cast<float *>(draft_io_stage_.take(per_lane_embed_bytes));
        float * dst = pinned;
        if (!dst) {
            embed_fallback.resize(static_cast<size_t>(hidden) * static_cast<size_t>(draft_width));
            dst = embed_fallback.data();
        }
        if (!b_.w_.embedder.embed(&root_token, 1, dst)) return false;
        const size_t row_bytes = sizeof(float) * static_cast<size_t>(hidden);
        for (int r = 1; r < draft_width; ++r) {
            std::memcpy(dst + static_cast<size_t>(r) * static_cast<size_t>(hidden),
                        mask_embed_.data(), row_bytes);
        }
        if (pinned) {
            ggml_backend_tensor_set_async(
                b_.draft_backend_, state->inp_embed, pinned, 0, per_lane_embed_bytes);
        } else {
            ggml_backend_tensor_set(
                state->inp_embed, dst, 0, per_lane_embed_bytes);
        }
        return true;
    };

    for (const Lane & lane : lanes) {
        if (!draft_kv_begin_step(
                *lane.state, dw, b_.draft_backend_, *lane.mirror,
                slots_.slot(lane.slot).cur_pos, draft_width, active_a_step,
                /*defer_sync=*/true, &draft_io_stage_) ||
            !upload_lane_embed(lane.state, lane.root)) {
            reset_lanes();
            return std::nullopt;
        }
    }

    std::vector<DraftKvState *> batch_states;
    std::vector<int32_t> real_roots;
    batch_states.reserve(static_cast<size_t>(bucket));
    real_roots.reserve(lanes.size());
    for (const Lane & lane : lanes) {
        batch_states.push_back(lane.state);
        real_roots.push_back(lane.root);
    }

    for (int i = 0; i < dummy_count; ++i) {
        DraftKvState * dummy = dummies[static_cast<size_t>(i)].get();
        if (!draft_kv_begin_step(
                *dummy, dw, b_.draft_backend_, *lanes.front().mirror, 1,
                draft_width, active_a_step, /*defer_sync=*/true,
                &draft_io_stage_) ||
            !upload_lane_embed(dummy, lanes.front().root)) {
            reset_lanes();
            return std::nullopt;
        }
        batch_states.push_back(dummy);
    }

    // One copy-stream sync after queuing feature copies for every lane.
    if (!sync_feature_ring_copy_stream(lanes.front().mirror->device)) {
        reset_lanes();
        return std::nullopt;
    }

    // Find or allocate a cached DraftKvBatchGraph matching (lane_states, width, a_step).
    DraftKvBatchGraph * draft_graph = nullptr;
    for (auto it = batch_draft_graphs_.begin(); it != batch_draft_graphs_.end(); ++it) {
        DraftKvBatchGraph & g = **it;
        if (draft_kv_batch_reusable(g, dw, b_.draft_backend_, batch_states,
                                    draft_width, active_a_step)) {
            if (it + 1 != batch_draft_graphs_.end()) {
                auto found = std::move(*it);
                batch_draft_graphs_.erase(it);
                batch_draft_graphs_.push_back(std::move(found));
            }
            draft_graph = batch_draft_graphs_.back().get();
            break;
        }
    }
    if (!draft_graph) {
        constexpr size_t kMaxDraftGraphs = 16;
        if (batch_draft_graphs_.size() >= kMaxDraftGraphs) {
            draft_kv_batch_free(*batch_draft_graphs_.front());
            batch_draft_graphs_.erase(batch_draft_graphs_.begin());
        }
        batch_draft_graphs_.push_back(std::make_unique<DraftKvBatchGraph>());
        draft_graph = batch_draft_graphs_.back().get();
    }

    bool drafted = draft_kv_batch_compute_device(
        *draft_graph, dw, b_.draft_backend_,
        batch_states, draft_width, active_a_step);
    // A graph that could not be allocated (build failed: no gf) may fit once
    // the coldest cached shapes give their buffers back.
    while (!drafted && !draft_graph->gf && batch_draft_graphs_.size() > 1) {
        draft_kv_batch_free(*batch_draft_graphs_.front());
        batch_draft_graphs_.erase(batch_draft_graphs_.begin());
        drafted = draft_kv_batch_compute_device(
            *draft_graph, dw, b_.draft_backend_,
            batch_states, draft_width, active_a_step);
    }
    if (!drafted || draft_graph->hidden_by_lane.size() < lanes.size()) {
        reset_lanes();
        return std::nullopt;
    }

    // Only pass the real lanes (skip dummy padding lanes) and keep their
    // hidden tensors on-device straight into the lm_head + selector graphs.
    std::vector<ggml_tensor *> real_hidden_tensors(
        draft_graph->hidden_by_lane.begin(),
        draft_graph->hidden_by_lane.begin() + static_cast<std::ptrdiff_t>(lanes.size()));
    std::vector<std::vector<int32_t>> proposals;
    if (!dflash2_select_chains_batched_device(
            dw, b_.draft_backend_, b_.w_.output, real_hidden_tensors,
            draft_width, real_roots, proposals) ||
        proposals.size() != lanes.size()) {
        reset_lanes();
        return std::nullopt;
    }

    for (size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
        const Lane & lane = lanes[lane_index];
        std::vector<int32_t> & tokens = proposals[lane_index];
        if (tokens.size() != static_cast<size_t>(draft_width) ||
            tokens.front() != lane.root) {
            reset_lanes();
            return std::nullopt;
        }
        PreparedChainDraft & prepared =
            round.drafts[lane.input_index];
        prepared.tokens = std::move(tokens);
    }
    return round;
}

bool Qwen35SeqEngine::token_is_eos(int32_t token) const {
    return b_.token_is_eos(token);
}

SeqEngine::AdmitResult Qwen35SeqEngine::admit(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    AdmitResult result = slots_.admit(request_id, prompt, sampler);
    while (short_of_blocks(result) && evict_one_gpu_prefix()) {
        result = slots_.admit(request_id, prompt, sampler);
    }
    if (result.status == AdmitResult::Status::admitted) {
        if (result.slot >= 0 && (size_t)result.slot < restored_kv_checkpoint_.size()) {
            restored_kv_checkpoint_[(size_t)result.slot] = -1;
            restored_kv_tokens_[(size_t)result.slot] = 0;
        }
        clear_slot_images(result.slot);
        reset_recurrent_slot(b_.cache_, result.slot);
        reset_slot_draft_kv(result.slot);
    }
    return result;
}

bool Qwen35SeqEngine::supports_images() const {
    return b_.supports_images();
}

SeqEngine::AdmitResult Qwen35SeqEngine::admit_images(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler,
        const ImagePromptHandle & images) {
    AdmitResult refused;
    refused.status = AdmitResult::Status::failed;
    const auto * payload = dynamic_cast<const Qwen35ImagePrompt *>(images.get());
    if (!payload || payload->owner != &b_ || !payload->matches(prompt)) {
        refused.error = "image binding does not match this prompt";
        return refused;
    }
    // Claim the slot first: a busy pool defers the request and retries it,
    // and encoding before that would rerun the tower on every retry. The
    // tower runs on this (the scheduler) thread, so no step sees the slot
    // before its images are in place.
    AdmitResult result = admit(request_id, prompt, sampler);
    if (result.status != AdmitResult::Status::admitted) return result;
    Qwen35ImageRows rows;
    std::string error = "image encoding failed";
    if (!b_.encode_images(*payload, rows, error)) {
        retire(result.slot);
        refused.error = error;
        return refused;
    }
    if (slot_images_.size() < static_cast<size_t>(slots_.slot_count())) {
        slot_images_.resize(static_cast<size_t>(slots_.slot_count()));
    }
    SlotImages & state = slot_images_[static_cast<size_t>(result.slot)];
    state.payload = images;
    state.rows = std::move(rows);
    state.rows.prompt = payload;
    state.rope_delta = payload->positions.next - static_cast<int>(prompt.size());
    return result;
}

size_t Qwen35SeqEngine::estimate_prefix_store_bytes(int tokens) const {
    return estimate_paged_target_cache_snapshot_bytes(b_.cache_, tokens);
}

int Qwen35SeqEngine::checkpoint_index(PrefixStoreRef checkpoint) const {
    if (!checkpoint.valid() ||
        checkpoint.id > (uint64_t)Qwen35Backend::PREFIX_SLOTS) return -1;
    return (int)checkpoint.id - 1;
}

void Qwen35SeqEngine::unpin_gpu_prefix(int index) {
    if (index < 0 || static_cast<size_t>(index) >= gpu_prefix_pins_.size()) return;
    GpuPinnedPrefix & pin = gpu_prefix_pins_[static_cast<size_t>(index)];
    if (!pin.blocks.empty()) {
        pool_.unpin_prefix_blocks(pin.blocks);
        pin.blocks.clear();
    }
    pin.tokens = 0;
    pin.last_use = 0;
}

const Qwen35SeqEngine::GpuPinnedPrefix * Qwen35SeqEngine::live_gpu_prefix(int index) {
    if (index < 0 || static_cast<size_t>(index) >= gpu_prefix_pins_.size()) return nullptr;
    const GpuPinnedPrefix & pin = gpu_prefix_pins_[static_cast<size_t>(index)];
    if (pin.blocks.empty()) return nullptr;
    if (pin.snapshot_generation != b_.prefix_snapshots_[index].generation) {
        unpin_gpu_prefix(index);
        return nullptr;
    }
    return &pin;
}

// Pin the full blocks of `slot`'s first `tokens` positions for checkpoint
// `index`, whose payload holds those same positions now.
bool Qwen35SeqEngine::pin_gpu_prefix(int index, int slot, int tokens) {
    const uint32_t full_blocks = static_cast<uint32_t>(tokens) / pool_.block_size();
    if (full_blocks == 0 || index < 0 ||
        static_cast<size_t>(index) >= gpu_prefix_pins_.size()) return false;
    unpin_gpu_prefix(index);
    GpuPinnedPrefix & pin = gpu_prefix_pins_[static_cast<size_t>(index)];
    if (pool_.pin_prefix_blocks(slots_.slot(slot).handle, full_blocks,
                                pin.blocks) != PagedKvStatus::Ok) {
        return false;
    }
    pin.tokens = tokens;
    pin.last_use = ++gpu_prefix_clock_;
    pin.snapshot_generation = b_.prefix_snapshots_[index].generation;
    return true;
}

void Qwen35SeqEngine::unpin_all_gpu_prefixes() {
    for (size_t i = 0; i < gpu_prefix_pins_.size(); ++i) {
        unpin_gpu_prefix(static_cast<int>(i));
    }
}

bool Qwen35SeqEngine::evict_one_gpu_prefix(int skip_index) {
    if (pool_.pinned_only_block_count() == 0) return false;
    int victim = -1;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < gpu_prefix_pins_.size(); ++i) {
        if (static_cast<int>(i) == skip_index) continue;
        const GpuPinnedPrefix & pin = gpu_prefix_pins_[i];
        if (pin.blocks.empty()) continue;
        if (victim < 0 || pin.last_use < oldest) {
            oldest = pin.last_use;
            victim = static_cast<int>(i);
        }
    }
    if (victim < 0) return false;
    unpin_gpu_prefix(victim);
    return true;
}

bool Qwen35SeqEngine::short_of_blocks(const AdmitResult & result) const {
    if (result.status != AdmitResult::Status::busy) return false;
    bool free_slot = false;
    for (int i = 0; i < slots_.slot_count(); ++i) {
        if (!slots_.is_active(i)) free_slot = true;
        else if (slots_.slot(i).parked()) return false;
    }
    return free_slot;
}

void Qwen35SeqEngine::discard_prefix_store(PrefixStoreRef checkpoint) {
    {
        const int drop_idx = checkpoint_index(checkpoint);
        if (drop_idx >= 0) {
            unpin_gpu_prefix(drop_idx);
            for (size_t s = 0; s < restored_kv_checkpoint_.size(); ++s) {
                if (restored_kv_checkpoint_[s] == drop_idx) {
                    restored_kv_checkpoint_[s] = -1;
                    restored_kv_tokens_[s] = 0;
                }
            }
        }
    }
    const int index = checkpoint_index(checkpoint);
    if (index >= 0 &&
        b_.prefix_snapshots_[index].cur_pos == checkpoint.tokens) {
        free_prefix_snapshot(b_.prefix_snapshots_[index]);
    }
}

size_t Qwen35SeqEngine::estimate_prefix_state_bytes(int tokens) const {
    return tokens > 0 ? estimate_paged_target_cache_state_bytes(b_.cache_) : 0;
}

bool Qwen35SeqEngine::drop_prefix_kv(PrefixStoreRef checkpoint) {
    {
        const int drop_idx = checkpoint_index(checkpoint);
        if (drop_idx >= 0) {
            unpin_gpu_prefix(drop_idx);
            for (size_t s = 0; s < restored_kv_checkpoint_.size(); ++s) {
                if (restored_kv_checkpoint_[s] == drop_idx) {
                    restored_kv_checkpoint_[s] = -1;
                    restored_kv_tokens_[s] = 0;
                }
            }
        }
    }
    const int index = checkpoint_index(checkpoint);
    return index >= 0 &&
        b_.prefix_snapshots_[index].layout == PrefixSnapshot::Layout::paged &&
        b_.prefix_snapshots_[index].cur_pos == checkpoint.tokens &&
        drop_prefix_snapshot_kv(b_.prefix_snapshots_[index]);
}

bool Qwen35SeqEngine::arm_capture(
        int slot, PrefixCaptureTicket ticket, int restored_tokens) {
    if (slot < 0 || slot >= slots_.slot_count()) return false;
    const Qwen35Slot & sequence = slots_.slot(slot);
    if (!ticket.valid() || checkpoint_index(ticket.checkpoint) < 0 ||
        ticket.checkpoint.tokens <= restored_tokens ||
        ticket.checkpoint.tokens > sequence.prompt_len) return false;
    slots_.slot(slot).pending_capture = ticket;
    return true;
}

SeqEngine::AdmitResult Qwen35SeqEngine::admit_with_prefix(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler,
        const PrefixStorePlan & plan) {
    const uint32_t block_sz = pool_.block_size();
    int restore_index = -1;
    int kv_index = -1;
    int pin_src_idx = -1;
    uint32_t pre_shared_blocks = 0;
    PrefixSnapshot * snap = nullptr;
    const PrefixSnapshot * kv_from = nullptr;

    if (plan.restore.valid()) {
        restore_index = checkpoint_index(plan.restore);
        const bool metadata_valid =
            restore_index >= 0 &&
            plan.restore.tokens < (int)prompt.size();
        snap = metadata_valid ? &b_.prefix_snapshots_[restore_index] : nullptr;
        // The rows of a state-only checkpoint come from the deeper one the
        // plan names; a stale name fails the restore below.
        kv_index = checkpoint_index(plan.restore_kv);
        kv_from =
            kv_index >= 0 && kv_index != restore_index &&
            b_.prefix_snapshots_[kv_index].cur_pos == plan.restore_kv.tokens
                ? &b_.prefix_snapshots_[kv_index] : nullptr;

        if (snap && snap->ctx &&
            snap->layout == PrefixSnapshot::Layout::paged &&
            snap->cur_pos == plan.restore.tokens) {
            const uint32_t full_blocks =
                static_cast<uint32_t>(plan.restore.tokens) / block_sz;
            if (full_blocks > 0) {
                const GpuPinnedPrefix * p = live_gpu_prefix(restore_index);
                if (p && p->tokens == plan.restore.tokens &&
                    p->blocks.size() >= full_blocks) {
                    pin_src_idx = restore_index;
                    pre_shared_blocks = full_blocks;
                }
                p = kv_from ? live_gpu_prefix(kv_index) : nullptr;
                if (pin_src_idx < 0 && p &&
                    p->tokens >= plan.restore.tokens &&
                    p->blocks.size() >= full_blocks) {
                    pin_src_idx = kv_index;
                    pre_shared_blocks = full_blocks;
                }
            }
        }
    }

    AdmitResult result = slots_.admit(request_id, prompt, sampler, pre_shared_blocks);
    while (short_of_blocks(result) && evict_one_gpu_prefix(pin_src_idx)) {
        result = slots_.admit(request_id, prompt, sampler, pre_shared_blocks);
    }
    if (short_of_blocks(result) && pin_src_idx >= 0) {
        // Last resort: drop the target pin too if even discounted admission
        // cannot fit and retry with full pool headroom. No other pin is left
        // to evict after this one.
        unpin_gpu_prefix(pin_src_idx);
        pin_src_idx = -1;
        pre_shared_blocks = 0;
        result = slots_.admit(request_id, prompt, sampler, 0);
    }
    if (result.status != AdmitResult::Status::admitted) return result;
    clear_slot_images(result.slot);

    const int slot = result.slot;
    slots_.slot(slot).pending_capture = {};
    bool restored = false;
    size_t shared_blocks_used = 0;
    if (plan.restore.valid()) {
        const auto restore_started = std::chrono::steady_clock::now();
        if (snap && snap->ctx &&
            snap->layout == PrefixSnapshot::Layout::paged &&
            snap->cur_pos == plan.restore.tokens) {
            Qwen35SlotManager::PrefillChunk seeded;
            if (pin_src_idx >= 0) {
                GpuPinnedPrefix & pin =
                    gpu_prefix_pins_[static_cast<size_t>(pin_src_idx)];
                seeded = slots_.seed_shared_prefix(
                    slot, pin.blocks.data(), pre_shared_blocks,
                    plan.restore.tokens);
                if (seeded.ok) {
                    shared_blocks_used = pre_shared_blocks;
                    pin.last_use = ++gpu_prefix_clock_;
                }
            } else {
                seeded = slots_.seed_restored_prefix(slot, plan.restore.tokens);
            }
            PagedKvSequenceSnapshot sequence;
            const bool pool_ok = seeded.ok &&
                pool_.sequence(slots_.slot(slot).handle, sequence) ==
                    PagedKvStatus::Ok;
            const bool table_ok = pool_ok && upload_block_table_delta(
                slot, seeded.first_new_block, seeded.new_blocks.data(),
                seeded.new_blocks.size());
            restored = table_ok && restore_paged_target_cache(
                *snap, b_.cache_, slot, sequence.block_table,
                (int)block_sz, kv_from, shared_blocks_used);
            // Warm the GPU pin table after a host/disk restore so subsequent
            // concurrent requests matching this prefix share blocks zero-copy.
            if (restored && shared_blocks_used == 0 &&
                !live_gpu_prefix(restore_index)) {
                pin_gpu_prefix(restore_index, slot, plan.restore.tokens);
            }
        }
        const uint64_t restore_elapsed_us =
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - restore_started).count();

        if (!restored) {
            discard_prefix_store(plan.restore);
            slots_.retire(slot);
            result = admit(request_id, prompt, sampler);
            result.prefix_store.invalidated = plan.restore;
            if (result.status != AdmitResult::Status::admitted) {
                result.error =
                    "cold admission failed after stale prefix restore";
            }
        } else {
            const int kv_src_idx = kv_from ? kv_index : restore_index;
            if (slot >= 0 && (size_t)slot < restored_kv_checkpoint_.size()) {
                restored_kv_checkpoint_[(size_t)slot] = kv_src_idx;
                restored_kv_tokens_[(size_t)slot] = plan.restore.tokens;
            }
            result.prefix_store.restored = plan.restore;
            const size_t total_blocks =
                (static_cast<size_t>(plan.restore.tokens) + block_sz - 1) / block_sz;
            std::fprintf(stderr,
                "[parallel-pc] restored checkpoint=%llu seq_slot=%d "
                "tokens=%d shared_blocks=%zu/%zu time_ms=%.1f\n",
                (unsigned long long)plan.restore.id, slot,
                plan.restore.tokens, shared_blocks_used, total_blocks,
                (double)restore_elapsed_us / 1000.0);
        }
        result.prefix_store.restore_attempted = true;
        result.prefix_store.restore_elapsed_us = restore_elapsed_us;
        if (result.status != AdmitResult::Status::admitted) return result;
    } else {
        if (slot >= 0 && (size_t)slot < restored_kv_checkpoint_.size()) {
            restored_kv_checkpoint_[(size_t)slot] = -1;
            restored_kv_tokens_[(size_t)slot] = 0;
        }
        reset_recurrent_slot(b_.cache_, slot);
    }

    const int admitted_slot = result.slot;
    if (!result.prefix_store.invalidated.valid() &&
        arm_capture(
            admitted_slot, plan.capture,
            result.prefix_store.restored.tokens)) {
        result.prefix_store.capture = plan.capture;
    }
    return result;
}

PrefixStoreEvent Qwen35SeqEngine::capture_prefix(
        int slot, PrefixCaptureTicket ticket) {
    PrefixStoreEvent event;
    event.ticket = ticket;
    event.status = PrefixStoreEvent::Status::failed;
    const int checkpoint = checkpoint_index(ticket.checkpoint);
    if (!ticket.valid() || checkpoint < 0 ||
        !slots_.is_prefilling(slot) ||
        slots_.slot(slot).cur_pos != ticket.checkpoint.tokens) {
        event.error = "invalid prefix capture boundary";
        return event;
    }
    PagedKvSequenceSnapshot sequence;
    if (pool_.sequence(slots_.slot(slot).handle, sequence) !=
            PagedKvStatus::Ok ||
        sequence.kv_seq_len != (uint32_t)ticket.checkpoint.tokens) {
        event.error = "prefix capture page table is incomplete";
        return event;
    }
    const auto capture_started = std::chrono::steady_clock::now();
    PrefixSnapshot & snapshot = b_.prefix_snapshots_[checkpoint];
    const PrefixSnapshot * base_kv = nullptr;
    if (slot >= 0 && (size_t)slot < restored_kv_checkpoint_.size()) {
        const int bidx = restored_kv_checkpoint_[(size_t)slot];
        const int btok = restored_kv_tokens_[(size_t)slot];
        if (bidx >= 0 && bidx != checkpoint && btok > 0 && btok < ticket.checkpoint.tokens) {
            const PrefixSnapshot & cand = b_.prefix_snapshots_[bidx];
            if (cand.ctx && cand.layout == PrefixSnapshot::Layout::paged &&
                !cand.state_only && cand.cur_pos > 0 && cand.cur_pos <= btok) {
                base_kv = &cand;
            }
        }
    }
    if (!replace_paged_target_cache(
            b_.cache_, slot, sequence.block_table,
            (int)pool_.block_size(), ticket.checkpoint.tokens, snapshot,
            &b_.displaced_snapshots_[checkpoint], base_kv)) {
        event.elapsed_us =
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - capture_started).count();
        event.error = luce_last_error();
        if (event.error.empty()) {
            event.error = "paged prefix capture failed";
        }
        return event;
    }
    // Pin full physical blocks of this checkpoint on GPU for zero-copy sharing.
    pin_gpu_prefix(checkpoint, slot, ticket.checkpoint.tokens);
    if (slot >= 0 && (size_t)slot < restored_kv_checkpoint_.size()) {
        restored_kv_checkpoint_[(size_t)slot] = checkpoint;
        restored_kv_tokens_[(size_t)slot] = ticket.checkpoint.tokens;
    }
    event.status = PrefixStoreEvent::Status::saved;
    event.elapsed_us =
        (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - capture_started).count();
    event.bytes = snapshot.buf
        ? ggml_backend_buffer_get_size(snapshot.buf) : 0;
    std::fprintf(stderr,
        "[parallel-pc] saved checkpoint=%llu seq_slot=%d tokens=%d "
        "bytes=%zu time_ms=%.1f\n",
        (unsigned long long)ticket.checkpoint.id, slot,
        ticket.checkpoint.tokens, event.bytes,
        (double)event.elapsed_us / 1000.0);
    return event;
}

int32_t Qwen35SeqEngine::sample_graph_row(
        int slot, int logits_row, int32_t argmax,
        std::vector<float> & logits,
        const ggml_tensor * src_logits,
        const int32_t * top_ids,
        const float * top_vals,
        int top_n) {
    const TargetWeights & w = b_.w_;
    const int vocab = w.n_vocab;
    Qwen35Slot & seq = slots_.slot(slot);
    int32_t token = -1;
    if (seq.sampler.needs_logit_processing()) {
        const auto hist = seq.penalty_history_view();
        if (top_n > 0 && top_ids && top_vals) {
            token = sample_logits_top(
                top_vals + (size_t)logits_row * (size_t)top_n,
                top_ids + (size_t)logits_row * (size_t)top_n,
                top_n, vocab, seq.sampler, hist.data, hist.size, seq.rng);
        }
        if (token < 0) {
            if ((int)logits.size() < vocab) logits.resize((size_t)vocab);
            ggml_backend_tensor_get(
                src_logits, logits.data(),
                (size_t)logits_row * (size_t)vocab * sizeof(float),
                sizeof(float) * (size_t)vocab);
            token = sample_logits(logits.data(), vocab, seq.sampler,
                                  hist.data, hist.size, seq.rng);
        }
    } else {
        token = argmax;
    }
    return b_.apply_min_tokens_floor(
        token, seq.generated_tokens(),
        (size_t)logits_row * (size_t)vocab * sizeof(float),
        src_logits);
}

bool Qwen35SeqEngine::upload_block_table_delta(
        int slot, int first_block, const int32_t * blocks, size_t count) {
    if (count == 0) return true;
    ggml_tensor * table = b_.cache_.paged_block_table;
    if (!table || slot < 0 || slot >= table->ne[1] || first_block < 0 ||
        (uint64_t)first_block + count > (uint64_t)table->ne[0]) {
        return false;
    }
    // `blocks` commonly points into a temporary PrefillChunk vector or a
    // stack-local StepAppend. Keep this tiny metadata write synchronous so
    // the backend never observes a source whose lifetime has ended.
    ggml_backend_tensor_set(
        table, blocks,
        (size_t)slot * table->nb[1] +
            (size_t)first_block * sizeof(int32_t),
        count * sizeof(int32_t));
    return true;
}

void Qwen35SeqEngine::fail_prefill(
        int slot, std::vector<PrefillOutput> & prefill_outputs,
        const char * log_message, const char * client_message) {
    if (!slots_.is_prefilling(slot)) return;
    std::fprintf(stderr, "[parallel] %s — failing slot %d\n",
                 log_message, slot);
    PrefillOutput out;
    out.slot = slot;
    out.status = PrefillOutput::Status::failed;
    out.error = client_message;
    prefill_outputs.push_back(std::move(out));
}

Qwen35SeqEngine::PrefillStage Qwen35SeqEngine::stage_prefill_chunk(
        int slot, int max_tokens, float * embeddings,
        std::vector<PrefillOutput> & prefill_outputs) {
    PrefillStage stage;
    if (!slots_.is_prefilling(slot)) return stage;

    Qwen35Slot & seq = slots_.slot(slot);
    stage.kv_pos = seq.cur_pos;
    stage.chunk = std::min(
        max_tokens, seq.prompt_len - stage.kv_pos);
    const PrefixCaptureTicket capture = slots_.slot(slot).pending_capture;
    if (capture.valid() &&
        stage.kv_pos < capture.checkpoint.tokens) {
        // Recurrent state is only checkpoint-consistent after compute, so a
        // selected capture boundary must be the exact end of this graph slice.
        stage.chunk = std::min(
            stage.chunk, capture.checkpoint.tokens - stage.kv_pos);
    }
    if (stage.chunk <= 0) return PrefillStage{};
    stage.commit = stage.kv_pos + stage.chunk >= seq.prompt_len;

    Qwen35SlotManager::PrefillChunk chunk =
        slots_.append_prefill(slot, stage.chunk);
    if (!chunk.ok || chunk.rows.size() != (size_t)stage.chunk) {
        fail_prefill(slot, prefill_outputs, "prefill K/V allocation failed",
                     "prefill K/V allocation failed");
        return PrefillStage{};
    }
    if (!upload_block_table_delta(
            slot, chunk.first_new_block, chunk.new_blocks.data(),
            chunk.new_blocks.size())) {
        // step() reports a clean failure: no rows of this chunk may stay.
        slots_.rollback_prefill(slot, stage.chunk);
        fail_prefill(
            slot, prefill_outputs, "prefill block-table delta exceeds device capacity",
            "prefill block-table update failed");
        return PrefillStage{};
    }

    stage.rows = std::move(chunk.rows);
    if (!b_.w_.embedder.embed(
            seq.sample_history.data() + stage.kv_pos, stage.chunk,
            embeddings)) {
        slots_.rollback_prefill(slot, stage.chunk);
        fail_prefill(slot, prefill_outputs, "prefill embed failed",
                     "prefill embedding failed");
        return PrefillStage{};
    }
    if (slot < static_cast<int>(slot_images_.size()) &&
        slot_images_[static_cast<size_t>(slot)].payload) {
        const SlotImages & images = slot_images_[static_cast<size_t>(slot)];
        images.rows.overwrite(embeddings, stage.kv_pos, stage.chunk, b_.w_.n_embd);
        stage.positions.assign(static_cast<size_t>(4) * stage.chunk, 0);
        images.rows.prompt->positions.fill(stage.positions.data(), stage.kv_pos, stage.chunk);
    }
    stage.ready = true;
    return stage;
}

namespace {
// Offers per width before the controller decides for a new lane count.
// Rounds that rebuild their graph are not cost samples.
constexpr int kChainWidthCalibrationSamples = 2;
// A width that keeps landing on rounds with AR peers or a graph rebuild
// stops being forced after this many offers.
constexpr int kChainWidthCalibrationMaxOffers = 8;
// Round times are wall clock: a busy host can stretch them 2-3x for a while.
// A clean round slower than this many times its width's cost is not a cost
// sample, unless kChainWidthSlowStreak rounds in a row are (the new normal).
constexpr float kChainWidthOutlierRatio = 1.5f;
constexpr int kChainWidthSlowStreak = 32;
// Rounds per bucket between full recalibrations of every width's cost.
constexpr long long kChainWidthRecalibrationRounds = 1024;
// Per-round decay of the depth statistics, so they follow the live mix.
constexpr double kChainDepthDecay = 0.98;
// Beta prior on each conditional acceptance (mean 0.7, weight 2 trials).
constexpr double kChainDepthPriorAccepts = 1.4;
constexpr double kChainDepthPriorTrials = 2.0;
}  // namespace

int Qwen35SeqEngine::choose_chain_width(int bucket, int cap) {
    const int requested = cap > 0 ? std::min(cap, fixed_chain_.width)
                                  : fixed_chain_.width;
    if (!adaptive_chain_width_ || chain_width_choices_.empty()) return requested;
    ChainWidthState & state =
        chain_width_by_bucket_.try_emplace(bucket, requested).first->second;
    // A bucket keeps the width it was set up for: batched buckets draft the
    // shorter batched block, so their controller never offers more.
    const int full = std::min(
        requested, static_cast<int>(state.depth_trials.size()));
    // A width the controller stopped offering keeps the cost it was last
    // timed at; time every width again now and then, so a cost measured
    // while the host was busy cannot keep a width out for good.
    if (state.rounds >= state.recalibrate_at) {
        if (state.rounds > 0) {
            std::fill(state.offers.begin(), state.offers.end(), 0);
            std::fill(state.samples.begin(), state.samples.end(), 0);
            state.controller.set_measured_costs({});
        }
        state.recalibrate_at = state.rounds + kChainWidthRecalibrationRounds;
    }
    for (int width : chain_width_choices_) {
        if (width > full) break;
        const size_t w = static_cast<size_t>(width);
        if (state.samples[w] < kChainWidthCalibrationSamples &&
            state.offers[w] < kChainWidthCalibrationMaxOffers) {
            return width;
        }
    }
    // Conditional acceptance of each candidate depth. Depths never tried
    // inherit the deepest measured rate (geometric continuation).
    std::vector<float> & conditional = state.conditional;
    conditional.assign(static_cast<size_t>(full - 1), 0.7f);
    float last = 0.7f;
    for (int depth = 1; depth < full; ++depth) {
        const double trials = state.depth_trials[static_cast<size_t>(depth)];
        if (trials >= 1.0) {
            last = static_cast<float>(
                (state.depth_accepts[static_cast<size_t>(depth)] +
                 kChainDepthPriorAccepts) /
                (trials + kChainDepthPriorTrials));
        }
        conditional[static_cast<size_t>(depth - 1)] = last;
    }
    const int wanted =
        state.controller.next_width_cost_aware(conditional, full);
    int width = chain_width_choices_.front();
    for (int choice : chain_width_choices_) {
        if (choice <= wanted && choice <= full) width = choice;
    }
    return width;
}

void Qwen35SeqEngine::observe_chain_width(
        int bucket, int width, const std::vector<size_t> & accepted,
        const std::vector<uint8_t> & sampled, double step_ms,
        double readback_ms, double walk_ms) {
    if (!adaptive_chain_width_) return;
    auto it = chain_width_by_bucket_.find(bucket);
    if (it == chain_width_by_bucket_.end()) return;
    ChainWidthState & state = it->second;
    // Saturating counters: they only gate calibration.
    int & offers = state.offers[static_cast<size_t>(width)];
    offers = std::min(offers + 1, kChainWidthCalibrationMaxOffers);
    // Every lane is an acceptance sample; the round is one cost sample when
    // it is clean (step_ms < 0 marks a round that is not).
    float cost = step_ms > 0.0 ? static_cast<float>(step_ms) : -1.0f;
    if (cost > 0.0f) {
        const std::vector<float> & costs = state.controller.measured_costs();
        const float known = static_cast<size_t>(width) < costs.size()
            ? costs[static_cast<size_t>(width)] : 0.0f;
        if (std::isfinite(known) && known > 0.0f &&
            cost > kChainWidthOutlierRatio * known &&
            ++state.slow_streak < kChainWidthSlowStreak) {
            cost = -1.0f;
        } else {
            state.slow_streak = 0;
        }
    }
    if (cost > 0.0f) {
        int & samples = state.samples[static_cast<size_t>(width)];
        samples = std::min(samples + 1, kChainWidthCalibrationSamples);
    }
    // The controller's clean-draft flag comes from its last observe(), so
    // the round's majority outcome is observed last (with the cost sample);
    // otherwise one arbitrary lane would drive the next exploration step.
    size_t clean = 0;
    for (size_t accepted_width : accepted) {
        clean += accepted_width >= static_cast<size_t>(width);
    }
    const bool round_clean = 2 * clean > accepted.size();
    size_t last = accepted.size();
    for (size_t lane = 0; lane < accepted.size(); ++lane) {
        if ((accepted[lane] >= static_cast<size_t>(width)) == round_clean) {
            last = lane;
            break;
        }
    }
    for (size_t lane = 0; lane < accepted.size(); ++lane) {
        if (lane == last) continue;
        state.controller.observe(static_cast<int>(accepted[lane]), width, -1.0f);
    }
    if (last < accepted.size()) {
        state.controller.observe(static_cast<int>(accepted[last]), width, cost);
    }
    for (double & trials : state.depth_trials) trials *= kChainDepthDecay;
    for (double & accepts : state.depth_accepts) accepts *= kChainDepthDecay;
    for (size_t lane = 0; lane < accepted.size(); ++lane) {
        // accepted includes the seed: candidates 1..accepted-1 passed, and
        // candidate `accepted` (if offered) was the first rejection.
        const int passed = static_cast<int>(accepted[lane]) - 1;
        for (int depth = 1; depth < width; ++depth) {
            if (depth <= passed) {
                state.depth_trials[static_cast<size_t>(depth)] += 1.0;
                state.depth_accepts[static_cast<size_t>(depth)] += 1.0;
            } else {
                state.depth_trials[static_cast<size_t>(depth)] += 1.0;
                break;
            }
        }
    }
    ++state.rounds;
    bool any_sampled = false;
    for (size_t lane = 0; lane < accepted.size(); ++lane) {
        const double lane_accepted = static_cast<double>(accepted[lane]);
        if (lane < sampled.size() && sampled[lane]) {
            state.sampled_accepted_sum += lane_accepted;
            ++state.sampled_lane_rounds;
            any_sampled = true;
        } else {
            state.greedy_accepted_sum += lane_accepted;
            ++state.greedy_lane_rounds;
        }
    }
    if (any_sampled) {
        ++state.sampled_rounds;
        state.readback_ms_sum += readback_ms;
        state.walk_ms_sum += walk_ms;
    }
    if (state.rounds % 1024 == 0) {
        const auto mean = [](double sum, double count) {
            return count > 0.0 ? sum / count : 0.0;
        };
        const double sampled_rounds =
            static_cast<double>(state.sampled_rounds);
        std::fprintf(stderr, "[parallel] adaptive width bucket=%d rounds=%lld "
                     "mean_accepted=%.2f greedy=%.2f sampled=%.2f "
                     "sampled_readback_ms=%.2f sampled_walk_ms=%.2f offers:",
                     bucket, (long long) state.rounds,
                     mean(state.greedy_accepted_sum + state.sampled_accepted_sum,
                          static_cast<double>(state.greedy_lane_rounds +
                                              state.sampled_lane_rounds)),
                     mean(state.greedy_accepted_sum,
                          static_cast<double>(state.greedy_lane_rounds)),
                     mean(state.sampled_accepted_sum,
                          static_cast<double>(state.sampled_lane_rounds)),
                     mean(state.readback_ms_sum, sampled_rounds),
                     mean(state.walk_ms_sum, sampled_rounds));
        for (int choice : chain_width_choices_) {
            // a batched bucket only tracks widths up to its own block
            if (static_cast<size_t>(choice) >= state.offers.size()) break;
            std::fprintf(stderr, " w%d=%d", choice,
                         state.offers[static_cast<size_t>(choice)]);
        }
        std::fprintf(stderr, " next=%d\n",
                     choose_chain_width(bucket));
    }
}

SeqEngine::StepResult Qwen35SeqEngine::step_chain_spec(
        const StepPlan & plan, const std::vector<uint8_t> & selected,
        PreparedChainRound && prepared_round) {
    StepResult result;
    const std::vector<StepInput> & inputs = plan.decode;
    if (!plan.prefills.empty() || selected.size() != inputs.size()) {
        result.error = "invalid fixed chain round";
        return result;
    }

    struct Proposal {
        size_t input_index = 0;
        int slot = -1;
        std::vector<int32_t> tokens;
        size_t accepted = 0;
        size_t verified = 0;  // target-verified prefix before policy clamps
        int32_t pending = -1;
        // A sampled lane: is_sampled marks it for stats & sampled_verify_next;
        // sampled_rows is non-null only when full logits rows were copied back
        // (GPU top_n settled lanes leave sampled_rows null).
        bool is_sampled = false;
        const float * sampled_rows = nullptr;
        SampledVerifyWalk walk;
        Qwen35SlotManager::StepAppend append;
        int seq_len = -1;
    };
    struct ArLane {
        size_t input_index = 0;
        int slot = -1;
        int32_t token = -1;
        int position = -1;
        int64_t physical_row = -1;
        int32_t pending = -1;
    };

    const int full_width = fixed_chain_.width;
    const int hidden = b_.w_.n_embd;
    const int n_head_kv = b_.w_.n_head_kv;
    const int n_slots = slots_.slot_count();
    const int vocab = b_.w_.n_vocab;
    static const int min_tokens = []() {
        const char * value = std::getenv("LUCE_MIN_TOKENS");
        return value ? std::max(0, std::atoi(value)) : 0;
    }();

    std::vector<Proposal> proposals;
    std::vector<ArLane> ar_lanes;
    std::vector<int> proposal_for_input(inputs.size(), -1);
    std::vector<int> ar_for_input(inputs.size(), -1);
    proposals.reserve(inputs.size());
    ar_lanes.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        const StepInput & input = inputs[i];
        if (selected[i]) {
            if (i >= prepared_round.drafts.size()) {
                result.error = "prepared DFlash2 chain round is incomplete";
                return result;
            }
            PreparedChainDraft & prepared = prepared_round.drafts[i];
            if (prepared.tokens.size() < 2 ||
                prepared.tokens.size() > static_cast<size_t>(full_width) ||
                prepared.tokens.front() != input.token) {
                result.error = "prepared DFlash2 chain is invalid";
                return result;
            }
            Proposal proposal;
            proposal.input_index = i;
            proposal.slot = input.slot;
            proposal.tokens = std::move(prepared.tokens);
            proposal_for_input[i] = static_cast<int>(proposals.size());
            proposals.push_back(std::move(proposal));
        } else {
            ArLane lane;
            lane.input_index = i;
            lane.slot = input.slot;
            lane.token = input.token;
            ar_for_input[i] = static_cast<int>(ar_lanes.size());
            ar_lanes.push_back(lane);
        }
    }
    if (proposals.empty()) {
        result.error = "fixed chain round has no speculative lanes";
        return result;
    }

    struct StagedRound {
        Qwen35SlotManager & slots;
        std::vector<int32_t> & seq_lens;
        ggml_tensor * device_seq_lens;
        std::vector<int> staged_slots;
        std::vector<int32_t> saved_seq_lens;
        bool device_lengths_dirty = false;
        bool committed = false;

        StagedRound(
                Qwen35SlotManager & slots,
                std::vector<int32_t> & seq_lens,
                ggml_tensor * device_seq_lens,
                size_t lane_count)
            : slots(slots), seq_lens(seq_lens),
              device_seq_lens(device_seq_lens),
              saved_seq_lens(seq_lens) {
            staged_slots.reserve(lane_count);
        }

        ~StagedRound() {
            if (committed) return;
            for (auto it = staged_slots.rbegin();
                 it != staged_slots.rend(); ++it) {
                if (!slots.rollback_step(*it)) {
                    std::fprintf(stderr,
                        "[parallel] staged round rollback failed for slot %d\n",
                        *it);
                    std::abort();
                }
            }
            seq_lens = saved_seq_lens;
            if (!device_lengths_dirty) return;
            ggml_backend_tensor_set(
                device_seq_lens, seq_lens.data(), 0,
                sizeof(int32_t) * seq_lens.size());
        }

        void track(int slot) { staged_slots.push_back(slot); }
    } staged_round(
        slots_, seq_lens_, b_.cache_.paged_kv_seq_lens,
        inputs.size());

    // AR peers write their ordinary rows in the same target forward. Host
    // history and positions remain staged until every promotion succeeds.
    for (ArLane & lane : ar_lanes) {
        Qwen35SlotManager::StepAppend append =
            slots_.append_token(lane.slot, lane.token);
        while (!append.ok && append.busy &&
               evict_one_gpu_prefix()) {
            append = slots_.append_token(lane.slot, lane.token);
        }
        if (!append.ok) {
            result.error = append.busy
                ? "paged KV pool exhausted during compact AR staging"
                : "compact AR K/V staging failed";
            return result;
        }
        staged_round.track(lane.slot);
        if (append.new_block >= 0 &&
            !upload_block_table_delta(
                lane.slot, append.new_block_index,
                &append.new_block, 1)) {
            result.error = "compact AR block-table update failed";
            return result;
        }
        lane.position = append.position;
        lane.physical_row = append.physical_row;
    }

    const int spec_count = static_cast<int>(proposals.size());
    const int ar_count = static_cast<int>(ar_lanes.size());
    const int tree_bucket = chain_decode_bucket_width(spec_count);
    // prepare_chain_drafts chose the width from this bucket's controller.
    // Controllers are keyed by the verify graph bucket: the graph is built
    // for tree_bucket lanes (missing lanes are padding), so a round's cost is
    // set by the bucket and width, not by how many lanes are real.
    // A lane alone drafted the full block; a narrower round verifies its
    // leading prefix, which is the same chain truncated.
    const int tree_width = prepared_round.width;
    for (Proposal & proposal : proposals) {
        if (tree_width < 2 ||
            proposal.tokens.size() < static_cast<size_t>(tree_width)) {
            result.error = "prepared DFlash2 chain is shorter than its width";
            return result;
        }
        proposal.tokens.resize(static_cast<size_t>(tree_width));
    }
    const int tree_rows_count = tree_width * tree_bucket;
    const int total_rows = ar_count + tree_rows_count;

    int max_prefix = 1;
    for (const Proposal & proposal : proposals) {
        max_prefix = std::max(
            max_prefix, slots_.slot(proposal.slot).cur_pos);
    }
    for (const ArLane & lane : ar_lanes) {
        max_prefix = std::max(max_prefix, lane.position + 1);
    }

    int max_top_n = 0;
    if (gpu_argmax_enabled()) {
        for (const Proposal & proposal : proposals) {
            const Qwen35Slot & seq = slots_.slot(proposal.slot);
            if (seq.sampler.needs_logit_processing()) {
                max_top_n = std::max(
                    max_top_n, sample_logits_top_n(seq.sampler, vocab));
            }
        }
        for (const ArLane & lane : ar_lanes) {
            const Qwen35Slot & seq = slots_.slot(lane.slot);
            if (seq.sampler.needs_logit_processing()) {
                max_top_n = std::max(
                    max_top_n, sample_logits_top_n(seq.sampler, vocab));
            }
        }
    }

    bool graph_rebuilt = false;
    StepGraph * graph_ptr = acquire_paged_tree_graph(
        tree_width, tree_bucket, max_prefix, ar_count, max_top_n, graph_rebuilt);
    if (!graph_ptr) {
        result.error = "fixed chain target graph build failed";
        return result;
    }
    StepGraph & graph = *graph_ptr;

    const int real_rows = ar_count + spec_count * tree_width;
    chain_tokens_.assign(static_cast<size_t>(real_rows), 0);
    chain_parents_.assign(static_cast<size_t>(tree_rows_count), -1);
    chain_tree_sizes_.assign(static_cast<size_t>(tree_bucket), 0);
    chain_mapped_slots_.assign(static_cast<size_t>(ar_count + tree_bucket), -1);
    chain_state_slots_.assign(static_cast<size_t>(ar_count + tree_bucket), 0);
    chain_query_slots_.assign(static_cast<size_t>(total_rows), -1);
    chain_query_positions_.assign(static_cast<size_t>(total_rows), -1);
    chain_write_rows_.assign(
        static_cast<size_t>(total_rows) * n_head_kv, scratch_row_);
    chain_positions_.assign(static_cast<size_t>(4) * total_rows, 0);
    seq_lens_.assign(static_cast<size_t>(n_slots), 0);

    for (int lane_index = 0; lane_index < ar_count; ++lane_index) {
        const ArLane & lane = ar_lanes[static_cast<size_t>(lane_index)];
        chain_mapped_slots_[static_cast<size_t>(lane_index)] = lane.slot;
        chain_state_slots_[static_cast<size_t>(lane_index)] = lane.slot;
        chain_tokens_[static_cast<size_t>(lane_index)] = lane.token;
        chain_query_slots_[static_cast<size_t>(lane_index)] = lane.slot;
        chain_query_positions_[static_cast<size_t>(lane_index)] = lane.position;
        seq_lens_[static_cast<size_t>(lane.slot)] = lane.position + 1;
        for (int axis = 0; axis < 3; ++axis) {
            chain_positions_[static_cast<size_t>(axis) * total_rows + lane_index] =
                lane.position + rope_delta(lane.slot);
        }
        chain_write_rows_[static_cast<size_t>(lane_index)] = lane.physical_row;
    }

    for (int lane_index = 0; lane_index < spec_count; ++lane_index) {
        const Proposal & proposal =
            proposals[static_cast<size_t>(lane_index)];
        const int tree_base = lane_index * tree_width;
        const int row_base = ar_count + tree_base;
        const int mapped_lane = ar_count + lane_index;
        chain_tree_sizes_[static_cast<size_t>(lane_index)] = tree_width;
        chain_mapped_slots_[static_cast<size_t>(mapped_lane)] = proposal.slot;
        chain_state_slots_[static_cast<size_t>(mapped_lane)] = proposal.slot;
        seq_lens_[static_cast<size_t>(proposal.slot)] =
            slots_.slot(proposal.slot).cur_pos;
        const int64_t slot_scratch_base =
            static_cast<int64_t>(fixed_chain_.scratch_base) +
            static_cast<int64_t>(proposal.slot) * fixed_chain_.scratch_stride;
        const int base_pos =
            slots_.slot(proposal.slot).cur_pos + rope_delta(proposal.slot);
        for (int node = 0; node < tree_width; ++node) {
            const int tree_row = tree_base + node;
            const int row = row_base + node;
            chain_tokens_[static_cast<size_t>(row)] =
                proposal.tokens[static_cast<size_t>(node)];
            chain_parents_[static_cast<size_t>(tree_row)] =
                node == 0 ? -1 : node - 1;
            chain_query_slots_[static_cast<size_t>(row)] = proposal.slot;
            const int position = base_pos + node;
            chain_positions_[static_cast<size_t>(row)] = position;
            chain_positions_[static_cast<size_t>(total_rows) + row] = position;
            chain_positions_[static_cast<size_t>(2) * total_rows + row] = position;
            chain_write_rows_[static_cast<size_t>(row)] = slot_scratch_base + node;
        }
    }
    for (int head = 1; head < n_head_kv; ++head) {
        std::memcpy(
            chain_write_rows_.data() + static_cast<size_t>(head) * total_rows,
            chain_write_rows_.data(),
            sizeof(int64_t) * static_cast<size_t>(total_rows));
    }

    const bool use_pinned_embed =
        static_cast<size_t>(total_rows) <= embed_rows_cap_;
    float * embed_dst = use_pinned_embed
        ? embed_rows_
        : (chain_embeddings_.resize(static_cast<size_t>(hidden) * total_rows),
           chain_embeddings_.data());
    if (real_rows > 0 &&
        !b_.w_.embedder.embed(chain_tokens_.data(), real_rows, embed_dst)) {
        result.error = "fixed chain embedding failed";
        return result;
    }
    if (total_rows > real_rows) {
        std::memset(
            embed_dst + static_cast<size_t>(hidden) * real_rows, 0,
            sizeof(float) * static_cast<size_t>(hidden) *
                static_cast<size_t>(total_rows - real_rows));
    }

    const bool have_top_n = max_top_n > 0 && graph.top_ids && graph.top_vals;
    const size_t top_elems = have_top_n
        ? static_cast<size_t>(max_top_n) * static_cast<size_t>(total_rows)
        : 0;
    const size_t stage_bytes =
        sizeof(int32_t) * (chain_positions_.size() + chain_parents_.size() +
                           chain_tree_sizes_.size() + chain_mapped_slots_.size() +
                           chain_state_slots_.size() + chain_query_slots_.size() +
                           chain_query_positions_.size() + seq_lens_.size() +
                           static_cast<size_t>(total_rows) + top_elems) +
        sizeof(int64_t) * chain_write_rows_.size() +
        sizeof(float) * top_elems + 4096;
    target_io_stage_.reserve(b_.target_backend_, stage_bytes);
    target_io_stage_.reset();

    if (use_pinned_embed) {
        ggml_backend_tensor_set_async(
            b_.target_backend_, graph.inp_embed, embed_dst, 0,
            sizeof(float) * static_cast<size_t>(hidden) * total_rows);
    } else {
        ggml_backend_tensor_set(
            graph.inp_embed, embed_dst, 0,
            sizeof(float) * static_cast<size_t>(hidden) * total_rows);
    }
    target_io_stage_.set(
        b_.target_backend_, graph.positions, chain_positions_.data(), 0,
        sizeof(int32_t) * chain_positions_.size());
    target_io_stage_.set(
        b_.target_backend_, graph.parent_ids, chain_parents_.data(), 0,
        sizeof(int32_t) * chain_parents_.size());
    target_io_stage_.set(
        b_.target_backend_, graph.tree_sizes, chain_tree_sizes_.data(), 0,
        sizeof(int32_t) * chain_tree_sizes_.size());
    if (detail::target_paged_tree_active_slots_need_upload(graph)) {
        target_io_stage_.set(
            b_.target_backend_, graph.active_slot_ids, chain_mapped_slots_.data(), 0,
            sizeof(int32_t) * chain_mapped_slots_.size());
    }
    target_io_stage_.set(
        b_.target_backend_, graph.state_slot_ids, chain_state_slots_.data(), 0,
        sizeof(int32_t) * chain_state_slots_.size());
    target_io_stage_.set(
        b_.target_backend_, graph.paged_query_seq_ids, chain_query_slots_.data(), 0,
        sizeof(int32_t) * chain_query_slots_.size());
    if (graph.paged_query_positions) {
        target_io_stage_.set(
            b_.target_backend_, graph.paged_query_positions,
            chain_query_positions_.data(), 0,
            sizeof(int32_t) * chain_query_positions_.size());
    }
    target_io_stage_.set(
        b_.target_backend_, graph.kv_write_rows, chain_write_rows_.data(), 0,
        sizeof(int64_t) * chain_write_rows_.size());
    target_io_stage_.set(
        b_.target_backend_, b_.cache_.paged_kv_seq_lens, seq_lens_.data(), 0,
        sizeof(int32_t) * seq_lens_.size());
    staged_round.device_lengths_dirty = true;
    // A batched verify of two or more lanes may split k in its MMQ calls
    // (ggml_backend_cuda_set_mmq_splitk_scope): not bit-exact against the
    // unsplit sum, which a batch of concurrent requests never was. A single
    // lane keeps the unsplit kernels. The launches happen inside
    // graph_compute, so the scope closes after it.
    ggml_backend_cuda_set_mmq_splitk_scope(spec_count >= 2);
    const ggml_status verify_status =
        ggml_backend_graph_compute(b_.target_backend_, graph.gf);
    ggml_backend_cuda_set_mmq_splitk_scope(false);
    if (verify_status != GGML_STATUS_SUCCESS) {
        result.error = "fixed chain target compute failed";
        return result;
    }

    const size_t argmax_bytes = sizeof(int32_t) * static_cast<size_t>(total_rows);
    const int32_t * posterior_ptr = static_cast<const int32_t *>(
        target_io_stage_.get(b_.target_backend_, graph.argmax_tokens, 0, argmax_bytes));
    if (!posterior_ptr) {
        chain_posterior_.assign(static_cast<size_t>(total_rows), -1);
        ggml_backend_tensor_get_async(
            b_.target_backend_, graph.argmax_tokens, chain_posterior_.data(), 0, argmax_bytes);
        posterior_ptr = chain_posterior_.data();
    }
    const int32_t * top_ids_ptr = nullptr;
    const float * top_vals_ptr = nullptr;
    if (have_top_n) {
        const size_t top_id_bytes = sizeof(int32_t) * top_elems;
        const size_t top_val_bytes = sizeof(float) * top_elems;
        top_ids_ptr = static_cast<const int32_t *>(
            target_io_stage_.get(b_.target_backend_, graph.top_ids, 0, top_id_bytes));
        top_vals_ptr = static_cast<const float *>(
            target_io_stage_.get(b_.target_backend_, graph.top_vals, 0, top_val_bytes));
        if (!top_ids_ptr) {
            top_ids_buf_.resize(top_elems);
            ggml_backend_tensor_get_async(
                b_.target_backend_, graph.top_ids, top_ids_buf_.data(), 0, top_id_bytes);
            top_ids_ptr = top_ids_buf_.data();
        }
        if (!top_vals_ptr) {
            top_vals_buf_.resize(top_elems);
            ggml_backend_tensor_get_async(
                b_.target_backend_, graph.top_vals, top_vals_buf_.data(), 0, top_val_bytes);
            top_vals_ptr = top_vals_buf_.data();
        }
    }
    ggml_backend_synchronize(b_.target_backend_);

    // Sampled lanes verify by the single-lane sampled walk over their own
    // rows (sampled_verify.h); greedy lanes keep the argmax prefix.
    // Lanes with GPU top_n candidates verify directly from the compact top-N
    // readback, falling back to full-row
    // logits D2H only when top_n cannot settle a row or the sampler needs
    // full-vocab access (e.g. pure top_p without top_k).
    const size_t lane_floats =
        static_cast<size_t>(tree_width) * static_cast<size_t>(vocab);
    std::vector<int> top_lanes;
    std::vector<int> full_lanes;
    for (int lane_index = 0; lane_index < spec_count; ++lane_index) {
        Proposal & proposal = proposals[static_cast<size_t>(lane_index)];
        const SamplerCfg & cfg = slots_.slot(proposal.slot).sampler;
        if (!cfg.needs_logit_processing()) continue;
        proposal.is_sampled = true;
        if (have_top_n && sample_logits_top_n(cfg, vocab) > 0) {
            top_lanes.push_back(lane_index);
        } else {
            full_lanes.push_back(lane_index);
        }
    }
    double readback_ms = 0.0;
    double walk_ms = 0.0;
    if (!top_lanes.empty()) {
        const auto t_top0 = std::chrono::steady_clock::now();
        std::vector<uint8_t> top_fallback(top_lanes.size(), 0);
        const auto run_top_walk = [&](int k) {
            const int lane_index = top_lanes[static_cast<size_t>(k)];
            Proposal & proposal = proposals[static_cast<size_t>(lane_index)];
            Qwen35Slot & seq = slots_.slot(proposal.slot);
            const auto saved_rng = seq.rng;
            std::vector<int32_t> history = seq.penalty_history();
            const int generated = seq.generated_tokens();
            const int row_base = ar_count + lane_index * tree_width;
            const SampledVerifyWalk w = sampled_verify_walk_draws(
                [&](int row, const SamplerCfg & round,
                    const std::vector<int32_t> & h, std::mt19937_64 & r) {
                    const size_t off =
                        static_cast<size_t>(row_base + row) *
                        static_cast<size_t>(max_top_n);
                    return sample_logits_top(
                        top_vals_ptr + off, top_ids_ptr + off,
                        max_top_n, vocab, round, h.data(), h.size(), r);
                },
                tree_width, proposal.tokens.data(), seq.sampler, history,
                seq.rng,
                [&](int32_t token, int position) {
                    return token_is_eos(token) &&
                           generated + position < min_tokens;
                });
            if (w.accepted <= 0) {
                seq.rng = saved_rng;
                top_fallback[static_cast<size_t>(k)] = 1;
            } else {
                proposal.walk = w;
                proposal.verified = static_cast<size_t>(w.accepted);
            }
        };
        if (top_lanes.size() >= 2) {
            if (!chain_sampled_pool_) {
                chain_sampled_pool_ = std::make_unique<BlockingRowPool>();
            }
            chain_sampled_pool_->run_custom(
                static_cast<int>(top_lanes.size()), run_top_walk);
        } else {
            run_top_walk(0);
        }
        for (size_t k = 0; k < top_lanes.size(); ++k) {
            if (top_fallback[k]) full_lanes.push_back(top_lanes[k]);
        }
        walk_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_top0).count();
    }
    if (!full_lanes.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        const size_t floats = full_lanes.size() * lane_floats;
        if (chain_sampled_rows_.size() < floats) {
            chain_sampled_rows_.resize(floats);
        }
        for (size_t k = 0; k < full_lanes.size(); ++k) {
            Proposal & proposal =
                proposals[static_cast<size_t>(full_lanes[k])];
            const int row_base = ar_count + full_lanes[k] * tree_width;
            proposal.sampled_rows = chain_sampled_rows_.data() + k * lane_floats;
            ggml_backend_tensor_get_async(
                b_.target_backend_, graph.logits,
                chain_sampled_rows_.data() + k * lane_floats,
                static_cast<size_t>(row_base) * static_cast<size_t>(vocab) *
                    sizeof(float),
                sizeof(float) * lane_floats);
        }
        ggml_backend_synchronize(b_.target_backend_);
        const auto t1 = std::chrono::steady_clock::now();
        // Lanes are independent (own rows, sampler, RNG and history), so
        // their walks run in parallel, except a lane whose sampler may take
        // a GPU path: that one runs here, on the thread that owns the
        // device. Each walk extends a copy of its slot's penalty history;
        // commit_step appends the kept tokens below. A draw the min-token
        // floor would replace ends the walk, so the floor rewrites it once,
        // as the pending token.
        const auto walk = [&](int k) {
            Proposal & proposal =
                proposals[static_cast<size_t>(
                    full_lanes[static_cast<size_t>(k)])];
            Qwen35Slot & seq = slots_.slot(proposal.slot);
            std::vector<int32_t> history = seq.penalty_history();
            const int generated = seq.generated_tokens();
            proposal.walk = sampled_verify_walk(
                proposal.sampled_rows, vocab, tree_width,
                proposal.tokens.data(), seq.sampler, history, seq.rng,
                [&](int32_t token, int position) {
                    return token_is_eos(token) &&
                           generated + position < min_tokens;
                });
            proposal.verified = static_cast<size_t>(proposal.walk.accepted);
        };
        std::vector<int> pooled;
        std::vector<int> on_thread;
        for (size_t k = 0; k < full_lanes.size(); ++k) {
            const Proposal & proposal =
                proposals[static_cast<size_t>(full_lanes[k])];
            const bool host_only = sample_logits_host_only(
                slots_.slot(proposal.slot).sampler, vocab);
            (host_only ? pooled : on_thread).push_back(static_cast<int>(k));
        }
        if (pooled.size() >= 2) {
            if (!chain_sampled_pool_) {
                chain_sampled_pool_ = std::make_unique<BlockingRowPool>();
            }
            chain_sampled_pool_->run_custom(
                static_cast<int>(pooled.size()),
                [&](int j) { walk(pooled[static_cast<size_t>(j)]); });
        } else {
            for (int k : pooled) walk(k);
        }
        for (int k : on_thread) walk(k);
        const auto t2 = std::chrono::steady_clock::now();
        readback_ms +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        walk_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    for (int lane_index = 0; lane_index < spec_count; ++lane_index) {
        Proposal & proposal = proposals[static_cast<size_t>(lane_index)];
        const int row_base = ar_count + lane_index * tree_width;
        const int32_t * lane_posterior =
            posterior_ptr + static_cast<size_t>(row_base);
        if (!proposal.is_sampled) {
            proposal.verified = chain_verified_prefix(
                proposal.tokens, lane_posterior,
                static_cast<size_t>(tree_width));
        }
        size_t accepted = proposal.verified;
        const int room =
            slots_.max_context() - slots_.slot(proposal.slot).cur_pos;
        accepted = std::min(accepted, static_cast<size_t>(std::max(0, room)));
        if (accepted == 0) {
            result.error = "fixed chain accepted path is empty";
            return result;
        }
        proposal.accepted = chain_min_tokens_safe_prefix(
            proposal.tokens.data(), accepted,
            slots_.slot(proposal.slot).generated_tokens(), min_tokens,
            [&](int32_t token) { return token_is_eos(token); });
        if (proposal.accepted == 0) {
            result.error = "fixed chain min-token clamp removed the root";
            return result;
        }
    }

    chain_accepted_prefixes_.assign(static_cast<size_t>(tree_bucket), 0);
    chain_commit_slots_.assign(static_cast<size_t>(tree_bucket), -1);
    chain_commit_rows_.assign(static_cast<size_t>(tree_rows_count), -1);
    chain_feature_commit_rows_.assign(static_cast<size_t>(total_rows), -1);
    const int feature_cap = b_.cache_.target_feat_cap;

    for (int lane_index = 0; lane_index < ar_count; ++lane_index) {
        const ArLane & lane = ar_lanes[static_cast<size_t>(lane_index)];
        chain_feature_commit_rows_[static_cast<size_t>(lane_index)] =
            lane.slot * feature_cap + lane.position % feature_cap;
    }

    const auto block_table_delta_fits = [&](int slot, int first_block,
                                             size_t count) {
        if (count == 0) return true;
        const ggml_tensor * table = b_.cache_.paged_block_table;
        return table && slot >= 0 && slot < table->ne[1] &&
            first_block >= 0 &&
            static_cast<uint64_t>(first_block) + count <=
                static_cast<uint64_t>(table->ne[0]);
    };

    for (int lane_index = 0; lane_index < spec_count; ++lane_index) {
        Proposal & proposal = proposals[static_cast<size_t>(lane_index)];
        proposal.append = slots_.append_tokens(
            proposal.slot, proposal.tokens.data(),
            static_cast<int>(proposal.accepted));
        while (!proposal.append.ok && proposal.append.busy &&
               evict_one_gpu_prefix()) {
            proposal.append = slots_.append_tokens(
                proposal.slot, proposal.tokens.data(),
                static_cast<int>(proposal.accepted));
        }
        const Qwen35SlotManager::StepAppend & append = proposal.append;
        if (!append.ok ||
            append.physical_rows.size() != proposal.accepted) {
            result.error = append.busy
                ? "paged KV pool exhausted during chain staging"
                : "accepted chain K/V staging failed";
            return result;
        }
        staged_round.track(proposal.slot);
        if (!block_table_delta_fits(
                proposal.slot, append.first_new_block,
                append.new_blocks.size())) {
            result.error = "accepted chain block-table update failed";
            return result;
        }

        chain_accepted_prefixes_[static_cast<size_t>(lane_index)] =
            static_cast<int32_t>(proposal.accepted);
        chain_commit_slots_[static_cast<size_t>(lane_index)] = proposal.slot;
        for (size_t depth = 0; depth < proposal.accepted; ++depth) {
            const int flat = lane_index * tree_width + static_cast<int>(depth);
            const int graph_row = ar_count + flat;
            chain_commit_rows_[static_cast<size_t>(flat)] =
                append.physical_rows[depth];
            chain_feature_commit_rows_[static_cast<size_t>(graph_row)] =
                proposal.slot * feature_cap +
                (append.position + static_cast<int>(depth)) % feature_cap;
        }
        proposal.seq_len =
            append.position + static_cast<int>(proposal.accepted);
    }

    // Previous step's commit_io_stage_ copies already finished before the
    // verify ggml_backend_synchronize() above, so resetting here is race-free.
    const size_t commit_stage_bytes =
        sizeof(int32_t) * (chain_accepted_prefixes_.size() + chain_commit_slots_.size() +
                           chain_feature_commit_rows_.size() + seq_lens_.size()) +
        sizeof(int64_t) * chain_commit_rows_.size() + 1024;
    commit_io_stage_.reserve(b_.target_backend_, commit_stage_bytes);
    commit_io_stage_.reset();

    commit_io_stage_.set(
        b_.target_backend_, graph.accepted_prefixes,
        chain_accepted_prefixes_.data(), 0,
        sizeof(int32_t) * chain_accepted_prefixes_.size());
    commit_io_stage_.set(
        b_.target_backend_, graph.commit_slot_ids,
        chain_commit_slots_.data(), 0,
        sizeof(int32_t) * chain_commit_slots_.size());
    commit_io_stage_.set(
        b_.target_backend_, graph.commit_rows,
        chain_commit_rows_.data(), 0,
        sizeof(int64_t) * chain_commit_rows_.size());
    commit_io_stage_.set(
        b_.target_backend_, graph.feature_commit_rows,
        chain_feature_commit_rows_.data(), 0,
        sizeof(int32_t) * chain_feature_commit_rows_.size());
    // Tree commit kernels run on gdn-replay-log's dedicated commit_stream,
    // which does not implicitly order after b_.target_backend_'s non-blocking
    // stream; sync once so all four uploaded index tables are visible.
    ggml_backend_synchronize(b_.target_backend_);

    const size_t n_delta = b_.cache_.ssm_state.size();
    if (n_delta == 0 || graph.delta_captures.size() != n_delta ||
        b_.cache_.conv_state.size() != n_delta ||
        !graph.tree_features || !b_.cache_.target_feat) {
        result.error = "fixed chain capture set is incomplete";
        return result;
    }
    chain_replay_logs_.clear();
    chain_states_.clear();
    chain_conv_inputs_.clear();
    chain_conv_states_.clear();
    chain_replay_logs_.reserve(n_delta);
    chain_states_.reserve(n_delta);
    chain_conv_inputs_.reserve(n_delta);
    chain_conv_states_.reserve(n_delta);
    for (size_t layer = 0; layer < n_delta; ++layer) {
        const DeltaNetCapture & capture = graph.delta_captures[layer];
        if (!capture.replay_log || !capture.conv_input ||
            !b_.cache_.ssm_state[layer] ||
            !b_.cache_.conv_state[layer]) {
            result.error = "fixed chain layer capture is incomplete";
            return result;
        }
        chain_replay_logs_.push_back(capture.replay_log);
        chain_states_.push_back(b_.cache_.ssm_state[layer]);
        chain_conv_inputs_.push_back(capture.conv_input);
        chain_conv_states_.push_back(b_.cache_.conv_state[layer]);
    }

    chain_caches_.clear();
    chain_caches_.reserve(b_.cache_.attn_k.size() + b_.cache_.attn_v.size());
    for (ggml_tensor * tensor : b_.cache_.attn_k) {
        if (tensor) chain_caches_.push_back(tensor);
    }
    for (ggml_tensor * tensor : b_.cache_.attn_v) {
        if (tensor) chain_caches_.push_back(tensor);
    }
    if (chain_caches_.empty()) {
        result.error = "fixed chain K/V cache set is empty";
        return result;
    }

    // Validate from the host copies just uploaded (no read-back) and commit
    // with one launch per kind. A refused preflight changes nothing, so the
    // device-validated transaction then runs as before.
    bool committed = false;
    if (static_cast<int64_t>(chain_accepted_prefixes_.size()) ==
            ggml_nelements(graph.accepted_prefixes) &&
        static_cast<int64_t>(chain_commit_slots_.size()) ==
            ggml_nelements(graph.commit_slot_ids) &&
        static_cast<int64_t>(chain_commit_rows_.size()) ==
            ggml_nelements(graph.commit_rows) &&
        static_cast<int64_t>(chain_feature_commit_rows_.size()) ==
            ggml_nelements(graph.feature_commit_rows)) {
        committed = ggml_backend_cuda_tree_commit_transaction_host(
            chain_caches_.data(), static_cast<int>(chain_caches_.size()),
            graph.tree_features, b_.cache_.target_feat,
            graph.feature_commit_rows,
            chain_replay_logs_.data(), chain_states_.data(),
            chain_conv_inputs_.data(), chain_conv_states_.data(),
            static_cast<int>(n_delta),
            graph.commit_rows, graph.accepted_prefixes,
            graph.commit_slot_ids,
            fixed_chain_.scratch_base, fixed_chain_.scratch_stride,
            chain_commit_rows_.data(), chain_accepted_prefixes_.data(),
            chain_commit_slots_.data(), chain_feature_commit_rows_.data());
        static std::atomic<bool> warned{false};
        if (!committed && !warned.exchange(true)) {
            std::fprintf(stderr,
                "[qwen35] host-validated commit preflight refused; "
                "using the device-validated commit transaction\n");
        }
    }
    if (!committed &&
        !ggml_backend_cuda_tree_commit_transaction(
            chain_caches_.data(), static_cast<int>(chain_caches_.size()),
            graph.tree_features, b_.cache_.target_feat,
            graph.feature_commit_rows,
            chain_replay_logs_.data(), chain_states_.data(),
            chain_conv_inputs_.data(), chain_conv_states_.data(),
            static_cast<int>(n_delta),
            graph.commit_rows, graph.accepted_prefixes,
            graph.commit_slot_ids,
            fixed_chain_.scratch_base, fixed_chain_.scratch_stride)) {
        result.error = "fixed chain commit preflight failed";
        return result;
    }

    for (const Proposal & proposal : proposals) {
        if (!upload_block_table_delta(
                proposal.slot, proposal.append.first_new_block,
                proposal.append.new_blocks.data(),
                proposal.append.new_blocks.size())) {
            std::fprintf(stderr,
                "[parallel] validated chain block-table upload failed\n");
            std::abort();
        }
        seq_lens_[static_cast<size_t>(proposal.slot)] = proposal.seq_len;
    }
    commit_io_stage_.set(
        b_.target_backend_, b_.cache_.paged_kv_seq_lens, seq_lens_.data(), 0,
        sizeof(int32_t) * seq_lens_.size());

    for (const StepInput & input : inputs) {
        slots_.commit_step(input.slot);
    }
    staged_round.committed = true;
    for (int lane_index = 0; lane_index < spec_count; ++lane_index) {
        Proposal & proposal = proposals[static_cast<size_t>(lane_index)];
        const int row_base = ar_count + lane_index * tree_width;
        const int graph_row = row_base + static_cast<int>(proposal.accepted) - 1;
        if (proposal.is_sampled) {
            // The walk's draw at the kept tip, or one fresh draw from the
            // tip's row (via GPU top_n or host logits) with the committed history.
            Qwen35Slot & seq = slots_.slot(proposal.slot);
            const auto hist = seq.penalty_history_view();
            const int32_t next = sampled_verify_next(
                proposal.walk, proposal.tokens.data(),
                static_cast<int>(proposal.accepted), [&](int row) -> int32_t {
                    if (proposal.sampled_rows) {
                        return sample_logits(
                            proposal.sampled_rows +
                                static_cast<size_t>(row) *
                                    static_cast<size_t>(vocab),
                            vocab, seq.sampler, hist.data, hist.size, seq.rng);
                    }
                    if (have_top_n) {
                        const size_t off =
                            static_cast<size_t>(row_base + row) *
                            static_cast<size_t>(max_top_n);
                        const int32_t t = sample_logits_top(
                            top_vals_ptr + off, top_ids_ptr + off,
                            max_top_n, vocab, seq.sampler,
                            hist.data, hist.size, seq.rng);
                        if (t >= 0) return t;
                    }
                    if ((int)logits_buf_.size() < vocab) {
                        logits_buf_.resize(static_cast<size_t>(vocab));
                    }
                    ggml_backend_tensor_get(
                        graph.logits, logits_buf_.data(),
                        static_cast<size_t>(row_base + row) *
                            static_cast<size_t>(vocab) * sizeof(float),
                        sizeof(float) * static_cast<size_t>(vocab));
                    return sample_logits(
                        logits_buf_.data(), vocab, seq.sampler,
                        hist.data, hist.size, seq.rng);
                });
            proposal.pending = b_.apply_min_tokens_floor(
                next, seq.generated_tokens(),
                static_cast<size_t>(graph_row) * static_cast<size_t>(vocab) *
                    sizeof(float),
                graph.logits);
        } else {
            proposal.pending = sample_graph_row(
                proposal.slot, graph_row,
                posterior_ptr[static_cast<size_t>(graph_row)], logits_buf_,
                graph.logits, top_ids_ptr, top_vals_ptr, have_top_n ? max_top_n : 0);
        }
        if (proposal.pending < 0) {
            result.error = "fixed chain sampling failed";
            return result;
        }
    }
    for (int lane_index = 0; lane_index < ar_count; ++lane_index) {
        ArLane & lane = ar_lanes[static_cast<size_t>(lane_index)];
        lane.pending = sample_graph_row(
            lane.slot, lane_index,
            posterior_ptr[static_cast<size_t>(lane_index)], logits_buf_,
            graph.logits, top_ids_ptr, top_vals_ptr, have_top_n ? max_top_n : 0);
        if (lane.pending < 0) {
            result.error = "compact AR sampling failed";
            return result;
        }
    }

    if (adaptive_chain_width_) {
        std::vector<size_t> & accepted = chain_accepted_scratch_;
        std::vector<uint8_t> & sampled = chain_sampled_scratch_;
        accepted.clear();
        sampled.clear();
        // Model acceptance, not the committed length: the room and
        // min-token clamps are serving policy, not rejections.
        for (const Proposal & proposal : proposals) {
            accepted.push_back(proposal.verified);
            sampled.push_back(proposal.is_sampled ? 1 : 0);
        }
        const double step_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - chain_round_t0_).count();
        // Rounds with AR peers or a fresh graph still teach acceptance, but
        // their time is not the cost of this width at this lane count.
        observe_chain_width(tree_bucket, tree_width, accepted, sampled,
                            ar_count == 0 && !graph_rebuilt ? step_ms : -1.0,
                            readback_ms, walk_ms);
    }

    result.decode.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        DecodeOutput output;
        output.slot = inputs[i].slot;
        if (selected[i]) {
            Proposal & proposal =
                proposals[static_cast<size_t>(proposal_for_input[i])];
            output.token = proposal.pending;
            output.committed_tokens.assign(
                proposal.tokens.begin() + 1,
                proposal.tokens.begin() + proposal.accepted);
            output.spec_offered = static_cast<int>(proposal.tokens.size());
            output.spec_accepted = static_cast<int>(proposal.verified);
        } else {
            output.token =
                ar_lanes[static_cast<size_t>(ar_for_input[i])].pending;
        }
        result.decode.push_back(std::move(output));
    }
    return result;
}

SeqEngine::StepResult Qwen35SeqEngine::step(const StepPlan & plan) {
    StepResult result;
    std::vector<DecodeOutput> & decode_outputs = result.decode;
    std::vector<PrefillOutput> & prefill_outputs = result.prefills;
    const std::vector<StepInput> & inputs = plan.decode;
    const int n_slots = slots_.slot_count();

    auto fail_step = [&](const std::string & error) {
        result.decode.clear();
        result.prefills.clear();
        result.error = error;
        return std::move(result);
    };

    if ((int)inputs.size() != slots_.decoding_count()) {
        return fail_step("decode plan does not cover every live slot");
    }
    constexpr uint8_t kDecodeSeen = 1u;
    constexpr uint8_t kPrefillSeen = 2u;
    std::fill(step_seen_.begin(), step_seen_.end(), 0);
    for (const StepInput & in : inputs) {
        if (in.slot < 0 || in.slot >= n_slots || in.token < 0 ||
            (step_seen_[(size_t)in.slot] & kDecodeSeen) ||
            !slots_.slot(in.slot).decoding()) {
            return fail_step("invalid or duplicate decode row in step plan");
        }
        step_seen_[(size_t)in.slot] |= kDecodeSeen;
    }

    int planned_prefill_tokens = 0;
    if (!plan.prefills.empty()) {
        const StepPlanLimits limits = step_plan_limits((int)inputs.size());
        if ((int)plan.prefills.size() > limits.max_prefill_sequences) {
            return fail_step("prefill plan exceeds engine sequence capacity");
        }
        for (const PrefillSlice & slice : plan.prefills) {
            if (slice.slot < 0 || slice.slot >= n_slots ||
                slice.max_tokens <= 0 ||
                slice.max_tokens > limits.max_prefill_tokens_per_sequence ||
                (step_seen_[(size_t)slice.slot] & (kDecodeSeen | kPrefillSeen)) ||
                !slots_.is_prefilling(slice.slot)) {
                return fail_step("invalid or duplicate prefill slice in step plan");
            }
            step_seen_[(size_t)slice.slot] |= kPrefillSeen;
            planned_prefill_tokens += slice.max_tokens;
            if (planned_prefill_tokens > limits.max_prefill_tokens_total) {
                return fail_step("prefill plan exceeds engine total-token capacity");
            }
        }
    }
    if (inputs.empty() && plan.prefills.empty()) return result;

    const Qwen35RoctxMetadata service_round_metadata{
        static_cast<int>(inputs.size()),
        -1,
        planned_prefill_tokens,
        static_cast<int>(plan.prefills.size()),
        -1,
        -1,
    };
    const Qwen35RoctxRange roctx_service_round(
        "qwen35.service_round", service_round_metadata);

    std::vector<uint8_t> chain_lanes = select_chain_lanes(plan);
    const bool has_chain_lane = std::any_of(
        chain_lanes.begin(), chain_lanes.end(),
        [](uint8_t selected) { return selected != 0; });
    if (has_chain_lane) {
        chain_round_t0_ = std::chrono::steady_clock::now();
        std::optional<PreparedChainRound> prepared =
            prepare_chain_drafts(inputs, chain_lanes);
        if (prepared) {
            return step_chain_spec(plan, chain_lanes, std::move(*prepared));
        }
    }

    const TargetWeights & w = b_.w_;
    StepGraph & sg = b_.sg_;
    const int hidden = w.n_embd;
    const int n_head_kv = w.n_head_kv;

    decode_outputs.reserve(inputs.size());
    prefill_outputs.reserve(plan.prefills.size());
    output_rows_.clear();
    live_tokens_.clear();
    live_positions_.clear();
    live_physical_rows_.clear();
    live_slot_ids_.clear();
    output_rows_.reserve(inputs.size());
    live_tokens_.reserve(inputs.size());
    live_positions_.reserve(inputs.size());
    live_physical_rows_.reserve(inputs.size());
    live_slot_ids_.reserve(inputs.size());

    // Checked above against step_plan_limits(), so this always holds.
    if ((size_t)planned_prefill_tokens +
            (size_t)chain_decode_bucket_width((int)inputs.size()) >
        embed_rows_cap_) {
        result.state_clean = true;
        return fail_step("step exceeds the input embedding rows");
    }

    // A failure before the compute (staging, graph build or alloc OOM) undoes
    // the appends, so the scheduler can shed prefills and keep the decoders.
    SeqSlotManager::StagedAppends staged_rollback(slots_, inputs.size(), plan.prefills.size());

    int max_kv_len = 1;
    for (const StepInput & in : inputs) {
        DecodeOutput out;
        out.slot = in.slot;
        out.failed = true;
        int compact_row = -1;
        Qwen35SlotManager::StepAppend app =
            slots_.append_token(in.slot, in.token);
        while (!app.ok && app.busy &&
               evict_one_gpu_prefix()) {
            app = slots_.append_token(in.slot, in.token);
        }
        if (!app.ok) {
            out.error = app.busy
                ? "paged KV pool exhausted during decode; raise "
                  "--kv-pool-tokens or lower --max-ctx/--max-concurrency"
                : "decode K/V append failed";
            decode_outputs.push_back(std::move(out));
            output_rows_.push_back(compact_row);
            continue;
        }
        if (app.new_block >= 0 &&
            !upload_block_table_delta(
                in.slot, app.new_block_index, &app.new_block, 1)) {
            slots_.rollback_step(in.slot);
            out.error = "decode block-table entry exceeds device capacity";
            decode_outputs.push_back(std::move(out));
            output_rows_.push_back(compact_row);
            continue;
        }
        staged_rollback.decode(in.slot);
        compact_row = (int)live_tokens_.size();
        live_tokens_.push_back(in.token);
        live_positions_.push_back(app.position);
        live_physical_rows_.push_back(app.physical_row);
        live_slot_ids_.push_back(in.slot);
        max_kv_len = std::max(max_kv_len, app.position + 1);
        out.failed = false;
        decode_outputs.push_back(std::move(out));
        output_rows_.push_back(compact_row);
    }

    std::vector<PrefillStage> prefills;
    prefills.reserve(plan.prefills.size());
    int staged_rows = 0;
    for (const PrefillSlice & slice : plan.prefills) {
        const size_t outputs_before = prefill_outputs.size();
        PrefillStage prefill = stage_prefill_chunk(
            slice.slot, slice.max_tokens,
            embed_rows_ + (size_t)w.n_embd * staged_rows, prefill_outputs);
        if (!prefill.ready) {
            if (prefill_outputs.size() == outputs_before) {
                fail_prefill(
                    slice.slot, prefill_outputs,
                    "prefill made no progress despite reserved capacity",
                    "prefill scheduler made no progress");
            }
            result.state_clean = true;
            return fail_step("selected prefill work made no progress");
        }
        staged_rollback.prefill(slice.slot, prefill.chunk);
        staged_rows += prefill.chunk;
        prefills.push_back(std::move(prefill));
    }

    const int live_count = (int)live_tokens_.size();
    const bool with_decode = live_count > 0;
    const int decode_bucket = with_decode
        ? chain_decode_bucket_width(live_count)
        : 0;

    dec_tokens_.assign((size_t)decode_bucket, 0);
    dec_rows_.assign((size_t)decode_bucket * n_head_kv, scratch_row_);
    active_slot_ids_.assign((size_t)decode_bucket, -1);
    state_slot_ids_.assign((size_t)decode_bucket, 0);
    seq_lens_.assign((size_t)n_slots, 0);
    for (int row = 0; row < live_count; ++row) {
        dec_tokens_[(size_t)row] = live_tokens_[(size_t)row];
        const int pos = live_positions_[(size_t)row];
        active_slot_ids_[(size_t)row] = live_slot_ids_[(size_t)row];
        state_slot_ids_[(size_t)row] = live_slot_ids_[(size_t)row];
        seq_lens_[(size_t)live_slot_ids_[(size_t)row]] = pos + 1;
        for (int h = 0; h < n_head_kv; ++h) {
            dec_rows_[(size_t)h * decode_bucket + row] =
                live_physical_rows_[(size_t)row];
        }
    }

    int n_prefill = 0;
    int n_commits = 0;
    std::vector<QwenPrefillSegment> segments;
    segments.reserve(prefills.size());
    for (size_t i = 0; i < prefills.size(); ++i) {
        const PrefillStage & prefill = prefills[i];
        const int slot = plan.prefills[i].slot;
        segments.push_back({n_prefill, prefill.chunk, slot});
        n_prefill += prefill.chunk;
        n_commits += prefill.commit ? 1 : 0;
        max_kv_len = std::max(max_kv_len, prefill.kv_pos + prefill.chunk);
        seq_lens_[(size_t)slot] = prefill.kv_pos + prefill.chunk;
    }
    const bool with_prefill = n_prefill > 0;
    const int n_total = n_prefill + decode_bucket;
    const Qwen35RoctxMetadata roctx_metadata{
        live_count, decode_bucket, n_prefill, (int)segments.size(),
        n_total, max_kv_len};
    const Qwen35RoctxRange roctx_step("qwen35.concurrent_step", roctx_metadata);
    const int gather_rows = with_prefill
        ? (with_decode ? n_commits + decode_bucket
                       : std::max(1, n_commits))
        : 0;

    const int vocab = w.n_vocab;
    int max_top_n = 0;
    if (gpu_argmax_enabled()) {
        for (int row = 0; row < live_count; ++row) {
            const SamplerCfg & cfg = slots_.slot(live_slot_ids_[(size_t)row]).sampler;
            if (cfg.needs_logit_processing()) {
                max_top_n = std::max(max_top_n, sample_logits_top_n(cfg, vocab));
            }
        }
        for (size_t i = 0; i < prefills.size(); ++i) {
            if (!prefills[i].commit) continue;
            const SamplerCfg & cfg = slots_.slot(plan.prefills[i].slot).sampler;
            if (cfg.needs_logit_processing()) {
                max_top_n = std::max(max_top_n, sample_logits_top_n(cfg, vocab));
            }
        }
    }

    bool built = false;
    if (with_prefill) {
        built = build_target_step(
            sg, w, b_.cache_, b_.target_backend_,
            /*kv_start=*/0, /*n_tokens=*/n_total,
            /*with_mask=*/false, /*capture=*/fixed_chain_ready_,
            /*capture_delta_intermediate=*/false,
            /*fa_window=*/0, /*logits_tail_rows=*/0,
            b_.cfg_.kq_stride_pad,
            /*capture_moe_router=*/false,
            /*kvflash_mask=*/false,
            /*capture_qk=*/false,
            /*paged_attention=*/true,
            /*n_seqs=*/with_decode ? decode_bucket : 1,
            /*seq_slot=*/0,
            /*paged_max_kv_len=*/max_kv_len,
            /*n_prefill_tokens=*/n_prefill,
            segments.data(), (int)segments.size(), gather_rows,
            /*compact_slots=*/with_decode,
            /*top_n=*/max_top_n);
    } else {
        built = build_target_step(
            sg, w, b_.cache_, b_.target_backend_,
            /*kv_start=*/0, /*n_tokens=*/decode_bucket,
            /*with_mask=*/false, /*capture=*/fixed_chain_ready_,
            /*capture_delta_intermediate=*/false,
            /*fa_window=*/0, /*logits_tail_rows=*/0,
            b_.cfg_.kq_stride_pad,
            /*capture_moe_router=*/false,
            /*kvflash_mask=*/false,
            /*capture_qk=*/false,
            /*paged_attention=*/true,
            /*n_seqs=*/decode_bucket,
            /*seq_slot=*/0,
            /*paged_max_kv_len=*/max_kv_len,
            /*n_prefill_tokens=*/0,
            /*prefill_segments=*/nullptr,
            /*n_prefill_segments=*/0,
            /*n_logits_rows=*/0,
            /*compact_slots=*/true,
            /*top_n=*/max_top_n);
    }
    if (!built || !sg.kv_write_rows ||
        (fixed_chain_ready_ && !sg.target_feat_rows) ||
        (with_prefill &&
         (!sg.paged_query_seq_ids || !sg.paged_query_positions ||
          !sg.logits_row_indices))) {
        result.state_clean = true;
        return fail_step("packed prefill/decode graph build failed");
    }

    // The prompt rows are already in place (stage_prefill_chunk). Only embed
    // the live decode rows and zero-fill any padding rows in the bucket.
    if (with_decode) {
        float * dec_embed = embed_rows_ + (size_t)hidden * n_prefill;
        if (!w.embedder.embed(dec_tokens_.data(), live_count, dec_embed)) {
            result.state_clean = true;
            return fail_step("decode embedding failed");
        }
        if (decode_bucket > live_count) {
            std::memset(
                dec_embed + (size_t)hidden * live_count, 0,
                sizeof(float) * (size_t)hidden *
                    static_cast<size_t>(decode_bucket - live_count));
        }
    }

    const int decode_row0 = with_prefill ? n_commits : 0;
    const int argmax_rows = with_prefill ? gather_rows : decode_bucket;
    const bool have_top_n = max_top_n > 0 && sg.top_ids && sg.top_vals;
    const size_t top_elems = have_top_n
        ? (size_t)max_top_n * (size_t)argmax_rows : 0;
    const size_t stage_bytes =
        sizeof(int32_t) * ((size_t)4 * n_total + (size_t)n_total +
                           (with_prefill ? (size_t)2 * n_total + (size_t)gather_rows : 0) +
                           (with_decode ? (size_t)2 * decode_bucket : 0) +
                           seq_lens_.size() + (size_t)argmax_rows + top_elems) +
        sizeof(int64_t) * (size_t)n_total * n_head_kv +
        sizeof(float) * top_elems + 4096;
    target_io_stage_.reserve(b_.target_backend_, stage_bytes);
    target_io_stage_.reset();

    ggml_backend_tensor_set_async(
        b_.target_backend_, sg.inp_embed, embed_rows_, 0,
        sizeof(float) * (size_t)hidden * n_total);

    pos_buf_.assign((size_t)4 * n_total, 0);
    int token_offset = 0;
    for (const PrefillStage & prefill : prefills) {
        if (prefill.positions.empty()) {
            fill_qwen35_mrope_positions(
                pos_buf_.data(), n_total, token_offset,
                prefill.kv_pos, prefill.chunk);
        } else {
            for (int axis = 0; axis < 4; ++axis) {
                std::copy_n(prefill.positions.data() + (size_t)axis * prefill.chunk, prefill.chunk,
                            pos_buf_.data() + (size_t)axis * n_total + token_offset);
            }
        }
        token_offset += prefill.chunk;
    }
    if (with_decode) {
        for (int row = 0; row < live_count; ++row) {
            // Rotary positions run ahead of KV positions after an image.
            const int pos = live_positions_[(size_t)row] +
                rope_delta(live_slot_ids_[(size_t)row]);
            const int packed_row = n_prefill + row;
            pos_buf_[(size_t)0 * n_total + packed_row] = pos;
            pos_buf_[(size_t)1 * n_total + packed_row] = pos;
            pos_buf_[(size_t)2 * n_total + packed_row] = pos;
        }
    }
    target_io_stage_.set(
        b_.target_backend_, sg.positions, pos_buf_.data(), 0,
        sizeof(int32_t) * pos_buf_.size());

    if (fixed_chain_ready_) {
        const int cap = b_.cache_.target_feat_cap;
        const int dead_row = cap * n_slots;
        feature_rows_.assign(static_cast<size_t>(n_total), dead_row);
        token_offset = 0;
        for (size_t i = 0; i < prefills.size(); ++i) {
            const PrefillStage & prefill = prefills[i];
            const int slot = plan.prefills[i].slot;
            for (int row = 0; row < prefill.chunk; ++row) {
                feature_rows_[static_cast<size_t>(token_offset + row)] =
                    slot * cap + (prefill.kv_pos + row) % cap;
            }
            token_offset += prefill.chunk;
        }
        for (int row = 0; row < live_count; ++row) {
            const int slot = live_slot_ids_[static_cast<size_t>(row)];
            feature_rows_[static_cast<size_t>(n_prefill + row)] =
                slot * cap +
                live_positions_[static_cast<size_t>(row)] % cap;
        }
        target_io_stage_.set(
            b_.target_backend_, sg.target_feat_rows,
            feature_rows_.data(), 0,
            sizeof(int32_t) * feature_rows_.size());
    }

    rows_buf_.resize((size_t)n_total * n_head_kv);
    token_offset = 0;
    for (const PrefillStage & prefill : prefills) {
        std::memcpy(
            rows_buf_.data() + token_offset, prefill.rows.data(),
            sizeof(int64_t) * static_cast<size_t>(prefill.chunk));
        token_offset += prefill.chunk;
    }
    if (decode_bucket > 0) {
        std::memcpy(
            rows_buf_.data() + n_prefill, dec_rows_.data(),
            sizeof(int64_t) * static_cast<size_t>(decode_bucket));
    }
    for (int h = 1; h < n_head_kv; ++h) {
        std::memcpy(
            rows_buf_.data() + (size_t)h * n_total, rows_buf_.data(),
            sizeof(int64_t) * static_cast<size_t>(n_total));
    }
    target_io_stage_.set(
        b_.target_backend_, sg.kv_write_rows, rows_buf_.data(), 0,
        sizeof(int64_t) * rows_buf_.size());

    if (with_prefill) {
        query_slot_ids_.assign((size_t)n_total, -1);
        query_positions_.assign((size_t)n_total, -1);
        logits_rows_.clear();
        logits_rows_.reserve((size_t)gather_rows);
        token_offset = 0;
        for (size_t i = 0; i < prefills.size(); ++i) {
            const PrefillStage & prefill = prefills[i];
            const int slot = plan.prefills[i].slot;
            for (int row = 0; row < prefill.chunk; ++row) {
                query_slot_ids_[(size_t)(token_offset + row)] = slot;
                query_positions_[(size_t)(token_offset + row)] =
                    prefill.kv_pos + row;
            }
            if (prefill.commit) {
                logits_rows_.push_back(token_offset + prefill.chunk - 1);
            }
            token_offset += prefill.chunk;
        }
        for (int row = 0; row < live_count; ++row) {
            query_slot_ids_[(size_t)(n_prefill + row)] =
                live_slot_ids_[(size_t)row];
            query_positions_[(size_t)(n_prefill + row)] =
                live_positions_[(size_t)row];
        }
        for (int row = 0; row < decode_bucket; ++row) {
            logits_rows_.push_back(n_prefill + row);
        }
        if (logits_rows_.empty()) {
            logits_rows_.push_back(n_total - 1);
        }
        target_io_stage_.set(
            b_.target_backend_, sg.paged_query_seq_ids,
            query_slot_ids_.data(), 0,
            sizeof(int32_t) * query_slot_ids_.size());
        target_io_stage_.set(
            b_.target_backend_, sg.paged_query_positions,
            query_positions_.data(), 0,
            sizeof(int32_t) * query_positions_.size());
        target_io_stage_.set(
            b_.target_backend_, sg.logits_row_indices,
            logits_rows_.data(), 0,
            sizeof(int32_t) * logits_rows_.size());
    }
    if (with_decode) {
        target_io_stage_.set(
            b_.target_backend_, sg.active_slot_ids,
            active_slot_ids_.data(), 0,
            sizeof(int32_t) * active_slot_ids_.size());
        target_io_stage_.set(
            b_.target_backend_, sg.state_slot_ids,
            state_slot_ids_.data(), 0,
            sizeof(int32_t) * state_slot_ids_.size());
    }
    target_io_stage_.set(
        b_.target_backend_, b_.cache_.paged_kv_seq_lens,
        seq_lens_.data(), 0, sizeof(int32_t) * seq_lens_.size());

    staged_rollback.commit();
    ggml_status st = GGML_STATUS_FAILED;
    {
        const Qwen35RoctxRange roctx_compute(
            "qwen35.graph_compute", roctx_metadata);
        st = ggml_backend_graph_compute(b_.target_backend_, sg.gf);
    }
    if (st != GGML_STATUS_SUCCESS) {
        // The pinned embedding rows may still be in flight.
        ggml_backend_synchronize(b_.target_backend_);
        return fail_step("packed prefill/decode compute failed");
    }

    const size_t argmax_bytes = sizeof(int32_t) * (size_t)argmax_rows;
    const int32_t * argmax_ptr = static_cast<const int32_t *>(
        target_io_stage_.get(b_.target_backend_, sg.argmax_tokens, 0, argmax_bytes));
    if (!argmax_ptr) {
        argmax_buf_.assign((size_t)argmax_rows, -1);
        ggml_backend_tensor_get_async(
            b_.target_backend_, sg.argmax_tokens, argmax_buf_.data(), 0, argmax_bytes);
        argmax_ptr = argmax_buf_.data();
    }
    const int32_t * top_ids_ptr = nullptr;
    const float * top_vals_ptr = nullptr;
    if (have_top_n) {
        const size_t id_bytes = sizeof(int32_t) * top_elems;
        const size_t val_bytes = sizeof(float) * top_elems;
        top_ids_ptr = static_cast<const int32_t *>(
            target_io_stage_.get(b_.target_backend_, sg.top_ids, 0, id_bytes));
        top_vals_ptr = static_cast<const float *>(
            target_io_stage_.get(b_.target_backend_, sg.top_vals, 0, val_bytes));
        if (!top_ids_ptr) {
            top_ids_buf_.resize(top_elems);
            ggml_backend_tensor_get_async(
                b_.target_backend_, sg.top_ids, top_ids_buf_.data(), 0, id_bytes);
            top_ids_ptr = top_ids_buf_.data();
        }
        if (!top_vals_ptr) {
            top_vals_buf_.resize(top_elems);
            ggml_backend_tensor_get_async(
                b_.target_backend_, sg.top_vals, top_vals_buf_.data(), 0, val_bytes);
            top_vals_ptr = top_vals_buf_.data();
        }
    }
    {
        const Qwen35RoctxRange roctx_sync(
            "qwen35.argmax_readback", roctx_metadata);
        ggml_backend_synchronize(b_.target_backend_);
    }

    // Batch sampled rows: try GPU top_n first, then copy any remaining full
    // logits rows asynchronously with one sync and sample host-only rows in
    // parallel.
    struct PendingSample {
        bool is_decode = true;
        size_t output_idx = 0;
        int slot = -1;
        int logits_row = -1;
        int32_t token = -1;
    };
    std::vector<PendingSample> need_full;
    for (size_t oi = 0; oi < inputs.size(); ++oi) {
        DecodeOutput & out = decode_outputs[oi];
        if (out.failed) continue;
        slots_.commit_step(out.slot);
        const int row = decode_row0 + output_rows_[oi];
        Qwen35Slot & seq = slots_.slot(out.slot);
        if (!seq.sampler.needs_logit_processing()) {
            out.token = b_.apply_min_tokens_floor(
                argmax_ptr[(size_t)row], seq.generated_tokens(),
                (size_t)row * (size_t)vocab * sizeof(float));
            continue;
        }
        if (have_top_n) {
            const auto hist = seq.penalty_history_view();
            const int32_t t = sample_logits_top(
                top_vals_ptr + (size_t)row * (size_t)max_top_n,
                top_ids_ptr + (size_t)row * (size_t)max_top_n,
                max_top_n, vocab, seq.sampler, hist.data, hist.size, seq.rng);
            if (t >= 0) {
                out.token = b_.apply_min_tokens_floor(
                    t, seq.generated_tokens(),
                    (size_t)row * (size_t)vocab * sizeof(float));
                continue;
            }
        }
        need_full.push_back({true, oi, out.slot, row, -1});
    }

    int commit_row = 0;
    for (size_t i = 0; i < prefills.size(); ++i) {
        const int slot = plan.prefills[i].slot;
        PrefillOutput out;
        out.slot = slot;
        const PrefixCaptureTicket capture = slots_.slot(slot).pending_capture;
        if (capture.valid() &&
            slots_.slot(slot).cur_pos == capture.checkpoint.tokens) {
            out.prefix_store = capture_prefix(slot, capture);
            slots_.slot(slot).pending_capture = {};
        }
        if (prefills[i].commit) {
            out.status = PrefillOutput::Status::completed;
            Qwen35Slot & seq = slots_.slot(slot);
            if (!seq.sampler.needs_logit_processing()) {
                out.token = b_.apply_min_tokens_floor(
                    argmax_ptr[(size_t)commit_row], seq.generated_tokens(),
                    (size_t)commit_row * (size_t)vocab * sizeof(float));
            } else {
                int32_t t = -1;
                if (have_top_n) {
                    const auto hist = seq.penalty_history_view();
                    t = sample_logits_top(
                        top_vals_ptr + (size_t)commit_row * (size_t)max_top_n,
                        top_ids_ptr + (size_t)commit_row * (size_t)max_top_n,
                        max_top_n, vocab, seq.sampler, hist.data, hist.size, seq.rng);
                }
                if (t >= 0) {
                    out.token = b_.apply_min_tokens_floor(
                        t, seq.generated_tokens(),
                        (size_t)commit_row * (size_t)vocab * sizeof(float));
                } else {
                    need_full.push_back(
                        {false, prefill_outputs.size(), slot, commit_row, -1});
                }
            }
            ++commit_row;
            slots_.commit_prefill(slot);
        }
        prefill_outputs.push_back(std::move(out));
    }

    if (!need_full.empty()) {
        chain_sampled_rows_.resize(need_full.size() * (size_t)vocab);
        for (size_t k = 0; k < need_full.size(); ++k) {
            ggml_backend_tensor_get_async(
                b_.target_backend_, sg.logits,
                chain_sampled_rows_.data() + k * (size_t)vocab,
                (size_t)need_full[k].logits_row * (size_t)vocab * sizeof(float),
                sizeof(float) * (size_t)vocab);
        }
        ggml_backend_synchronize(b_.target_backend_);

        std::vector<size_t> pooled;
        std::vector<size_t> on_thread;
        for (size_t k = 0; k < need_full.size(); ++k) {
            const bool host_only = sample_logits_host_only(
                slots_.slot(need_full[k].slot).sampler, vocab);
            (host_only ? pooled : on_thread).push_back(k);
        }
        const auto sample_one = [&](size_t k) {
            PendingSample & p = need_full[k];
            Qwen35Slot & seq = slots_.slot(p.slot);
            const auto hist = seq.penalty_history_view();
            p.token = sample_logits(
                chain_sampled_rows_.data() + k * (size_t)vocab,
                vocab, seq.sampler, hist.data, hist.size, seq.rng);
        };
        if (pooled.size() >= 2) {
            if (!chain_sampled_pool_) {
                chain_sampled_pool_ = std::make_unique<BlockingRowPool>();
            }
            chain_sampled_pool_->run_custom(
                static_cast<int>(pooled.size()),
                [&](int j) { sample_one(pooled[static_cast<size_t>(j)]); });
        } else {
            for (size_t k : pooled) sample_one(k);
        }
        for (size_t k : on_thread) sample_one(k);

        for (const PendingSample & p : need_full) {
            Qwen35Slot & seq = slots_.slot(p.slot);
            const int32_t tok = b_.apply_min_tokens_floor(
                p.token, seq.generated_tokens(),
                (size_t)p.logits_row * (size_t)vocab * sizeof(float));
            if (p.is_decode) {
                decode_outputs[p.output_idx].token = tok;
            } else {
                prefill_outputs[p.output_idx].token = tok;
            }
        }
    }
    return result;
}

bool Qwen35SeqEngine::reserve_decode(const StepPlan & plan) {
    std::fill(reserve_growth_.begin(), reserve_growth_.end(), 0);
    const auto chains = select_chain_lanes(plan);
    const int chain_width = chain_draft_width(chains);      // the most the round can draft and verify
    for (size_t i = 0; i < plan.decode.size(); ++i) {
        const int slot = plan.decode[i].slot;
        if (slot < 0 || slot >= slots_.slot_count() ||
            reserve_growth_[(size_t)slot]) return false;
        reserve_growth_[(size_t)slot] = chains[i] ? chain_width : 1;
    }
    bool ok = slots_.reserve_decode(reserve_growth_);
    while (!ok && evict_one_gpu_prefix()) {
        ok = slots_.reserve_decode(reserve_growth_);
    }
    return ok;
}

bool Qwen35SeqEngine::restore_kv(int slot, std::string & error) {
    if (slots_.is_active(slot) && slots_.slot(slot).recomputing()) {
        // Parked without a checkpoint: resume as an ordinary chunked prefill
        // over the folded history, which rebuilds paged KV and slot-local
        // state together — so it must be reset exactly as at admission.
        bool ok = slots_.resume_recompute(slot);
        while (!ok && evict_one_gpu_prefix()) {
            ok = slots_.resume_recompute(slot);
        }
        if (!ok) return false;
        reset_recurrent_slot(b_.cache_, slot);
        reset_slot_draft_kv(slot);
        return true;
    }
    std::vector<int32_t> blocks;
    bool ok = offload_.restore(slot, blocks, error);
    while (!ok && evict_one_gpu_prefix()) {
        ok = offload_.restore(slot, blocks, error);
    }
    if (!ok) return false;
    if (!upload_block_table_delta(slot, 0, blocks.data(), blocks.size())) {
        error = "restored Qwen KV block table exceeds device capacity";
        return false;
    }
    // step() uploads current per-slot lengths from the preserved host state.
    return true;
}

bool Qwen35SeqEngine::evict_kv(int slot, int32_t pending_token,
                               std::string & error) {
    if (!slots_.evict_for_recompute(slot, pending_token)) {
        error = "slot history cannot be re-prefilled within the paged KV pool";
        return false;
    }
    offload_.discard(slot);
    return true;
}

void Qwen35SeqEngine::retire(int slot) {
    if (slot >= 0 && (size_t)slot < restored_kv_checkpoint_.size()) {
        restored_kv_checkpoint_[(size_t)slot] = -1;
        restored_kv_tokens_[(size_t)slot] = 0;
    }
    offload_.discard(slot);
    clear_slot_images(slot);
    if (!slots_.is_active(slot)) return;
    reset_slot_draft_kv(slot);
    slots_.retire(slot);
}

}  // namespace luce::common
