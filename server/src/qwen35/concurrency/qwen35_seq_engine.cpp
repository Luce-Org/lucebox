// Concurrent slot engine for the paged Qwen3.5/3.6 backend
// (--max-concurrency N).
//
// All calls come from the HTTP scheduler thread, which is also the only
// caller of the pool, step graph, and device metadata uploads.

#include "common/cuda_graph_overrides.h"
#include "qwen35_seq_engine.h"

#include "qwen35_backend.h"
#include "qwen35_roctx.h"
#include "graph_builders.h"
#include "attn_masks.h"
#include "prefill_helpers.h"
#include "common/blocking_row_pool.h"
#include "common/concurrency/chain_spec_shapes.h"
#include "common/kvflash_chain_plan.h"
#include "common/dflash2_head.h"
#include "common/kvflash_exact_fast.h"
#include "common/kvflash_prefill_reselect.h"
#include "common/sampled_verify.h"
#include "common/sampler.h"
#include "internal.h"
#include "ggml-cuda.h"
#include "server/kv/session_tenant.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>

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
      prefill_quantum_(std::max(1, prefill_quantum)), pool_(&pool),
      b_(backend), slots_(pool, max_ctx),
      offload_(std::make_unique<PagedKvOffload>(
          slots_, pool, backend.target_backend_,
          qwen_paged_kv_planes(backend.cache_, pool.block_size()))),
      scratch_row_(scratch_row),
      fixed_chain_(fixed_chain) {
    init_slots_();
}

Qwen35SeqEngine::Qwen35SeqEngine(
        Qwen35Backend & backend, KvFlashSlotKv & kv, KvFlashStepDriver & driver,
        int max_ctx, int64_t scratch_row, FixedChainConfig fixed_chain, int max_prefills,
        int mixed_prefill_tokens, int long_mixed_prefill_tokens,
        int long_prefill_threshold, int idle_prefill_tokens)
    : max_prefills_(std::max(1, max_prefills)),
      mixed_prefill_tokens_(std::max(1, mixed_prefill_tokens)),
      long_mixed_prefill_tokens_(std::max(1, long_mixed_prefill_tokens)),
      long_prefill_threshold_(std::max(1, long_prefill_threshold)),
      idle_prefill_tokens_(std::max(1, idle_prefill_tokens)),
      prefill_quantum_(kKvFlashCutU),
      b_(backend), slots_(kv, max_ctx), kvf_kv_(&kv), kvf_(&driver),
      scratch_row_(scratch_row),
      fixed_chain_(fixed_chain) {
    init_slots_();
}

namespace {
// The head files' disk view never touches a checkpoint: a head goes in and out as host bytes (OwnedSnapshot).
class NoCheckpoints final : public CheckpointStore {
public:
    int granularity() const override { return 1; }
    int restore_point_spacing() const override { return 0; }
    bool prefill_cuts_at_restore_points() const override { return false; }
    size_t estimate_bytes(int) const override { return 0; }
    bool used(Checkpoint) const override { return false; }
    CheckpointKind kind(Checkpoint) const override { return CheckpointKind::dense; }
    int position(Checkpoint) const override { return 0; }
    void free(Checkpoint) override {}
    Checkpoint staging() const override { return {}; }
    ModelBackend::SnapshotRef ref(Checkpoint) const override { return {}; }
    OwnedSnapshot detach(Checkpoint) override { return {}; }
    bool adopt(Checkpoint, OwnedSnapshot &) override { return false; }
};
constexpr size_t kHeadsHost = 16, kHeadsPinned = 16;           // heads kept in host memory, head files pinned on disk
constexpr uint64_t kHeadRequest = 0x4b56480000000000ULL;      // ledger ids of an attached head: no HTTP request reaches them
}  // namespace

void Qwen35SeqEngine::init_slots_() {
    const int n_slots = slots_.slot_count();
    slot_draft_kv_.resize(static_cast<size_t>(n_slots));
    seq_lens_.assign(static_cast<size_t>(n_slots), 0);
    reserve_growth_.assign(static_cast<size_t>(n_slots), 0);
    recall_.assign(static_cast<size_t>(n_slots), SlotRecall{});
    uq_store_.assign(static_cast<size_t>(n_slots), KvFlashUqStore{});
    parked_.assign(static_cast<size_t>(n_slots), ParkedSession{});
    ckpt_.assign(static_cast<size_t>(n_slots), SlotCkpt{});
    trunk_.assign(static_cast<size_t>(n_slots), SlotCkpt{});
    extra_.assign(static_cast<size_t>(n_slots), SlotCkpt{});
    stash_pos_.assign(static_cast<size_t>(n_slots), -1);
    book_.init(n_slots);
    bundle_failed_ = std::vector<std::atomic<bool>>(static_cast<size_t>(n_slots));
    recall_on_ = kvf_ && b_.kvflash_slot_recall_profile();
    uq_queries_ = recall_on_ && b_.kvflash_prefill_reselect_ == 4 ? b_.kvflash_uq_queries_ : 0;
    if (kvf_) {
        sticky_on_ = kvflash_slot_sticky_env();
        const char * rung = std::getenv("LUCE_KVFLASH_END_RUNG_MIN");
        end_rung_min_ = rung && *rung ? std::atoi(rung) : -1;
        const char * hr = std::getenv("LUCE_KVFLASH_HEADER_RUNG");
        header_rung_ = hr && hr[0] == '1' && !hr[1];
        if (header_rung_) std::fprintf(stderr, "[kvflash] slot replay rung at the last cut before the header on recall turns (LUCE_KVFLASH_HEADER_RUNG)\n");
        const KvFlashParkCfg pc = kvflash_park_cfg(sticky_on_, std::getenv("LUCE_KVFLASH_PARK_SHORT"), std::getenv("LUCE_KVFLASH_PARK_TAIL"), std::getenv("LUCE_KVFLASH_PARK_DECODE"));
        park_short_on_ = pc.any() && slot_state_stash_alloc(b_.cache_, b_.target_backend_, stash_);   // the stash exists only when a mode uses it
        park_decode_on_ = park_short_on_ && pc.decode_on;
        park_tail_ = pc.tail;
        std::fprintf(stderr, "[kvflash] slot sessions: %s (LUCE_KVFLASH_SLOT_STICKY)\n", sticky_on_ ? "sticky" : "cold");
        if (sticky_on_) std::fprintf(stderr, "[kvflash] slot park short: %s, tail %d, decode stash %s (LUCE_KVFLASH_PARK_SHORT/_TAIL/_DECODE)\n", park_short_on_ ? "on" : "off", park_tail_, park_decode_on_ ? "on" : "off");
        if (end_rung_min_ >= 0)
            std::fprintf(stderr, "[kvflash] slot header rung on recall turns with end - R >= %d (LUCE_KVFLASH_END_RUNG_MIN)\n", end_rung_min_);
        init_bundles_();
        init_heads_();
        rebind_restored_heads_();   // the start-up adopts ran before the binder and head_k_max_ existed: bind their heads now
    }
    if (kvf_)
        std::fprintf(stderr, "[kvflash] slot recall: %s (mode=%d queries=%d k=%d unit=%d uq=%d preamble=%d)\n",
                     recall_on_ ? "on" : "off", b_.kvflash_prefill_reselect_,
                     b_.kvflash_select_queries_, b_.kvflash_recall_blocks_, b_.kvflash_select_unit_, uq_queries_, b_.kvflash_preamble_rows_);

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

    fixed_chain_ready_ = chain_config_ready(
        fixed_chain_.enabled, fixed_chain_.width, b_.dw_.block_size, fixed_chain_.scratch_stride,
        b_.cache_.target_feat && b_.cache_.target_feat_cap > 0);
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
            slot_draft_kv_batched_.resize(static_cast<size_t>(n_slots));
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
    bundle_writer_.reset();   // the pending saves land before the store is detached
    if (disk_ && tenant_id_) disk_->detach_tenant(tenant_id_);
    slot_state_stash_free(stash_);
    release_draft_graphs();
    for (DraftFeatureMirror & mirror : slot_feature_mirrors_) {
        draft_feature_mirror_free(mirror);
    }
    if (feature_view_ctx_) {
        ggml_free(feature_view_ctx_);
        feature_view_ctx_ = nullptr;
    }
}

void Qwen35SeqEngine::release_draft_graphs() {
    draft_kv_batch_free(batch_draft_graph_);
    for (auto & entry : batch_draft_graphs_batched_) {
        draft_kv_batch_free(entry.second);
    }
    for (auto * dummies : {&dummy_draft_kv_, &dummy_draft_kv_batched_}) {
        for (std::unique_ptr<DraftKvState> & state : *dummies) {
            if (state) draft_kv_free(*state);
        }
        dummies->clear();
    }
    for (auto * states : {&slot_draft_kv_, &slot_draft_kv_batched_}) {
        for (std::unique_ptr<DraftKvState> & state : *states) {
            if (state) {
                draft_kv_free(*state);
                state.reset();
            }
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
    for (auto * states : {&slot_draft_kv_, &slot_draft_kv_batched_}) {
        if (slot >= 0 && slot < static_cast<int>(states->size()) &&
            (*states)[static_cast<size_t>(slot)]) {
            draft_kv_reset(*(*states)[static_cast<size_t>(slot)]);
        }
    }
}

DraftFeatureMirror * Qwen35SeqEngine::slot_feature_mirror(int slot) {
    if (!fixed_chain_ready_ || slot < 0 ||
        slot >= static_cast<int>(slot_feature_mirrors_.size())) {
        return nullptr;
    }
    return &slot_feature_mirrors_[static_cast<size_t>(slot)];
}

DraftKvState * Qwen35SeqEngine::ensure_slot_draft_kv(int slot, bool batched) {
    DraftFeatureMirror * mirror = slot_feature_mirror(slot);
    std::vector<std::unique_ptr<DraftKvState>> & states =
        batched ? slot_draft_kv_batched_ : slot_draft_kv_;
    const DraftWeights & dw = batched ? batched_dw_ : b_.dw_;
    if (!mirror || slot < 0 ||
        slot >= static_cast<int>(states.size())) {
        return nullptr;
    }
    std::unique_ptr<DraftKvState> & state =
        states[static_cast<size_t>(slot)];
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
    // batched_draft_width_) with the live draft tensors; one lane drafts the
    // configured block and verifies its leading prefix.
    const int max_width = chain_draft_width(selected);
    const int lane_count = static_cast<int>(
        std::count(selected.begin(), selected.end(), uint8_t{1}));
    round.width = choose_chain_width(
        chain_decode_bucket_width(lane_count), max_width);
    const bool batched = max_width != fixed_chain_.width;
    const int draft_width = batched ? round.width : max_width;
    if (batched) {
        batched_dw_ = b_.dw_;
        batched_dw_.block_size = batched_draft_width_;
    }
    const DraftWeights & dw = batched ? batched_dw_ : b_.dw_;
    std::vector<std::unique_ptr<DraftKvState>> & dummies =
        batched ? dummy_draft_kv_batched_ : dummy_draft_kv_;
    DraftKvBatchGraph & draft_graph = batched
        ? batch_draft_graphs_batched_[draft_width]
        : batch_draft_graph_;

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
    std::vector<int32_t> noise(
        static_cast<size_t>(draft_width), b_.w_.mask_token_id);
    std::vector<float> noise_embed(
        static_cast<size_t>(hidden) * draft_width);

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
        DraftKvState * state = ensure_slot_draft_kv(input.slot, batched);
        DraftFeatureMirror * mirror = slot_feature_mirror(input.slot);
        if (!state || !mirror) {
            if (state) draft_kv_reset(*state);
            reset_lanes();
            return std::nullopt;
        }
        lanes.push_back({i, input.slot, input.token, state, mirror});
        if (!draft_kv_begin_step(
                *state, dw, b_.draft_backend_, *mirror,
                slots_.slot(input.slot).cur_pos, draft_width, feat_from_of_(input.slot))) {
            reset_lanes();
            return std::nullopt;
        }
        noise[0] = input.token;
        std::fill(
            noise.begin() + 1, noise.end(), b_.w_.mask_token_id);
        if (!b_.w_.embedder.embed(
                noise.data(), draft_width, noise_embed.data())) {
            reset_lanes();
            return std::nullopt;
        }
        ggml_backend_tensor_set(
            state->inp_embed, noise_embed.data(), 0,
            sizeof(float) * noise_embed.size());
    }
    if (lanes.empty()) return round;

    const int bucket = chain_draft_bucket_width(
        static_cast<int>(lanes.size()));
    std::vector<DraftKvState *> batch_states;
    std::vector<int32_t> roots;
    batch_states.reserve(static_cast<size_t>(bucket));
    roots.reserve(static_cast<size_t>(bucket));
    for (const Lane & lane : lanes) {
        batch_states.push_back(lane.state);
        roots.push_back(lane.root);
    }

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
    noise[0] = lanes.front().root;
    std::fill(noise.begin() + 1, noise.end(), b_.w_.mask_token_id);
    if (!b_.w_.embedder.embed(
            noise.data(), draft_width, noise_embed.data())) {
        reset_lanes();
        return std::nullopt;
    }
    for (int i = 0; i < dummy_count; ++i) {
        DraftKvState * dummy =
            dummies[static_cast<size_t>(i)].get();
        if (!draft_kv_begin_step(
                *dummy, dw, b_.draft_backend_,
                *lanes.front().mirror, 1, draft_width)) {
            reset_lanes();
            return std::nullopt;
        }
        ggml_backend_tensor_set(
            dummy->inp_embed, noise_embed.data(), 0,
            sizeof(float) * noise_embed.size());
        batch_states.push_back(dummy);
        roots.push_back(lanes.front().root);
    }

    std::vector<std::vector<float>> hidden_blocks;
    std::vector<std::vector<int32_t>> proposals;
    if (!draft_kv_batch_compute(
            draft_graph, dw, b_.draft_backend_,
            batch_states, hidden_blocks, draft_width) ||
        hidden_blocks.size() != batch_states.size()) {
        reset_lanes();
        return std::nullopt;
    }
    std::vector<const float *> hidden_by_lane;
    hidden_by_lane.reserve(hidden_blocks.size());
    for (const std::vector<float> & block : hidden_blocks) {
        hidden_by_lane.push_back(block.data());
    }
    if (!dflash2_select_chains_batched(
            dw, b_.draft_backend_, b_.w_.output, hidden_by_lane,
            draft_width, roots, proposals) ||
        proposals.size() != batch_states.size()) {
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
    // No bound given: reserve for a generation up to max_ctx (conservative).
    return admit_bounded(request_id, prompt, sampler, std::max(0, slots_.max_context() - (int)prompt.size()), AdmitMeta{});
}

SeqEngine::AdmitResult Qwen35SeqEngine::admit_bounded(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler,
        int max_new_tokens,
        const AdmitMeta & meta) {
    AdmitResult result = admit_bounded_(request_id, prompt, sampler, max_new_tokens, meta);
    if (result.status == AdmitResult::Status::admitted) set_gen_cap_(result.slot, max_new_tokens);
    return result;
}

SeqEngine::AdmitResult Qwen35SeqEngine::admit_bounded_(
        uint64_t request_id,
        const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler,
        int max_new_tokens,
        const AdmitMeta & meta) {
    const int64_t total = (int64_t)prompt.size() + max_new_tokens;   // host backing for the whole request (R5)
    if (sticky_on_) return admit_sticky_(request_id, prompt, sampler, total, meta);
    if (kvf_kv_) kvf_kv_->set_next_total_tokens(total);
    AdmitResult result = slots_.admit(request_id, prompt, sampler);
    if (result.status == AdmitResult::Status::admitted) {
        clear_slot_images(result.slot);
        reset_recurrent_slot(b_.cache_, result.slot);
        reset_slot_draft_kv(result.slot);
        set_feat_from_(result.slot, 0);
        if (kvf_) plan_slot_turn_(result.slot, prompt, 0);
    }
    return result;
}

// Sticky admission: a parked session the prompt extends resumes in its slot at L (kvflash_resume_point): the whole
// history when the prompt still matches it, else a rung (the prompt-end checkpoint, the trunk or the third), to which
// the session is first rewound (slab, pager rows, allocator length).
// The session's identity: the request's, else anonymous (ownerless: resumable by content, never a stated identity's session).
// e_role: where the preamble ends.
SeqEngine::AdmitResult Qwen35SeqEngine::admit_sticky_(
        uint64_t request_id, const std::vector<int32_t> & prompt, const SamplerCfg & sampler,
        int64_t total, const AdmitMeta & meta) {
    AdmitMeta who = meta;
    if (who.sid.empty()) { who.sid = "anon:" + std::to_string(request_id); who.client = "anon"; }
    AdmitResult refused;
    const KvFlashEarlyRefusal early = kvflash_early_refusal((int64_t)prompt.size(), slots_.max_context(), [&] { return kvf_kv_->exceeds_host_budget(0, total); });
    if (early != KvFlashEarlyRefusal::none) {   // before any eviction or resume: no session is touched
        refused.status = AdmitResult::Status::capacity_exceeded;
        refused.error = early == KvFlashEarlyRefusal::prompt_over_max_ctx ? "prompt exceeds max_ctx"
                                                                          : "request needs more host memory than the whole KVFlash host budget (LUCE_KVFLASH_HOST_MIB)";
        std::fprintf(stderr, "[kvflash] admit %s/%s refused: %s (no session touched)\n", who.client.c_str(), who.sid.c_str(), refused.error.c_str());
        return refused;
    }
    const int e_role = kvflash_preamble_end(prompt, b_.w_.im_start_id, b_.w_.system_id);
    {   // a second request under a running identity admits as its own anonymous session, not as a busy retry that holds the line
        const KvFlashSessionId id = kvflash_admit_identity(book_, {who.sid, who.client}, active_slots_(), request_id);
        who.sid = id.sid; who.client = id.client;
    }
    std::vector<KvFlashResumeWait> waits((size_t)slots_.slot_count());   // a parked session serves this request (its prompt extends it, or a busy resume left it rewound): no disk session is restored first
    for (int i = 0; i < slots_.slot_count(); ++i) waits[(size_t)i] = parked_[(size_t)i].wait;
    const bool resident = kvflash_resident_serves(parked_list_(), waits, who.client, who.sid, prompt, e_role);
    bool restored = false;   // the restore outcome, for the admit plan below
    switch (kvflash_admit_gate(book_, {who.sid, who.client}, active_slots_(), resident, [&] {   // a session evicted to disk comes back into a slot, then resumes below
                if (bundle_dir_.empty()) return KvFlashRestore::absent;
                const KvFlashRestore rs = restore_from_disk_(who, prompt, e_role);
                restored = rs == KvFlashRestore::restored;
                return rs;
            })) {
        case KvFlashAdmitVerdict::invalid_identity:   // the HTTP boundary drops such an id first; an engine caller is held to the same bound
            refused.status = AdmitResult::Status::failed;
            refused.error = "session id longer than " + std::to_string(kKvFlashSidMax) + " bytes";
            return refused;
        case KvFlashAdmitVerdict::identity_busy:      // not reached: a running identity was made anonymous above
            refused.status = AdmitResult::Status::busy;
            refused.error = "session " + who.client + "/" + who.sid + " is running another request";
            return refused;
        case KvFlashAdmitVerdict::restore_busy:
            refused.status = AdmitResult::Status::busy;
            refused.error = "the session on disk cannot be restored yet (no slot to leave, or the cache budget has it claimed)";
            return refused;
        case KvFlashAdmitVerdict::proceed:
            break;
    }
    // The admit table: resume, or cold.
    const std::vector<KvFlashOwnedSession> owners = parked_list_();
    KvFlashSessionPick pick = kvflash_pick_parked_owner(owners, who.client, who.sid, prompt, e_role);
    for (const auto & o : owners)   // a session a busy resume left rewound is picked again by the request that rewound it
        if (parked_[(size_t)o.parked.slot].wait.L > 0) {
            const KvFlashSessionPick w = kvflash_pick_with_wait(pick, o.parked.slot, (int)o.parked.history->size(), parked_[(size_t)o.parked.slot].wait, prompt, who.client, who.sid);
            if (w.slot == o.parked.slot) pick = w;
        }
    KvFlashAdmitIn plan_in;
    plan_in.e_role = e_role; plan_in.can_archive = book_.persistent(); plan_in.from_disk = restored;
    // The pinned head: the server's cut rounded to the chunk grid (the role end when sharing is off), attached when
    // known, captured at its cut otherwise. k depends on the prompt alone; what is known only picks attach or capture.
    const int hct = (int)kvf_kv_->block_size();
    if (head_k_max_ > 0)
        plan_in.pd = kvflash_head_decide(prompt, kvflash_head_boundary(who.head_cut, e_role, hct, kKvFlashCutU), hct,
                                         kvflash_head_unit(hct, kKvFlashCutU), head_k_max_,
                                         [&](uint64_t, int k) { return find_head_(prompt, k) != nullptr; });
    if (pick.exact()) {
        const ParkedSession & ps = parked_[(size_t)pick.slot];
        plan_in.pick_slot = pick.slot; plan_in.lcp = pick.lcp;
        plan_in.L = kvflash_resume_point(pick.lcp, (int)ps.history.size(), ps.ckpt.pos, ps.trunk.pos, (int)prompt.size(), ps.extra.pos);
    }
    plan_in.same_sid_slot = kvflash_same_sid_slot(owners, who.client, who.sid, pick.slot);
    if (plan_in.same_sid_slot >= 0) plan_in.same_sid_key = book_[plan_in.same_sid_slot].key;
    KvFlashAdmitPlan plan = kvflash_admit_plan(plan_in);
    std::fprintf(stderr, "[kvflash] admit %s/%s row %s: slot=%d pick_lcp=%d L=%d e_role=%d prompt=%zu%s%s\n", who.client.c_str(), who.sid.c_str(),
                 kvflash_admit_row(plan.action), plan.slot >= 0 ? plan.slot : pick.slot, pick.lcp, plan.L, e_role, prompt.size(),
                 plan.archive_key.empty() ? "" : " archive=", plan.archive_key.c_str());
    if (plan.action == KvFlashAdmitAction::resume || plan.action == KvFlashAdmitAction::resume_disk) {
        ParkedSession & ps = parked_[(size_t)pick.slot];
        const int history = (int)ps.history.size();
        const int L = plan.L;
        bool ready = L > 0;
        if (ready && L < history) {
            const SlotCkpt & at = ps.ckpt.pos == L ? ps.ckpt : ps.trunk.pos == L ? ps.trunk : ps.extra;
            const char * which = ps.ckpt.pos == L ? "checkpoint" : ps.trunk.pos == L ? "trunk" : "third rung";
            ready = restore_slot_state(b_.cache_, pick.slot, at.slab) && kvf_kv_->truncate_parked(pick.slot, (uint32_t)L);
            if (ready) {
                ps.history.resize((size_t)L);   // the rung stays: it is the state at L
                // rungs past L go now; a rewind below the checkpoint is the branch event: the rung at L is pinned in
                // the third slot (rule v5). A Busy retry resumes at L = the history: no new event
                const KvFlashLadderMove mv = kvflash_ladder_resume(ladder_of_(ps.ckpt, ps.trunk, ps.extra), L, history);
                apply_ladder_(mv, ps.ckpt, ps.trunk, ps.extra, SlotCkpt{});
                if (mv.to.pinned) std::fprintf(stderr, "[kvflash] slot %d branch point %d pinned (third rung)\n", pick.slot, L);
            }
            std::fprintf(stderr, "[kvflash] slot %d session rewound to its %s %d (lcp=%d history=%d)%s\n",
                         pick.slot, which, L, pick.lcp, history, ready ? "" : ": FAILED");
        }
        if (ready) {
            AdmitResult result;
            for (;;) {
                kvf_kv_->set_next_total_tokens(total);
                kvf_kv_->set_next_resume(pick.slot, (uint32_t)L);
                result = slots_.admit_at(pick.slot, request_id, prompt, sampler, L);
                if (result.status == AdmitResult::Status::admitted) break;
                const bool busy = result.status == AdmitResult::Status::busy;
                bool other_parked = false;
                for (const auto & o : parked_list_()) other_parked |= o.parked.slot != pick.slot;
                const KvFlashResumeRefused act = kvflash_resume_refused(busy, other_parked, result.status == AdmitResult::Status::capacity_exceeded);
                if (act == KvFlashResumeRefused::refuse) {   // the backstop of the early refusal: the session stays parked, the HTTP boundary answers the refusal
                    std::fprintf(stderr, "[kvflash] slot %d resume of sid=%s refused: %s (the session stays parked)\n", pick.slot, who.sid.c_str(), result.error.c_str());
                    return result;
                }
                if (act == KvFlashResumeRefused::evict_other_retry && evict_oldest_("host budget needed to resume a session", pick.slot) >= 0) continue;
                if (act != KvFlashResumeRefused::drop_cold) {
                    ps.wait = kvflash_resume_wait_make(L, pick.lcp, prompt, who.client, who.sid);   // the retry resumes here, rewound or not
                    std::fprintf(stderr, "[kvflash] slot %d resume of sid=%s waits: %s (the session stays parked)\n", pick.slot, who.sid.c_str(), result.error.c_str());
                    return result;
                }
                break;
            }
            if (result.status == AdmitResult::Status::admitted) {
                // Rule B: the rung at L (or the highest below it) is the fallback until this turn's checkpoint, the one below
                // it the trunk, the third rung rides along; a rung above L went with its rows (the rewind above, or none was needed)
                const KvFlashRungs rr{ps.ckpt.pos, ps.trunk.pos};
                ckpt_[(size_t)result.slot] = std::move(ps.ckpt);
                trunk_[(size_t)result.slot] = std::move(ps.trunk);
                extra_[(size_t)result.slot] = std::move(ps.extra);
                book_[result.slot].resume_L = L;         // the park rewrites the chunks from L on
                book_.promote_owner(result.slot, who.client, who.sid);   // a session with no sid of its own learns its client's
                ps = ParkedSession{};
                clear_slot_images(result.slot);
                reset_slot_draft_kv(result.slot);
                set_feat_from_(result.slot, L);   // KV and state are the session's; the ring still holds the last occupant's features below L
                std::fprintf(stderr, "[kvflash] slot %d resume L=%d history=%d new=%d rungs=%d/%d/%d\n",
                             result.slot, L, history, (int)prompt.size() - L, rr.ckpt, rr.trunk, extra_[(size_t)result.slot].pos);
                begin_resume_(result.slot);
                plan_slot_turn_(result.slot, prompt, L);
                return result;
            }
            evict_session_(pick.slot, result.error.c_str());   // the pool refused (host budget): cold below
        } else {
            evict_session_(pick.slot, "checkpoint rewind failed");
        }
        plan_in.pick_slot = plan_in.same_sid_slot = -1;   // it failed and the session is gone: plan the cold admit without it
        plan = kvflash_admit_plan(plan_in);
    } else if (pick.exact() && !plan.archives() && plan.action != KvFlashAdmitAction::summary_cold) {
        const ParkedSession & ps = parked_[(size_t)pick.slot];
        std::fprintf(stderr, "[kvflash] slot %d session not resumable: lcp=%d history=%d ckpt=%d trunk=%d third=%d prompt=%zu%s\n",
                     pick.slot, pick.lcp, (int)ps.history.size(), ps.ckpt.pos, ps.trunk.pos, ps.extra.pos, prompt.size(),
                     plan.slot >= 0 ? " (replaced by this request: the first to leave)" : "");
        // Not dropped here: under one sid for several conversations (a gateway's constant header) each would drop the other.
        if (plan.slot >= 0) parked_[(size_t)plan.slot].tick = 0;   // older than every park (ticks start at 1)
    }
    // The sid's parked instance: archived (it leaves for disk, the new instance names it) or, for a summary-generation
    // request, left alone and not linked to (the request is no successor of it).
    std::optional<std::string> archive_of;
    int keep_slot = plan.action == KvFlashAdmitAction::summary_cold ? plan.slot : -1;
    if (plan.archives()) {
        // the link names the instance only once it is on disk with a committed head; an eviction that fails leaves it parked in
        // its slot (kept from the slot choice below) and the new instance starts unlinked
        archive_of = evict_session_(plan.slot, "archived by the next instance of its sid") ? book_.archive_target(plan.archive_key) : std::string();
        if (archive_of->empty()) keep_slot = plan.slot;
    } else if (plan.action == KvFlashAdmitAction::summary_cold) {
        archive_of = std::string();
    }
    // Cold: a slot with no session, else the oldest session's slot.
    int slot = -1;
    for (int i = 0; i < slots_.slot_count() && slot < 0; ++i)
        if (!slots_.is_active(i) && !parked_[(size_t)i].valid) slot = i;
    if (slot < 0) slot = kvflash_evict_for_cold([&](int keep) { return evict_oldest_("slot needed for a new session", keep); }, keep_slot, any_slot_live_());
    if (slot >= 0 && !book_[slot].sid.empty()) evict_session_(slot, "slot reused");   // a session that left it without parking
    AdmitResult result;
    if (slot < 0) {   // every slot is live (a plain admit would pair a manager slot with another pager slot)
        result.status = AdmitResult::Status::busy;
        result.error = "all sequence slots are live";
        return result;
    }
    // Rows 4 and 6: a new instance starts from the pinned head (L = k chunks) instead of prefilling it. Any failure leaves the
    // slot free and the request cold below.
    int attached_at = 0;
    if ((plan.action == KvFlashAdmitAction::attach || plan.action == KvFlashAdmitAction::archive_attach) && plan.L > 0) {
        const KvFlashHead * head = find_head_(prompt, plan.L / hct);
        SlotStateSnapshot hs;
        if (head && kvflash_slab_unpack(head->slab.data(), head->slab.size(), hs.ssm, hs.conv) &&
            kvflash_head_attach(*kvf_kv_, head_binder_, slot, kHeadRequest + (uint64_t)slot, *head)) {
            if (restore_slot_state(b_.cache_, slot, hs)) {
                for (;;) {
                    kvf_kv_->set_next_total_tokens(total);
                    kvf_kv_->set_next_resume(slot, (uint32_t)plan.L);
                    result = slots_.admit_at(slot, request_id, prompt, sampler, plan.L);
                    const int old = result.status == AdmitResult::Status::busy ? evict_oldest_("host budget needed to attach a head", keep_slot) : -1;
                    if (old < 0) break;
                }
                if (result.status == AdmitResult::Status::admitted) attached_at = plan.L;
            }
            if (!attached_at) { kvf_kv_->drop_parked(slot); head_binder_.unbind(slot); }
        }
        std::fprintf(stderr, "[kvflash] slot %d pinned head attach %s (L=%d)\n", slot, attached_at ? "ok" : "FAILED: cold", plan.L);
    }
    if (!attached_at) for (;;) {
        kvf_kv_->set_next_total_tokens(total);
        kvf_kv_->set_next_slot(slot);
        result = slots_.admit_at(slot, request_id, prompt, sampler, 0);
        // Parked reservations are a cache, never a reason to refuse: the oldest go until the budget fits.
        const int old = result.status == AdmitResult::Status::busy
            ? kvflash_evict_for_cold([&](int keep) { return evict_oldest_("host budget needed for a new session (cold)", keep); }, keep_slot, any_slot_live_()) : -1;
        if (old < 0) break;
    }
    if (result.status == AdmitResult::Status::admitted) {
        ckpt_[(size_t)result.slot] = SlotCkpt{};
        trunk_[(size_t)result.slot] = SlotCkpt{};
        extra_[(size_t)result.slot] = SlotCkpt{};
        book_.begin_session(result.slot, who.client, who.sid, (int64_t)std::time(nullptr), archive_of);
        bundle_failed_[(size_t)result.slot] = false;
        book_[result.slot].resume_L = 0;
        clear_slot_images(result.slot);
        if (attached_at) {   // KV and state are the head's; the feature ring holds the last occupant's below L, as on a resume
            reset_slot_draft_kv(result.slot);
            set_feat_from_(result.slot, attached_at);
            begin_resume_(result.slot);
            plan_slot_turn_(result.slot, prompt, attached_at);
            return result;
        }
        reset_recurrent_slot(b_.cache_, result.slot);
        reset_slot_draft_kv(result.slot);
        set_feat_from_(result.slot, 0);
        plan_slot_turn_(result.slot, prompt, 0, plan.capture_at > 0 ? plan.capture_at : -1);
    }
    return result;
}

// The session's park cannot be written (the cache budget has no room, a write failed): it leaves host memory and the slot, but
// whatever the writer last committed stays on disk with its store entry and owner: the next turn restores that older head or,
// with none, prefills cold.
void Qwen35SeqEngine::release_host_(int slot, const char * why) {
    if (slot < 0 || slot >= slots_.slot_count()) return;
    if (!bundle_dir_.empty()) bundle_writer_->wait(slot);
    const std::string key = book_[slot].key;
    if (book_.release_to_last_commit(slot)) {
        std::fprintf(stderr, "[kvflash] slot %d session released from host memory, its last committed bundle %s stays on disk: its newer park cannot be written (%s)\n", slot, key.c_str(), why);
        parked_[(size_t)slot] = ParkedSession{};
        uq_store_[(size_t)slot].clear();   // the rows go with the session: the next occupant must not save them
        kvf_kv_->drop_parked(slot);
        bundle_failed_[(size_t)slot] = false;
        return;
    }
    std::fprintf(stderr, "[kvflash] slot %d session dropped from host memory: its park cannot be written and nothing was committed, the next turn prefills cold (%s)\n", slot, why);
    drop_session_(slot, why);
}

void Qwen35SeqEngine::drop_session_(int slot, const char * why) {
    if (slot < 0 || slot >= slots_.slot_count()) return;
    if (parked_[(size_t)slot].valid)
        std::fprintf(stderr, "[kvflash] slot %d session dropped: history=%zu (%s)\n",
                     slot, parked_[(size_t)slot].history.size(), why);
    parked_[(size_t)slot] = ParkedSession{};
    uq_store_[(size_t)slot].clear();
    kvf_kv_->drop_parked(slot);
    if (!bundle_dir_.empty()) {
        bundle_writer_->wait(slot);   // a queued park of this slot would recreate the directory after the removal
        const std::string key = book_[slot].key;
        if (!key.empty()) { kvflash_bundle_remove(book_.store().dir(key)); book_.store().erase(key); book_.store().write_index(); }
    }
    book_.reset(slot);   // whether or not sessions are on disk: the slot's owner goes with its session
    bundle_failed_[(size_t)slot] = false;
}

// The session leaves its slot and its bundle stays: the directory the last park committed is what a later request restores.
// Only a session whose latest park is on disk leaves: one whose park was skipped or failed is parked again from scratch first,
// and stays in its slot when that fails too (false). A session with no committed park and nothing in its slot (a resume that
// failed) is dropped.
bool Qwen35SeqEngine::evict_session_(int slot, const char * why) {
    if (slot < 0 || slot >= slots_.slot_count()) return false;
    if (bundle_dir_.empty() || book_[slot].key.empty()) { drop_session_(slot, why); return true; }
    bundle_writer_->wait(slot);
    KvFlashSlotBundle & sb = book_[slot];
    bool durable = book_.durable(slot, bundle_failed_[(size_t)slot].load());
    if (!durable && parked_[(size_t)slot].valid) {
        sb.rewrite = true;   // from scratch: the directory's head, if any, is older than this session
        save_bundle_(slot, parked_[(size_t)slot]);
        bundle_writer_->wait(slot);
        durable = book_.durable(slot, bundle_failed_[(size_t)slot].load());
        if (!durable) {
            std::fprintf(stderr, "[kvflash] slot %d session %s stays in its slot: its park is not on disk (%s)\n", slot, sb.key.c_str(), why);
            return false;
        }
    }
    const KvFlashSlotHead * on_disk = book_.committed_head(slot, bundle_failed_[(size_t)slot].load());
    if (!on_disk) { drop_session_(slot, why); return true; }
    std::fprintf(stderr, "[kvflash] slot %d session evicted to disk: sid=%s key=%s history=%zu (%s)\n",
                 slot, sb.sid.c_str(), sb.key.c_str(), on_disk->history.size(), why);
    const std::string evicted_key = sb.key;
    if (!book_.evict(slot, bundle_failed_[(size_t)slot].load())) std::fprintf(stderr, "[kvflash] slot %d session %s: no committed head or meta and index not written\n", slot, evicted_key.c_str());
    parked_[(size_t)slot] = ParkedSession{};
    uq_store_[(size_t)slot].clear();
    kvf_kv_->drop_parked(slot);
    bundle_failed_[(size_t)slot] = false;
    return true;
}

int Qwen35SeqEngine::evict_oldest_(const char * why, int keep) {
    std::vector<KvFlashParkedSession> list;
    for (const auto & o : parked_list_()) if (o.parked.slot != keep) list.push_back(o.parked);
    const int slot = kvflash_oldest_session(list);
    if (slot >= 0 && !evict_session_(slot, why)) release_host_(slot, why);   // the park was refused: the parked set is a cache and never blocks service
    return slot;
}

bool Qwen35SeqEngine::any_slot_live_() const {
    for (int i = 0; i < slots_.slot_count(); ++i) if (slots_.is_active(i)) return true;
    return false;
}

std::vector<bool> Qwen35SeqEngine::active_slots_() const {
    std::vector<bool> v((size_t)slots_.slot_count(), false);
    for (int i = 0; i < slots_.slot_count(); ++i) v[(size_t)i] = slots_.is_active(i);
    return v;
}

std::vector<KvFlashOwnedSession> Qwen35SeqEngine::parked_list_() const {
    std::vector<KvFlashOwnedSession> v;
    for (int slot = 0; slot < slots_.slot_count(); ++slot)
        if (parked_[(size_t)slot].valid)
            v.push_back({{slot, &parked_[(size_t)slot].history, parked_[(size_t)slot].tick}, book_[slot].client, book_[slot].sid});
    return v;
}

void Qwen35SeqEngine::apply_ladder_(const KvFlashLadderMove & m, SlotCkpt & ck, SlotCkpt & tr, SlotCkpt & ex, SlotCkpt live) {
    SlotCkpt old[4] = {std::move(ck), std::move(tr), std::move(ex), std::move(live)};
    SlotCkpt * dst[3] = {&ck, &tr, &ex};
    const int pos[3] = {m.to.ckpt, m.to.trunk, m.to.extra};
    for (int i = 0; i < 3; ++i) *dst[i] = SlotCkpt{};
    // the plan frees the dead sources first, so a copy never sits beside them (the ladder's charge is three slabs)
    for (const KvFlashSlabOp & op : kvflash_ladder_ops(m)) {
        if (op.kind == KvFlashSlabOp::Free) { old[op.from] = SlotCkpt{}; continue; }
        SlotCkpt & d = *dst[op.to];
        if (op.kind == KvFlashSlabOp::Move) d = std::move(old[op.from]); else d.slab = old[op.from].slab;
        d.pos = pos[op.to];
        d.pinned = op.to == 2 && m.to.pinned;
    }
}

// C1.5 F4, Rule A (kvflash_ladder_ckpt): a new checkpoint replaces the one at its position, or pushes the ones below it down
// (checkpoint to trunk, trunk to the third slot, which a pinned branch point keeps); the rung that falls off goes.
void Qwen35SeqEngine::place_ckpt_(int slot, SlotCkpt nu) {
    SlotCkpt & ck = ckpt_[(size_t)slot];
    SlotCkpt & tr = trunk_[(size_t)slot];
    SlotCkpt & ex = extra_[(size_t)slot];
    const KvFlashLadderMove m = kvflash_ladder_ckpt(ladder_of_(ck, tr, ex), nu.pos);
    apply_ladder_(m, ck, tr, ex, std::move(nu));
}

// The header rung: the slot's recurrent state now, at pos. The rungs Rule A drops are freed before the
// snapshot, so the slot holds three rungs at most counting the one in flight (the ladder's charge).
void Qwen35SeqEngine::take_ckpt_(int slot, int pos) {
    SlotCkpt & ck = ckpt_[(size_t)slot];
    SlotCkpt & tr = trunk_[(size_t)slot];
    SlotCkpt & ex = extra_[(size_t)slot];
    const KvFlashLadderMove m = kvflash_ladder_ckpt(ladder_of_(ck, tr, ex), pos);
    SlotCkpt * rs[3] = {&ck, &tr, &ex};
    for (int i = 0; i < 3; ++i) {
        const bool kept = (m.to.trunk > 0 && m.trunk_from == i) || (m.to.extra > 0 && m.extra_from == i);
        if (!kept) *rs[i] = SlotCkpt{};
    }
    SlotCkpt nu;
    if (!snapshot_slot_state(b_.cache_, slot, nu.slab)) {
        std::fprintf(stderr, "[kvflash] slot %d header rung at %d: slab snapshot failed\n", slot, pos);
        // what survived, highest first, is the checkpoint, the trunk and the third rung
        std::vector<SlotCkpt> live;
        for (SlotCkpt * r : rs) if (r->pos > 0) live.push_back(std::move(*r));
        std::sort(live.begin(), live.end(), [](const SlotCkpt & a, const SlotCkpt & b) { return a.pos > b.pos; });
        for (int i = 0; i < 3; ++i) { *rs[i] = i < (int)live.size() ? std::move(live[(size_t)i]) : SlotCkpt{}; }
        ex.pinned = ex.pos > 0 && ex.pinned;
        return;
    }
    nu.pos = pos;
    apply_ladder_(m, ck, tr, ex, std::move(nu));
}

// The bundle identity: the model file (path, size, mtime), the KV types and rotation, the pager
// geometry, the context and the paged-attention route. A bundle from any other server is skipped, never adopted.
std::string Qwen35SeqEngine::identity_key_() const {
    struct stat st{};
    const std::string & model = b_.cfg_.target_path;
    const bool have = ::stat(model.c_str(), &st) == 0;
    const KvFlashSlotPager & p0 = kvf_kv_->pager(0);
    KvFlashIdentityParts id;
    id.model = model;
    id.model_size = have ? (long long)st.st_size : -1LL;
    id.model_mtime = have ? (long long)st.st_mtime : -1LL;
    id.kv_k_type = (int)b_.cache_.kv_k_type;
    id.kv_v_type = (int)b_.cache_.kv_v_type;
    id.kv_k_rotated = b_.cache_.kv_k_rotated;
    id.chunk_tokens = p0.chunk_tokens();
    id.n_pool_blocks = p0.n_pool_blocks();
    id.max_context = slots_.max_context();
    // The route the autoregressive graphs dispatch (tree verify graphs always run V_DOT2).
    id.paged_wmma_route = ggml_backend_cuda_paged_attn_wmma_route(
        b_.cfg_.device.gpu, /*tree_mode=*/false, b_.w_.n_embd_head_k, b_.w_.n_head, b_.w_.n_head_kv,
        b_.cache_.kv_k_type, b_.cache_.kv_v_type, PAGED_BLOCK_SIZE);
    return kvflash_identity_key(id);
}

size_t Qwen35SeqEngine::slab_bytes_() const {
    size_t slab = 0;
    for (const ggml_tensor * t : b_.cache_.ssm_state)  if (t) slab += ggml_nbytes(t) / (size_t)slots_.slot_count();
    for (const ggml_tensor * t : b_.cache_.conv_state) if (t) slab += ggml_nbytes(t) / (size_t)slots_.slot_count();
    return slab;
}

void Qwen35SeqEngine::init_bundles_() {
    const char * dir = std::getenv("LUCE_KVFLASH_SESSIONS_DIR");
    if (!sticky_on_ || !dir || !*dir) return;
    disk_ = DiskStore::find(dir);
    if (!disk_) {
        std::fprintf(stderr, "[kvflash] sessions: no open cache directory %s: sessions stay in memory\n", dir);
        return;
    }
    bundle_dir_ = dir;
    bundle_writer_ = std::make_unique<KvFlashBundleWriter>([this](const KvFlashBundleWriter::Job & j, bool ok, double ms) {
        size_t bytes = j.commit.head.size() + (j.commit.new_log ? kKvFlashLogHeader : 0) + j.commit.copies.size() * j.commit.record_bytes;
        for (const auto & r : j.commit.records) bytes += kKvFlashRecordTag + r.second.size();
        for (const auto & f : j.commit.files) bytes += f.second.size();
        std::fprintf(stderr, "[kvflash] slot %d bundle gen %u %s: %.1f MiB %.0f ms (writer)\n", j.slot, j.commit.gen,
                     ok ? "written" : "NOT written", bytes / 1048576.0, ms);
        if (!ok) bundle_failed_[(size_t)j.slot] = true;   // the next park of the slot starts the directory over
    });
    kvf_kv_->set_park_in_flight([this](int slot) { return bundle_writer_ && bundle_writer_->in_flight(slot); });   // a resume keeps the staging's reservation
    const KvFlashSlotPager & p0 = kvf_kv_->pager(0);
    bundle_identity_ = kvflash_fnv1a(identity_key_());
    std::fprintf(stderr, "[kvflash] sessions on disk: %s/sessions (identity %016llx)\n", dir, (unsigned long long)bundle_identity_);
    const size_t slab = slab_bytes_();   // the read cap: a file past what this server can write is skipped before it is loaded
    const size_t max_ctx = (size_t)slots_.max_context(), ct = (size_t)p0.chunk_tokens();
    bundle_lim_ = KvFlashBundleLimits{max_ctx, (max_ctx + ct - 1) / ct, kvflash_slab_max_bytes(slab), (size_t)p0.n_pool_blocks()};
    // The session store is a tenant of the cache directory's DiskStore (one budget, one LRU, one TTL); every session
    // directory is listed, the newest ones fill the slots and the rest stay on disk.
    book_.enable_store(KvFlashSessionStore(dir), bundle_identity_);
    tenant_id_ = attach_session_store(disk_, book_.store());
    book_.resume_claim = [this](const std::string & key) { return book_.store().claim(key); };   // the budget cannot evict a session a restore is about to read
    std::error_code mk;
    std::filesystem::create_directories(book_.store().dir(""), mk);
    const size_t head_max = kvflash_bundle_head_max(bundle_lim_);
    KvFlashScanReport rep;
    book_.store().load(kvflash_session_scan(dir, bundle_identity_, kvflash_record_bytes(p0.chunk_bytes()), head_max, rep, bundle_lim_.slab_max));
    std::fprintf(stderr, "[kvflash] sessions: %zu on disk, %zu empty removed, %zu skipped (another server's)\n",
                 book_.store().entries().size(), rep.removed.size(), rep.skipped.size());
    for (const auto & k : rep.failed) std::fprintf(stderr, "[kvflash] sessions: metadata of %s not durable (the next start repeats it)\n", k.c_str());
    std::vector<std::string> adopt;   // the scan returns them newest first: the N newest, oldest adopted first so the newest has the latest tick
    for (const auto & e : book_.store().entries()) if (e.state == "disk" && (int)adopt.size() < slots_.slot_count()) adopt.push_back(e.key);
    std::reverse(adopt.begin(), adopt.end());
    int slot = 0;
    for (const std::string & key : adopt) {
        bool refused = false;
        const std::string why = adopt_bundle_(slot, key, &refused);
        if (why.empty()) { ++slot; continue; }
        std::fprintf(stderr, "[kvflash] bundle %s not adopted: %s%s\n", book_.store().dir(key).c_str(), why.c_str(), refused ? " (listed as foreign, left in place)" : "");
        if (refused) book_.store().mark_foreign(key);
    }
    book_.store().write_index();
}

// LUCE_KVFLASH_BUNDLE_DIGEST=1 (kvflash_bundle_digest_on): the R3 digest of a session's saved state. The one staging of both sides: the
// live pager's header, ledger and every materialized chunk (read_chunk) beside the end slab, the history and the rungs it is given,
// so a save (the slab being staged) and a restore (the slab read back from the slot) digest the same way. Flushed at once.
// t0: when the caller's own staging began, so a restore's ms includes its slab readback.
void Qwen35SeqEngine::log_bundle_digest_(const char * tag, int slot, const std::string & key, uint32_t gen, const std::vector<int32_t> & history, const SlotStateSnapshot & slab,
                                           const SlotCkpt & ck, const SlotCkpt & tr, const SlotCkpt & ex, const KvFlashUqStore * uq, std::chrono::steady_clock::time_point t0) {
    KvFlashSlotPager & p = kvf_kv_->pager(slot);
    p.synchronize_paging();
    const std::vector<uint8_t> pk = ck.pos > 0 ? kvflash_slab_pack(ck.slab.ssm, ck.slab.conv) : std::vector<uint8_t>(),
                               pt = tr.pos > 0 ? kvflash_slab_pack(tr.slab.ssm, tr.slab.conv) : std::vector<uint8_t>(),
                               pe = ex.pos > 0 ? kvflash_slab_pack(ex.slab.ssm, ex.slab.conv) : std::vector<uint8_t>();
    const KvFlashDigestRungs rungs = kvflash_digest_rungs(ck.pos, &pk, tr.pos, &pt, ex.pos, ex.pinned, &pe);   // raw, as the session holds them
    const std::vector<uint8_t> pool = p.pool_layout_blob();   // the live pager's, at save and again after a restore: a restore that lost it shows
    const KvFlashStateDigest d = kvflash_state_digest(history, kvflash_slab_pack(slab.ssm, slab.conv), p.blob_header(), p.ledger_blob(), rungs, p.chunk_bytes(),
                                                      [&](int ci, uint8_t * dst) { p.read_chunk(ci, dst); }, uq, &pool);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "%s\n", kvflash_state_digest_line(tag, key, gen, d, ms).c_str());
    std::fflush(stderr);
}

// The session directory sessions/<key>/ into the slot's pager as a parked session: the same restore
// at start and at a request that names a disk session. "" when adopted, else why it was refused.
std::string Qwen35SeqEngine::adopt_bundle_(int slot, const std::string & key, bool * refused, bool evict_for_room, bool * busy) {
    constexpr uint64_t kBundleRequest = 0x4b56460000000000ULL;   // ledger ids no HTTP request reaches
    const std::string sdir = book_.store().dir(key);
    const KvFlashSlotPager & p0 = kvf_kv_->pager(0);
    KvFlashBundleLoaded l;
    SlotStateSnapshot end;
    std::string why;
    auto load = [&] {
        l = KvFlashBundleLoaded{}; end = SlotStateSnapshot{};
        *refused = true;   // until the bundle is read: a load that fails is the bundle's, not the host's
        if (!kvflash_bundle_load(sdir, bundle_identity_, p0.chunk_bytes(), bundle_lim_, l, why)) return false;
        *refused = false;
        end.ssm = std::move(l.end_ssm); end.conv = std::move(l.end_conv);
        return true;
    };
    // The host reservation of the returning session, asked of the ledger from the head alone (a few KiB) before the bundle (GiB)
    // is read: a retry that cannot fit returns Busy without a read. Anything the probe cannot decide (no head, a head over this
    // server's limits, an occupied slot) falls through to the load, which refuses with its own reason.
    auto read_head = [&](KvFlashSlotHead & head) {
        std::vector<uint8_t> bytes;
        return kvflash_bundle_read(sdir + "/head", bytes, kvflash_bundle_head_max(bundle_lim_)) &&
               kvflash_head_unpack(bytes.data(), bytes.size(), bundle_identity_, head) &&
               head.history.size() <= bundle_lim_.history_max && head.map.size() <= bundle_lim_.nc_max;
    };
    auto probe = [&] {
        KvFlashSlotHead head;
        if (!read_head(head)) return KvFlashAdopt::Refused;
        return kvflash_adopt_of(kvf_kv_->adopt_probe(slot, kBundleRequest + (uint64_t)slot, (uint32_t)head.history.size()));
    };
    // Before the first host-budget eviction: a bundle with a valid head but an unreadable log or slab is refused here (streamed once,
    // nothing held), so no healthy parked session leaves for a restore that cannot happen.
    auto validate = [&] {
        KvFlashSlotHead head;
        *refused = true;   // as load(): a damaged bundle is the bundle's, not the host's
        if (!read_head(head)) { why = "head refused: another server, a bad format or over this server's limits"; return false; }
        if (!kvflash_bundle_validate(sdir, head, p0.chunk_bytes(), bundle_lim_.slab_max, why)) return false;
        *refused = false;
        return true;
    };
    auto has_victim = [&] { for (const auto & o : parked_list_()) if (o.parked.slot != slot) return true; return false; };
    // When the ledger lacks room the returning bundle's memory (the pager blob and the slabs, not charged yet) is freed, the
    // oldest parked session leaves (to disk, as an admit does), the bundle is read again and the adopt retried, so the session
    // resumes from disk instead of going cold and no victim is staged beside an uncharged bundle
    const KvFlashRestoreResult res = kvflash_restore_adopt(
        probe, has_victim, validate, load,
        [&] {
            const PagedKvStatus st = kvf_kv_->adopt_parked(slot, kBundleRequest + (uint64_t)slot, (uint32_t)l.head.history.size(), l.pager.data(), l.pager.size());
            return kvflash_adopt_of(st);
        },
        [&] { l = KvFlashBundleLoaded{}; end = SlotStateSnapshot{}; },
        [&] { return evict_oldest_("host budget needed to restore a session", slot) >= 0; },
        evict_for_room);
    if (res.load_failed) return why;
    const KvFlashAdopt got = res.adopt;
    if (got == KvFlashAdopt::Busy && busy) { *busy = true; return "the host budget is full"; }
    if (got != KvFlashAdopt::Ok)
        return "the pager or the host budget refused it";
    if (!restore_slot_state(b_.cache_, slot, end)) {
        kvf_kv_->drop_parked(slot);
        *refused = true;
        return "the recurrent slab does not fit this model";
    }
    ParkedSession ps;
    ps.tick = ++park_tick_; ps.valid = true;
    if (l.head.ckpt_pos > 0) {
        ps.ckpt.pos = (int)l.head.ckpt_pos; ps.ckpt.file = l.head.ckpt_gen;
        ps.ckpt.slab.ssm = std::move(l.ckpt_ssm); ps.ckpt.slab.conv = std::move(l.ckpt_conv);
    }
    if (l.head.trunk_pos > 0) {
        ps.trunk.pos = (int)l.head.trunk_pos; ps.trunk.file = l.head.trunk_gen;
        ps.trunk.slab.ssm = std::move(l.trunk_ssm); ps.trunk.slab.conv = std::move(l.trunk_conv);
    }
    if (l.head.extra_pos > 0) {
        ps.extra.pos = (int)l.head.extra_pos; ps.extra.file = l.head.extra_gen;
        ps.extra.slab.ssm = std::move(l.extra_ssm); ps.extra.slab.conv = std::move(l.extra_conv);
    }
    // the pin comes back from the head (kvflash_ladder_restored): with the third, or as a copy of the ckpt or trunk slab it names
    apply_ladder_(kvflash_ladder_restored(ps.ckpt.pos, ps.trunk.pos, ps.extra.pos, (int)l.head.pin_pos), ps.ckpt, ps.trunk, ps.extra, SlotCkpt{});
    const KvFlashSlotPager & p = kvf_kv_->pager(slot);
    std::fprintf(stderr, "[kvflash] slot %d bundle restored: gen=%u history=%zu ckpt=%d trunk=%d third=%d resident=%d/%d (%.1f MiB)\n",
                 slot, l.head.gen, l.head.history.size(), ps.ckpt.pos, ps.trunk.pos, ps.extra.pos, p.resident_blocks(), p.n_pool_blocks(), l.pager.size() / 1048576.0);
    const uint32_t gen = l.head.gen;   // l.head moves into the book below
    kvflash_bundle_clean(sdir, &l.head);   // a park cut short by the last run, a torn tail
    ps.history = book_.adopt(slot, key, std::move(l.head));
    parked_[(size_t)slot] = std::move(ps);
    bundle_failed_[(size_t)slot] = false;
    // the slot's question rows come back with it (an older head or a lost file: none, the next question recaptures them)
    if (!l.uq_why.empty()) std::fprintf(stderr, "[kvflash] slot %d bundle %s\n", slot, l.uq_why.c_str());
    uq_store_[(size_t)slot] = std::move(l.uq);
    if (!uq_store_[(size_t)slot].empty())
        std::fprintf(stderr, "[kvflash] slot %d bundle restored uq rows lo=%d hi=%d n=%d\n", slot, uq_store_[(size_t)slot].lo, uq_store_[(size_t)slot].hi, uq_store_[(size_t)slot].n);
    const bool rebind = head_k_max_ > 0;   // the digest below is taken before the rebind: the hard pins came back with the pager's layout, the rebind only adds the head's again
    if (kvflash_bundle_digest_on()) {   // the installed session, re-read through the save's staging; before the rebind and any reselect
        ParkedSession & q = parked_[(size_t)slot];
        if (const std::string & fault = kvflash_digest_fault(); !fault.empty()) {   // R3 negative control (test hook): corrupt one byte of the live restored slot
            KvFlashSlotPager & fp = kvf_kv_->pager(slot);
            fp.synchronize_paging();
            bool hit = false;
            if (fault == "kv") { const auto lg = fp.ledger_blob(); for (int c = 0; c < (int)(lg.size() / kKvFlashLedgerEntry) && !hit; ++c) if (lg[(size_t)c * kKvFlashLedgerEntry + 1]) hit = fp.corrupt_chunk_byte(c); }
            else { SlotStateSnapshot f; if (snapshot_slot_state(b_.cache_, slot, f) && !f.ssm.empty() && !f.ssm[0].empty()) { f.ssm[0][0] ^= 1; hit = restore_slot_state(b_.cache_, slot, f); } }
            if (hit) { std::fprintf(stderr, "%s\n", kvflash_fault_line(fault, key, gen).c_str()); std::fflush(stderr); }
        }
        const auto t0 = std::chrono::steady_clock::now();
        SlotStateSnapshot live;
        if (snapshot_slot_state(b_.cache_, slot, live)) log_bundle_digest_("restore", slot, key, gen, q.history, live, q.ckpt, q.trunk, q.extra, &uq_store_[(size_t)slot], t0);
        else std::fprintf(stderr, "[kvflash-digest] restore sid=%s gen=%u unavailable: slab snapshot failed\n", key.c_str(), gen);
    }
    if (rebind) {   // the session pinned a head when it parked: pin it again (its history must still start with that head)
        const KvFlashSessionEntry * me = book_.store().find_key(key);
        if (me && !kvflash_rebind_restored(*kvf_kv_, head_binder_, head_k_max_, {{slot, &parked_[(size_t)slot].history, me->preamble, me->preamble_chunks}}) && !me->preamble.empty() && me->preamble_chunks > 0)
            std::fprintf(stderr, "[kvflash] slot %d session %s: its pinned head does not hold, the slot runs unpinned\n", slot, key.c_str());
    }
    return "";
}

void Qwen35SeqEngine::rebind_restored_heads_() {
    std::vector<KvFlashRestoredSession> rs;
    for (int slot = 0; slot < slots_.slot_count(); ++slot) {
        const KvFlashSessionEntry * me = parked_[(size_t)slot].valid ? book_.store().find_key(book_[slot].key) : nullptr;
        if (me) rs.push_back({slot, &parked_[(size_t)slot].history, me->preamble, me->preamble_chunks});
    }
    const int n = kvflash_rebind_restored(*kvf_kv_, head_binder_, head_k_max_, rs);
    if (!rs.empty()) std::fprintf(stderr, "[kvflash] start-up: %d of %zu restored sessions re-bound their pinned head\n", n, rs.size());
}

void Qwen35SeqEngine::init_heads_() {
    const KvFlashSlotPager & p0 = kvf_kv_->pager(0);
    head_binder_ = KvFlashHeadBinder(slots_.slot_count());
    kvf_kv_->set_binder(&head_binder_);
    head_k_max_ = sticky_on_ ? kvflash_head_k_max(p0.n_pool_blocks()) : 0;
    if (head_k_max_ <= 0) { std::fprintf(stderr, "[kvflash] pinned heads: off\n"); return; }
    if (disk_) {
        DiskCacheConfig dc;
        dc.cache_dir = bundle_dir_;
        dc.min_tokens = 1;
        head_nostore_ = std::make_unique<NoCheckpoints>();
        auto disk = std::make_unique<DiskPrefixCache>(dc, *head_nostore_);
        if (disk->init() && !disk->disabled()) {
            disk->use_layout(kvflash_head_layout_id(identity_key_()));
            head_disk_ = std::move(disk);
        }
    }
    std::fprintf(stderr, "[kvflash] pinned heads: up to %d chunks per slot, %s\n", head_k_max_,
                 head_disk_ ? "persisted in the cache directory" : "host memory only");
}

const KvFlashHead * Qwen35SeqEngine::keep_head_(KvFlashHead h) {
    const uint64_t hash = h.hash;
    if (heads_.emplace(hash, std::move(h)).second) head_order_.push_back(hash);
    while (heads_.size() > kHeadsHost) { heads_.erase(head_order_.front()); head_order_.pop_front(); }   // the file, if any, stays
    const auto it = heads_.find(hash);
    return it == heads_.end() ? nullptr : &it->second;
}

const KvFlashHead * Qwen35SeqEngine::find_head_(const std::vector<int32_t> & prompt, int k) {
    const int ct = (int)kvf_kv_->block_size();
    const std::vector<uint64_t> path = kvflash_head_hashes(prompt, k, ct);
    if (k <= 0 || (int)path.size() < k) return nullptr;
    if (const auto it = heads_.find(path.back()); it != heads_.end()) return it->second.k == k ? &it->second : nullptr;
    if (!head_disk_) return nullptr;
    const std::vector<int32_t> key(prompt.begin(), prompt.begin() + (size_t)k * (size_t)ct);
    if (!head_disk_->contains(key)) return nullptr;
    KvFlashHead h;
    const OwnedSnapshot snap = head_disk_->load(key);
    if (!kvflash_head_from_snapshot(snap, key, ct, kvf_kv_->pager(0).chunk_bytes(), h) || h.hash != path.back()) return nullptr;
    std::fprintf(stderr, "[kvflash] pinned head of %d chunks read from the cache directory\n", k);
    return keep_head_(std::move(h));
}

// The slot's prefill stopped at the head's capture cut: it holds exactly the head's chunks and its recurrent state is the
// head's. The head is kept once, the slot pinned to it, and the file written (pinned in the directory up to kHeadsPinned).
void Qwen35SeqEngine::capture_head_(int slot) {
    SlotRecall & rc = recall_[(size_t)slot];
    const std::vector<int32_t> ids = std::move(rc.capture_ids);
    rc.capture_ids.clear();
    KvFlashSlotPager & p = kvf_kv_->pager(slot);
    const int ct = p.chunk_tokens(), k = (int)ids.size() / std::max(1, ct);
    SlotStateSnapshot snap;
    if (!snapshot_slot_state(b_.cache_, slot, snap)) { std::fprintf(stderr, "[kvflash] slot %d head not captured: slab snapshot failed\n", slot); return; }
    KvFlashHead h = kvflash_head_capture(head_binder_, slot, p, ids, k, head_k_max_, kvflash_slab_pack(snap.ssm, snap.conv));
    if (h.k == 0) { std::fprintf(stderr, "[kvflash] slot %d head of %d chunks not captured (past the %d-chunk bound or the pager lacks them)\n", slot, k, head_k_max_); return; }
    std::fprintf(stderr, "[kvflash] slot %d pinned head captured: %d chunks (%d tokens), %.1f MiB\n", slot, h.k, h.k * ct,
                 (h.hdr.size() + h.chunks.size() + h.slab.size()) / 1048576.0);
    if (heads_.count(h.hash)) return;   // another slot kept the same head first
    if (head_disk_) {
        OwnedSnapshot s = kvflash_head_to_snapshot(h, ct);
        if (head_disk_->save_async(std::move(s), ids) && head_pinned_files_ < (int)kHeadsPinned && !head_disk_->pin(ids, true).empty())
            ++head_pinned_files_;
    }
    keep_head_(std::move(h));
}

// A request whose sid names a session on disk (or a migrated one its prompt extends) brings it back
// into a slot: a free one, else the oldest parked session's, which is evicted to disk in turn. The
// resume then runs through the same pick as a session that never left.
KvFlashRestore Qwen35SeqEngine::restore_from_disk_(const AdmitMeta & who, const std::vector<int32_t> & prompt, int e_role) {
    const KvFlashRestorePlan plan = book_.plan_restore(who.client, who.sid, prompt, e_role, active_slots_());
    if (plan.pick == KvFlashRestorePick::none) return KvFlashRestore::absent;
    if (plan.pick == KvFlashRestorePick::busy) return KvFlashRestore::busy;
    const std::string key = plan.key;
    // the claim was taken by book_.resume_claim, so the cache budget cannot evict the session while it is read: given back when this restore ends
    struct Unclaim { KvFlashSessionStore & st; const std::string & key; ~Unclaim() { st.unclaim(key); } } unclaim{book_.store(), key};
    int slot = -1;
    for (int i = 0; i < slots_.slot_count() && slot < 0; ++i)
        if (!slots_.is_active(i) && !parked_[(size_t)i].valid) slot = i;
    if (slot < 0) {
        slot = evict_oldest_("slot needed to restore a session");
    }
    if (slot < 0) return KvFlashRestore::busy;   // every slot is live, or no parked session could leave
    if (!book_[slot].key.empty() && !evict_session_(slot, "slot reused")) return KvFlashRestore::busy;
    bool refused = false;
    bool busy = false;
    const std::string why = adopt_bundle_(slot, key, &refused, /*evict_for_room=*/true, &busy);
    if (busy) {   // live reservations hold the room and no parked session is left to give it: the request waits, it does not go cold
        std::fprintf(stderr, "[kvflash] session %s not restored from disk yet: %s\n", key.c_str(), why.c_str());
        return kvflash_restore_of(KvFlashAdopt::Busy);
    }
    if (!why.empty()) {
        std::fprintf(stderr, "[kvflash] session %s not restored from disk: %s%s\n", key.c_str(), why.c_str(), refused ? " (listed as foreign)" : "");
        if (refused) book_.store().mark_foreign(key);
        return KvFlashRestore::refused;
    }
    std::fprintf(stderr, "[kvflash] slot %d session restored from disk: sid=%s key=%s\n", slot, book_[slot].sid.c_str(), key.c_str());
    return KvFlashRestore::restored;
}

// At park the step thread stages the commit (the end slab D2H, the changed chunks' bytes, the head)
// and hands it to the writer thread; the file writes are off the critical path. A resumed session
// appends only the chunks from the one holding its resume point on and keeps the slab file of a rung
// the directory holds; the first park of a slot, a park after a failed or skipped one and a log
// whose dead records outnumber the live ones write a fresh log. The head the next park builds on is
// the last one the writer acknowledged, so a failed park leaves nothing but files to unlink.
void Qwen35SeqEngine::save_bundle_(int slot, ParkedSession & ps) {
    if (bundle_dir_.empty()) return;
    bundle_writer_->wait(slot);   // one staged park per slot, the ladder's charge: the earlier one is on disk first
    const auto t0 = std::chrono::steady_clock::now();
    KvFlashSlotBundle & sb = book_[slot];
    const bool failed = bundle_failed_[(size_t)slot].exchange(false);
    KvFlashBundleCommit c;
    if (sb.pending) {   // the earlier park's verdict
        sb.pending = false;
        if (failed) {   // its files join the debt: a failed park never reaches its unlinks, so the debt it carried is still owed
            const auto left = kvflash_bundle_leftovers(sb.head, sb.staged);
            sb.debt.insert(sb.debt.end(), left.begin(), left.end());
        } else {
            sb.head = std::move(sb.staged);
            sb.debt.clear();   // it unlinked what it carried
        }
        sb.staged = KvFlashSlotHead{};
    }
    c.unlink = sb.debt;
    const KvFlashSlotHead & prev = sb.head;
    const bool fresh = sb.fresh_park(failed);
    sb.rewrite = false;
    SlotStateSnapshot end;
    if (!snapshot_slot_state(b_.cache_, slot, end)) {   // nothing staged: the next park rewrites the slot from the acknowledged head
        std::fprintf(stderr, "[kvflash] slot %d bundle not saved: slab snapshot failed\n", slot);
        sb.rewrite = true;
        return;
    }
    KvFlashSlotPager & p = kvf_kv_->pager(slot);
    p.synchronize_paging();
    const int nc = p.n_chunks();
    const size_t cb = p.chunk_bytes(), rb = kvflash_record_bytes(cb);
    if (sb.key.empty()) {   // a session that was never given an identity (cannot happen from admit_bounded)
        sb.sid = "anon:" + std::to_string(park_tick_); sb.client = "anon"; sb.created = (int64_t)std::time(nullptr);
        sb.key = book_.store().new_key(sb.client, sb.sid);
    }
    c.dir = book_.store().dir(sb.key);
    c.new_dir = fresh;
    c.gen = sb.next_gen++;
    c.record_bytes = rb;
    KvFlashSlotHead h;
    h.identity = bundle_identity_; h.gen = c.gen;
    h.history = ps.history;
    kvflash_head_take_pager(h, p);
    h.map.assign((size_t)nc, 0);
    // the records: a chunk from the resume point on, or one the previous park had no record for
    const int d0 = fresh ? 0 : kvflash_bundle_dirty_chunks(sb.resume_L, nc, p.chunk_tokens()).first;
    std::vector<int> write;
    size_t kept = 0;
    for (int ci = 0; ci < nc; ++ci) {
        if (h.ledger[(size_t)ci * kKvFlashLedgerEntry + 1] == 0) continue;   // not materialized: zeros at load, no record
        const bool keep = ci < d0 && ci < (int)prev.map.size() && prev.map[(size_t)ci] != 0;
        if (keep) { h.map[(size_t)ci] = prev.map[(size_t)ci]; ++kept; } else write.push_back(ci);
    }
    // compaction when the log's dead records would outnumber the live ones after this append
    const bool compact = !fresh && kvflash_bundle_compacts(prev.log_bytes, kept, write.size(), rb);
    uint64_t off;
    if (fresh || compact) {
        c.new_log = true; c.log_gen = c.gen; h.log_gen = c.gen;
        off = kKvFlashLogHeader;
        if (compact) {
            c.copy_from = prev.log_gen;
            for (int ci = 0; ci < nc; ++ci) if (h.map[(size_t)ci] != 0) { c.copies.push_back(h.map[(size_t)ci]); h.map[(size_t)ci] = off; off += rb; }
        }
        if (prev.gen != 0) c.unlink.push_back(kvflash_bundle_log_name(prev.log_gen));
    } else {
        c.log_gen = prev.log_gen; h.log_gen = prev.log_gen; c.log_bytes = prev.log_bytes;
        off = prev.log_bytes;
    }
    for (int ci : write) {
        std::vector<uint8_t> bytes(cb);
        p.read_chunk(ci, bytes.data());
        c.records.push_back({(uint32_t)ci, std::move(bytes)});
        h.map[(size_t)ci] = off; off += rb;
    }
    h.log_bytes = off;
    // the slabs: the end state every park (a new file), a rung's when the directory does not hold its
    // file (a rung taken this turn, or every one after a fresh log); a rung that moved from checkpoint
    // to trunk keeps its file
    h.end_gen = sb.next_gen++;
    c.files.push_back({kvflash_bundle_slab_name(h.end_gen), kvflash_slab_pack(end.ssm, end.conv)});
    auto rung = [&](SlotCkpt & r, uint32_t & pos, uint32_t & file) {
        if (r.pos <= 0) return;
        pos = (uint32_t)r.pos;
        const auto named = kvflash_head_gens(prev);
        const bool on_disk = r.file != 0 && r.file != prev.end_gen && std::find(named.begin(), named.end(), r.file) != named.end();
        if (!on_disk) {
            r.file = sb.next_gen++;
            c.files.push_back({kvflash_bundle_slab_name(r.file), kvflash_slab_pack(r.slab.ssm, r.slab.conv)});
        }
        file = r.file;
    };
    rung(ps.ckpt, h.ckpt_pos, h.ckpt_gen);
    rung(ps.trunk, h.trunk_pos, h.trunk_gen);
    const KvFlashHeadRungs hr = kvflash_ladder_to_head(ladder_of_(ps.ckpt, ps.trunk, ps.extra));   // a third at a rung's position (parked mid-turn) has no slab of its own: the head keeps its pin
    if (hr.extra > 0) rung(ps.extra, h.extra_pos, h.extra_gen);
    h.pin_pos = (uint32_t)hr.pin;
    // the slot's question rows, when they are this session's (the key is the hash of its own prompt ids up to the window's end): a
    // new file only when the directory does not already hold these
    const KvFlashUqStore & uq = uq_store_[(size_t)slot];
    const bool uq_rows = uq_queries_ > 0 && kvflash_uq_belongs(uq, ps.history);
    if (uq_rows) {
        h.uq_lo = (uint32_t)uq.lo; h.uq_hi = (uint32_t)uq.hi; h.uq_n = (uint32_t)uq.n; h.uq_hash = uq.hash;
        if (prev.uq_gen != 0 && prev.uq_lo == h.uq_lo && prev.uq_hi == h.uq_hi && prev.uq_n == h.uq_n && prev.uq_hash == h.uq_hash && !fresh) h.uq_gen = prev.uq_gen;
        else { h.uq_gen = sb.next_gen++; c.files.push_back({kvflash_bundle_slab_name(h.uq_gen), kvflash_uq_file_pack(uq)}); }
    }
    const auto kept_gens = kvflash_head_owned_gens(h);
    for (uint32_t id : kvflash_head_owned_gens(prev))
        if (id != 0 && std::find(kept_gens.begin(), kept_gens.end(), id) == kept_gens.end()) c.unlink.push_back(kvflash_bundle_slab_name(id));
    c.head = kvflash_head_pack(h);
    size_t staged = c.head.size();
    for (const auto & r : c.records) staged += r.second.size();
    for (const auto & f : c.files) staged += f.second.size();
    std::fprintf(stderr, "[kvflash] slot %d bundle staged: gen=%u %s history=%zu ckpt=%d trunk=%d third=%d chunks=%zu/%d L=%d slabs=%zu %.1f MiB %.0f ms\n",
                 slot, c.gen, fresh ? "fresh" : compact ? "compacted" : "append", ps.history.size(), ps.ckpt.pos, ps.trunk.pos, (int)h.extra_pos, write.size(), nc,
                 fresh ? 0 : sb.resume_L, c.files.size(), staged / 1048576.0,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    // the session's meta.json and the index go before the commit: a crash between them leaves a directory
    // with a meta and no head, which the start-up scan removes (nothing was committed). publish_park asks the one cache
    // budget for the park's bytes first; a refusal skips the park (the session stays in its slot, the next park rewrites
    // from the acknowledged head).
    {
        size_t live = 0;
        for (uint64_t o : h.map) live += o != 0;
        KvFlashParkStats st;
        st.tokens = ps.history.size();
        st.bytes_live = live * rb + c.head.size() + kvflash_head_slab_bytes(h, [&](uint32_t id) {   // a file this park wrote is staged, one kept from the last park is on disk
            const std::string name = kvflash_bundle_slab_name(id);
            for (const auto & f : c.files) if (f.first == name) return (uint64_t)f.second.size();
            return detail::file_size_or0(c.dir + "/" + name);
        });
        st.bytes_dead = h.log_bytes - kKvFlashLogHeader - live * rb;
        if (const KvFlashHeadBinding hb = head_binder_.binding(slot); hb.k > 0) { st.preamble = kvflash_hex16(hb.hash); st.preamble_chunks = hb.k; }   // the head it pins
        if (!book_.publish_park(slot, st, (int64_t)std::time(nullptr))) {   // no room in the cache budget, or the session's identity is not durable
            std::fprintf(stderr, "[kvflash] slot %d bundle not saved: session %s has no room in the cache budget or its identity is not durable\n", slot, sb.key.c_str());
            return;   // publish_park marked the session `rewrite`: it is not evictable until a park lands
        }
    }
    sb.staged = std::move(h);
    sb.pending = true;
    if (kvflash_bundle_digest_on()) log_bundle_digest_("save", slot, sb.key, c.gen, ps.history, end, ps.ckpt, ps.trunk, ps.extra, uq_rows ? &uq_store_[(size_t)slot] : nullptr, std::chrono::steady_clock::now());   // before the writer: a kill after the head commit cannot lose the line
    bundle_writer_->submit({slot, std::move(c)});
}

// A resumed turn starts from the recent residency (the newest chunks), as the
// single-slot SCORE_POOL_RECENT does, so its first pass and scores do not depend
// on the previous turn's picks. Deferred frees are drained around the page work.
void Qwen35SeqEngine::begin_resume_(int slot) {
    KvFlashSlotPager & p = kvf_kv_->pager(slot);
    p.synchronize_paging();
    kvf_->complete_all();
    if (!recall_on_) return;
    p.set_deferred_free(false);
    p.reselect_recent();
    p.set_deferred_free(true);
    p.synchronize_paging();
}

// C1.4: the slot's turn (kvflash_slot_turn.h). R > kv_offset snapshots at the
// cut ending at R; R == kv_offset snapshots the resume state now. A recall turn
// has a sealed chunk before R (the plan), so R > 0.
void Qwen35SeqEngine::plan_slot_turn_(int slot, const std::vector<int32_t> & prompt, int kv_offset, int capture_at) {
    SlotRecall & rc = recall_[(size_t)slot];
    rc = SlotRecall{};
    KvFlashSlotPager & p = kvf_kv_->pager(slot);
    p.score_hook = nullptr;
    if (b_.kvflash_preamble_rows_ > 0) {
        // LUCE_KVFLASH_PREAMBLE_ROWS: the slot's system+tools preamble and the rest of the recall floor are never victims.
        // Every turn: a new session's pager starts without pins.
        const int e = kvflash_preamble_end(prompt, b_.w_.im_start_id, b_.w_.system_id);
        std::fprintf(stderr, "[kvflash] slot %d preamble pin end=%d pinned_chunks=%d reserve=%d\n",
                     slot, e, kvflash_pin_preamble(p, e, b_.kvflash_preamble_rows_), b_.kvflash_preamble_rows_);
    }
    // the recall budget kvflash_prefill_apply_on plans this turn with: a longer turn replays its tail only
    const int ct = p.chunk_tokens(), usable = luce::common::kvflash_usable_blocks(p.n_pool_blocks(), kvf_->headroom());
    const int max_turn = ct > 0 ? luce::common::kvflash_max_turn_tokens(ct, usable, b_.kvflash_preamble_reserve_ / ct,
                                                                        luce::common::kKvFlashRecallUnit * b_.kvflash_recall_blocks_) : 0;
    rc.turn = kvflash_slot_turn_plan(prompt, kv_offset, b_.w_.im_start_id, b_.w_.user_id, ct, usable, kKvFlashCutU,
                                     b_.kvflash_select_queries_, recall_on_, end_rung_min_, b_.kvflash_prefill_reselect_ == 5, max_turn, capture_at);
    if (rc.turn.capture_at > 0) rc.capture_ids.assign(prompt.begin(), prompt.begin() + rc.turn.capture_at);
    if (uq_queries_ > 0) {
        // The latest human question: its end is a cut of every pass and its last queries join the scorer's.
        // A prefill that covers the window captures them (whatever the slot's store holds: the same request
        // runs the same cuts and scores with its own rows); a later turn of the session reads the store.
        const KvFlashHumanTurn q = kvflash_last_human_turn(prompt, b_.w_.im_start_id, b_.w_.user_id, b_.w_.newline_id,
                                                           b_.w_.im_end_id, b_.w_.tool_response_id);
        kvflash_slot_turn_force_cut(rc.turn, kvflash_uq_cut(q, uq_queries_, rc.turn.kv_offset, rc.turn.end));
        rc.uq = kvflash_uq_plan(q, uq_queries_, rc.turn.kv_offset, rc.turn.end);
        rc.uq_hash = kvflash_uq_hash(prompt, rc.uq.cut);
        if (!rc.uq.capture) rc.uq_miss = rc.uq.n == 0 ? "no-human-turn" : "window-not-in-prefill";   // unless the store holds it
    }
    // L4: after the turn's last cut is in (the UQ cut above), with the replay's cut rule (the park tail of the step loop)
    if (header_rung_) kvflash_slot_turn_header_rung(rc.turn, kKvFlashCutU, park_short_on_ ? park_tail_ : 0);
    if (rc.turn.apply_at >= 0) {
        // Mode 5: score the prompt ids with the drafter's session now; an unavailable drafter leaves an LRU turn.
        if (b_.kvflash_pflash2_scores(prompt, p.chunk_tokens(), rc.scores, rc.score_ms)) {
            std::fprintf(stderr, "[kvflash] slot %d prefill reselect mode=5: R=%d (end %d, prefill %d tokens once)\n",
                         slot, rc.turn.r, rc.turn.end, rc.turn.end - rc.turn.kv_offset);
            if (kvflash_slot_turn_apply_due(rc.turn, kv_offset)) apply_recall5_(slot);
        } else {
            rc.turn.apply_at = -1;
            std::fprintf(stderr, "[kvflash] slot %d prefill reselect mode=5: drafter scoring failed (LRU)\n", slot);
        }
    } else if (rc.turn.first_pass()) {
        if (rc.turn.r == rc.turn.kv_offset) rc.snap_ok = snapshot_slot_state(b_.cache_, slot, rc.snap);
        std::fprintf(stderr, "[kvflash] slot %d prefill reselect mode=4: R4=%d (end %d, re-prefill %d tokens, %zu cuts%s)\n",
                     slot, rc.turn.r, rc.turn.end, rc.turn.end - rc.turn.r, rc.turn.cuts.size(), rc.turn.end_rung ? ", header rung" : rc.turn.header_rung ? ", replay rung" : "");
        if (rc.turn.header_rung) std::fprintf(stderr, "[kvflash] slot %d header rung: the replay checkpoints at %d\n", slot, rc.turn.ckpt);
    } else if (recall_on_) {
        std::fprintf(stderr, "[kvflash] slot %d prefill reselect mode=4: skip (%d tokens fit %d blocks or no turn)\n",
                     slot, rc.turn.end - rc.turn.kv_offset, p.n_pool_blocks());
    }
}

void Qwen35SeqEngine::apply_recall5_(int slot) {
    SlotRecall & rc = recall_[(size_t)slot];
    KvFlashSlotPager & p = kvf_kv_->pager(slot);
    // Pending frees must drain first and the apply's page-outs free at once, as in the mode-4 rewind.
    p.synchronize_paging();
    kvf_->complete_all();
    p.set_deferred_free(false);
    b_.kvflash_prefill_apply_on(p, rc.scores, rc.turn.apply_at, rc.turn.end, rc.turn.reserve_chunks,
                                rc.turn.apply_at - 1, rc.score_ms, "; pflash2", slot, head_binder_.pins(slot));
    p.set_deferred_free(true);
    rc.turn.apply_at = -1;
}

// C1.4: the first pass of a slot ended this step: score its chunks before R with
// the captured queries, restore the R state, rewind the allocator and pager to R,
// apply the picks (synchronous page work) and replay from R.
bool Qwen35SeqEngine::rewind_slot_turn_(int slot) {
    using clk = std::chrono::steady_clock;
    SlotRecall & rc = recall_[(size_t)slot];
    KvFlashSlotPager & p = kvf_kv_->pager(slot);
    const int ct = p.chunk_tokens(), r = rc.turn.r, end = rc.turn.end;
    const auto t0 = clk::now();
    const KvFlashQkDims d = b_.kvflash_exact_dims();
    const size_t qsize = (size_t)d.n_layers * d.n_q_heads * d.head_dim;
    const std::vector<float> q_all = std::move(rc.q_tail);   // accumulated over the first pass's cuts; consumed here
    rc.q_tail.clear();
    const int n_q = qsize ? (int)(q_all.size() / qsize) : 0;
    std::vector<const float *> qs;
    for (int t = 0; t < n_q; t++) qs.push_back(q_all.data() + (size_t)t * qsize);
    const int sealed = r / ct;
    bool scored = false;
    rc.scores.clear();
    if (uq_queries_ > 0) {
        const KvFlashUqStore & st = uq_store_[(size_t)slot];
        kvflash_uq_join(qs, kvflash_uq_holds(st, rc.uq.cut, rc.uq_hash) ? &st : nullptr, rc.uq_miss, qsize, qs.size());
    }
    if (!qs.empty()) {
        std::vector<std::vector<float>> o;
        std::vector<KvFlashMassPlanes> planes;
        if (b_.kvflash_exact_scores_gpu_on(p, sealed, qs, o, planes) || b_.kvflash_exact_scores_on(p, sealed, qs, o, &planes)) {
            rc.scores = kvflash_select_scores(planes);
            scored = true;
        }
    }
    const double score_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    {   // L4 pre-condition: the bits the first pass's cuts produce (the captured queries, the scores); FNV-1a 64
        auto fnv = [](const std::vector<float> & v) { uint64_t h = 1469598103934665603ull; const uint8_t * b = (const uint8_t *)v.data(); for (size_t i = 0; i < v.size() * sizeof(float); ++i) h = (h ^ b[i]) * 1099511628211ull; return h; };
        std::fprintf(stderr, "[kvflash] slot %d mode4 first-pass bits: q_tail n_q=%d fnv=%016llx scores n=%zu fnv=%016llx\n",
                     slot, n_q, (unsigned long long)fnv(q_all), rc.scores.size(), (unsigned long long)fnv(rc.scores));
    }
    if (!scored)
        std::fprintf(stderr, "[kvflash] slot %d prefill reselect mode=4 @%d: scoring failed (n_q=%d sealed=%d); LRU replay\n",
                     slot, r, n_q, sealed);
    // The replay runs the question's window again from the new pool: its rows replace the store's when its
    // cut reaches the question's end (a replay that never gets there leaves the first pass's rows).
    if (kvflash_uq_rewind_recaptures(rc.uq, r)) {
        rc.uq_rows.clear();
        rc.uq.capture = true;
    }
    if (!rc.snap_ok || !restore_slot_state(b_.cache_, slot, rc.snap)) {
        std::fprintf(stderr, "[kvflash] slot %d prefill reselect mode=4 @%d: state restore failed (snap %d)\n", slot, r, rc.snap_ok ? 1 : 0);
        return false;
    }
    // The R state is this turn's sticky checkpoint (the next turn resumes at R and prefills the question
    // with the recent window, as a cold turn does: a replay state there cost recall), the resume-time
    // fallback below it the trunk (Rule A); else freed (~150 MiB).
    if (sticky_on_) place_ckpt_(slot, SlotCkpt{r, std::move(rc.snap)});
    rc.snap = SlotStateSnapshot{};
    rc.snap_ok = false;
    // Pending frees would be double-freed by the truncate's relayout; the apply's page-outs must free at once
    // so every wanted page-in gets a block.
    p.synchronize_paging();
    kvf_->complete_all();
    p.set_deferred_free(false);
    const int cur_before_rewind = slots_.slot(slot).cur_pos;
    const bool rewound = slots_.rewind_prefill(slot, r);
    if (rewound) set_feat_from_(slot, feature_valid_after_rewind(feat_from_of_(slot), cur_before_rewind, r, b_.cache_.target_feat_cap));
    if (rewound && scored)
        b_.kvflash_prefill_apply_on(p, rc.scores, r, end, rc.turn.reserve_chunks, end - 1, score_ms, "; rewind", slot, head_binder_.pins(slot));
    p.set_deferred_free(true);
    if (!rewound) {
        std::fprintf(stderr, "[kvflash] slot %d prefill reselect mode=4 @%d: rewind failed\n", slot, r);
        return false;
    }
    rc.turn.phase = KvFlashTurnPhase::Replay;
    std::fprintf(stderr, "[kvflash] slot %d mode4 rewound to %d: replay %d tokens, resident %s\n",
                 slot, r, end - r, p.debug_resident(sealed).c_str());
    return true;
}

// LUCE_KVFLASH_UQ_QUERIES: the cut [kv_pos, kv_pos + n) the step just ran holds rows of the question's window
// (its last rows); the cut that ends at the question's end completes the store.
void Qwen35SeqEngine::take_uq_rows_(int slot, int kv_pos, int n) {
    SlotRecall & rc = recall_[(size_t)slot];
    const int want = kvflash_uq_cut_rows(rc.uq, kv_pos, n);
    if (want <= 0) return;
    const KvFlashQkDims qd = b_.kvflash_exact_dims();
    std::vector<float> rows;
    kvflash_read_captured_queries_tail(b_.cache_, qd, capture_rows_, want, rows);
    rc.uq_rows.insert(rc.uq_rows.end(), rows.begin(), rows.end());
    if (kv_pos + n < rc.uq.cut) return;
    rc.uq.capture = false;
    KvFlashUqStore st;
    st.lo = rc.uq.win_lo; st.hi = rc.uq.cut; st.n = rc.uq.n; st.hash = rc.uq_hash;
    st.rows = std::move(rc.uq_rows);
    rc.uq_rows.clear();
    if (st.rows.size() != (size_t)st.n * qd.n_layers * qd.n_q_heads * qd.head_dim) { rc.uq_miss = "capture-incomplete"; return; }
    uq_store_[(size_t)slot] = std::move(st);
    rc.uq_miss = "";
}

bool Qwen35SeqEngine::supports_images() const {
    return !kvf_ && b_.supports_images();
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

void Qwen35SeqEngine::discard_prefix_store(PrefixStoreRef checkpoint) {
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
    if (kvf_) return admit(request_id, prompt, sampler);   // no prefix store with KVFlash slots (the scheduler never plans one)
    AdmitResult result = slots_.admit(request_id, prompt, sampler);
    if (result.status != AdmitResult::Status::admitted) return result;
    clear_slot_images(result.slot);

    const int slot = result.slot;
    slots_.slot(slot).pending_capture = {};
    bool restored = false;
    if (plan.restore.valid()) {
        const int restore_index = checkpoint_index(plan.restore);
        const bool metadata_valid =
            restore_index >= 0 &&
            plan.restore.tokens < (int)prompt.size();
        PrefixSnapshot * snap = metadata_valid
            ? &b_.prefix_snapshots_[restore_index] : nullptr;
        // The rows of a state-only checkpoint come from the deeper one the
        // plan names; a stale name fails the restore below.
        const int kv_index = checkpoint_index(plan.restore_kv);
        const PrefixSnapshot * kv_from =
            kv_index >= 0 && kv_index != restore_index &&
            b_.prefix_snapshots_[kv_index].cur_pos == plan.restore_kv.tokens
                ? &b_.prefix_snapshots_[kv_index] : nullptr;
        const auto restore_started = std::chrono::steady_clock::now();
        if (snap && snap->ctx &&
            snap->layout == PrefixSnapshot::Layout::paged &&
            snap->cur_pos == plan.restore.tokens) {
            Qwen35SlotManager::PrefillChunk seeded =
                slots_.seed_restored_prefix(slot, plan.restore.tokens);
            PagedKvSequenceSnapshot sequence;
            const bool pool_ok = seeded.ok &&
                pool_->sequence(slots_.slot(slot).handle, sequence) ==
                    PagedKvStatus::Ok;
            const bool table_ok = pool_ok && upload_block_table_delta(
                slot, seeded.first_new_block, seeded.new_blocks.data(),
                seeded.new_blocks.size());
            restored = table_ok && restore_paged_target_cache(
                *snap, b_.cache_, slot, sequence.block_table,
                (int)pool_->block_size(), kv_from);
        }
        const uint64_t restore_elapsed_us =
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - restore_started).count();

        if (!restored) {
            discard_prefix_store(plan.restore);
            slots_.retire(slot);
            result = slots_.admit(request_id, prompt, sampler);
            result.prefix_store.invalidated = plan.restore;
            if (result.status == AdmitResult::Status::admitted) {
                reset_recurrent_slot(b_.cache_, result.slot);
            } else {
                result.error =
                    "cold admission failed after stale prefix restore";
            }
        } else {
            result.prefix_store.restored = plan.restore;
            set_feat_from_(slot, plan.restore.tokens);   // the checkpoint restores KV and state, not the ring's features
            std::fprintf(stderr,
                "[parallel-pc] restored checkpoint=%llu seq_slot=%d "
                "tokens=%d time_ms=%.1f\n",
                (unsigned long long)plan.restore.id, slot,
                plan.restore.tokens, (double)restore_elapsed_us / 1000.0);
        }
        result.prefix_store.restore_attempted = true;
        result.prefix_store.restore_elapsed_us = restore_elapsed_us;
        if (result.status != AdmitResult::Status::admitted) return result;
    } else {
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
    if (pool_->sequence(slots_.slot(slot).handle, sequence) !=
            PagedKvStatus::Ok ||
        sequence.kv_seq_len != (uint32_t)ticket.checkpoint.tokens) {
        event.error = "prefix capture page table is incomplete";
        return event;
    }
    const auto capture_started = std::chrono::steady_clock::now();
    PrefixSnapshot & snapshot = b_.prefix_snapshots_[checkpoint];
    if (!replace_paged_target_cache(
            b_.cache_, slot, sequence.block_table,
            (int)pool_->block_size(), ticket.checkpoint.tokens, snapshot,
            &b_.displaced_snapshots_[checkpoint])) {
        event.elapsed_us =
            (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - capture_started).count();
        event.error = luce_last_error();
        if (event.error.empty()) {
            event.error = "paged prefix capture failed";
        }
        return event;
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
        int slot, int logits_row, const int32_t * cached_argmax,
        std::vector<float> * logits_scratch) {
    const TargetWeights & w = b_.w_;
    const int vocab = w.n_vocab;
    Qwen35Slot & seq = slots_.slot(slot);
    int32_t token = -1;
    if (seq.sampler.needs_logit_processing()) {
        std::vector<float> local_logits;
        std::vector<float> & logits = logits_scratch
            ? *logits_scratch
            : local_logits;
        if (logits.empty()) logits.resize((size_t)vocab);
        ggml_backend_tensor_get_async(
            b_.target_backend_, b_.sg_.logits, logits.data(),
            (size_t)logits_row * (size_t)vocab * sizeof(float),
            sizeof(float) * (size_t)vocab);
        ggml_backend_synchronize(b_.target_backend_);
        token = sample_logits(logits.data(), vocab, seq.sampler,
                              seq.penalty_history(), seq.rng);
    } else if (cached_argmax) {
        token = *cached_argmax;
    } else {
        ggml_backend_tensor_get_async(
            b_.target_backend_, b_.sg_.argmax_tokens, &token,
            (size_t)logits_row * sizeof(int32_t), sizeof(int32_t));
        ggml_backend_synchronize(b_.target_backend_);
    }
    return b_.apply_min_tokens_floor(
        token, seq.generated_tokens(),
        (size_t)logits_row * (size_t)vocab * sizeof(float));
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

// A chain round's pager plan opens the chunks of its whole width; rows it did not accept leave a chunk the slot's length
// does not reach. Drop it (also after a planned round fell back to AR), so a park records ceil(len/64) chunks and the
// next crossing allocates and uploads afresh.
void Qwen35SeqEngine::trim_unaccepted_chunks_() {
    if (!kvf_kv_) return;
    for (int slot = 0; slot < slots_.slot_count(); ++slot)
        if (slots_.is_active(slot)) kvf_kv_->trim_to_len(slot);
}

// After the readback: queue the page-outs that return every live slot to its
// usable quota, so the next step finds its blocks free (no-op without KVFlash).
bool Qwen35SeqEngine::settle_kvflash_(std::vector<PrefillOutput> & prefill_outputs) {
    if (!kvf_) return true;
    kvf_->finish_step();   // the step pins end here, after the readback
    for (int slot : pending_rewind_) {
        if (rewind_slot_turn_(slot)) continue;
        // That slot's turn is lost and its session is never parked; the other slots keep their step.
        recall_[(size_t)slot].tainted = true;
        fail_prefill_output(prefill_outputs, slot, "KVFlash recall could not rewind the turn (state snapshot, restore or rewind failed)");
    }
    pending_rewind_.clear();
    trim_unaccepted_chunks_();
    std::vector<int> live;
    for (int slot = 0; slot < slots_.slot_count(); ++slot) {
        if (!slots_.is_active(slot)) continue;
        const SeqSlot & s = slots_.slot(slot);
        if (s.prefilling() && kvflash_slot_turn_apply_due(recall_[(size_t)slot].turn, s.cur_pos)) apply_recall5_(slot);
        live.push_back(slot);
    }
    if (kvf_->settle(live) < 0) return false;
    for (const PrefillOutput & o : prefill_outputs) {   // what the answer decodes against, as the serial engine prints it
        if (o.status != PrefillOutput::Status::completed || !std::getenv("LUCE_KVFLASH_DEBUG_RESIDENT")) continue;
        const KvFlashSlotPager & p = kvf_kv_->pager(o.slot);
        const int committed = slots_.slot(o.slot).cur_pos;
        std::fprintf(stderr, "[kvflash-dbg] after prefill kv_offset=%d committed=%d resident=%s slot=%d\n",
                     recall_[(size_t)o.slot].turn.kv_offset, committed, p.debug_resident(committed / p.chunk_tokens()).c_str(), o.slot);
    }
    return true;
}

// A KVFlash slot's whole column from its pager: 4 entries per resident chunk,
// -1 for an evicted one. Replaces the per-block deltas of the pool path.
bool Qwen35SeqEngine::upload_kvflash_column(int slot, int kv_len) {
    ggml_tensor * table = b_.cache_.paged_block_table;
    if (!table || slot < 0 || slot >= table->ne[1]) return false;
    const int ct = (int)kvf_kv_->block_size();
    const int entries = kvflash_column_entries(kv_len, (int)table->ne[0], ct);
    const int chunks = (kv_len + ct - 1) / ct;
    column_buf_.assign((size_t)chunks * 4, -1);
    kvf_->write_table(slot, column_buf_.data(), chunks, PAGED_BLOCK_SIZE);
    return upload_block_table_delta(slot, 0, column_buf_.data(), (size_t)entries);
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
    // C1.4: a first-pass endpoint is progress only; the prefill completes after the replay.
    stage.commit = stage.kv_pos + stage.chunk >= seq.prompt_len &&
                   !(kvf_ && recall_[(size_t)slot].turn.first_pass());

    Qwen35SlotManager::PrefillChunk chunk =
        slots_.append_prefill(slot, stage.chunk);
    if (!chunk.ok || chunk.rows.size() != (size_t)stage.chunk) {
        fail_prefill(slot, prefill_outputs, "prefill K/V allocation failed",
                     "prefill K/V allocation failed");
        return PrefillStage{};
    }
    if (!kvf_ && !upload_block_table_delta(
            slot, chunk.first_new_block, chunk.new_blocks.data(),
            chunk.new_blocks.size())) {
        fail_prefill(
            slot, prefill_outputs, "prefill block-table delta exceeds device capacity",
            "prefill block-table update failed");
        return PrefillStage{};
    }

    stage.rows = std::move(chunk.rows);
    if (!b_.w_.embedder.embed(
            seq.sample_history.data() + stage.kv_pos, stage.chunk,
            embeddings)) {
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
        PreparedChainRound && prepared_round, const ChainTablePlan * kvf_plan) {
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
        // A sampled lane: its verify rows on the host, and the walk.
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
    const int min_tokens = []() {
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
        const Qwen35SlotManager::StepAppend append =
            slots_.append_token(lane.slot, lane.token);
        if (!append.ok) {
            result.error = append.busy
                ? "paged KV pool exhausted during compact AR staging"
                : "compact AR K/V staging failed";
            return result;
        }
        staged_round.track(lane.slot);
        if (!kvf_ && append.new_block >= 0 &&
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

    StepGraph & graph = b_.sg_;
    const ggml_cgraph * graph_before = graph.gf;
    if (!build_target_step_paged_tree(
            graph, b_.w_, b_.cache_, b_.target_backend_,
            tree_width, tree_bucket, max_prefix,
            fixed_chain_.scratch_base, fixed_chain_.scratch_stride,
            b_.cfg_.kq_stride_pad, ar_count)) {
        result.error = "fixed chain target graph build failed";
        return result;
    }
    // A rebuilt graph (new width, context bucket or AR mix) puts its build
    // time into this round; such a round is not a cost sample.
    const bool graph_rebuilt = graph.gf != graph_before;

    std::vector<int32_t> tokens(static_cast<size_t>(total_rows), 0);
    std::vector<int32_t> parents(
        static_cast<size_t>(tree_rows_count), -1);
    std::vector<int32_t> tree_sizes(
        static_cast<size_t>(tree_bucket), 0);
    std::vector<int32_t> mapped_slots(
        static_cast<size_t>(ar_count + tree_bucket), -1);
    std::vector<int32_t> state_slots(
        static_cast<size_t>(ar_count + tree_bucket), 0);
    std::vector<int32_t> query_slots(
        static_cast<size_t>(total_rows), -1);
    std::vector<int32_t> query_positions(
        static_cast<size_t>(total_rows), -1);
    std::vector<int64_t> write_rows(
        static_cast<size_t>(total_rows) * n_head_kv, scratch_row_);
    std::vector<int32_t> positions(
        static_cast<size_t>(4) * total_rows, 0);
    std::vector<float> embeddings(
        static_cast<size_t>(hidden) * total_rows, 0.0f);
    seq_lens_.assign(static_cast<size_t>(n_slots), 0);

    for (int lane_index = 0; lane_index < ar_count; ++lane_index) {
        const ArLane & lane = ar_lanes[static_cast<size_t>(lane_index)];
        mapped_slots[static_cast<size_t>(lane_index)] = lane.slot;
        state_slots[static_cast<size_t>(lane_index)] = lane.slot;
        tokens[static_cast<size_t>(lane_index)] = lane.token;
        query_slots[static_cast<size_t>(lane_index)] = lane.slot;
        query_positions[static_cast<size_t>(lane_index)] = lane.position;
        seq_lens_[static_cast<size_t>(lane.slot)] = lane.position + 1;
        for (int axis = 0; axis < 3; ++axis) {
            positions[static_cast<size_t>(axis) * total_rows + lane_index] =
                lane.position + rope_delta(lane.slot);
        }
        for (int head = 0; head < n_head_kv; ++head) {
            write_rows[static_cast<size_t>(head) * total_rows + lane_index] =
                lane.physical_row;
        }
    }

    for (int lane_index = 0; lane_index < spec_count; ++lane_index) {
        const Proposal & proposal =
            proposals[static_cast<size_t>(lane_index)];
        const int tree_base = lane_index * tree_width;
        const int row_base = ar_count + tree_base;
        const int mapped_lane = ar_count + lane_index;
        tree_sizes[static_cast<size_t>(lane_index)] = tree_width;
        mapped_slots[static_cast<size_t>(mapped_lane)] = proposal.slot;
        state_slots[static_cast<size_t>(mapped_lane)] = proposal.slot;
        seq_lens_[static_cast<size_t>(proposal.slot)] =
            slots_.slot(proposal.slot).cur_pos;
        for (int node = 0; node < tree_width; ++node) {
            const int tree_row = tree_base + node;
            const int row = row_base + node;
            tokens[static_cast<size_t>(row)] =
                proposal.tokens[static_cast<size_t>(node)];
            parents[static_cast<size_t>(tree_row)] = node == 0
                ? -1
                : node - 1;
            query_slots[static_cast<size_t>(row)] = proposal.slot;
            const int position = slots_.slot(proposal.slot).cur_pos + node +
                rope_delta(proposal.slot);
            for (int axis = 0; axis < 3; ++axis) {
                positions[static_cast<size_t>(axis) * total_rows + row] =
                    position;
            }
            for (int head = 0; head < n_head_kv; ++head) {
                write_rows[static_cast<size_t>(head) * total_rows + row] =
                    static_cast<int64_t>(fixed_chain_.scratch_base) +
                    static_cast<int64_t>(proposal.slot) *
                        fixed_chain_.scratch_stride + node;
            }
        }
    }

    if (!b_.w_.embedder.embed(
            tokens.data(), total_rows, embeddings.data())) {
        result.error = "fixed chain embedding failed";
        return result;
    }
    ggml_backend_tensor_set(
        graph.inp_embed, embeddings.data(), 0,
        sizeof(float) * embeddings.size());
    ggml_backend_tensor_set(
        graph.positions, positions.data(), 0,
        sizeof(int32_t) * positions.size());
    ggml_backend_tensor_set(
        graph.parent_ids, parents.data(), 0,
        sizeof(int32_t) * parents.size());
    ggml_backend_tensor_set(
        graph.tree_sizes, tree_sizes.data(), 0,
        sizeof(int32_t) * tree_sizes.size());
    if (detail::target_paged_tree_active_slots_need_upload(graph)) {
        ggml_backend_tensor_set(
            graph.active_slot_ids, mapped_slots.data(), 0,
            sizeof(int32_t) * mapped_slots.size());
    }
    ggml_backend_tensor_set(
        graph.state_slot_ids, state_slots.data(), 0,
        sizeof(int32_t) * state_slots.size());
    ggml_backend_tensor_set(
        graph.paged_query_seq_ids, query_slots.data(), 0,
        sizeof(int32_t) * query_slots.size());
    if (graph.paged_query_positions) {
        ggml_backend_tensor_set(
            graph.paged_query_positions, query_positions.data(), 0,
            sizeof(int32_t) * query_positions.size());
    }
    ggml_backend_tensor_set(
        graph.kv_write_rows, write_rows.data(), 0,
        sizeof(int64_t) * write_rows.size());
    if (kvf_) {   // (b): the arena column from the pager, covering every chunk the plan allocated; never a pool delta
        if (!kvf_plan) { result.error = "KVFlash chain round without a pager plan"; return result; }
        for (const auto & [slot, cover] : kvf_plan->cover_len) {
            if (kvf_->table_stale(slot) && !upload_kvflash_column(slot, cover)) {
                result.error = "KVFlash block-table column exceeds device capacity";
                return result;
            }
        }
    }
    ggml_backend_tensor_set(
        b_.cache_.paged_kv_seq_lens, seq_lens_.data(), 0,
        sizeof(int32_t) * seq_lens_.size());
    staged_round.device_lengths_dirty = true;
    // A batched verify of two or more lanes may split k in its MMQ calls
    // (ggml_backend_cuda_set_mmq_splitk_scope): not bit-exact against the
    // unsplit sum, which a batch of concurrent requests never was. A single
    // lane keeps the unsplit kernels. The launches happen inside
    // graph_compute, so the scope closes after it.
    // LUCE_BATCH_INVARIANT=1 keeps the unsplit kernels at every lane count.
    ggml_backend_cuda_set_mmq_splitk_scope(spec_count >= 2 && !luce::common::batch_invariant_requested());
    const ggml_status verify_status =
        luce::common::target_graph_compute(b_.target_backend_, graph.gf);
    ggml_backend_cuda_set_mmq_splitk_scope(false);
    if (verify_status != GGML_STATUS_SUCCESS) {
        result.error = "fixed chain target compute failed";
        return result;
    }

    std::vector<int32_t> posterior(
        static_cast<size_t>(total_rows), -1);
    ggml_backend_tensor_get(
        graph.argmax_tokens, posterior.data(), 0,
        sizeof(int32_t) * posterior.size());

    // Sampled lanes verify by the single-lane sampled walk over their own
    // rows (sampled_verify.h); greedy lanes keep the argmax prefix.
    // A lane's tree_width rows are contiguous: one copy per sampled lane.
    const int vocab = b_.w_.n_vocab;
    const size_t lane_floats =
        static_cast<size_t>(tree_width) * static_cast<size_t>(vocab);
    std::vector<int> sampled_lanes;
    for (int lane_index = 0; lane_index < spec_count; ++lane_index) {
        const Proposal & proposal = proposals[static_cast<size_t>(lane_index)];
        if (slots_.slot(proposal.slot).sampler.needs_logit_processing()) {
            sampled_lanes.push_back(lane_index);
        }
    }
    double readback_ms = 0.0;
    double walk_ms = 0.0;
    if (!sampled_lanes.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        const size_t floats = sampled_lanes.size() * lane_floats;
        if (chain_sampled_rows_.size() < floats) {
            chain_sampled_rows_.resize(floats);
        }
        for (size_t k = 0; k < sampled_lanes.size(); ++k) {
            Proposal & proposal =
                proposals[static_cast<size_t>(sampled_lanes[k])];
            const int row_base = ar_count + sampled_lanes[k] * tree_width;
            proposal.sampled_rows = chain_sampled_rows_.data() + k * lane_floats;
            ggml_backend_tensor_get(
                graph.logits, chain_sampled_rows_.data() + k * lane_floats,
                static_cast<size_t>(row_base) * static_cast<size_t>(vocab) *
                    sizeof(float),
                sizeof(float) * lane_floats);
        }
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
                    sampled_lanes[static_cast<size_t>(k)])];
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
        for (size_t k = 0; k < sampled_lanes.size(); ++k) {
            const Proposal & proposal =
                proposals[static_cast<size_t>(sampled_lanes[k])];
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
        readback_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        walk_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    }

    for (int lane_index = 0; lane_index < spec_count; ++lane_index) {
        Proposal & proposal = proposals[static_cast<size_t>(lane_index)];
        const int row_base = ar_count + lane_index * tree_width;
        const int32_t * lane_posterior =
            posterior.data() + static_cast<size_t>(row_base);
        if (!proposal.sampled_rows) {
            proposal.verified = chain_verified_prefix(
                proposal.tokens, lane_posterior,
                static_cast<size_t>(tree_width));
        }
        size_t accepted = proposal.verified;
        int room =
            slots_.max_context() - slots_.slot(proposal.slot).cur_pos;
        if (kvf_kv_) room = static_cast<int>(std::min<int64_t>(room, kvf_kv_->append_room(proposal.slot)));   // never past the reserved block
        accepted = std::min(accepted, static_cast<size_t>(std::max(0, room)));
        accepted = chain_budget_cap(accepted, slot_value_or(gen_cap_, proposal.slot, 1 << 30),
                                    slots_.slot(proposal.slot).generated_tokens());   // P1-1: never past the output the scheduler will take
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

    std::vector<int32_t> accepted_prefixes(
        static_cast<size_t>(tree_bucket), 0);
    std::vector<int32_t> commit_slots(
        static_cast<size_t>(tree_bucket), -1);
    std::vector<int64_t> commit_rows(
        static_cast<size_t>(tree_rows_count), -1);
    std::vector<int32_t> feature_commit_rows(
        static_cast<size_t>(total_rows), -1);
    const int feature_cap = b_.cache_.target_feat_cap;

    for (int lane_index = 0; lane_index < ar_count; ++lane_index) {
        const ArLane & lane = ar_lanes[static_cast<size_t>(lane_index)];
        feature_commit_rows[static_cast<size_t>(lane_index)] =
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
        const Qwen35SlotManager::StepAppend & append = proposal.append;
        if (!append.ok ||
            append.physical_rows.size() != proposal.accepted) {
            result.error = append.busy
                ? "paged KV pool exhausted during chain staging"
                : "accepted chain K/V staging failed";
            return result;
        }
        staged_round.track(proposal.slot);
        if (!kvf_ && !block_table_delta_fits(
                proposal.slot, append.first_new_block,
                append.new_blocks.size())) {
            result.error = "accepted chain block-table update failed";
            return result;
        }

        accepted_prefixes[static_cast<size_t>(lane_index)] =
            static_cast<int32_t>(proposal.accepted);
        commit_slots[static_cast<size_t>(lane_index)] = proposal.slot;
        for (size_t depth = 0; depth < proposal.accepted; ++depth) {
            const int flat = lane_index * tree_width + static_cast<int>(depth);
            const int graph_row = ar_count + flat;
            commit_rows[static_cast<size_t>(flat)] =
                append.physical_rows[depth];
            feature_commit_rows[static_cast<size_t>(graph_row)] =
                proposal.slot * feature_cap +
                (append.position + static_cast<int>(depth)) % feature_cap;
        }
        proposal.seq_len =
            append.position + static_cast<int>(proposal.accepted);
    }

    ggml_backend_tensor_set(
        graph.accepted_prefixes, accepted_prefixes.data(), 0,
        sizeof(int32_t) * accepted_prefixes.size());
    ggml_backend_tensor_set(
        graph.commit_slot_ids, commit_slots.data(), 0,
        sizeof(int32_t) * commit_slots.size());
    ggml_backend_tensor_set(
        graph.commit_rows, commit_rows.data(), 0,
        sizeof(int64_t) * commit_rows.size());
    ggml_backend_tensor_set(
        graph.feature_commit_rows, feature_commit_rows.data(), 0,
        sizeof(int32_t) * feature_commit_rows.size());

    const size_t n_delta = b_.cache_.ssm_state.size();
    if (n_delta == 0 || graph.delta_captures.size() != n_delta ||
        b_.cache_.conv_state.size() != n_delta ||
        !graph.tree_features || !b_.cache_.target_feat) {
        result.error = "fixed chain capture set is incomplete";
        return result;
    }
    std::vector<const ggml_tensor *> replay_logs;
    std::vector<ggml_tensor *> states;
    std::vector<const ggml_tensor *> conv_inputs;
    std::vector<ggml_tensor *> conv_states;
    replay_logs.reserve(n_delta);
    states.reserve(n_delta);
    conv_inputs.reserve(n_delta);
    conv_states.reserve(n_delta);
    for (size_t layer = 0; layer < n_delta; ++layer) {
        const DeltaNetCapture & capture = graph.delta_captures[layer];
        if (!capture.replay_log || !capture.conv_input ||
            !b_.cache_.ssm_state[layer] ||
            !b_.cache_.conv_state[layer]) {
            result.error = "fixed chain layer capture is incomplete";
            return result;
        }
        replay_logs.push_back(capture.replay_log);
        states.push_back(b_.cache_.ssm_state[layer]);
        conv_inputs.push_back(capture.conv_input);
        conv_states.push_back(b_.cache_.conv_state[layer]);
    }

    std::vector<ggml_tensor *> caches;
    caches.reserve(b_.cache_.attn_k.size() + b_.cache_.attn_v.size());
    for (ggml_tensor * tensor : b_.cache_.attn_k) {
        if (tensor) caches.push_back(tensor);
    }
    for (ggml_tensor * tensor : b_.cache_.attn_v) {
        if (tensor) caches.push_back(tensor);
    }
    if (caches.empty()) {
        result.error = "fixed chain K/V cache set is empty";
        return result;
    }

    // Validate from the host copies just uploaded (no read-back) and commit
    // with one launch per kind. A refused preflight changes nothing, so the
    // device-validated transaction then runs as before.
    bool committed = false;
    if (static_cast<int64_t>(accepted_prefixes.size()) ==
            ggml_nelements(graph.accepted_prefixes) &&
        static_cast<int64_t>(commit_slots.size()) ==
            ggml_nelements(graph.commit_slot_ids) &&
        static_cast<int64_t>(commit_rows.size()) ==
            ggml_nelements(graph.commit_rows) &&
        static_cast<int64_t>(feature_commit_rows.size()) ==
            ggml_nelements(graph.feature_commit_rows)) {
        committed = ggml_backend_cuda_tree_commit_transaction_host(
            caches.data(), static_cast<int>(caches.size()),
            graph.tree_features, b_.cache_.target_feat,
            graph.feature_commit_rows,
            replay_logs.data(), states.data(), conv_inputs.data(),
            conv_states.data(), static_cast<int>(n_delta),
            graph.commit_rows, graph.accepted_prefixes,
            graph.commit_slot_ids,
            fixed_chain_.scratch_base, fixed_chain_.scratch_stride,
            commit_rows.data(), accepted_prefixes.data(),
            commit_slots.data(), feature_commit_rows.data());
        static std::atomic<bool> warned{false};
        if (!committed && !warned.exchange(true)) {
            std::fprintf(stderr,
                "[qwen35] host-validated commit preflight refused; "
                "using the device-validated commit transaction\n");
        }
    }
    if (!committed &&
        !ggml_backend_cuda_tree_commit_transaction(
            caches.data(), static_cast<int>(caches.size()),
            graph.tree_features, b_.cache_.target_feat,
            graph.feature_commit_rows,
            replay_logs.data(), states.data(), conv_inputs.data(),
            conv_states.data(), static_cast<int>(n_delta),
            graph.commit_rows, graph.accepted_prefixes,
            graph.commit_slot_ids,
            fixed_chain_.scratch_base, fixed_chain_.scratch_stride)) {
        result.error = "fixed chain commit preflight failed";
        return result;
    }

    for (const Proposal & proposal : proposals) {
        if (!kvf_ && !upload_block_table_delta(
                proposal.slot, proposal.append.first_new_block,
                proposal.append.new_blocks.data(),
                proposal.append.new_blocks.size())) {
            std::fprintf(stderr,
                "[parallel] validated chain block-table upload failed\n");
            std::abort();
        }
        seq_lens_[static_cast<size_t>(proposal.slot)] = proposal.seq_len;
    }
    ggml_backend_tensor_set(
        b_.cache_.paged_kv_seq_lens, seq_lens_.data(), 0,
        sizeof(int32_t) * seq_lens_.size());

    for (const StepInput & input : inputs) {
        slots_.commit_step(input.slot);
    }
    staged_round.committed = true;
    for (int lane_index = 0; lane_index < spec_count; ++lane_index) {
        Proposal & proposal = proposals[static_cast<size_t>(lane_index)];
        const int graph_row = ar_count + lane_index * tree_width +
            static_cast<int>(proposal.accepted) - 1;
        if (proposal.sampled_rows) {
            // The walk's draw at the kept tip, or one fresh draw from the
            // tip's row (already on the host) with the committed history.
            Qwen35Slot & seq = slots_.slot(proposal.slot);
            const int32_t next = sampled_verify_next(
                proposal.walk, proposal.tokens.data(),
                static_cast<int>(proposal.accepted), [&](int row) {
                    return sample_logits(
                        proposal.sampled_rows +
                            static_cast<size_t>(row) *
                                static_cast<size_t>(vocab),
                        vocab, seq.sampler, seq.penalty_history(), seq.rng);
                });
            proposal.pending = b_.apply_min_tokens_floor(
                next, seq.generated_tokens(),
                static_cast<size_t>(graph_row) * static_cast<size_t>(vocab) *
                    sizeof(float));
        } else {
            proposal.pending = sample_graph_row(
                proposal.slot, graph_row,
                &posterior[static_cast<size_t>(graph_row)], &logits_buf_);
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
            &posterior[static_cast<size_t>(lane_index)], &logits_buf_);
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
            sampled.push_back(proposal.sampled_rows ? 1 : 0);
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

void Qwen35SeqEngine::log_kvflash_chain_round_(const StepResult & result, const std::vector<uint8_t> & selected) {
    kvf_chain_stats_.resize((size_t)slots_.slot_count(), {0, 0});
    for (size_t i = 0; i < result.decode.size() && i < selected.size(); ++i) {
        if (!selected[i]) continue;
        auto & st = kvf_chain_stats_[(size_t)result.decode[i].slot];
        st.first++;
        st.second += 1 + result.decode[i].committed_tokens.size();   // the root and the accepted drafts
    }
    if (++kvf_chain_rounds_ % 64 != 1) return;   // the first round, then every 64th
    std::fprintf(stderr, "[kvflash-chain] %llu chain rounds", (unsigned long long)kvf_chain_rounds_);
    for (size_t slot = 0; slot < kvf_chain_stats_.size(); ++slot) {
        const auto & st = kvf_chain_stats_[slot];
        if (st.first) std::fprintf(stderr, "; slot %zu: %llu rounds, %.2f tokens/round", slot, (unsigned long long)st.first, (double)st.second / (double)st.first);
    }
    std::fprintf(stderr, "\n");
}

SeqEngine::StepResult Qwen35SeqEngine::step(const StepPlan & plan) {
    StepResult result;
    std::vector<DecodeOutput> & decode_outputs = result.decode;
    std::vector<PrefillOutput> & prefill_outputs = result.prefills;
    const std::vector<StepInput> & inputs = plan.decode;
    const int n_slots = slots_.slot_count();

    auto fail_step = [&](const std::string & error) {
        if (kvf_) {
            kvf_->abort_step();   // D3: one exit path for a staged KVFlash step
            for (int slot = 0; slot < n_slots; ++slot)
                if (slots_.is_active(slot)) recall_[(size_t)slot].tainted = true;   // never parked
        }
        result.decode.clear();
        result.prefills.clear();
        result.error = error;
        return std::move(result);
    };

    if ((int)inputs.size() != slots_.decoding_count()) {
        return fail_step("decode plan does not cover every live slot");
    }
    std::vector<uint8_t> decode_seen((size_t)n_slots, 0);
    for (const StepInput & in : inputs) {
        if (in.slot < 0 || in.slot >= n_slots || in.token < 0 ||
            decode_seen[(size_t)in.slot] ||
            !slots_.slot(in.slot).decoding()) {
            return fail_step("invalid or duplicate decode row in step plan");
        }
        decode_seen[(size_t)in.slot] = 1;
    }

    const StepPlanLimits limits = step_plan_limits((int)inputs.size());
    if ((int)plan.prefills.size() > limits.max_prefill_sequences) {
        return fail_step("prefill plan exceeds engine sequence capacity");
    }
    int planned_prefill_tokens = 0;
    std::vector<uint8_t> prefill_seen((size_t)n_slots, 0);
    for (const PrefillSlice & slice : plan.prefills) {
        if (slice.slot < 0 || slice.slot >= n_slots ||
            slice.max_tokens <= 0 ||
            slice.max_tokens > limits.max_prefill_tokens_per_sequence ||
            prefill_seen[(size_t)slice.slot] ||
            decode_seen[(size_t)slice.slot] ||
            !slots_.is_prefilling(slice.slot)) {
            return fail_step("invalid or duplicate prefill slice in step plan");
        }
        prefill_seen[(size_t)slice.slot] = 1;
        planned_prefill_tokens += slice.max_tokens;
        if (planned_prefill_tokens > limits.max_prefill_tokens_total) {
            return fail_step("prefill plan exceeds engine total-token capacity");
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
            // KVFlash (Stage C-4 a): the pager plans this round's rows first, `width` per spec lane and one per AR
            // lane, so the accepted prefix finds its rows already allocated. A lane the pager could not fit sends the
            // round down the AR path below (BlocksExhausted is an AR round, never an error).
            ChainTablePlan kvf_plan;
            ChainRoundVerdict verdict = ChainRoundVerdict::Chain;
            if (kvf_) {
                std::vector<ChainLaneIn> lanes;
                lanes.reserve(inputs.size());
                for (size_t i = 0; i < inputs.size(); ++i) {
                    const bool spec = chain_lanes[i] != 0;
                    const int width = spec && i < prepared->drafts.size() ? (int)prepared->drafts[i].tokens.size() : 1;
                    lanes.push_back({inputs[i].slot, slots_.slot(inputs[i].slot).cur_pos, spec, width});
                }
                kvf_plan = chain_table_plan(lanes, slots_.max_context());
                verdict = ChainRoundVerdict::FallbackAr;
                if (kvf_plan.valid) {
                    const KvFlashStepDriver::Result ran = kvf_->run_step(kvf_plan.work);
                    if (!ran.ok) return fail_step("KVFlash step failed (page-out or allocation)");
                    verdict = chain_round_verdict(kvf_plan, ran.ran);
                    if (verdict == ChainRoundVerdict::FallbackAr) { kvf_->finish_step(); trim_unaccepted_chunks_(); }   // the AR step plans its own rows; a lane's planned width must not leave a chunk
                }
                if (verdict == ChainRoundVerdict::FallbackAr) {
                    static int n_fallback = 0;
                    if (n_fallback++ % 64 == 0)
                        std::fprintf(stderr, "[kvflash-chain] round fell back to AR (the pager had no free block for a lane): %d so far\n", n_fallback);
                }
            }
            if (verdict == ChainRoundVerdict::Chain) {
                for (const StepInput & in : inputs) stash_pos_[(size_t)in.slot] = -1;   // a chain round commits several tokens: no one-token-back state
                StepResult chain = step_chain_spec(plan, chain_lanes, std::move(*prepared), kvf_ ? &kvf_plan : nullptr);
                if (kvf_) {
                    if (!chain.error.empty()) return fail_step(chain.error);   // abort_step, the live slots are never parked
                    if (!settle_kvflash_(chain.prefills)) return fail_step("KVFlash page-out failed");   // (d): after the readback; also rewinds
                    log_kvflash_chain_round_(chain, chain_lanes);
                }
                return chain;
            }
        }
    }

    const TargetWeights & w = b_.w_;
    StepGraph & sg = b_.sg_;
    const int hidden = w.n_embd;
    const int n_head_kv = w.n_head_kv;

    // KVFlash (D1/D2): this step's rows, the decode rows and one cut per prefill
    // slot, go through the pager plan first. A slot that does not fit idles.
    using kvf_clock = std::chrono::steady_clock;
    const auto kvf_t0 = kvf_clock::now();
    auto kvf_ms = [](kvf_clock::time_point a, kvf_clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    kvf_clock::time_point kvf_t_run{}, kvf_t_up{};
    int kvf_uploads = 0;
    std::vector<PrefillSlice> run_prefills;
    run_prefills.reserve(plan.prefills.size());
    std::vector<uint8_t> kvf_ran;
    int capture_slot = -1;   // C1.4: the one slot whose first pass ends this step (its queries are captured)
    capture_rows_ = 0;
    if (kvf_) {
        std::vector<KvFlashStepWork> work;
        std::vector<int> cut((size_t)n_slots, 0);
        for (const StepInput & in : inputs) work.push_back({in.slot, slots_.slot(in.slot).cur_pos, 1});
        for (const PrefillSlice & slice : plan.prefills) {
            const Qwen35Slot & seq = slots_.slot(slice.slot);
            // the slot's turn plans the cut (a first pass keeps its R and capture cuts, otherwise the next
            // multiple of U after cur_pos or the prompt end: a function of position only)
            const SlotRecall & rc = recall_[(size_t)slice.slot];
            int c = kvflash_slot_turn_next_cut(rc.turn, seq.cur_pos, kKvFlashCutU, park_short_on_ ? park_tail_ : 0);
            const KvFlashTurnStep cs = kvflash_slot_turn_step(rc.turn, seq.cur_pos, c, b_.kvflash_select_queries_);
            if (cs.capture || cs.q_rows > 0 || kvflash_uq_cut_rows(rc.uq, seq.cur_pos, c) > 0) {
                if (capture_slot >= 0) c = 0;        // one capture per step: this slot idles
                else capture_slot = slice.slot;
            }
            cut[(size_t)slice.slot] = c;
            if (c > 0 && c <= slice.max_tokens) work.push_back({slice.slot, seq.cur_pos, c});
        }
        const KvFlashStepDriver::Result ran = kvf_->run_step(work);
        if (!ran.ok) return fail_step("KVFlash step failed (page-out or allocation)");
        kvf_t_run = kvf_clock::now();
        kvf_ran.assign((size_t)n_slots, 0);
        for (int slot : ran.ran) kvf_ran[(size_t)slot] = 1;
        if (capture_slot >= 0 && !kvf_ran[(size_t)capture_slot]) capture_slot = -1;
        for (const PrefillSlice & slice : plan.prefills) {
            if (kvf_ran[(size_t)slice.slot]) {
                run_prefills.push_back({slice.slot, cut[(size_t)slice.slot]});
            } else {   // idle: its page-outs settle for the next step
                PrefillOutput idle;
                idle.slot = slice.slot;
                prefill_outputs.push_back(std::move(idle));
            }
        }
    } else {
        run_prefills = plan.prefills;
    }

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
        return fail_step("step exceeds the input embedding rows");
    }
    int max_kv_len = 1;
    for (const StepInput & in : inputs) {
        DecodeOutput out;
        out.slot = in.slot;
        out.failed = true;
        int compact_row = -1;
        if (kvf_ && !kvf_ran[(size_t)in.slot]) {
            out.error = "KVFlash slice has no free block for the decode row";
            decode_outputs.push_back(std::move(out));
            output_rows_.push_back(compact_row);
            continue;
        }
        const Qwen35SlotManager::StepAppend app =
            slots_.append_token(in.slot, in.token);
        if (!app.ok && kvf_) return fail_step("KVFlash decode append failed after the step plan");   // D3: internal error, cohort
        if (!app.ok) {
            out.error = app.busy
                ? "paged KV pool exhausted during decode; raise "
                  "--kv-pool-tokens or lower --max-ctx/--max-concurrency"
                : "decode K/V append failed";
            decode_outputs.push_back(std::move(out));
            output_rows_.push_back(compact_row);
            continue;
        }
        if (!kvf_ && app.new_block >= 0 &&
            !upload_block_table_delta(
                in.slot, app.new_block_index, &app.new_block, 1)) {
            out.error = "decode block-table entry exceeds device capacity";
            decode_outputs.push_back(std::move(out));
            output_rows_.push_back(compact_row);
            continue;
        }
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
    prefills.reserve(run_prefills.size());
    int staged_rows = 0;
    for (const PrefillSlice & slice : run_prefills) {
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
            return fail_step("selected prefill work made no progress");
        }
        staged_rows += prefill.chunk;
        prefills.push_back(std::move(prefill));
    }

    const int live_count = (int)live_tokens_.size();
    if (kvf_ && live_count == 0 && prefills.empty()) {   // every slot idled
        if (!settle_kvflash_(prefill_outputs)) return fail_step("KVFlash page-out failed");
        return result;
    }
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
    int capture_end = 0;   // C1.4: where the capturing slot's segment ends in the packed batch
    std::vector<QwenPrefillSegment> segments;
    segments.reserve(prefills.size());
    for (size_t i = 0; i < prefills.size(); ++i) {
        const PrefillStage & prefill = prefills[i];
        const int slot = run_prefills[i].slot;
        segments.push_back({n_prefill, prefill.chunk, slot});
        n_prefill += prefill.chunk;
        if (slot == capture_slot) { capture_end = n_prefill; capture_rows_ = std::min(n_prefill, b_.cache_.q_cap_all_tokens); }
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

    if (kvf_) {
        for (int slot = 0; slot < n_slots; ++slot) {
            if (seq_lens_[(size_t)slot] > 0 && kvf_->table_stale(slot)) {
                if (!upload_kvflash_column(slot, seq_lens_[(size_t)slot])) {
                    return fail_step("KVFlash block-table column exceeds device capacity");
                }
                ++kvf_uploads;
            }
        }
        kvf_t_up = kvf_clock::now();
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
            /*capture_qk=*/capture_slot >= 0,
            /*paged_attention=*/true,
            /*n_seqs=*/with_decode ? decode_bucket : 1,
            /*seq_slot=*/0,
            /*paged_max_kv_len=*/max_kv_len,
            /*n_prefill_tokens=*/n_prefill,
            segments.data(), (int)segments.size(), gather_rows,
            /*compact_slots=*/with_decode,
            /*top_n=*/0, /*keep_base_ssm_state=*/false,
            /*single_replay_slot_ids=*/nullptr, /*capture_q_stage=*/false,
            /*q_capture_end=*/capture_end);
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
            /*compact_slots=*/true);
    }
    if (!built || !sg.kv_write_rows ||
        (fixed_chain_ready_ && !sg.target_feat_rows) ||
        (with_prefill &&
         (!sg.paged_query_seq_ids || !sg.paged_query_positions ||
          !sg.logits_row_indices))) {
        return fail_step("packed prefill/decode graph build failed");
    }

    // The prompt rows are already in place (stage_prefill_chunk).
    if (with_decode &&
        !w.embedder.embed(
            dec_tokens_.data(), decode_bucket,
            embed_rows_ + (size_t)hidden * n_prefill)) {
        return fail_step("decode embedding failed");
    }
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
    ggml_backend_tensor_set_async(
        b_.target_backend_, sg.positions, pos_buf_.data(), 0,
        sizeof(int32_t) * pos_buf_.size());

    if (fixed_chain_ready_) {
        const int cap = b_.cache_.target_feat_cap;
        const int dead_row = cap * n_slots;
        feature_rows_.assign(static_cast<size_t>(n_total), dead_row);
        token_offset = 0;
        for (size_t i = 0; i < prefills.size(); ++i) {
            const PrefillStage & prefill = prefills[i];
            const int slot = run_prefills[i].slot;
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
        ggml_backend_tensor_set_async(
            b_.target_backend_, sg.target_feat_rows,
            feature_rows_.data(), 0,
            sizeof(int32_t) * feature_rows_.size());
    }

    rows_buf_.assign((size_t)n_total * n_head_kv, scratch_row_);
    for (int h = 0; h < n_head_kv; ++h) {
        token_offset = 0;
        for (const PrefillStage & prefill : prefills) {
            for (int i = 0; i < prefill.chunk; ++i) {
                rows_buf_[(size_t)h * n_total + token_offset + i] =
                    prefill.rows[(size_t)i];
            }
            token_offset += prefill.chunk;
        }
        for (int row = 0; row < decode_bucket; ++row) {
            rows_buf_[(size_t)h * n_total + n_prefill + row] =
                dec_rows_[(size_t)h * decode_bucket + row];
        }
    }
    ggml_backend_tensor_set_async(
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
            const int slot = run_prefills[i].slot;
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
        ggml_backend_tensor_set_async(
            b_.target_backend_, sg.paged_query_seq_ids,
            query_slot_ids_.data(), 0,
            sizeof(int32_t) * query_slot_ids_.size());
        ggml_backend_tensor_set_async(
            b_.target_backend_, sg.paged_query_positions,
            query_positions_.data(), 0,
            sizeof(int32_t) * query_positions_.size());
        ggml_backend_tensor_set_async(
            b_.target_backend_, sg.logits_row_indices,
            logits_rows_.data(), 0,
            sizeof(int32_t) * logits_rows_.size());
    }
    if (with_decode) {
        ggml_backend_tensor_set_async(
            b_.target_backend_, sg.active_slot_ids,
            active_slot_ids_.data(), 0,
            sizeof(int32_t) * active_slot_ids_.size());
        ggml_backend_tensor_set_async(
            b_.target_backend_, sg.state_slot_ids,
            state_slot_ids_.data(), 0,
            sizeof(int32_t) * state_slot_ids_.size());
    }
    ggml_backend_tensor_set_async(
        b_.target_backend_, b_.cache_.paged_kv_seq_lens,
        seq_lens_.data(), 0, sizeof(int32_t) * seq_lens_.size());

    if (park_short_on_) {   // the state before each live slot's token: what a park one token short keeps
        for (int row = 0; park_decode_on_ && row < live_count; ++row) {
            const int slot = live_slot_ids_[(size_t)row];
            stash_pos_[(size_t)slot] = slot_state_stash_copy(b_.target_backend_, b_.cache_, stash_, slot, true)
                ? live_positions_[(size_t)row] : -1;
        }
        for (size_t i = 0; i < prefills.size(); ++i) {   // a turn's last prompt tokens (a cut of their own, the tail): the state before them
            const int slot = run_prefills[i].slot;
            if (prefills[i].commit && prefills[i].chunk <= park_tail_)
                stash_pos_[(size_t)slot] = slot_state_stash_copy(b_.target_backend_, b_.cache_, stash_, slot, true) ? prefills[i].kv_pos : -1;
        }
    }
    ggml_status st = GGML_STATUS_FAILED;
    {
        const Qwen35RoctxRange roctx_compute(
            "qwen35.graph_compute", roctx_metadata);
        st = luce::common::target_graph_compute(b_.target_backend_, sg.gf);
    }
    if (st != GGML_STATUS_SUCCESS) {
        // The pinned embedding rows may still be in flight.
        ggml_backend_synchronize(b_.target_backend_);
        return fail_step("packed prefill/decode compute failed");
    }

    const int decode_row0 = with_prefill ? n_commits : 0;
    const int argmax_rows = with_prefill ? gather_rows : decode_bucket;
    argmax_buf_.assign((size_t)argmax_rows, -1);
    ggml_backend_tensor_get_async(
        b_.target_backend_, sg.argmax_tokens, argmax_buf_.data(), 0,
        sizeof(int32_t) * argmax_buf_.size());
    {
        const Qwen35RoctxRange roctx_sync(
            "qwen35.argmax_readback", roctx_metadata);
        ggml_backend_synchronize(b_.target_backend_);
    }

    for (size_t oi = 0; oi < inputs.size(); ++oi) {
        DecodeOutput & out = decode_outputs[oi];
        if (out.failed) continue;
        slots_.commit_step(out.slot);
        const int row = decode_row0 + output_rows_[oi];
        out.token = sample_graph_row(
            out.slot, row, &argmax_buf_[(size_t)row], &logits_buf_);
    }

    int commit_row = 0;
    for (size_t i = 0; i < prefills.size(); ++i) {
        const int slot = run_prefills[i].slot;
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
            out.token = sample_graph_row(
                slot, commit_row, &argmax_buf_[(size_t)commit_row],
                &logits_buf_);
            ++commit_row;
            slots_.commit_prefill(slot);
        }
        prefill_outputs.push_back(std::move(out));
        if (kvf_) {   // C1.4: the R state, the first-pass end, the sticky checkpoint
            SlotRecall & rc = recall_[(size_t)slot];
            const KvFlashTurnStep ts = kvflash_slot_turn_step(rc.turn, prefills[i].kv_pos, prefills[i].chunk, b_.kvflash_select_queries_);
            if (slot == capture_slot && ts.q_rows > 0) {   // this cut holds some of the last select_queries prompt tokens: keep their queries
                const KvFlashQkDims qd = b_.kvflash_exact_dims();
                std::vector<float> rows;
                const int got = kvflash_read_captured_queries_tail(b_.cache_, qd, capture_rows_, ts.q_rows, rows);
                kvflash_q_tail_push(rc.q_tail, rows.data(), got, (size_t)qd.n_layers * qd.n_q_heads * qd.head_dim, b_.kvflash_select_queries_);
            }
            if (slot == capture_slot) take_uq_rows_(slot, prefills[i].kv_pos, prefills[i].chunk);
            if (ts.snapshot) rc.snap_ok = snapshot_slot_state(b_.cache_, slot, rc.snap);
            if (ts.capture && slot == capture_slot) pending_rewind_.push_back(slot);
            if (ts.ckpt && sticky_on_) take_ckpt_(slot, prefills[i].kv_pos + prefills[i].chunk);   // the header rung (Rule A)
            if (ts.preamble && !rc.capture_ids.empty()) capture_head_(slot);                       // the pinned head
        }
    }
    const auto kvf_t_gr = kvf_clock::now();
    if (!settle_kvflash_(prefill_outputs)) return fail_step("KVFlash page-out failed");
    if (kvf_) {   // LUCE_KVFLASH_STEP_TRACE=1: where a step's host time goes (ms), summed over 64 steps
        static const bool trace = std::getenv("LUCE_KVFLASH_STEP_TRACE") != nullptr;
        thread_local double sum[5] = {0, 0, 0, 0, 0}; thread_local int n = 0, ups = 0, dec = 0, pre = 0;   // per scheduler thread
        if (trace) {
            const auto t_end = kvf_clock::now();
            sum[0] += kvf_ms(kvf_t0, kvf_t_run); sum[1] += kvf_ms(kvf_t_run, kvf_t_up);
            sum[2] += kvf_ms(kvf_t_up, kvf_t_gr); sum[3] += kvf_ms(kvf_t_gr, t_end); sum[4] += kvf_ms(kvf_t0, t_end);
            ups += kvf_uploads; dec += (int)decode_outputs.size(); pre += (int)prefills.size();
            if (++n == 64) {
                std::fprintf(stderr, "[kvflash-trace] 64 steps: run_step %.1f upload %.1f graph+readback %.1f settle %.1f total %.1f ms; column uploads %d decode rows %d prefill cuts %d\n",
                             sum[0], sum[1], sum[2], sum[3], sum[4], ups, dec, pre);
                for (double & x : sum) x = 0;
                n = ups = dec = pre = 0;
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
    return slots_.reserve_decode(reserve_growth_);
}

bool Qwen35SeqEngine::restore_kv(int slot, std::string & error) {
    if (slots_.is_active(slot) && slots_.slot(slot).recomputing()) {
        // Parked without a checkpoint: resume as an ordinary chunked prefill
        // over the folded history, which rebuilds paged KV and slot-local
        // state together — so it must be reset exactly as at admission.
        if (!slots_.resume_recompute(slot)) return false;
        reset_recurrent_slot(b_.cache_, slot);
        reset_slot_draft_kv(slot);
        return true;
    }
    std::vector<int32_t> blocks;
    if (!offload_ || !offload_->restore(slot, blocks, error)) return false;
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
    if (offload_) offload_->discard(slot);
    stash_pos_[(size_t)slot] = -1;   // the history is folded and re-prefilled: the stashed state is not its one-back state
    return true;
}

void Qwen35SeqEngine::retire(int slot) {
    if (offload_) offload_->discard(slot);
    clear_slot_images(slot);
    if (!slots_.is_active(slot)) return;
    reset_slot_draft_kv(slot);
    set_feat_from_(slot, 0);
    set_gen_cap_(slot, 1 << 30);
    const int stash_pos = stash_pos_[(size_t)slot];   // the token count of the state the last decode step stashed
    stash_pos_[(size_t)slot] = -1;
    if (kvf_) {
        const bool tainted = recall_[(size_t)slot].tainted;
        recall_[(size_t)slot] = SlotRecall{};
        SlotCkpt ckpt = std::move(ckpt_[(size_t)slot]);
        SlotCkpt trunk = std::move(trunk_[(size_t)slot]);
        SlotCkpt extra = std::move(extra_[(size_t)slot]);
        ckpt_[(size_t)slot] = SlotCkpt{};
        trunk_[(size_t)slot] = SlotCkpt{};
        extra_[(size_t)slot] = SlotCkpt{};
        std::vector<int32_t> history;
        if (sticky_on_ && !tainted && slots_.park(slot, history)) {   // the KV and the recurrent state stay in the slot
            const int fed = (int)history.size();
            const int P = kvflash_park_point(fed, stash_pos, ckpt.pos, std::max(trunk.pos, extra.pos), park_tail_);
            if (P < fed) {   // one token short: the rows, the state and the history all stop at P
                if (!kvf_kv_->truncate_parked(slot, (uint32_t)P) ||
                    !slot_state_stash_copy(b_.target_backend_, b_.cache_, stash_, slot, false)) {
                    std::fprintf(stderr, "[kvflash] slot %d park one token short failed: session dropped\n", slot);
                    kvf_kv_->drop_parked(slot);
                    return;
                }
                history.resize((size_t)P);
            }
            const KvFlashSlotPager & p = kvf_kv_->pager(slot);
            std::fprintf(stderr, "[kvflash] slot %d parked: history=%zu%s ckpt=%d trunk=%d third=%d resident=%d/%d page_outs=%lld\n",
                         slot, history.size(), P < fed ? " (short)" : "", ckpt.pos, trunk.pos, extra.pos, p.resident_blocks(), p.n_pool_blocks(),
                         (long long)kvf_kv_->page_outs(slot));
            parked_[(size_t)slot] = ParkedSession{std::move(history), ++park_tick_, true, std::move(ckpt), std::move(trunk), std::move(extra), KvFlashResumeWait{}};
            save_bundle_(slot, parked_[(size_t)slot]);
            return;
        }
        // a queued save of this slot (a session resumed and then cancelled or tainted) is charged to the reservation the
        // retire releases: on disk first
        if (bundle_writer_) bundle_writer_->wait(slot);
    }
    slots_.retire(slot);
}

}  // namespace luce::common
