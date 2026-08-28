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
#include "common/concurrency/chain_spec_shapes.h"
#include "common/dflash2_head.h"
#include "common/sampler.h"
#include "internal.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace dflash::common {

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
      prefill_quantum_(std::max(1, prefill_quantum)), b_(backend),
      slots_(pool, max_ctx), scratch_row_(scratch_row),
      fixed_chain_(fixed_chain) {
    const int n_slots = slots_.slot_count();
    slot_draft_kv_.resize(static_cast<size_t>(n_slots));
    seq_lens_.assign(static_cast<size_t>(n_slots), 0);

    fixed_chain_ready_ = fixed_chain_.enabled && fixed_chain_.width > 1 &&
        fixed_chain_.width <= 16 &&
        fixed_chain_.width == b_.dw_.block_size &&
        fixed_chain_.scratch_stride >= fixed_chain_.width &&
        b_.cache_.target_feat && b_.cache_.target_feat_cap > 0;
    if (!fixed_chain_ready_) return;

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
            const int draft_cap = std::min(
                cap, std::max(1, b_.cfg_.draft_ctx_max));
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

DraftFeatureMirror * Qwen35SeqEngine::slot_feature_mirror(int slot) {
    if (!fixed_chain_ready_ || slot < 0 ||
        slot >= static_cast<int>(slot_feature_mirrors_.size())) {
        return nullptr;
    }
    return &slot_feature_mirrors_[static_cast<size_t>(slot)];
}

DraftKvState * Qwen35SeqEngine::ensure_slot_draft_kv(int slot) {
    DraftFeatureMirror * mirror = slot_feature_mirror(slot);
    if (!mirror || slot < 0 ||
        slot >= static_cast<int>(slot_draft_kv_.size())) {
        return nullptr;
    }
    std::unique_ptr<DraftKvState> & state =
        slot_draft_kv_[static_cast<size_t>(slot)];
    if (state && state->mem_buf &&
        state->built_for == static_cast<const void *>(&b_.dw_)) {
        return state.get();
    }
    if (state) draft_kv_free(*state);
    state = std::make_unique<DraftKvState>();
    const int cap = std::min(
        mirror->cap, std::max(1, b_.cfg_.draft_ctx_max));
    if (!draft_kv_init_batched(
            *state, b_.dw_, b_.draft_backend_, cap)) {
        draft_kv_free(*state);
        state.reset();
        return nullptr;
    }
    return state.get();
}

bool Qwen35SeqEngine::chain_spec_input_capable(
        const StepInput & input) const {
    if (!fixed_chain_ready_ || !input.allow_speculation ||
        input.slot < 0 || input.slot >= slots_.slot_count()) {
        return false;
    }
    const Qwen35Slot & slot = slots_.slot(input.slot);
    return slot.decoding() && !slot.sampler.needs_logit_processing() &&
           slot.cur_pos >= 1 &&
           slot.cur_pos + fixed_chain_.width <= slots_.max_context();
}

std::vector<uint8_t>
Qwen35SeqEngine::select_chain_lanes(const StepPlan & plan) const {
    std::vector<uint8_t> selected(plan.decode.size(), 0);
    if (!plan.prefills.empty()) return selected;
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
        static_cast<size_t>(fixed_chain_.width), b_.w_.mask_token_id);
    std::vector<float> noise_embed(
        static_cast<size_t>(hidden) * fixed_chain_.width);

    auto reset_lanes = [&]() {
        for (const Lane & lane : lanes) {
            if (lane.state) draft_kv_reset(*lane.state);
        }
        for (const std::unique_ptr<DraftKvState> & state : dummy_draft_kv_) {
            if (state) draft_kv_reset(*state);
        }
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!selected[i]) continue;
        const StepInput & input = inputs[i];
        if (!chain_spec_input_capable(input)) {
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
        if (!draft_kv_begin_step(
                *state, b_.dw_, b_.draft_backend_, *mirror,
                slots_.slot(input.slot).cur_pos)) {
            reset_lanes();
            return std::nullopt;
        }
        noise[0] = input.token;
        std::fill(
            noise.begin() + 1, noise.end(), b_.w_.mask_token_id);
        if (!b_.w_.embedder.embed(
                noise.data(), fixed_chain_.width, noise_embed.data())) {
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
    const int cap = std::min(
        lanes.front().mirror->cap,
        std::max(1, b_.cfg_.draft_ctx_max));
    while (static_cast<int>(dummy_draft_kv_.size()) < dummy_count) {
        auto dummy = std::make_unique<DraftKvState>();
        if (!draft_kv_init_batched(
                *dummy, b_.dw_, b_.draft_backend_, cap)) {
            draft_kv_free(*dummy);
            reset_lanes();
            return std::nullopt;
        }
        dummy_draft_kv_.push_back(std::move(dummy));
    }
    noise[0] = lanes.front().root;
    std::fill(noise.begin() + 1, noise.end(), b_.w_.mask_token_id);
    if (!b_.w_.embedder.embed(
            noise.data(), fixed_chain_.width, noise_embed.data())) {
        reset_lanes();
        return std::nullopt;
    }
    for (int i = 0; i < dummy_count; ++i) {
        DraftKvState * dummy =
            dummy_draft_kv_[static_cast<size_t>(i)].get();
        if (!draft_kv_begin_step(
                *dummy, b_.dw_, b_.draft_backend_,
                *lanes.front().mirror, 1)) {
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
            batch_draft_graph_, b_.dw_, b_.draft_backend_,
            batch_states, hidden_blocks) ||
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
            b_.dw_, b_.draft_backend_, b_.w_.output, hidden_by_lane,
            fixed_chain_.width, roots, proposals) ||
        proposals.size() != batch_states.size()) {
        reset_lanes();
        return std::nullopt;
    }

    for (size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
        const Lane & lane = lanes[lane_index];
        std::vector<int32_t> & tokens = proposals[lane_index];
        if (tokens.size() != static_cast<size_t>(fixed_chain_.width) ||
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
    if (result.status == AdmitResult::Status::admitted) {
        reset_recurrent_slot(b_.cache_, result.slot);
        if (result.slot >= 0 &&
            result.slot < static_cast<int>(slot_draft_kv_.size()) &&
            slot_draft_kv_[static_cast<size_t>(result.slot)]) {
            draft_kv_reset(*slot_draft_kv_[static_cast<size_t>(result.slot)]);
        }
    }
    return result;
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
                              seq.sample_history, seq.rng);
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
        int slot, int max_tokens,
        std::vector<PrefillOutput> & prefill_outputs) {
    PrefillStage stage;
    if (!slots_.is_prefilling(slot)) return stage;

    Qwen35Slot & seq = slots_.slot(slot);
    stage.kv_pos = seq.cur_pos;
    stage.chunk = std::min(
        max_tokens, seq.prompt_len - stage.kv_pos);
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
        fail_prefill(
            slot, prefill_outputs, "prefill block-table delta exceeds device capacity",
            "prefill block-table update failed");
        return PrefillStage{};
    }

    stage.rows = std::move(chunk.rows);
    stage.embeddings.resize((size_t)b_.w_.n_embd * stage.chunk);
    if (!b_.w_.embedder.embed(
            seq.sample_history.data() + stage.kv_pos, stage.chunk,
            stage.embeddings.data())) {
        fail_prefill(slot, prefill_outputs, "prefill embed failed",
                     "prefill embedding failed");
        return PrefillStage{};
    }
    stage.ready = true;
    return stage;
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
        int32_t pending = -1;
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

    const int tree_width = fixed_chain_.width;
    const int hidden = b_.w_.n_embd;
    const int n_head_kv = b_.w_.n_head_kv;
    const int n_slots = slots_.slot_count();
    const int min_tokens = []() {
        const char * value = std::getenv("DFLASH_MIN_TOKENS");
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
            if (prepared.tokens.size() !=
                    static_cast<size_t>(tree_width) ||
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
    if (!build_target_step_paged_tree(
            graph, b_.w_, b_.cache_, b_.target_backend_,
            tree_width, tree_bucket, max_prefix,
            fixed_chain_.scratch_base, fixed_chain_.scratch_stride,
            b_.cfg_.kq_stride_pad, ar_count)) {
        result.error = "fixed chain target graph build failed";
        return result;
    }

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
                lane.position;
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
            const int position = slots_.slot(proposal.slot).cur_pos + node;
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
    ggml_backend_tensor_set(
        b_.cache_.paged_kv_seq_lens, seq_lens_.data(), 0,
        sizeof(int32_t) * seq_lens_.size());
    staged_round.device_lengths_dirty = true;
    if (ggml_backend_graph_compute(b_.target_backend_, graph.gf) !=
        GGML_STATUS_SUCCESS) {
        result.error = "fixed chain target compute failed";
        return result;
    }

    std::vector<int32_t> posterior(
        static_cast<size_t>(total_rows), -1);
    ggml_backend_tensor_get(
        graph.argmax_tokens, posterior.data(), 0,
        sizeof(int32_t) * posterior.size());

    for (int lane_index = 0; lane_index < spec_count; ++lane_index) {
        Proposal & proposal = proposals[static_cast<size_t>(lane_index)];
        const int row_base = ar_count + lane_index * tree_width;
        const int32_t * lane_posterior =
            posterior.data() + static_cast<size_t>(row_base);
        size_t accepted = chain_verified_prefix(
            proposal.tokens, lane_posterior, static_cast<size_t>(tree_width));
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
        if (!block_table_delta_fits(
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

    if (!ggml_backend_cuda_tree_commit_transaction(
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
        proposal.pending = sample_graph_row(
            proposal.slot, graph_row,
            &posterior[static_cast<size_t>(graph_row)], &logits_buf_);
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
    std::vector<uint8_t> decode_seen((size_t)n_slots, 0);
    for (const StepInput & in : inputs) {
        if (in.slot < 0 || in.slot >= n_slots || in.token < 0 ||
            decode_seen[(size_t)in.slot] ||
            !slots_.is_active(in.slot) || slots_.is_prefilling(in.slot)) {
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

    int max_kv_len = 1;
    for (const StepInput & in : inputs) {
        DecodeOutput out;
        out.slot = in.slot;
        out.failed = true;
        int compact_row = -1;
        const Qwen35SlotManager::StepAppend app =
            slots_.append_token(in.slot, in.token);
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
    prefills.reserve(plan.prefills.size());
    for (const PrefillSlice & slice : plan.prefills) {
        const size_t outputs_before = prefill_outputs.size();
        PrefillStage prefill =
            stage_prefill_chunk(slice.slot, slice.max_tokens, prefill_outputs);
        if (!prefill.ready) {
            if (prefill_outputs.size() == outputs_before) {
                fail_prefill(
                    slice.slot, prefill_outputs,
                    "prefill made no progress despite reserved capacity",
                    "prefill scheduler made no progress");
            }
            return fail_step("selected prefill work made no progress");
        }
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
            /*compact_slots=*/with_decode);
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

    embed_buf_.resize((size_t)hidden * n_total);
    int token_offset = 0;
    for (const PrefillStage & prefill : prefills) {
        std::copy(prefill.embeddings.begin(), prefill.embeddings.end(),
                  embed_buf_.begin() + (size_t)hidden * token_offset);
        token_offset += prefill.chunk;
    }
    if (with_decode &&
        !w.embedder.embed(
            dec_tokens_.data(), decode_bucket,
            embed_buf_.data() + (size_t)hidden * n_prefill)) {
        return fail_step("decode embedding failed");
    }
    ggml_backend_tensor_set_async(
        b_.target_backend_, sg.inp_embed, embed_buf_.data(), 0,
        sizeof(float) * (size_t)hidden * n_total);

    pos_buf_.assign((size_t)4 * n_total, 0);
    token_offset = 0;
    for (const PrefillStage & prefill : prefills) {
        fill_qwen35_mrope_positions(
            pos_buf_.data(), n_total, token_offset,
            prefill.kv_pos, prefill.chunk);
        token_offset += prefill.chunk;
    }
    if (with_decode) {
        for (int row = 0; row < live_count; ++row) {
            const int pos = live_positions_[(size_t)row];
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

    ggml_status st = GGML_STATUS_FAILED;
    {
        const Qwen35RoctxRange roctx_compute(
            "qwen35.graph_compute", roctx_metadata);
        st = ggml_backend_graph_compute(b_.target_backend_, sg.gf);
    }
    if (st != GGML_STATUS_SUCCESS) {
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
        const int slot = plan.prefills[i].slot;
        PrefillOutput out;
        out.slot = slot;
        if (prefills[i].commit) {
            out.status = PrefillOutput::Status::completed;
            out.token = sample_graph_row(
                slot, commit_row, &argmax_buf_[(size_t)commit_row],
                &logits_buf_);
            ++commit_row;
            slots_.commit_prefill(slot);
        }
        prefill_outputs.push_back(std::move(out));
    }
    return result;
}

void Qwen35SeqEngine::retire(int slot) {
    if (!slots_.is_active(slot)) return;
    if (slot >= 0 && slot < static_cast<int>(slot_draft_kv_.size()) &&
        slot_draft_kv_[static_cast<size_t>(slot)]) {
        draft_kv_reset(*slot_draft_kv_[static_cast<size_t>(slot)]);
    }
    slots_.retire(slot);
}

}  // namespace dflash::common
