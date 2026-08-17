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
#include "common/sampler.h"
#include "internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace dflash::common {

namespace {

// Denser than pure power-of-2: reduces padding waste at non-power-of-2
// live counts (e.g. C=5 uses bucket=6 at 17% waste vs bucket=8 at 37.5%).
int decode_bucket_width(int live_count) {
    static constexpr int buckets[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    for (int b : buckets)
        if (b >= live_count) return b;
    return 64;
}

} // namespace

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
    const int decode_bucket = with_decode ? decode_bucket_width(live_count) : 0;

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
            /*with_mask=*/false, /*capture=*/false,
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
            /*with_mask=*/false, /*capture=*/false,
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
    slots_.retire(slot);
}

}  // namespace dflash::common
