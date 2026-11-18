#include "deepseek4_seq_engine.h"

#include "deepseek4_backend.h"
#include "deepseek4_page_layout.h"
#include "deepseek4_snapshot.h"
#include "common/model_backend.h"
#include "common/sampler.h"

#include <algorithm>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace luce::common {

namespace {
std::vector<PagedKvTensor> ds4_paged_kv_planes(const DeepSeek4PagedCache & cache) {
    std::vector<PagedKvTensor> planes;
    for (const auto & layer : cache.layers) {
        if (!layer.ratio) continue;
        for (ggml_tensor * tensor : {layer.comp_kv, layer.index_comp_kv}) {
            if (tensor) planes.push_back({tensor, 0,
                (DS4_PAGE_TOKENS / layer.ratio) * tensor->nb[1]});
        }
    }
    return planes;
}
} // namespace

DeepSeek4SeqEngine::DeepSeek4SeqEngine(
        DeepSeek4Backend & backend, PagedKvPool & pool, int max_ctx,
        uint32_t table_stride)
    : b_(backend), slots_(pool, max_ctx),
      offload_(slots_, pool, backend.backend_, ds4_paged_kv_planes(backend.paged_cache_)),
      stride_(table_stride),
      host_tables_((size_t)pool.max_sequences() * table_stride, -1),
      reserve_growth_((size_t)pool.max_sequences(), 0),
      slot_has_images_((size_t)pool.max_sequences(), 0),
      checkpoints_(ModelBackend::kMaxSlots),
      capture_events_((size_t)pool.max_sequences()) {}

DeepSeek4SeqEngine::~DeepSeek4SeqEngine() {
    for (Checkpoint & checkpoint : checkpoints_) free_deepseek4_snapshot(checkpoint.snap);
}

bool DeepSeek4SeqEngine::token_is_eos(int32_t token) const {
    return deepseek4_is_eos_tok(token, b_.w_);
}

StepPlanLimits DeepSeek4SeqEngine::step_plan_limits(
        int decode_rows) const {
    // The gathered graph accepts at most six independent sequences. A live
    // decoder owns one row. Concurrent prompts advance four rows per turn;
    // a lone prompt can use all sixteen rows without delaying another request.
    // Each gathered lane reads earlier rows in graph order.
    // Heterogeneous execution keeps one prompt token per lane.
    const int available = std::max(
        0, DEEPSEEK4_MAX_PAGED_SEQUENCES - decode_rows);
    if (b_.moe_hybrid_) return {available, 1, available, 1};
    const int rows = std::max(0, DEEPSEEK4_MAX_GATHERED_ROWS - decode_rows);
    const int sequences = std::min(available, rows);
    int prompt_rows = 4;
    if (decode_rows == 0) {
        int prompts = 0;
        for (int slot = 0; slot < slots_.slot_count(); ++slot) {
            prompts += slots_.is_prefilling(slot);
        }
        if (prompts == 1) prompt_rows = rows;
    }
    return {sequences, prompt_rows, rows, /*prefill_allocation_quantum=*/4};
}

SeqEngine::AdmitResult DeepSeek4SeqEngine::admit(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    return admit_text(request_id, prompt, sampler, {});
}

SeqEngine::AdmitResult DeepSeek4SeqEngine::admit_text(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler, const std::vector<int> & restore_points) {
    AdmitResult result = admit_slot(request_id, prompt, sampler);
    if (result.status == AdmitResult::Status::admitted &&
        !stage_text_prompt(request_id, result.slot, prompt, restore_points) &&
        !slots_.is_active(result.slot)) {
        result.status = AdmitResult::Status::failed;
        result.slot = -1;
        result.error = "text prompt could not be staged";
    }
    return result;
}

bool DeepSeek4SeqEngine::stage_text_prompt(
        uint64_t request_id, int slot, const std::vector<int32_t> & prompt,
        const std::vector<int> & restore_points, const DeepSeek4Snapshot * from,
        const DeepSeek4Snapshot * rows_from) {
    // Short prompts prefill in the gathered batch; staging pays off once a
    // prompt spans many sixteen-row (one-row on the hybrid path) steps.
    constexpr int kMinStagedTextTokens = 64;
    constexpr int min_rows = DS4_MIN_LAYER_MAJOR_PREFILL_TOKENS;
    int prefix = int(prompt.size()) - 1;
    // A chunk ends at every restore point, as in the one-request prefill, so
    // the chunks after a restore are the ones a cold prefill of the prompt
    // runs. A point closer than a layer-major chunk to the one kept before it
    // is skipped: that depends only on the points before it, which the prompt
    // shares with the one whose checkpoint it restores.
    std::vector<int> cuts;
    for (int point : restore_points) {
        if (point < prefix && point - (cuts.empty() ? 0 : cuts.back()) >= min_rows) cuts.push_back(point);
    }
    const int start = from ? from->cur_pos : 0;
    cuts.erase(cuts.begin(), std::lower_bound(cuts.begin(), cuts.end(), start + min_rows));
    // Rows too few for a chunk after the last cut go to the gathered prefill
    // with the last token.
    const int last = cuts.empty() ? start : cuts.back();
    if (prefix - last < min_rows) prefix = last;
    if (!b_.text_staging_available() || (!from && int(prompt.size()) < kMinStagedTextTokens) ||
        prompt.size() > size_t(b_.cache_.max_ctx) || staging_in_flight(slot)) {
        return false;
    }
    PendingImage pending;
    pending.slot = slot;
    pending.staged.request_id = request_id;
    pending.staged.prompt = prompt;
    pending.staged.prefix = prefix;
    pending.staged.cuts = std::move(cuts);
    pending.staged.staging = b_.image_staging_cache(slot);
    if (!pending.staged.staging || !b_.begin_staged_prefill(pending.staged, from, rows_from)) {
        return false;  // nothing seeded yet: the gathered prefill still works
    }
    SeqSlotManager::PrefillChunk seeded = slots_.seed_restored_prefix(slot, prefix);
    bool ok = seeded.ok && seeded.rows.size() == size_t(prefix);
    for (size_t i = 0; ok && i < seeded.new_blocks.size(); ++i) {
        ok = set_block(slot, seeded.first_new_block + int(i), seeded.new_blocks[i]);
    }
    if (!ok) {
        retire(slot);  // partially seeded: the request cannot continue
        return false;
    }
    pending_images_.push_back(std::move(pending));
    return true;
}

SeqEngine::AdmitResult DeepSeek4SeqEngine::admit_slot(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler) {
    using AdmitStatus = AdmitResult::Status;
    AdmitResult result = slots_.admit(
        request_id, prompt, sampler);
    if (result.status != AdmitStatus::admitted) return result;
    if (result.slot < 0 || result.slot >= slots_.slot_count()) {
        result.status = AdmitStatus::failed;
        result.error = "invalid DeepSeek4 serving slot";
        return result;
    }
    std::fill_n(host_tables_.data() + (size_t)result.slot * stride_,
                stride_, -1);
    reset_deepseek4_paged_slot(b_.paged_cache_, (uint32_t)result.slot);
    return result;
}

bool DeepSeek4SeqEngine::supports_images() const {
    return b_.image_capable_ && b_.vision_ && b_.cache_.buf;
}

SeqEngine::AdmitResult DeepSeek4SeqEngine::admit_images(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler, const ImagePromptHandle & images) {
    AdmitResult refused;
    refused.status = AdmitResult::Status::failed;
    // Image blocks prefill whole (V4's attend bidirectionally; V4.1's carry
    // the image router bias and Engram mask), which the 16-row gathered graph
    // cannot run. Admission claims a slot, starts the image encode and seeds
    // the slot with every prompt token but the last; later steps prefill that
    // prefix into the slot's staging cache (one shared layer-major pass at a
    // time, or one chunk at a time on a heterogeneous placement), copy it into
    // the paged slot, and the last (text) token then prefills in the batch,
    // yielding the first sampled token as usual.
    const int prefix = int(prompt.size()) - 1;
    if (!supports_images() || prefix < DS4_MIN_LAYER_MAJOR_PREFILL_TOKENS ||
        prompt.size() > size_t(b_.cache_.max_ctx)) {
        refused.error = "image support or prompt length is invalid";
        return refused;
    }
    // Claim the slot before encoding: a busy pool defers the request and
    // retries it, and encoding first would rerun the encoder on every retry.
    AdmitResult result = admit_slot(request_id, prompt, sampler);
    if (result.status != AdmitResult::Status::admitted) return result;
    if (staging_in_flight(result.slot)) {
        // The slot's staging cache still belongs to a pass in flight (its
        // previous request retired mid-pass): retry after the pass.
        retire(result.slot);
        result.status = AdmitResult::Status::busy;
        result.slot = -1;
        result.error = "image staging cache is still in use";
        return result;
    }
    PendingImage pending;
    pending.slot = result.slot;
    pending.staged.images = images;
    pending.staged.request_id = request_id;
    pending.staged.prompt = prompt;
    pending.staged.prefix = prefix;
    pending.staged.staging = b_.image_staging_cache(result.slot);
    std::string error;
    bool ok = pending.staged.staging && b_.encode_image_request(prompt, images, error) &&
              b_.begin_staged_prefill(pending.staged);
    SeqSlotManager::PrefillChunk seeded;
    if (ok) {
        seeded = slots_.seed_restored_prefix(result.slot, prefix);
        ok = seeded.ok && seeded.rows.size() == size_t(prefix);
    }
    for (size_t i = 0; ok && i < seeded.new_blocks.size(); ++i) {
        ok = set_block(result.slot, seeded.first_new_block + int(i), seeded.new_blocks[i]);
    }
    if (!ok) {
        retire(result.slot);
        refused.error = !error.empty() ? error : !pending.staged.error.empty() ? pending.staged.error
            : "image request could not be staged";
        return refused;
    }
    slot_has_images_[size_t(result.slot)] = 1;
    pending_images_.push_back(std::move(pending));
    return result;
}

bool DeepSeek4SeqEngine::supports_prefix_store() const {
    return b_.text_staging_available() && b_.snap_backend_;
}

size_t DeepSeek4SeqEngine::estimate_prefix_store_bytes(int tokens) const {
    return supports_prefix_store() ? b_.staged_checkpoint_bytes(tokens) : 0;
}

// The checkpoint `ref` names, if it still holds that many tokens. A restore
// checks its tokens too: from a capture's save until the scheduler hears of
// it, the id still names the entry the capture replaced.
DeepSeek4SeqEngine::Checkpoint * DeepSeek4SeqEngine::find_checkpoint(PrefixStoreRef ref) {
    if (!ref.valid() || ref.id > checkpoints_.size()) return nullptr;
    Checkpoint & checkpoint = checkpoints_[size_t(ref.id - 1)];
    return checkpoint.snap.ctx && checkpoint.snap.cur_pos == ref.tokens &&
           checkpoint.tokens.size() == size_t(ref.tokens) ? &checkpoint : nullptr;
}

void DeepSeek4SeqEngine::discard_prefix_store(PrefixStoreRef ref) {
    if (Checkpoint * checkpoint = find_checkpoint(ref)) {
        free_deepseek4_snapshot(checkpoint->snap);
        checkpoint->tokens.clear();
    }
}

size_t DeepSeek4SeqEngine::estimate_prefix_state_bytes(int tokens) const {
    return supports_prefix_store() && tokens > 0 ? b_.staged_checkpoint_state_bytes() : 0;
}

bool DeepSeek4SeqEngine::drop_prefix_kv(PrefixStoreRef ref) {
    Checkpoint * checkpoint = find_checkpoint(ref);
    return checkpoint && deepseek4_snapshot_drop_rows(checkpoint->snap);
}

SeqEngine::AdmitResult DeepSeek4SeqEngine::admit_with_prefix(
        uint64_t request_id, const std::vector<int32_t> & prompt,
        const SamplerCfg & sampler, const PrefixStorePlan & plan) {
    if (!plan.restore.valid()) {
        AdmitResult result = admit_text(request_id, prompt, sampler, plan.restore_points);
        if (result.status == AdmitResult::Status::admitted && arm_capture(result.slot, plan.capture)) {
            result.prefix_store.capture = plan.capture;
        }
        return result;
    }
    AdmitResult result = admit_slot(request_id, prompt, sampler);
    if (result.status != AdmitResult::Status::admitted) return result;
    if (staging_in_flight(result.slot)) {
        // Restoring needs this slot's staging cache, still in a shared pass.
        retire(result.slot);
        result.status = AdmitResult::Status::busy;
        result.slot = -1;
        result.error = "staging cache is still in use";
        return result;
    }
    const auto t0 = std::chrono::steady_clock::now();
    Checkpoint * from = find_checkpoint(plan.restore);
    const bool matches = from && plan.restore.tokens < int(prompt.size()) &&
        std::equal(from->tokens.begin(), from->tokens.end(), prompt.begin());
    // A state-only checkpoint takes its rows from the deeper one the plan
    // names, which must hold a prompt that extends it.
    Checkpoint * rows = matches && from->snap.state_only ? find_checkpoint(plan.restore_kv) : nullptr;
    if (rows && (rows == from || rows->tokens.size() < from->tokens.size() ||
                 !std::equal(from->tokens.begin(), from->tokens.end(), rows->tokens.begin()))) {
        rows = nullptr;
    }
    const bool restored = matches && stage_text_prompt(request_id, result.slot, prompt,
                                                       plan.restore_points, &from->snap,
                                                       rows ? &rows->snap : nullptr);
    const uint64_t elapsed_us = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count());
    if (restored) {
        result.prefix_store.restored = plan.restore;
        std::fprintf(stderr, "[parallel-pc] restored checkpoint=%llu seq_slot=%d tokens=%d time_ms=%.1f\n",
                     (unsigned long long)plan.restore.id, result.slot, plan.restore.tokens, elapsed_us / 1000.0);
    } else {
        // A payload holding other tokens is a newer capture's: keep it.
        if (matches) discard_prefix_store(plan.restore);
        retire(result.slot);
        result = admit_text(request_id, prompt, sampler, plan.restore_points);
        result.prefix_store.invalidated = plan.restore;
    }
    result.prefix_store.restore_attempted = true;
    result.prefix_store.restore_elapsed_us = elapsed_us;
    if (restored && arm_capture(result.slot, plan.capture)) result.prefix_store.capture = plan.capture;
    return result;
}

bool DeepSeek4SeqEngine::arm_capture(int slot, PrefixCaptureTicket ticket) {
    PendingImage * pending = pending_image(slot);
    if (!pending || pending->staged.images || !ticket.valid() ||
        ticket.checkpoint.id > checkpoints_.size()) return false;
    // Only where a chunk ends: at a cut or at the end of the staged rows.
    const DeepSeek4StagedPrefill & staged = pending->staged;
    const int cut = ticket.checkpoint.tokens;
    if (cut <= staged.done || (cut != staged.prefix &&
        !std::binary_search(staged.cuts.begin(), staged.cuts.end(), cut))) return false;
    pending->capture = ticket;
    return true;
}

void DeepSeek4SeqEngine::save_capture(PendingImage & pending) {
    DeepSeek4StagedPrefill & staged = pending.staged;
    const PrefixCaptureTicket ticket = pending.capture;
    pending.capture = {};
    PrefixStoreEvent event;
    event.ticket = ticket;
    event.status = PrefixStoreEvent::Status::failed;
    const auto t0 = std::chrono::steady_clock::now();
    const int tokens = ticket.checkpoint.tokens;
    Checkpoint & checkpoint = checkpoints_[size_t(ticket.checkpoint.id - 1)];
    if (staged.done != tokens || staged.staging->cur_pos != tokens) {
        event.error = "staged prefill did not stop at the capture cut";
    } else {
        ggml_backend_synchronize(b_.backend_);
        checkpoint.tokens.clear();
        if (deepseek4_snapshot_save(*staged.staging, b_.snap_backend_, checkpoint.snap)) {
            checkpoint.tokens.assign(staged.prompt.begin(), staged.prompt.begin() + tokens);
            event.status = PrefixStoreEvent::Status::saved;
            event.bytes = ggml_backend_buffer_get_size(checkpoint.snap.buf);
        } else {
            event.error = "prefix checkpoint could not be saved";
        }
    }
    event.elapsed_us = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - t0).count());
    if (event.status == PrefixStoreEvent::Status::saved) {
        std::fprintf(stderr, "[parallel-pc] saved checkpoint=%llu seq_slot=%d tokens=%d bytes=%zu time_ms=%.1f\n",
                     (unsigned long long)ticket.checkpoint.id, pending.slot, tokens, event.bytes,
                     event.elapsed_us / 1000.0);
    }
    capture_events_[size_t(pending.slot)] = std::move(event);
}

void DeepSeek4SeqEngine::report_captures(std::vector<PrefillOutput> & outputs) {
    for (PrefillOutput & out : outputs) {
        PrefixStoreEvent & event = capture_events_[size_t(out.slot)];
        if (!event.attempted() || out.status == PrefillOutput::Status::failed) continue;
        out.prefix_store = std::move(event);
        event = {};
    }
}

DeepSeek4SeqEngine::PendingImage * DeepSeek4SeqEngine::pending_image(int slot) {
    for (auto & pending : pending_images_) {
        if (pending.slot == slot) return &pending;
    }
    return nullptr;
}

bool DeepSeek4SeqEngine::staging_in_flight(int slot) const {
    if (!staged_pass_) return false;
    for (const auto & member : staged_pass_->members) {
        if (member.slot == slot) return true;
    }
    return false;
}

namespace {
constexpr int kHybridStagedRowsWithDecode = 1024;
constexpr int kHybridStagedRowsWithoutDecode = 2048;
// A prompt or context this long gets the decode graphs and pools cleared
// before its prefill and after it retires.
constexpr int kLongStagedPromptTokens = 4096;
// A prompt staging this many rows gets the decode and verify graphs retired
// before its first chunk (deepseek4_step_layer_range's bulk prefill cleanup,
// which runs only for a chunk this wide at position 0).
constexpr int kBulkPrefillRows = 512;
}  // namespace

void DeepSeek4SeqEngine::advance_pending_images(bool decoding, bool idle) {
    if (b_.staged_text_uses_chunks() || b_.staged_images_use_chunks()) {
        // Heterogeneous placement or a V4.1 model (see
        // staged_pass_supported()): stage one bounded chunk of the oldest
        // pending prompt that can run per step, so live decoders wait for at
        // most one chunk rather than the whole prompt. An image prompt whose
        // next image is still encoding lets the next pending prompt go first.
        const int rows = decoding ? kHybridStagedRowsWithDecode : kHybridStagedRowsWithoutDecode;
        bool ran = false;
        const DeepSeek4StagedPrefill * waiting_image = nullptr;
        for (auto & pending : pending_images_) {
            DeepSeek4StagedPrefill & staged = pending.staged;
            if (staged.finished() || !staged.staging) continue;
            const bool image = bool(staged.images);
            if (image ? !b_.staged_images_use_chunks() : !b_.staged_text_uses_chunks()) continue;
            if (!b_.staged_chunk_ready(staged, rows)) {
                if (!waiting_image) waiting_image = &staged;
                continue;
            }
            std::string error;
            if (staged.done == staged.restored && staged.prefix - staged.done >= kBulkPrefillRows) {
                b_.prepare_staged_prefill(*staged.staging);
            }
            if (staged.done == staged.restored && staged.prefix >= kLongStagedPromptTokens) {
                // The decode graphs cached for other slots' histories (about
                // 1 GiB once a lane is long) must not hold the memory this
                // prompt's prefill arenas need.
                b_.release_paged_decode_graphs();
            }
            const auto t0 = std::chrono::steady_clock::now();
            const int before = staged.done;
            // Each heterogeneous chunk pays a large fixed cost (every cold
            // expert's weights are read once per call), so chunks stay wide:
            // measured 306 rows in 3.2 s against 1842 rows in 7.7 s.
            if (!b_.run_staged_chunk(staged, rows, error)) {
                // Intentionally terminal: the slot already holds the prompt
                // as seeded rows, and a failure of this path (scratch or
                // expert allocation) would most likely recur in the gathered
                // prefill of the same rows. The request fails cleanly and its
                // slot is released.
                staged.error = error.empty() ? "staged prefill failed" : error;
            } else if (staged.done >= staged.prefix) {
                std::fprintf(stderr, "[deepseek4] staged %s prefill slot %d: %d rows, last chunk %d rows %.0f ms\n",
                             image ? "image" : "text", pending.slot, staged.prefix, staged.done - before,
                             std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t0).count());
            }
            ran = true;
            break;
        }
        // Nothing else to run: wait briefly for the encoder instead of
        // spinning the scheduler.
        if (!ran && idle && waiting_image) b_.wait_staged_ready(*waiting_image, rows, 20);
    }
    if (!staged_pass_ && !pending_images_.empty()) {
        std::vector<DeepSeek4StagedPrefill *> items;
        std::vector<int> slots;
        // Image blocks take the budget first: a long text prompt must not
        // hold co-pending image staging back for its whole prefill.
        for (int text = 0; text < 2; ++text) {
            for (auto & pending : pending_images_) {
                if (pending.staged.finished() || bool(pending.staged.images) == bool(text)) continue;
                if (text && b_.staged_text_uses_chunks()) continue;
                if (!text && b_.staged_images_use_chunks()) continue;
                items.push_back(&pending.staged);
                slots.push_back(pending.slot);
            }
        }
        auto staged = std::make_unique<StagedPass>();
        std::vector<int> rows;
        if (items.empty()) {
            // Nothing for the shared pass this step.
        } else if (b_.begin_staged_pass(items, decoding ? DS4_STAGED_PREFILL_ROWS_PER_STEP
                                                        : DS4_STAGED_PREFILL_ROWS_WITHOUT_DECODE,
                                        staged->pass, rows)) {
            for (size_t k = 0; k < items.size(); ++k) {
                if (rows[k] > 0) staged->members.push_back(
                    {slots[k], items[k]->images, items[k]->request_id, rows[k]});
            }
            staged->started = std::chrono::steady_clock::now();
            staged_pass_ = std::move(staged);
        } else if (idle) {
            // Nothing ready and nothing else to run: wait briefly for the
            // encoder instead of spinning the scheduler.
            b_.wait_staged_ready(*items.front(), DS4_STAGED_PREFILL_ROWS_PER_STEP, 20);
        }
    }
    if (staged_pass_) {
        // With live decoders, only a slice of the pass's layers: they then
        // decode after each slice instead of waiting for the whole pass.
        StagedPass & staged = *staged_pass_;
        std::string error;
        const bool ok = staged.pass.run_layers(
            decoding ? DS4_STAGED_PREFILL_LAYERS_PER_STEP : b_.w_.n_layer, error);
        ++staged.slices;
        if (!ok || staged.pass.done()) {
            int rows = 0;
            for (const auto & member : staged.members) {
                rows += member.rows;
                PendingImage * pending = pending_image(member.slot);
                // A member that retired (or whose slot now holds another
                // request) is skipped.
                if (!pending || pending->staged.request_id != member.request_id) continue;
                if (ok) pending->staged.done += member.rows;
                else pending->staged.error = error.empty() ? "staged prefill failed" : error;
            }
            std::fprintf(stderr, "[deepseek4] staged prefill pass: %zu requests, %d rows, %d slices, %.0f ms\n",
                         staged.members.size(), rows, staged.slices,
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - staged.started).count());
            staged_pass_.reset();
        }
    }
    // A shared pass still in flight keeps its members' staging caches; any
    // other request's can be released between its slices (each slice builds
    // its graphs anew, and the pass's hidden state is its own buffer).
    const auto in_staged_pass = [&](const PendingImage & pending) {
        if (!staged_pass_) return false;
        for (const auto & member : staged_pass_->members) {
            if (member.slot == pending.slot && member.request_id == pending.staged.request_id) return true;
        }
        return false;
    };
    for (auto & pending : pending_images_) {
        DeepSeek4StagedPrefill & staged = pending.staged;
        if (!staged.staging) continue;
        if (!staged.error.empty()) {
            // Failed: its arenas go now (the slot is failed when planned),
            // or once the shared pass holding them completes.
            if (!in_staged_pass(pending)) {
                b_.release_staged_prefill_scratch(*staged.staging);
                staged.staging = nullptr;
            }
            continue;
        }
        if (pending.capture.valid() && staged.done >= pending.capture.checkpoint.tokens) {
            save_capture(pending);
        }
        if (staged.done < staged.prefix) continue;
        if (!import_deepseek4_paged_slot(
                *staged.staging, staged.prefix, b_.paged_cache_, uint32_t(pending.slot),
                host_tables_.data() + size_t(pending.slot) * stride_, stride_, staged.error) &&
            staged.error.empty()) {
            staged.error = "staged image prefill could not be copied into the paged slot";
        }
        // The prompt is in the paged slot: its batch-width prefill arenas
        // (above 1 GiB after a 40K-token prompt) and the pools' cached
        // temporaries would otherwise stay until this staging cache's next
        // prompt, and another slot's long prompt then runs out of memory. A
        // finished prompt is never in the pass still in flight (members count
        // their rows when their pass completes), so this runs beside one too.
        b_.release_staged_prefill_scratch(*staged.staging);
        // Copied (or failed): the staging cache is free for the next request.
        staged.staging = nullptr;
    }
}

bool DeepSeek4SeqEngine::set_block(int slot, int logical, int32_t physical) {
    if (slot < 0 || slot >= slots_.slot_count() || logical < 0 ||
        (uint32_t) logical >= stride_) return false;
    host_tables_[(size_t)slot * stride_ + (size_t)logical] = physical;
    return true;
}

void DeepSeek4SeqEngine::fail_prefill(
        int slot, std::vector<PrefillOutput> & outputs,
        const std::string & error) {
    std::fprintf(stderr, "[deepseek4-parallel] prefill slot %d: %s\n",
                 slot, error.c_str());
    PrefillOutput out;
    out.slot = slot;
    out.status = PrefillOutput::Status::failed;
    out.error = error;
    outputs.push_back(std::move(out));
}

SeqEngine::StepResult DeepSeek4SeqEngine::step(const StepPlan & plan) {
    using Clock = std::chrono::steady_clock;
    const auto phase_t0 = Clock::now();
    const char * timing_env = std::getenv("LUCE_DS4_TIMING");
    const bool timing = timing_env && *timing_env && std::strcmp(timing_env, "0") != 0;
    DeepSeek4StepTelemetry telemetry;
    StepResult result;
    const std::vector<StepInput> & inputs = plan.decode;
    const int n_slots = slots_.slot_count();

    auto fail_step = [&](const std::string & error) {
        result.decode.clear();
        result.prefills.clear();
        result.error = error;
        return std::move(result);
    };

    if ((int)inputs.size() != slots_.decoding_count()) {
        return fail_step("decode plan does not cover every live DeepSeek4 slot");
    }
    std::vector<uint8_t> decode_seen((size_t)n_slots, 0);
    for (const StepInput & input : inputs) {
        if (input.slot < 0 || input.slot >= n_slots || input.token < 0 ||
            decode_seen[(size_t)input.slot] ||
            !slots_.slot(input.slot).decoding()) {
            return fail_step("invalid or duplicate DeepSeek4 decode row");
        }
        decode_seen[(size_t)input.slot] = 1;
    }

    const StepPlanLimits limits = step_plan_limits((int)inputs.size());
    if ((int)plan.prefills.size() > limits.max_prefill_sequences) {
        return fail_step("DeepSeek4 step exceeds the six-lane graph");
    }
    std::vector<uint8_t> prefill_seen((size_t)n_slots, 0);
    int prefill_rows = 0;
    for (const PrefillSlice & slice : plan.prefills) {
        prefill_rows += slice.max_tokens;
        if (slice.slot < 0 || slice.slot >= n_slots ||
            slice.max_tokens < 1 ||
            slice.max_tokens > limits.max_prefill_tokens_per_sequence ||
            prefill_rows > limits.max_prefill_tokens_total ||
            prefill_seen[(size_t)slice.slot] ||
            decode_seen[(size_t)slice.slot] ||
            !slots_.is_prefilling(slice.slot)) {
            return fail_step("invalid or duplicate DeepSeek4 prefill row");
        }
        prefill_seen[(size_t)slice.slot] = 1;
    }
    if (inputs.empty() && plan.prefills.empty()) return result;

    if (!pending_images_.empty()) {
        bool idle = inputs.empty();
        for (const PrefillSlice & slice : plan.prefills) idle = idle && pending_image(slice.slot);
        advance_pending_images(!inputs.empty(), idle);
    }

    std::vector<int32_t> lane_tokens;
    std::vector<int64_t> lane_positions;
    std::vector<int32_t> lane_slots;
    std::vector<int> decode_lanes;
    lane_tokens.reserve(inputs.size() + plan.prefills.size());
    lane_positions.reserve(inputs.size() + plan.prefills.size());
    lane_slots.reserve(inputs.size() + plan.prefills.size());
    decode_lanes.reserve(inputs.size());
    result.decode.reserve(inputs.size());
    result.prefills.reserve(plan.prefills.size());

    for (const StepInput & input : inputs) {
        DecodeOutput out;
        out.slot = input.slot;
        const SeqSlotManager::StepAppend append =
            slots_.append_token(input.slot, input.token);
        if (!append.ok) {
            out.failed = true;
            out.error = append.busy
                ? "paged KV pool exhausted during DeepSeek4 decode; raise "
                  "--kv-pool-tokens or lower --max-ctx/--max-concurrency"
                : "DeepSeek4 decode K/V append failed";
            decode_lanes.push_back(-1);
            result.decode.push_back(std::move(out));
            continue;
        }
        if (append.new_block >= 0 &&
            !set_block(input.slot, append.new_block_index,
                       append.new_block)) {
            out.failed = true;
            out.error = "DeepSeek4 decode block-table update failed";
            decode_lanes.push_back(-1);
            result.decode.push_back(std::move(out));
            continue;
        }
        decode_lanes.push_back((int)lane_tokens.size());
        lane_tokens.push_back(input.token);
        lane_positions.push_back(append.position);
        lane_slots.push_back(input.slot);
        result.decode.push_back(std::move(out));
    }

    struct PrefillLane {
        int slot = -1;
        int lane = -1;
        bool commit = false;
    };
    std::vector<PrefillLane> prefill_lanes;
    prefill_lanes.reserve(plan.prefills.size());
    for (const PrefillSlice & slice : plan.prefills) {
        if (PendingImage * pending = pending_image(slice.slot)) {
            // A staged image request reports its failure here, where its
            // slot is planned; until its prefix is copied in, it sits out.
            const DeepSeek4StagedPrefill & staged = pending->staged;
            const bool failed = !staged.error.empty();
            const bool copied = !failed && staged.done >= staged.prefix && !staged.staging;
            if (failed) {
                fail_prefill(slice.slot, result.prefills, staged.error);
                // Its queued encode must not keep the encoder from others.
                if (staged.images) b_.cancel_image_encode(*staged.images);
            }
            if (failed || copied) {
                pending_images_.erase(pending_images_.begin() + (pending - pending_images_.data()));
            }
            if (!copied) {
                // Still staging: report the slot as advanced, with no row run.
                if (!failed) {
                    PrefillOutput waiting;
                    waiting.slot = slice.slot;
                    result.prefills.push_back(std::move(waiting));
                }
                continue;
            }
        }
        const SeqSlot & before = slots_.slot(slice.slot);
        const int remaining = before.prompt_len - before.cur_pos;
        const int n_rows = std::min(slice.max_tokens, remaining);
        SeqSlotManager::PrefillChunk chunk =
            slots_.append_prefill(slice.slot, n_rows);
        if (n_rows < 1 || !chunk.ok || chunk.rows.size() != (size_t) n_rows) {
            fail_prefill(slice.slot, result.prefills,
                         "DeepSeek4 prefill K/V append failed");
            continue;
        }
        bool blocks_ok = true;
        for (size_t i = 0; i < chunk.new_blocks.size(); ++i) {
            blocks_ok = set_block(slice.slot, chunk.first_new_block + (int) i,
                                  chunk.new_blocks[i]) && blocks_ok;
        }
        if (!blocks_ok) {
            fail_prefill(slice.slot, result.prefills,
                         "DeepSeek4 prefill block-table update failed");
            continue;
        }
        const SeqSlot & slot = slots_.slot(slice.slot);
        const bool commit = slot.cur_pos == slot.prompt_len;
        // One gathered lane per prompt row, in chronological order. Only the
        // chunk's last row can need logits.
        const int first = slot.cur_pos - n_rows;
        for (int pos = first; pos < slot.cur_pos; ++pos) {
            lane_tokens.push_back(slot.sample_history[(size_t) pos]);
            lane_positions.push_back(pos);
            lane_slots.push_back(slice.slot);
        }
        prefill_lanes.push_back(
            {slice.slot, (int) lane_tokens.size() - 1, commit});
    }

    if (lane_tokens.empty()) {
        report_captures(result.prefills);
        return result;
    }
    if (lane_tokens.size() > (size_t) DEEPSEEK4_MAX_GATHERED_ROWS) {
        return fail_step("DeepSeek4 gathered step exceeds sixteen rows");
    }

    const auto embed_t0 = Clock::now();
    std::vector<float> embeddings(
        (size_t)b_.w_.n_embd * lane_tokens.size());
    if (!b_.w_.embedder.embed(lane_tokens.data(), (int)lane_tokens.size(),
                              embeddings.data())) {
        return fail_step("DeepSeek4 token embedding failed");
    }

    if (timing) telemetry.embed_us += std::chrono::duration_cast<
        std::chrono::microseconds>(Clock::now() - embed_t0).count();
    std::vector<int32_t> compact_tables(
        lane_tokens.size() * stride_, -1);
    for (size_t lane = 0; lane < lane_slots.size(); ++lane) {
        std::copy_n(
            host_tables_.data() + (size_t)lane_slots[lane] * stride_,
            stride_, compact_tables.data() + lane * stride_);
    }

    std::vector<uint8_t> logit_lanes(lane_tokens.size(), 0);
    for (size_t i = 0; i < inputs.size(); ++i) {
        const int lane = decode_lanes[i];
        if (lane >= 0 &&
            slots_.slot(inputs[i].slot).sampler.needs_logit_processing()) {
            logit_lanes[(size_t) lane] = 1;
        }
    }
    for (const PrefillLane & prefill : prefill_lanes) {
        if (prefill.commit &&
            slots_.slot(prefill.slot).sampler.needs_logit_processing()) {
            logit_lanes[(size_t) prefill.lane] = 1;
        }
    }

    std::vector<float> logits;
    std::vector<int32_t> argmax;
    // Bucketed histories pad each lane's rows to a stride (masked), which
    // changes the attention reduction. V4.1 buckets every hybrid step, so a
    // sequence's arithmetic never depends on whether another one is still
    // prefilling in the same step.
    const bool bucket_history = b_.moe_hybrid_ &&
        (plan.prefills.empty() || b_.w_.hc_staggered_pre);
    if (!deepseek4_paged_gathered_step(
            b_.backend_, b_.cfg_.device.gpu, b_.w_, b_.paged_cache_,
            embeddings.data(), lane_tokens.data(), lane_positions.data(),
            lane_slots.data(), (uint32_t)lane_tokens.size(),
            compact_tables.data(), stride_, bucket_history,
            logit_lanes.data(), logits, argmax,
            b_.moe_hybrid_.get(), b_.routing_stats_.get(),
            timing ? &telemetry : nullptr)) {
        return fail_step("DeepSeek4 gathered paged graph failed");
    }

    const auto sample_t0 = Clock::now();
    auto sample_lane = [&](int slot_id, int lane) {
        SeqSlot & slot = slots_.slot(slot_id);
        if (!slot.sampler.needs_logit_processing()) {
            return argmax[(size_t)lane];
        }
        return sample_logits(
            logits.data() + (size_t)lane * b_.w_.n_vocab,
            b_.w_.n_vocab, slot.sampler, slot.sample_history, slot.rng);
    };

    for (size_t i = 0; i < inputs.size(); ++i) {
        const int lane = decode_lanes[i];
        if (lane < 0) continue;
        DecodeOutput & out = result.decode[i];
        slots_.commit_step(out.slot);
        out.token = sample_lane(out.slot, lane);
    }
    for (const PrefillLane & prefill : prefill_lanes) {
        PrefillOutput out;
        out.slot = prefill.slot;
        if (prefill.commit) {
            out.status = PrefillOutput::Status::completed;
            out.token = sample_lane(prefill.slot, prefill.lane);
            slots_.commit_prefill(prefill.slot);
        }
        result.prefills.push_back(std::move(out));
    }
    if (timing) {
        telemetry.sample_us += std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - sample_t0).count();
        const char * phase = plan.prefills.empty() ? "paged-decode"
            : inputs.empty() ? "paged-prefill" : "paged-mixed";
        log_deepseek4_step_telemetry(phase, (int)lane_tokens.size(), 1,
            std::chrono::duration<double>(Clock::now() - phase_t0).count(), telemetry);
    }
    report_captures(result.prefills);
    return result;
}

bool DeepSeek4SeqEngine::reserve_decode(const StepPlan & plan) {
    std::fill(reserve_growth_.begin(), reserve_growth_.end(), 0);
    for (const auto & input : plan.decode) {
        if (input.slot < 0 || input.slot >= slots_.slot_count() ||
            reserve_growth_[(size_t)input.slot]) return false;
        reserve_growth_[(size_t)input.slot] = 1;
    }
    return slots_.reserve_decode(reserve_growth_);
}

bool DeepSeek4SeqEngine::restore_kv(int slot, std::string & error) {
    if (slots_.is_active(slot) && slots_.slot(slot).recomputing()) {
        // Parked without a checkpoint: resume as an ordinary chunked prefill
        // over the folded history, which rebuilds paged KV and slot-local
        // state together — so it must be reset exactly as at admission.
        if (!slots_.resume_recompute(slot)) return false;
        reset_deepseek4_paged_slot(b_.paged_cache_, (uint32_t)slot);
        std::fill_n(host_tables_.data() + (size_t)slot * stride_, stride_, -1);
        return true;
    }
    std::vector<int32_t> blocks;
    if (!offload_.restore(slot, blocks, error)) return false;
    if (blocks.size() > stride_) {
        error = "restored DeepSeek4 KV block table exceeds device capacity";
        return false;
    }
    auto * table = host_tables_.data() + static_cast<size_t>(slot) * stride_;
    std::fill_n(table, stride_, -1);
    std::copy(blocks.begin(), blocks.end(), table);
    return true;
}

bool DeepSeek4SeqEngine::kv_recomputable(int slot) const {
    return slot >= 0 && size_t(slot) < slot_has_images_.size() && !slot_has_images_[size_t(slot)];
}

bool DeepSeek4SeqEngine::offload_kv(int slot, size_t bytes, std::string & error) {
    // A slot still staging its image prefix has no paged KV to save yet.
    for (const auto & pending : pending_images_) {
        if (pending.slot == slot) {
            error = "image request is still prefilling";
            return false;
        }
    }
    return offload_.suspend(slot, bytes, error);
}

bool DeepSeek4SeqEngine::evict_kv(int slot, int32_t pending_token,
                                  std::string & error) {
    if (!kv_recomputable(slot)) {
        // Recompute replays token ids; image rows would come back as text.
        error = "image request KV cannot be rebuilt from its tokens";
        return false;
    }
    if (slot < 0 || slot >= slots_.slot_count() ||
        !slots_.evict_for_recompute(slot, pending_token)) {
        error = "slot history cannot be re-prefilled within the paged KV pool";
        return false;
    }
    offload_.discard(slot);
    return true;
}

void DeepSeek4SeqEngine::retire(int slot) {
    for (auto & pending : pending_images_) {
        if (pending.slot != slot) continue;
        // Stop its encode before its payload may go away.
        if (const auto * images = pending.staged.images.get()) b_.cancel_image_encode(*images);
    }
    pending_images_.erase(std::remove_if(pending_images_.begin(), pending_images_.end(),
                                         [slot](const PendingImage & p) { return p.slot == slot; }),
                          pending_images_.end());
    if (slot >= 0 && size_t(slot) < slot_has_images_.size()) {
        slot_has_images_[size_t(slot)] = 0;
        // A capture not yet reported: the scheduler cancels its transaction
        // with the slot, so nothing will commit the payload it saved.
        PrefixStoreEvent & event = capture_events_[size_t(slot)];
        if (event.status == PrefixStoreEvent::Status::saved) discard_prefix_store(event.ticket.checkpoint);
        event = {};
    }
    offload_.discard(slot);
    if (!slots_.is_active(slot)) return;
    const bool long_context = slots_.slot(slot).cur_pos >= kLongStagedPromptTokens;
    slots_.retire(slot);
    reset_deepseek4_paged_slot(b_.paged_cache_, (uint32_t)slot);
    std::fill_n(host_tables_.data() + (size_t)slot * stride_, stride_, -1);
    // Its long-history decode graphs serve no other slot.
    if (long_context) b_.release_paged_decode_graphs();
}

} // namespace luce::common
