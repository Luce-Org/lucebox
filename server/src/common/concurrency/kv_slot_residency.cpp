#include "common/concurrency/kv_slot_residency.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace luce::common {

KvSlotResidency::KvSlotResidency(
        PagedKvPool & pool, ggml_backend_t backend,
        std::vector<PagedKvTensor> tensors, Config cfg)
    : pool_(pool), backend_(backend), tensors_(std::move(tensors)), cfg_(cfg) {
    const int bs = std::max(1, (int)pool_.block_size());
    cfg_.chunk_tokens = std::max(bs, (cfg_.chunk_tokens / bs) * bs);
    blocks_per_chunk_ = cfg_.chunk_tokens / bs;
    budget_chunks_ = std::max(1, cfg_.per_slot_tokens / cfg_.chunk_tokens);
    const int pin_res_tok = kvflash_preamble_reserve_in_pool(
        budget_chunks_ * cfg_.chunk_tokens, cfg_.preamble_reserve_tokens,
        cfg_.chunk_tokens);
    pin_reserve_chunks_ = pin_res_tok / cfg_.chunk_tokens;
    size_t per_block = 0;
    for (const auto & t : tensors_) per_block += t.block_bytes;
    chunk_bytes_ = per_block * (size_t)blocks_per_chunk_;
    slots_.resize(pool_.max_sequences());
}

void KvSlotResidency::clear(int slot) {
    if (slot < 0 || slot >= (int)slots_.size()) return;
    SlotState & st = slots_[(size_t)slot];
    for (auto & c : st.chunks) {
        if (c.host) {
            host_bytes_used_ -= std::min(host_bytes_used_, c.host->size());
            c.host.reset();
        }
    }
    st = SlotState{};
}

void KvSlotResidency::init_slot(
        int slot, PagedKvSequenceHandle handle, int preamble_tokens,
        const uint32_t * shared_blocks, uint32_t shared_block_count,
        int shared_tokens) {
    if (slot < 0 || slot >= (int)slots_.size()) return;
    clear(slot);
    SlotState & st = slots_[(size_t)slot];
    st.active = true;
    st.handle = handle;
    st.epoch = 1;
    const int pin_rows = kvflash_preamble_pin_rows(
        preamble_tokens, pin_reserve_chunks_ * cfg_.chunk_tokens);
    const int max_pin = std::max(0, budget_chunks_ - cfg_.tail_window_chunks - 2);
    st.pinned_chunks = std::min(
        max_pin, (pin_rows + cfg_.chunk_tokens - 1) / cfg_.chunk_tokens);
    if (shared_tokens > 0) {
        on_tokens_written(slot, 0, shared_tokens);
        const int shared_chunks = (int)shared_block_count / blocks_per_chunk_;
        for (int c = 0; c < shared_chunks && c < (int)st.chunks.size(); ++c) {
            Chunk & ch = st.chunks[(size_t)c];
            ch.shared_pins.assign(
                shared_blocks + (size_t)c * (size_t)blocks_per_chunk_,
                shared_blocks + (size_t)(c + 1) * (size_t)blocks_per_chunk_);
            ch.dirty = false;
        }
    }
}

void KvSlotResidency::on_tokens_written(int slot, int old_pos, int new_pos) {
    if (slot < 0 || slot >= (int)slots_.size() || new_pos <= old_pos) return;
    SlotState & st = slots_[(size_t)slot];
    if (!st.active) return;
    const int first_c = old_pos / cfg_.chunk_tokens;
    const int last_c  = (new_pos - 1) / cfg_.chunk_tokens;
    if (last_c + 1 > (int)st.chunks.size()) {
        st.chunks.resize((size_t)(last_c + 1));
    }
    bool changed = false;
    for (int c = first_c; c <= last_c; ++c) {
        Chunk & ch = st.chunks[(size_t)c];
        if (!ch.resident) {
            ch.resident = true;
            ++st.resident_chunks;
            changed = true;
        }
        ch.dirty = true;
        ch.pinned = (c == 0) || (c < st.pinned_chunks);
        ch.last_use = ++st.clock;
    }
    const int bs = (int)pool_.block_size();
    const int old_blocks = (old_pos + bs - 1) / bs;
    const int new_blocks = (new_pos + bs - 1) / bs;
    if (changed || new_blocks != old_blocks) {
        ++st.epoch;
    }
}

bool KvSlotResidency::all_pins_still_live(const std::vector<uint32_t> & pins) const {
    if ((int)pins.size() != blocks_per_chunk_) return false;
    for (uint32_t pb : pins) {
        if (!pool_.is_block_pinned(pb)) return false;
    }
    return true;
}

bool KvSlotResidency::evict_chunks(
        SlotState & st, const std::vector<int> & chunk_indices,
        const std::vector<uint32_t> & block_table, std::string & error) {
    if (chunk_indices.empty()) return true;
    struct D2hItem {
        int c;
        uint32_t first_lb;
        std::vector<uint32_t> blocks;
    };
    std::vector<D2hItem> d2h_list;
    d2h_list.reserve(chunk_indices.size());

    for (int chunk_idx : chunk_indices) {
        Chunk & ch = st.chunks[(size_t)chunk_idx];
        if (!ch.resident) continue;
        const uint32_t first_lb = (uint32_t)chunk_idx * (uint32_t)blocks_per_chunk_;
        if (first_lb + (uint32_t)blocks_per_chunk_ > block_table.size()) {
            error = "evict_chunks out of block_table range";
            return false;
        }
        const bool has_host = (ch.host && ch.host->size() == chunk_bytes_);
        const bool skip_d2h_clean = (!ch.dirty && has_host);
        if (skip_d2h_clean) {
            ++st.stats.d2h_skipped;
        } else if (chunk_bytes_ > 0 && backend_) {
            const bool need_alloc = !has_host || ch.host.use_count() > 1;
            if (cfg_.max_host_bytes > 0 && !ch.host &&
                host_bytes_used_ + chunk_bytes_ > cfg_.max_host_bytes) {
                error = "kvflash host RAM archive budget exhausted";
                return false;
            }
            if (need_alloc) {
                if (ch.host) host_bytes_used_ -= std::min(host_bytes_used_, ch.host->size());
                ch.host = std::make_shared<std::vector<uint8_t>>(chunk_bytes_);
                host_bytes_used_ += ch.host->size();
            }
            d2h_list.push_back({chunk_idx, first_lb,
                std::vector<uint32_t>(
                    block_table.begin() + first_lb,
                    block_table.begin() + first_lb + blocks_per_chunk_)});
        }
    }

    // Single batched DMA + single GPU sync for all victims!
    if (!d2h_list.empty()) {
        auto xfer = transfer_.scope(backend_);
        for (const auto & item : d2h_list) {
            uint8_t * dst = st.chunks[(size_t)item.c].host->data();
            size_t dst_off = 0;
            for (const PagedKvTensor & t : tensors_) {
                if (!t.tensor || t.block_bytes == 0) continue;
                for (size_t i = 0; i < (size_t)blocks_per_chunk_;) {
                    const size_t run = contiguous_block_run(
                        item.blocks, i, (size_t)blocks_per_chunk_);
                    const size_t nbytes = run * t.block_bytes;
                    const size_t dev_off = t.offset + (size_t)item.blocks[i] * t.block_bytes;
                    xfer.d2h(t.tensor, dev_off, dst + dst_off + i * t.block_bytes, nbytes);
                    i += run;
                }
                dst_off += (size_t)blocks_per_chunk_ * t.block_bytes;
            }
        }
        xfer.finish();
    }

    for (int chunk_idx : chunk_indices) {
        Chunk & ch = st.chunks[(size_t)chunk_idx];
        if (!ch.resident) continue;
        const uint32_t first_lb = (uint32_t)chunk_idx * (uint32_t)blocks_per_chunk_;
        const PagedKvStatus ps = pool_.evict_logical_blocks(
            st.handle, first_lb, (uint32_t)blocks_per_chunk_);
        if (ps != PagedKvStatus::Ok) {
            error = std::string("evict_logical_blocks failed: ") + paged_kv_status_string(ps);
            return false;
        }
        ch.resident = false;
        ch.dirty = false;
        --st.resident_chunks;
        ++st.stats.evictions;
    }
    ++st.epoch;
    return true;
}

bool KvSlotResidency::restore_chunks(
        SlotState & st, const std::vector<int> & chunk_indices,
        std::string & error) {
    if (chunk_indices.empty()) return true;
    struct H2dItem {
        int c;
        std::vector<uint32_t> phys_blocks;
    };
    std::vector<H2dItem> h2d_list;
    h2d_list.reserve(chunk_indices.size());

    for (int chunk_idx : chunk_indices) {
        Chunk & ch = st.chunks[(size_t)chunk_idx];
        if (ch.resident) continue;
        const uint32_t first_lb = (uint32_t)chunk_idx * (uint32_t)blocks_per_chunk_;
        const bool pin_live = all_pins_still_live(ch.shared_pins);
        std::vector<uint32_t> phys_blocks;
        const PagedKvStatus ps = pool_.restore_logical_blocks(
            st.handle, first_lb, (uint32_t)blocks_per_chunk_, phys_blocks,
            pin_live ? ch.shared_pins.data() : nullptr);
        if (ps != PagedKvStatus::Ok) {
            error = std::string("restore_logical_blocks failed: ") + paged_kv_status_string(ps);
            return false;
        }
        if (pin_live) {
            ++st.stats.h2d_skipped;
            ++st.stats.shared_reused;
        } else if (chunk_bytes_ > 0 && backend_) {
            if (!ch.host || ch.host->size() != chunk_bytes_) {
                error = "restore_chunks missing host payload";
                return false;
            }
            h2d_list.push_back({chunk_idx, std::move(phys_blocks)});
        }
        ch.resident = true;
        ch.dirty = false;
        ch.last_use = ++st.clock;
        ++st.resident_chunks;
        ++st.stats.recalls;
    }

    // Single batched DMA + single GPU sync for all recalled chunks!
    if (!h2d_list.empty()) {
        auto xfer = transfer_.scope(backend_);
        for (const auto & item : h2d_list) {
            const uint8_t * src = st.chunks[(size_t)item.c].host->data();
            size_t src_off = 0;
            for (const PagedKvTensor & t : tensors_) {
                if (!t.tensor || t.block_bytes == 0) continue;
                for (size_t i = 0; i < (size_t)blocks_per_chunk_;) {
                    const size_t run = contiguous_block_run(
                        item.phys_blocks, i, (size_t)blocks_per_chunk_);
                    const size_t nbytes = run * t.block_bytes;
                    const size_t dev_off = t.offset + (size_t)item.phys_blocks[i] * t.block_bytes;
                    xfer.h2d(t.tensor, dev_off, src + src_off + i * t.block_bytes, nbytes);
                    i += run;
                }
                src_off += (size_t)blocks_per_chunk_ * t.block_bytes;
            }
        }
        xfer.finish();
    }
    ++st.epoch;
    return true;
}

int KvSlotResidency::pick_victim(
        const SlotState & st, int sealed_chunks, int cur_chunk,
        const std::vector<uint8_t> * keep_mask) const {
    const int tail_floor = std::max(0, cur_chunk - cfg_.tail_window_chunks);
    int best = -1;
    bool best_scored = false;
    float best_score = std::numeric_limits<float>::infinity();
    uint64_t best_use = std::numeric_limits<uint64_t>::max();
    const int lim = std::min(sealed_chunks, (int)st.chunks.size());
    for (int c = 0; c < lim; ++c) {
        const Chunk & ch = st.chunks[(size_t)c];
        if (!ch.resident || ch.pinned || c == 0 || c < st.pinned_chunks) continue;
        if (c >= tail_floor && !keep_mask) continue;
        if (keep_mask && c < (int)keep_mask->size() && (*keep_mask)[(size_t)c]) continue;
        if (ch.has_score) {
            if (best < 0 || !best_scored || ch.score < best_score ||
                (ch.score == best_score && ch.last_use < best_use)) {
                best = c; best_scored = true; best_score = ch.score; best_use = ch.last_use;
            }
        } else if (!best_scored) {
            if (best < 0 || ch.last_use < best_use) {
                best = c; best_use = ch.last_use;
            }
        }
    }
    return best;
}

bool KvSlotResidency::ensure_capacity(
        int slot, int cur_pos, int add_tokens, std::string & error) {
    if (slot < 0 || slot >= (int)slots_.size()) return false;
    SlotState & st = slots_[(size_t)slot];
    if (!st.active || add_tokens <= 0) return true;

    const int cur_chunk = cur_pos / cfg_.chunk_tokens;
    const int end_pos = cur_pos + add_tokens;
    const int new_chunks_needed =
        (end_pos + cfg_.chunk_tokens - 1) / cfg_.chunk_tokens -
        (cur_pos + cfg_.chunk_tokens - 1) / cfg_.chunk_tokens;
    if (new_chunks_needed <= 0) return true;

    const int sealed_chunks = cur_pos / cfg_.chunk_tokens;
    PagedKvSequenceSnapshot snap;
    if (pool_.sequence(st.handle, snap) != PagedKvStatus::Ok) {
        error = "ensure_capacity: stale sequence handle";
        return false;
    }

    const int need_evict = st.resident_chunks + new_chunks_needed - budget_chunks_;
    if (need_evict <= 0) return true;
    std::vector<uint8_t> picked_mask((size_t)std::max(0, sealed_chunks), 0);
    std::vector<int> victims;
    victims.reserve((size_t)need_evict);
    // Pick all victims upfront, then execute ONE batched D2H + GPU sync.
    const int tail_floor = std::max(0, cur_chunk - cfg_.tail_window_chunks);
    for (int k = 0; k < need_evict; ++k) {
        int best = -1;
        bool best_scored = false;
        float best_score = std::numeric_limits<float>::infinity();
        uint64_t best_use = std::numeric_limits<uint64_t>::max();
        const int lim = std::min(sealed_chunks, (int)st.chunks.size());
        for (int c = 0; c < lim; ++c) {
            if (picked_mask[(size_t)c]) continue;
            const Chunk & ch = st.chunks[(size_t)c];
            if (!ch.resident || ch.pinned || c == 0 || c < st.pinned_chunks || c >= tail_floor) continue;
            if (ch.has_score) {
                if (best < 0 || !best_scored || ch.score < best_score ||
                    (ch.score == best_score && ch.last_use < best_use)) {
                    best = c; best_scored = true; best_score = ch.score; best_use = ch.last_use;
                }
            } else if (!best_scored) {
                if (best < 0 || ch.last_use < best_use) {
                    best = c; best_use = ch.last_use;
                }
            }
        }
        if (best < 0) {
            error = "kvflash per-slot budget exhausted (no evictable sealed chunk)";
            return false;
        }
        picked_mask[(size_t)best] = 1;
        victims.push_back(best);
    }
    return evict_chunks(st, victims, snap.block_table, error);
}

bool KvSlotResidency::apply_recall(
        int slot, int r_pos, int prompt_end,
        const std::vector<float> & chunk_scores, int recall_unit,
        std::vector<int> * picked_out, std::string & error) {
    if (picked_out) picked_out->clear();
    if (slot < 0 || slot >= (int)slots_.size()) return false;
    SlotState & st = slots_[(size_t)slot];
    if (!st.active) return false;

    const int sealed = r_pos / cfg_.chunk_tokens;
    const int eff_pin_res = std::min(pin_reserve_chunks_, st.pinned_chunks);
    const KvFlashRecallGate gate = kvflash_recall_gate(
        /*turn_found=*/r_pos > 0, r_pos, prompt_end, cfg_.chunk_tokens,
        budget_chunks_, eff_pin_res, cfg_.recall_chunks);
    if (gate.skip != KvFlashRecallSkip::None) return true;

    const int score_n = std::min(sealed, (int)std::min(chunk_scores.size(), st.chunks.size()));
    for (int c = 0; c < score_n; ++c) {
        st.chunks[(size_t)c].score = chunk_scores[(size_t)c];
        st.chunks[(size_t)c].has_score = true;
    }

    KvFlashRecallIn rin;
    rin.n_pool_blocks = budget_chunks_;
    rin.cur_chunk = sealed;
    rin.recall_chunks = cfg_.recall_chunks;
    rin.unit = recall_unit > 0 ? recall_unit : kKvFlashRecallUnit;
    rin.reserve = gate.reserve;
    rin.pin_reserve_chunks = eff_pin_res;
    rin.pinned_chunks = st.pinned_chunks;
    // Close the [64, 640) dead zone when pinned_chunks < kKvFlashRecallUnit and unit < 10:
    // chunks [max(1, pinned_chunks)..first_window) are evictable, so allow recall on them.
    const int first_chunk = std::max(1, st.pinned_chunks);
    const int first_window = kvflash_recall_first_window(rin);
    const int k_units = rin.unit > 0 ? rin.recall_chunks / rin.unit : 0;
    const std::vector<int> picked = kvflash_recall_pick(
        chunk_scores, first_window, k_units, rin.unit, first_chunk);
    if (picked_out) *picked_out = picked;

    std::vector<uint8_t> keep((size_t)sealed, 0);
    for (int c = 0; c < std::min(sealed, std::max(1, st.pinned_chunks)); ++c) keep[(size_t)c] = 1;
    for (int c = std::max(0, first_window); c < sealed; ++c) keep[(size_t)c] = 1;
    for (int b : picked) {
        const int c0 = b * rin.unit;
        const int c1 = std::min(sealed, c0 + rin.unit);
        for (int c = std::max(0, c0); c < c1; ++c) keep[(size_t)c] = 1;
    }

    PagedKvSequenceSnapshot snap;
    if (pool_.sequence(st.handle, snap) != PagedKvStatus::Ok) {
        error = "apply_recall: stale sequence handle";
        return false;
    }

    std::vector<int> to_evict;
    to_evict.reserve((size_t)budget_chunks_);
    for (int c = 0; c < sealed && c < (int)st.chunks.size(); ++c) {
        if (st.chunks[(size_t)c].resident && !keep[(size_t)c]) {
            to_evict.push_back(c);
        }
    }
    if (!evict_chunks(st, to_evict, snap.block_table, error)) return false;

    std::vector<int> to_restore;
    const int room = std::max(0, budget_chunks_ - gate.reserve - st.resident_chunks);
    to_restore.reserve((size_t)room);
    for (int c = 0; c < sealed && c < (int)st.chunks.size() && (int)to_restore.size() < room; ++c) {
        if (keep[(size_t)c] && !st.chunks[(size_t)c].resident) {
            to_restore.push_back(c);
        }
    }
    return restore_chunks(st, to_restore, error);
}

bool KvSlotResidency::build_active_block_table(
        int slot, int cur_pos, std::vector<int32_t> & out_blocks,
        int & out_res_tokens, int & out_prefix_res_tokens,
        int chunk_start_pos) const {
    out_blocks.clear();
    out_res_tokens = 0;
    out_prefix_res_tokens = 0;
    if (slot < 0 || slot >= (int)slots_.size()) return false;
    const SlotState & st = slots_[(size_t)slot];
    if (!st.active) return false;

    PagedKvSequenceSnapshot snap;
    if (pool_.sequence(st.handle, snap) != PagedKvStatus::Ok) return false;

    const int bs = (int)pool_.block_size();
    const int total_logical_blocks = (cur_pos + bs - 1) / bs;
    if (total_logical_blocks > (int)snap.block_table.size()) return false;

    const int split_pos = (chunk_start_pos >= 0 && chunk_start_pos <= cur_pos)
        ? chunk_start_pos : cur_pos;
    const int split_block = split_pos / bs;
    out_blocks.reserve((size_t)budget_chunks_ * (size_t)blocks_per_chunk_ + 8);
    for (int lb = 0; lb < split_block; ++lb) {
        const uint32_t pb = snap.block_table[(size_t)lb];
        if (pb == PagedKvPool::kEvictedBlock) continue;
        out_blocks.push_back((int32_t)pb);
    }
    const int prefix_full_blocks = (int)out_blocks.size();
    out_prefix_res_tokens = prefix_full_blocks * bs + (split_pos % bs);
    for (int lb = split_block; lb < total_logical_blocks; ++lb) {
        const uint32_t pb = snap.block_table[(size_t)lb];
        if (pb == PagedKvPool::kEvictedBlock) return false;
        out_blocks.push_back((int32_t)pb);
    }
    const int tail_tokens = cur_pos - split_block * bs;
    out_res_tokens = prefix_full_blocks * bs + tail_tokens;
    return true;
}

bool KvSlotResidency::truncate(int slot, int new_pos) {
    if (slot < 0 || slot >= (int)slots_.size() || new_pos < 0) return false;
    SlotState & st = slots_[(size_t)slot];
    if (!st.active) return false;
    if (pool_.truncate_logical_blocks(st.handle, (uint32_t)new_pos) != PagedKvStatus::Ok) {
        return false;
    }
    const int keep_chunks = (new_pos + cfg_.chunk_tokens - 1) / cfg_.chunk_tokens;
    while ((int)st.chunks.size() > keep_chunks) {
        Chunk & ch = st.chunks.back();
        if (ch.resident && st.resident_chunks > 0) --st.resident_chunks;
        if (ch.host) {
            host_bytes_used_ -= std::min(host_bytes_used_, ch.host->size());
        }
        st.chunks.pop_back();
    }
    if (new_pos % cfg_.chunk_tokens != 0 && !st.chunks.empty()) {
        st.chunks.back().dirty = true;
    }
    ++st.epoch;
    return true;
}

uint64_t KvSlotResidency::epoch(int slot) const {
    return (slot >= 0 && slot < (int)slots_.size()) ? slots_[(size_t)slot].epoch : 0;
}
int KvSlotResidency::resident_chunks(int slot) const {
    return (slot >= 0 && slot < (int)slots_.size()) ? slots_[(size_t)slot].resident_chunks : 0;
}
int KvSlotResidency::pinned_chunks(int slot) const {
    return (slot >= 0 && slot < (int)slots_.size()) ? slots_[(size_t)slot].pinned_chunks : 0;
}
int KvSlotResidency::effective_pin_reserve_chunks(int slot) const {
    return (slot >= 0 && slot < (int)slots_.size())
        ? std::min(pin_reserve_chunks_, slots_[(size_t)slot].pinned_chunks) : 0;
}
int KvSlotResidency::total_chunks(int slot) const {
    return (slot >= 0 && slot < (int)slots_.size()) ? (int)slots_[(size_t)slot].chunks.size() : 0;
}
bool KvSlotResidency::is_chunk_resident(int slot, int chunk) const {
    if (slot < 0 || slot >= (int)slots_.size()) return false;
    const auto & cs = slots_[(size_t)slot].chunks;
    return chunk >= 0 && chunk < (int)cs.size() && cs[(size_t)chunk].resident;
}
const uint8_t * KvSlotResidency::host_chunk_data(int slot, int chunk) const {
    if (slot < 0 || slot >= (int)slots_.size()) return nullptr;
    const auto & cs = slots_[(size_t)slot].chunks;
    if (chunk < 0 || chunk >= (int)cs.size() || !cs[(size_t)chunk].host || cs[(size_t)chunk].host->empty()) return nullptr;
    return cs[(size_t)chunk].host->data();
}

bool KvSlotResidency::capture_snapshot(
        int slot, int tokens, Snapshot & out, std::string & error) {
    out = Snapshot{};
    if (slot < 0 || slot >= (int)slots_.size() || tokens <= 0) {
        error = "capture_snapshot: invalid slot or token count";
        return false;
    }
    SlotState & st = slots_[(size_t)slot];
    if (!st.active) {
        error = "capture_snapshot: slot inactive";
        return false;
    }
    PagedKvSequenceSnapshot snap;
    if (pool_.sequence(st.handle, snap) != PagedKvStatus::Ok ||
        snap.kv_seq_len < (uint32_t)tokens) {
        error = "capture_snapshot: stale handle or short sequence";
        return false;
    }
    const int bs = (int)pool_.block_size();
    const int total_blocks = (tokens + bs - 1) / bs;
    const int n_chunks = (tokens + cfg_.chunk_tokens - 1) / cfg_.chunk_tokens;
    if (n_chunks > (int)st.chunks.size() || total_blocks > (int)snap.block_table.size()) {
        error = "capture_snapshot: chunk metadata shorter than tokens";
        return false;
    }

    out.tokens = tokens;
    out.pinned_chunks = st.pinned_chunks;
    out.chunks.resize((size_t)n_chunks);
    out.resident.assign((size_t)n_chunks, 0);
    out.scores.assign((size_t)n_chunks, -1e30f);
    out.has_score.assign((size_t)n_chunks, 0);

    bool need_xfer = false;
    struct PendingD2h {
        int c;
        int n_lb;
        std::vector<uint32_t> blocks;
        std::shared_ptr<std::vector<uint8_t>> dst;
        bool cache_on_slot;
    };
    std::vector<PendingD2h> d2h_jobs;
    d2h_jobs.reserve((size_t)budget_chunks_ + 2);

    for (int c = 0; c < n_chunks; ++c) {
        Chunk & ch = st.chunks[(size_t)c];
        const int lb0 = c * blocks_per_chunk_;
        const int n_lb = std::min(blocks_per_chunk_, total_blocks - lb0);
        const bool full_chunk = (n_lb == blocks_per_chunk_) &&
            ((c + 1) * cfg_.chunk_tokens <= tokens);
        out.resident[(size_t)c] = ch.resident ? 1 : 0;
        out.scores[(size_t)c] = ch.score;
        out.has_score[(size_t)c] = ch.has_score ? 1 : 0;

        if (!ch.resident) {
            if (ch.host && ch.host->size() == chunk_bytes_) {
                out.chunks[(size_t)c] = ch.host;
            } else if (all_pins_still_live(ch.shared_pins) && chunk_bytes_ > 0 && backend_) {
                auto buf = std::make_shared<std::vector<uint8_t>>(chunk_bytes_, 0);
                d2h_jobs.push_back({c, blocks_per_chunk_, ch.shared_pins, buf, true});
                out.chunks[(size_t)c] = buf;
                need_xfer = true;
            } else if (chunk_bytes_ > 0 && backend_) {
                error = "capture_snapshot: evicted chunk missing host payload";
                return false;
            }
        } else {
            if (!ch.dirty && full_chunk && ch.host && ch.host->size() == chunk_bytes_) {
                out.chunks[(size_t)c] = ch.host;
            } else if (chunk_bytes_ > 0 && backend_) {
                std::vector<uint32_t> blks(
                    snap.block_table.begin() + lb0,
                    snap.block_table.begin() + lb0 + n_lb);
                for (uint32_t pb : blks) {
                    if (pb == PagedKvPool::kEvictedBlock) {
                        error = "capture_snapshot: resident chunk contains evicted block";
                        return false;
                    }
                }
                auto buf = std::make_shared<std::vector<uint8_t>>(chunk_bytes_, 0);
                d2h_jobs.push_back({c, n_lb, std::move(blks), buf, full_chunk});
                out.chunks[(size_t)c] = buf;
                need_xfer = true;
            }
        }
    }

    if (need_xfer) {
        auto xfer = transfer_.scope(backend_);
        for (const auto & job : d2h_jobs) {
            size_t dst_off = 0;
            for (const PagedKvTensor & t : tensors_) {
                if (!t.tensor || t.block_bytes == 0) continue;
                for (size_t i = 0; i < (size_t)job.n_lb;) {
                    const size_t run = contiguous_block_run(job.blocks, i, (size_t)job.n_lb);
                    const size_t nbytes = run * t.block_bytes;
                    const size_t dev_off = t.offset + (size_t)job.blocks[i] * t.block_bytes;
                    xfer.d2h(t.tensor, dev_off, job.dst->data() + dst_off + i * t.block_bytes, nbytes);
                    i += run;
                }
                dst_off += (size_t)blocks_per_chunk_ * t.block_bytes;
            }
        }
        xfer.finish();
        for (const auto & job : d2h_jobs) {
            out.delta_bytes += job.dst->size();
            if (job.cache_on_slot) {
                Chunk & ch = st.chunks[(size_t)job.c];
                if (!ch.host) host_bytes_used_ += job.dst->size();
                ch.host = job.dst;
                ch.dirty = false;
            }
        }
    }
    return true;
}

void KvSlotResidency::build_restore_resident_mask(
        const Snapshot & snap, int preamble_tokens,
        std::vector<uint8_t> & out_block_mask,
        std::vector<uint8_t> & out_chunk_mask) const {
    const int bs = (int)pool_.block_size();
    const int total_blocks = (snap.tokens + bs - 1) / bs;
    const int n_chunks = (int)snap.chunks.size();
    out_chunk_mask.assign((size_t)n_chunks, 0);
    out_block_mask.assign((size_t)total_blocks, 0);
    if (n_chunks == 0) return;

    const int pin_rows = kvflash_preamble_pin_rows(
        preamble_tokens, pin_reserve_chunks_ * cfg_.chunk_tokens);
    const int max_pin = std::max(0, budget_chunks_ - cfg_.tail_window_chunks - 2);
    const int n_pin = std::min(
        n_chunks, std::max(1, std::min(max_pin, (pin_rows + cfg_.chunk_tokens - 1) / cfg_.chunk_tokens)));

    int chosen = 0;
    auto pick = [&](int c) {
        if (c < 0 || c >= n_chunks || out_chunk_mask[(size_t)c] || chosen >= budget_chunks_) return;
        out_chunk_mask[(size_t)c] = 1;
        ++chosen;
    };
    // Always keep chunk 0 + pinned preamble + last chunk (may be partial tail)
    for (int c = 0; c < n_pin; ++c) pick(c);
    pick(n_chunks - 1);
    // Keep newest previously-resident chunks up to budget_chunks_
    for (int c = n_chunks - 2; c >= n_pin && chosen < budget_chunks_; --c) {
        if (c < (int)snap.resident.size() && snap.resident[(size_t)c]) pick(c);
    }
    for (int c = 0; c < n_chunks; ++c) {
        if (!out_chunk_mask[(size_t)c]) continue;
        const int lb0 = c * blocks_per_chunk_;
        const int lb1 = std::min(total_blocks, lb0 + blocks_per_chunk_);
        for (int lb = lb0; lb < lb1; ++lb) out_block_mask[(size_t)lb] = 1;
    }
}

bool KvSlotResidency::restore_snapshot(
        int slot, PagedKvSequenceHandle handle, const Snapshot & snap,
        int preamble_tokens, const std::vector<uint8_t> & chunk_mask,
        const std::vector<uint32_t> & block_table, std::string & error) {
    if (slot < 0 || slot >= (int)slots_.size() || !snap.valid()) {
        error = "restore_snapshot: invalid slot or snapshot";
        return false;
    }
    clear(slot);
    SlotState & st = slots_[(size_t)slot];
    st.active = true;
    st.handle = handle;
    st.epoch = 1;
    const int pin_rows = kvflash_preamble_pin_rows(
        preamble_tokens, pin_reserve_chunks_ * cfg_.chunk_tokens);
    const int max_pin = std::max(0, budget_chunks_ - cfg_.tail_window_chunks - 2);
    st.pinned_chunks = std::min(
        max_pin, (pin_rows + cfg_.chunk_tokens - 1) / cfg_.chunk_tokens);

    const int bs = (int)pool_.block_size();
    const int total_blocks = (snap.tokens + bs - 1) / bs;
    const int n_chunks = (int)snap.chunks.size();
    st.chunks.resize((size_t)n_chunks);

    bool need_h2d = false;
    for (int c = 0; c < n_chunks; ++c) {
        Chunk & ch = st.chunks[(size_t)c];
        const bool is_res = (c < (int)chunk_mask.size() && chunk_mask[(size_t)c] != 0);
        ch.resident = is_res;
        ch.pinned = (c == 0) || (c < st.pinned_chunks);
        ch.last_use = ++st.clock;
        if (c < (int)snap.scores.size() && c < (int)snap.has_score.size()) {
            ch.score = snap.scores[(size_t)c];
            ch.has_score = snap.has_score[(size_t)c] != 0;
        }
        const bool is_partial_last = (c == n_chunks - 1) && (snap.tokens % cfg_.chunk_tokens != 0);
        if (!is_partial_last && snap.chunks[(size_t)c]) {
            ch.host = snap.chunks[(size_t)c];
            host_bytes_used_ += ch.host->size();
            ch.dirty = false;
        } else {
            ch.dirty = is_res;
        }
        if (is_res) {
            ++st.resident_chunks;
            if (chunk_bytes_ > 0 && backend_) need_h2d = true;
        }
    }

    if (need_h2d) {
        auto xfer = transfer_.scope(backend_);
        for (int c = 0; c < n_chunks; ++c) {
            if (!st.chunks[(size_t)c].resident) continue;
            const auto & src = snap.chunks[(size_t)c];
            if (!src || src->size() != chunk_bytes_) {
                error = "restore_snapshot: missing chunk payload";
                return false;
            }
            const int lb0 = c * blocks_per_chunk_;
            const int n_lb = std::min(blocks_per_chunk_, total_blocks - lb0);
            std::vector<uint32_t> blks(
                block_table.begin() + lb0,
                block_table.begin() + lb0 + n_lb);
            size_t src_off = 0;
            for (const PagedKvTensor & t : tensors_) {
                if (!t.tensor || t.block_bytes == 0) continue;
                for (size_t i = 0; i < (size_t)n_lb;) {
                    const size_t run = contiguous_block_run(blks, i, (size_t)n_lb);
                    const size_t nbytes = run * t.block_bytes;
                    const size_t dev_off = t.offset + (size_t)blks[i] * t.block_bytes;
                    xfer.h2d(t.tensor, dev_off, src->data() + src_off + i * t.block_bytes, nbytes);
                    i += run;
                }
                src_off += (size_t)blocks_per_chunk_ * t.block_bytes;
            }
        }
        xfer.finish();
    }
    ++st.epoch;
    return true;
}
const KvSlotResidency::Stats & KvSlotResidency::stats(int slot) const {
    static const Stats kEmpty{};
    return (slot >= 0 && slot < (int)slots_.size()) ? slots_[(size_t)slot].stats : kEmpty;
}

}  // namespace luce::common
