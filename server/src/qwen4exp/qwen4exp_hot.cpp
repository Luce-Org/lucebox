#include "qwen4exp_hot.h"
#include "qwen4exp_internal.h"

#include "common/moe_hybrid_routing_stats.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace luce::common {

// Counts are read and swaps planned every kTickGraphs split decode or verify graphs: a read costs a small device
// copy and a host scan of every layer's counters, so reading more often follows a new request sooner for more host
// time per graph.
static constexpr int kTickGraphs = 16;
// A layer's scores lose this share of their weight at each read, so an expert's score is about its last
// kTickGraphs / (1 - kDecay) graphs of picks (~320 graphs): a longer memory holds a steadier set and follows a change of
// request more slowly.
static constexpr float kDecay = 0.95f;
// A swap moves in an expert picked more than kSwapRatio times as often as the one it evicts, plus kSwapMargin picks
// (a copy costs ~1 ms of the link; a swap back and forth on noise would waste it): lower values follow the picks more
// closely and spend more of the link on copies.
static constexpr float kSwapRatio = 1.5f;
static constexpr float kSwapMargin = 4.0f;
// At most this many experts are being copied at once (~3 MB each over the R9700's PCIe link): more fills the slots
// sooner, and the copies share the link with the decode steps' transfers.
static constexpr int kMaxCopies = 32;
// A start-up placement (--expert-placement) counts as this many picks per layer, so a request's own picks outweigh it
// after a few hundred tokens: a larger prior keeps the profiled set longer against the served picks.
static constexpr float kPriorPicks = 2000.0f;
// A prompt chunk's picks count as at most this many per layer (a 4096-row chunk makes ~41k), so the reply's own
// picks take over within a few hundred tokens: a larger share starts the reply with the prompt's experts and holds
// them longer.
static constexpr float kPromptPicks = 2000.0f;
// Copies a prompt chunk may start (it runs for a second or more; the decode steps' hops are not waiting on the link):
// more brings the prompt's experts in before the reply, at the cost of copies the reply may not use.
static constexpr int kMaxPromptCopies = 128;

static bool split_layer(const Qwen4ExpWeights & w, int il) {
    const Qwen4ExpLayer & L = w.layers[(size_t) il];
    return w.expert_buf && L.ffn_gate_exps && L.ffn_gate_exps->buffer == w.expert_buf && L.ffn_up_exps &&
           L.ffn_down_exps;
}

Qwen4ExpHotSet::~Qwen4ExpHotSet() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    for (Batch & b : todo_) if (b.ready) ggml_backend_event_free(b.ready);
    // The stage's slices must outlive the table writes that read them.
    if (lut_pending_) ggml_backend_event_synchronize(lut_written_);
    if (lut_written_) ggml_backend_event_free(lut_written_);
    if (staging_buf_) ggml_backend_buffer_free(staging_buf_);
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
}

bool Qwen4ExpHotSet::init(ggml_backend_t target, const Qwen4ExpWeights & w, uint64_t budget,
                          const MoeHybridRoutingStats * stats) {
    n_layer_ = w.n_layer;
    n_expert_ = w.n_expert;
    uint64_t per_slot = 0;   // one expert in every split layer
    size_t largest = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        if (!split_layer(w, il)) continue;
        const Qwen4ExpLayer & L = w.layers[(size_t) il];
        per_slot += L.ffn_gate_exps->nb[2] + L.ffn_up_exps->nb[2] + L.ffn_down_exps->nb[2];
        largest = std::max({largest, L.ffn_gate_exps->nb[2], L.ffn_up_exps->nb[2], L.ffn_down_exps->nb[2]});
    }
    if (per_slot == 0) return false;
    slots = (int) std::min<uint64_t>(budget / per_slot, (uint64_t) w.n_expert);
    if (slots < 1) return false;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (5 * (size_t) w.n_layer + 2);
    ip.no_alloc = true;
    ctx_ = ggml_init(ip);
    if (!ctx_) return false;
    gate.assign((size_t) w.n_layer, nullptr);
    up.assign((size_t) w.n_layer, nullptr);
    down.assign((size_t) w.n_layer, nullptr);
    lut.assign((size_t) w.n_layer, nullptr);
    hist.assign((size_t) w.n_layer, nullptr);
    lut_all_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_I32, w.n_expert, w.n_layer);
    hist_all_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_I32, w.n_expert, w.n_layer);
    for (int il = 0; il < w.n_layer; ++il) {
        if (!split_layer(w, il)) continue;
        const Qwen4ExpLayer & L = w.layers[(size_t) il];
        gate[(size_t) il] = ggml_new_tensor_3d(ctx_, L.ffn_gate_exps->type, L.ffn_gate_exps->ne[0], L.ffn_gate_exps->ne[1], slots);
        up[(size_t) il] = ggml_new_tensor_3d(ctx_, L.ffn_up_exps->type, L.ffn_up_exps->ne[0], L.ffn_up_exps->ne[1], slots);
        down[(size_t) il] = ggml_new_tensor_3d(ctx_, L.ffn_down_exps->type, L.ffn_down_exps->ne[0], L.ffn_down_exps->ne[1], slots);
        lut[(size_t) il] = ggml_view_1d(ctx_, lut_all_, w.n_expert, (size_t) il * lut_all_->nb[1]);
        hist[(size_t) il] = ggml_view_1d(ctx_, hist_all_, w.n_expert, (size_t) il * hist_all_->nb[1]);
    }
    buf_ = ggml_backend_alloc_ctx_tensors(ctx_, target);
    if (!buf_) return false;
    // ggml_backend_alloc_ctx_tensors has set up the lut/hist views.
    host_lut_.assign((size_t) w.n_layer * w.n_expert, -1);
    slot_expert_.assign((size_t) w.n_layer * slots, -1);
    filling_.assign((size_t) w.n_layer * slots, 0);
    incoming_.assign((size_t) w.n_layer * w.n_expert, 0);
    seen_.assign((size_t) w.n_layer * w.n_expert, 0);
    read_.assign(seen_.size(), 0);
    score_.assign(seen_.size(), 0.0f);
    initial.assign((size_t) w.n_layer, {});
    ggml_backend_buffer_clear(buf_, 0);

    staging_buf_ = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), largest);
    if (!staging_buf_) return false;
    staging_bytes_ = largest;
    // A tick writes at most every slot's table entry twice (an eviction and a finished copy).
    lut_written_ = ggml_backend_event_new(ggml_backend_get_device(target));
    if (!lut_written_ || !lut_stage_.reserve(target, 2 * (size_t) w.n_layer * slots * 64)) return false;
    w_ = &w;

    // The start-up placement: each layer's most-picked experts, copied before the first request.
    double share = 0.0;
    if (stats) {
        uint64_t hot_routes = 0, routes = 0;
        for (int il = 0; il < w.n_layer; ++il) {
            if (!gate[(size_t) il]) continue;
            const uint64_t total = std::max<uint64_t>(1, stats->layer_totals.empty() ? 0 : stats->layer_totals[(size_t) il]);
            for (int e = 0; e < w.n_expert; ++e) {
                score_[(size_t) il * w.n_expert + e] = kPriorPicks * (float) stats->count(il, e) / (float) total;
                routes += stats->count(il, e);
            }
            const std::vector<int> top = stats->hot_experts(il, slots);
            for (int s = 0; s < (int) top.size(); ++s) {
                copy_expert(w, {il, s, top[(size_t) s]}, (uint8_t *) ggml_backend_buffer_get_base(staging_buf_));
                slot_expert_[(size_t) il * slots + s] = top[(size_t) s];
                host_lut_[(size_t) il * w.n_expert + top[(size_t) s]] = s;
                hot_routes += stats->count(il, top[(size_t) s]);
                initial[(size_t) il].push_back(top[(size_t) s]);
            }
        }
        share = routes ? (double) hot_routes / (double) routes : 0.0;
    }
    ggml_backend_tensor_set(lut_all_, host_lut_.data(), 0, host_lut_.size() * sizeof(int32_t));
    // Lanes for 1.5x a token's mean hot picks (a lane's output crosses the link whether a pick used it or not; a pick
    // past the token's last lane runs on the expert device). Without a start-up profile: half the picks.
    lanes = stats ? (int) std::ceil(1.5 * share * w.n_expert_used) : w.n_expert_used / 2;
    lanes = std::clamp(lanes, 1, w.n_expert_used);
    std::fprintf(stderr, "[qwen4exp] hot experts: %d slots per layer (%.2f GiB) on the target, %d lanes, %s\n", slots,
                 (double) per_slot * slots / (1ull << 30), lanes,
                 stats ? "start-up placement, then the served picks" : "filled from the served picks");
    thread_ = std::thread([this] { copier(); });
    return true;
}

void Qwen4ExpHotSet::copy_expert(const Qwen4ExpWeights & w, const Job & j, uint8_t * staging) {
    const Qwen4ExpLayer & L = w.layers[(size_t) j.layer];
    const ggml_tensor * src[3] = {L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps};
    ggml_tensor * dst[3] = {gate[(size_t) j.layer], up[(size_t) j.layer], down[(size_t) j.layer]};
    for (int i = 0; i < 3; ++i) {
        const size_t bytes = src[i]->nb[2];
        ggml_backend_tensor_get(src[i], staging, (size_t) j.expert * bytes, bytes);
        ggml_backend_tensor_set(dst[i], staging, (size_t) j.slot * dst[i]->nb[2], bytes);
    }
}

// The background copies: a batch waits until the graphs that could still read its slots are done (the event follows
// the table writes that stopped naming them), then copies each expert from the expert device into its slot.
void Qwen4ExpHotSet::copier() {
    uint8_t * staging = (uint8_t *) ggml_backend_buffer_get_base(staging_buf_);
    for (;;) {
        Batch b;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] { return stop_ || !todo_.empty(); });
            if (stop_) return;
            b = std::move(todo_.front());
            todo_.pop_front();
        }
        ggml_backend_event_synchronize(b.ready);
        ggml_backend_event_free(b.ready);
        for (const Job & j : b.jobs) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                if (stop_) return;
            }
            copy_expert(*w_, j, staging);
            std::lock_guard<std::mutex> lock(mu_);
            done_.push_back(j);
        }
    }
}

void Qwen4ExpHotSet::write_lut(ggml_backend_t target, size_t e) {
    lut_stage_.set(target, lut_all_, &host_lut_[e], e * sizeof(int32_t), sizeof(int32_t));
    lut_writes_ = true;
}

void Qwen4ExpHotSet::lut_queued(ggml_backend_t target) {
    if (!lut_writes_) return;
    ggml_backend_event_record(lut_written_, target);
    lut_writes_ = false;
    lut_pending_ = true;
}

void Qwen4ExpHotSet::tick(ggml_backend_t target, const Qwen4ExpWeights & w, int prompt_rows) {
    // The previous tick's table writes ran (they were queued before the graphs since, so this rarely waits): their
    // slices are free again.
    if (lut_pending_) {
        ggml_backend_event_synchronize(lut_written_);
        lut_pending_ = false;
    }
    lut_stage_.reset();
    // Finished copies: their experts' table entries go live with the next graph.
    std::vector<Job> done;
    {
        std::lock_guard<std::mutex> lock(mu_);
        done.swap(done_);
    }
    for (const Job & j : done) {
        const size_t e = (size_t) j.layer * n_expert_ + j.expert;
        host_lut_[e] = j.slot;
        slot_expert_[(size_t) j.layer * slots + j.slot] = j.expert;
        filling_[(size_t) j.layer * slots + j.slot] = 0;
        incoming_[e] = 0;
        write_lut(target, e);
    }
    lut_queued(target);
    // The counts since the last read: after a prompt chunk (rows_ > 0) they are read at once, before the next graph;
    // otherwise every kTickGraphs decode or verify graphs.
    const int counted_rows = rows_;
    rows_ = prompt_rows;
    if (counted_rows == 0 && (prompt_rows > 0 || ++graphs_ % kTickGraphs != 0)) return;
    const float scale = counted_rows > 0
        ? std::min(1.0f, kPromptPicks / ((float) counted_rows * (float) w.n_expert_used)) : 1.0f;
    const int max_copies = counted_rows > 0 ? kMaxPromptCopies : kMaxCopies;

    int copying = 0;
    for (uint8_t f : filling_) copying += f;
    // The counters as the graphs left them (a graph still running may add picks after this read: they count next
    // time, or are lost when the read races them; the scores are a guide, not a ledger).
    ggml_backend_tensor_get(hist_all_, read_.data(), 0, read_.size() * sizeof(uint32_t));
    struct Swap {
        float gain;
        int layer, expert, slot;
    };
    std::vector<Swap> swaps;
    std::vector<int> order((size_t) n_expert_);
    for (int il = 0; il < n_layer_; ++il) {
        if (!gate[(size_t) il]) continue;
        float * sc = &score_[(size_t) il * n_expert_];
        for (int e = 0; e < n_expert_; ++e) {
            const size_t i = (size_t) il * n_expert_ + e;
            sc[e] = sc[e] * kDecay + scale * (float) (read_[i] - seen_[i]);
            seen_[i] = read_[i];
        }
        if (copying >= max_copies) continue;
        // The layer's best experts without a slot against its worst slots (an empty slot scores -1).
        for (int e = 0; e < n_expert_; ++e) order[(size_t) e] = e;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return sc[a] > sc[b] || (sc[a] == sc[b] && a < b); });
        std::vector<std::pair<float, int>> victims;   // (score, slot)
        for (int s = 0; s < slots; ++s) {
            if (filling_[(size_t) il * slots + s]) continue;
            const int held = slot_expert_[(size_t) il * slots + s];
            victims.emplace_back(held < 0 ? -1.0f : sc[held], s);
        }
        std::sort(victims.begin(), victims.end());
        size_t v = 0;
        for (int e : order) {
            if (v == victims.size()) break;
            if (host_lut_[(size_t) il * n_expert_ + e] >= 0 || incoming_[(size_t) il * n_expert_ + e]) continue;
            const float vs = victims[v].first;
            if (sc[e] <= kSwapRatio * std::max(vs, 0.0f) + kSwapMargin) break;
            swaps.push_back({sc[e] - std::max(vs, 0.0f), il, e, victims[v].second});
            ++v;
        }
    }
    if (swaps.empty() || copying >= max_copies) return;
    std::sort(swaps.begin(), swaps.end(), [](const Swap & a, const Swap & b) { return a.gain > b.gain; });
    swaps.resize(std::min<size_t>(swaps.size(), (size_t) (max_copies - copying)));
    Batch b;
    for (const Swap & s : swaps) {
        const size_t slot = (size_t) s.layer * slots + s.slot;
        const int held = slot_expert_[slot];
        if (held >= 0) {   // the evicted expert runs on the expert device from the next graph on
            const size_t e = (size_t) s.layer * n_expert_ + held;
            host_lut_[e] = -1;
            write_lut(target, e);
        }
        slot_expert_[slot] = -1;
        filling_[slot] = 1;
        incoming_[(size_t) s.layer * n_expert_ + s.expert] = 1;
        b.jobs.push_back({s.layer, s.slot, s.expert});
    }
    lut_queued(target);
    b.ready = ggml_backend_event_new(ggml_backend_get_device(target));
    if (!b.ready) {   // no event: copy nothing, the slots stay empty until the next tick tries again
        for (const Job & j : b.jobs) {
            filling_[(size_t) j.layer * slots + j.slot] = 0;
            incoming_[(size_t) j.layer * n_expert_ + j.expert] = 0;
        }
        return;
    }
    ggml_backend_event_record(b.ready, target);
    {
        std::lock_guard<std::mutex> lock(mu_);
        todo_.push_back(std::move(b));
    }
    cv_.notify_one();
}

}  // namespace luce::common
