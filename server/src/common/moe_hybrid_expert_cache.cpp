// Device cache for streamed MoE experts — implementation.

#include "moe_hybrid_expert_cache.h"
#include "moe_hybrid_ffn_eval.h"
#include "gpu_runtime_compat.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace luce::common {

namespace {

using Clock = std::chrono::steady_clock;

uint64_t elapsed_us(Clock::time_point t0, Clock::time_point t1) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

// Room after each role's stack for the quantized row padding a GPU buffer
// reserves past the last row of a tensor.
constexpr size_t kStackTail = 64 * 1024;
constexpr size_t kAlign = 256;

size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Expert byte ranges of one layer in file order: gate (or fused gate_up),
// up, down. Unused roles have size 0.
struct ExpertRanges {
    size_t off[3] = {0, 0, 0};
    size_t size[3] = {0, 0, 0};
};

ExpertRanges expert_ranges(const LayerExpertRegions & r, int expert) {
    ExpertRanges out;
    const size_t e = (size_t) expert;
    if (r.fused_gate_up) {
        out.off[0] = r.gate_up_exps.offset + e * r.expert_bytes_gate_up;
        out.size[0] = r.expert_bytes_gate_up;
    } else {
        out.off[0] = r.gate_exps.offset + e * r.expert_bytes_gate;
        out.size[0] = r.expert_bytes_gate;
        out.off[1] = r.up_exps.offset + e * r.expert_bytes_up;
        out.size[1] = r.expert_bytes_up;
    }
    out.off[2] = r.down_exps.offset + e * r.expert_bytes_down;
    out.size[2] = r.expert_bytes_down;
    return out;
}

void advise_willneed(const void * map, size_t map_size, const ExpertRanges & ranges) {
#if !defined(_WIN32)
    static const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    for (int i = 0; i < 3; ++i) {
        if (ranges.size[i] == 0 || ranges.off[i] + ranges.size[i] > map_size) continue;
        const size_t start = ranges.off[i] / page * page;
        ::madvise(const_cast<uint8_t *>(static_cast<const uint8_t *>(map)) + start,
                  ranges.size[i] + (ranges.off[i] - start), MADV_WILLNEED);
    }
#else
    (void) map; (void) map_size; (void) ranges;
#endif
}

}  // namespace

struct MoeStreamedExpertCache::Loader {
    void * staging = nullptr;  // pinned, one slot
    cudaStream_t stream = nullptr;
};

void MoeStreamedExpertCache::Graph::free() {
    if (alloc) ggml_gallocr_free(alloc);
    if (ctx) ggml_free(ctx);
    *this = {};
}

MoeStreamedExpertCache::~MoeStreamedExpertCache() {
    destroy();
}

bool MoeStreamedExpertCache::init(const MoeHybridConfig & cfg,
                                  const std::vector<MoeLayerDesc> & descs,
                                  const MoeHybridStorage & storage,
                                  const MoeExpertCacheOptions & opts,
                                  std::string * err) {
    destroy();
    const auto fail = [&](const std::string & msg) {
        if (err) *err = msg;
        destroy();
        return false;
    };
    if (!storage.has_mmap() || storage.layer_regions.size() != storage.layers.size() ||
        descs.size() != storage.layers.size()) {
        return fail("streamed expert cache needs the model file mapping and per-layer regions");
    }
    cfg_ = cfg;
    storage_ = &storage;
    device_ = opts.device;

    // Slot geometry: the largest expert of each role over the streamed layers.
    // Layers with other weight types get their own view of the same slots.
    bool fused = false, any = false;
    for (size_t il = 0; il < storage.layers.size(); ++il) {
        if (storage.layers[il].n_streamed == 0) continue;
        const LayerExpertRegions & r = storage.layer_regions[il];
        const MoeLayerDesc & d = descs[il];
        if (any && r.fused_gate_up != fused) return fail("mixed fused and split gate/up layers");
        fused = r.fused_gate_up;
        any = true;
        for (const ggml_tensor * t : {d.ffn_gate_exps, d.ffn_up_exps, d.ffn_down_exps, d.ffn_gate_up_exps}) {
            if (t && (t->type == GGML_TYPE_Q3_1_ROCMFP3_MIX || t->type == GGML_TYPE_Q2_1_ROCMFP2_MIX)) {
                return fail("mixed-codebook expert types need per-expert decode tables");
            }
        }
        stride_gate_ = std::max(stride_gate_, fused ? r.expert_bytes_gate_up : r.expert_bytes_gate);
        stride_up_ = std::max(stride_up_, fused ? (size_t) 0 : r.expert_bytes_up);
        stride_down_ = std::max(stride_down_, r.expert_bytes_down);
    }
    if (!any) return fail("no streamed experts");
    slot_bytes_ = stride_gate_ + stride_up_ + stride_down_;

    size_t pool = opts.pool_bytes;
    if (pool == 0) {
        size_t free_b = 0, total_b = 0;
        ggml_backend_cuda_get_device_memory(device_, &free_b, &total_b);
        pool = free_b > opts.reserve_bytes ? free_b - opts.reserve_bytes : 0;
    }
    const size_t fixed = 3 * (kStackTail + kAlign);
    n_slots_ = pool > fixed ? (int) ((pool - fixed) / slot_bytes_) : 0;
    const int min_slots = std::max(16, 2 * cfg.n_expert_used);
    if (n_slots_ < min_slots) {
        return fail("device " + std::to_string(device_) + " has room for " +
                    std::to_string(n_slots_) + " expert slots, need at least " +
                    std::to_string(min_slots));
    }

    backend_ = ggml_backend_cuda_init(device_);
    if (!backend_) return fail("failed to create a backend on device " + std::to_string(device_));

    const size_t off_up = align_up((size_t) n_slots_ * stride_gate_ + kStackTail, kAlign);
    const size_t off_down = off_up + (stride_up_ ? align_up((size_t) n_slots_ * stride_up_ + kStackTail, kAlign) : 0);
    const size_t total = off_down + (size_t) n_slots_ * stride_down_ + kStackTail;
    pool_buf_ = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device_), total);
    if (!pool_buf_) return fail("failed to allocate " + std::to_string(total >> 20) + " MiB of expert slots");
    ggml_backend_buffer_set_usage(pool_buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(pool_buf_));
    base_gate_ = base;
    base_up_ = stride_up_ ? base + off_up : nullptr;
    base_down_ = base + off_down;

    // One stacked tensor per role and weight-type signature, strided by slot.
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4 * storage.layers.size() + 1024;
    ip.no_alloc = true;
    pool_ctx_ = ggml_init(ip);
    if (!pool_ctx_) return fail("ggml_init failed for the expert slot views");
    const auto stack = [&](const ggml_tensor * like, size_t stride, uint8_t * at) -> ggml_tensor * {
        ggml_tensor * t = ggml_new_tensor_3d(pool_ctx_, like->type, like->ne[0], like->ne[1], n_slots_);
        t->nb[2] = stride;
        t->nb[3] = stride * (size_t) n_slots_;
        if (ggml_backend_tensor_alloc(pool_buf_, t, at) != GGML_STATUS_SUCCESS) return nullptr;
        return t;
    };
    using Sig = std::tuple<int, int, int, int, float, float, float, float>;
    std::map<Sig, int> view_by_sig;
    view_of_layer_.assign(storage.layers.size(), -1);
    for (size_t il = 0; il < storage.layers.size(); ++il) {
        if (storage.layers[il].n_streamed == 0) continue;
        const MoeLayerDesc & d = descs[il];
        const auto type_of = [](const ggml_tensor * t) { return t ? (int) t->type : -1; };
        const Sig sig{type_of(d.ffn_gate_exps), type_of(d.ffn_up_exps), type_of(d.ffn_down_exps),
                      type_of(d.ffn_gate_up_exps), d.ffn_gate_exps_s, d.ffn_up_exps_s,
                      d.ffn_down_exps_s, d.ffn_gate_up_exps_s};
        auto it = view_by_sig.find(sig);
        if (it == view_by_sig.end()) {
            PoolView v;
            if (fused) {
                v.gate_up = stack(d.ffn_gate_up_exps, stride_gate_, base_gate_);
            } else {
                v.gate = stack(d.ffn_gate_exps, stride_gate_, base_gate_);
                v.up = stack(d.ffn_up_exps, stride_up_, base_up_);
            }
            v.down = stack(d.ffn_down_exps, stride_down_, base_down_);
            if (!(v.gate_up || (v.gate && v.up)) || !v.down) return fail("failed to place the expert slot views");
            it = view_by_sig.emplace(sig, (int) views_.size()).first;
            views_.push_back(v);
        }
        view_of_layer_[il] = it->second;
    }

    slots_.assign((size_t) n_slots_, Slot{});
    predicted_.assign(storage.layers.size(), {});
    const int n_loaders = std::max(1, opts.n_loaders);
    for (int i = 0; i < n_loaders; ++i) {
        auto * l = new Loader();
        loaders_.push_back(l);
        if (cudaMallocHost(&l->staging, slot_bytes_) != cudaSuccess) {
            l->staging = nullptr;
            return fail("failed to allocate pinned expert staging");
        }
    }
    for (Loader * l : loaders_) threads_.emplace_back(&MoeStreamedExpertCache::loader_main, this, l);
    if (!mailbox_.init(this, (int) storage.layers.size(), cfg.n_expert, cfg.n_expert_used, err)) {
        return fail(err ? *err : "failed to set up the streamed mailbox");
    }

    std::fprintf(stderr,
                 "[moe-stream] expert cache on device %d: %d slots x %.2f MiB = %.2f GiB, "
                 "%d loaders, %zu weight view(s)\n",
                 device_, n_slots_, slot_bytes_ / 1048576.0, total / 1073741824.0,
                 n_loaders, views_.size());
    return true;
}

void MoeStreamedExpertCache::destroy() {
    mailbox_.destroy();  // its resolver calls into the cache
    {
        std::lock_guard<std::mutex> lk(mu_);
        stopping_ = true;
    }
    cv_.notify_all();
    for (std::thread & t : threads_) t.join();
    threads_.clear();
    for (Loader * l : loaders_) {
        if (l->staging) cudaFreeHost(l->staging);
        delete l;
    }
    loaders_.clear();
    for (auto & kv : graphs_) kv.second.free();
    graphs_.clear();
    views_.clear();
    view_of_layer_.clear();
    if (pool_ctx_) ggml_free(pool_ctx_);
    pool_ctx_ = nullptr;
    if (pool_buf_) ggml_backend_buffer_free(pool_buf_);
    pool_buf_ = nullptr;
    if (backend_) ggml_backend_free(backend_);
    backend_ = nullptr;
    slots_.clear();
    slot_of_.clear();
    jobs_.clear();
    warm_.clear();
    warm_next_ = 0;
    warm_loading_ = 0;
    warm_loads_ = warm_bytes_ = 0;
    predicted_.clear();
    staged_.clear();
    staged_layer_ = -1;
    failed_.clear();
    n_slots_ = 0;
    slot_bytes_ = stride_gate_ = stride_up_ = stride_down_ = 0;
    base_gate_ = base_up_ = base_down_ = nullptr;
    storage_ = nullptr;
    device_ = -1;
    tick_ = 0;
    stats_ = {};
    stopping_ = false;
}

MoeStreamedExpertCache::Stats MoeStreamedExpertCache::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stats_;
}

void MoeStreamedExpertCache::reset_stats() {
    std::lock_guard<std::mutex> lk(mu_);
    stats_ = {};
}

// ── Loaders ─────────────────────────────────────────────────────────────

void MoeStreamedExpertCache::loader_main(Loader * self) {
    cudaSetDevice(device_);
    cudaStreamCreateWithFlags(&self->stream, cudaStreamNonBlocking);
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
        cv_.wait(lk, [&] { return stopping_ || !jobs_.empty() || warm_next_ < warm_.size(); });
        if (stopping_) break;
        // Demand and prefetch loads first; the warm start only fills idle time.
        const bool warm = jobs_.empty();
        int slot;
        const auto report_warm = [&] {
            if (warm_loading_ > 0 || warm_next_ < warm_.size() || warm_loads_ == 0) return;
            const double ms = elapsed_us(warm_t0_, Clock::now()) / 1000.0;
            std::fprintf(stderr,
                         "[moe-stream] warm start: %" PRIu64 " experts %.2f GiB in %.0f ms (%.2f GB/s)\n",
                         warm_loads_, warm_bytes_ / 1073741824.0, ms,
                         ms > 0 ? warm_bytes_ / (ms * 1e6) : 0.0);
            warm_loads_ = 0;
        };
        if (warm) {
            if ((slot = next_warm_locked()) < 0) {
                report_warm();
                continue;
            }
        } else {
            slot = jobs_.front();
            jobs_.pop_front();
        }
        lk.unlock();
        uint64_t read_us = 0, upload_us = 0;
        const bool ok = load_slot(*self, slot, &read_us, &upload_us);
        lk.lock();
        slots_[(size_t) slot].state = SlotState::Ready;
        if (warm) {
            --warm_loading_;
            report_warm();
        } else {
            stats_.read_us += read_us;
            stats_.upload_us += upload_us;
        }
        Slot & s = slots_[(size_t) slot];
        if (!ok && s.warm && s.pins == 0) {
            // Nobody needs it yet: a failed warm load just leaves the slot empty.
            slot_of_.erase(key(s.layer, s.expert));
            s = Slot{};
        } else if (!ok && failed_.empty()) {
            failed_ = "failed to load expert " + std::to_string(slots_[(size_t) slot].expert) +
                      " of layer " + std::to_string(slots_[(size_t) slot].layer);
        }
        cv_.notify_all();
    }
    lk.unlock();
    if (self->stream) cudaStreamDestroy(self->stream);
    self->stream = nullptr;
}

// A loading slot is never evicted, so its layer/expert are stable here.
bool MoeStreamedExpertCache::load_slot(Loader & loader, int slot,
                                       uint64_t * read_us, uint64_t * upload_us) {
    const Slot & s = slots_[(size_t) slot];
    const ExpertRanges r = expert_ranges(storage_->layer_regions[(size_t) s.layer], s.expert);
    const auto * file = static_cast<const uint8_t *>(storage_->mmap_data);
    auto * staging = static_cast<uint8_t *>(loader.staging);
    uint8_t * dst[3] = {
        base_gate_ + (size_t) slot * stride_gate_,
        base_up_ ? base_up_ + (size_t) slot * stride_up_ : nullptr,
        base_down_ + (size_t) slot * stride_down_,
    };
    const auto t0 = Clock::now();
    size_t at = 0;
    size_t staged_at[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        if (r.size[i] == 0) continue;
        if (r.off[i] + r.size[i] > storage_->mmap_size) return false;
        std::memcpy(staging + at, file + r.off[i], r.size[i]);
        staged_at[i] = at;
        at += r.size[i];
    }
    const auto t1 = Clock::now();
    for (int i = 0; i < 3; ++i) {
        if (r.size[i] == 0) continue;
        if (cudaMemcpyAsync(dst[i], staging + staged_at[i], r.size[i], cudaMemcpyHostToDevice,
                            loader.stream) != cudaSuccess) {
            return false;
        }
    }
    const bool ok = cudaStreamSynchronize(loader.stream) == cudaSuccess;
    *read_us = elapsed_us(t0, t1);
    *upload_us = elapsed_us(t1, Clock::now());
    return ok;
}

// ── Slots ───────────────────────────────────────────────────────────────

int MoeStreamedExpertCache::evict_locked() {
    int best = -1;
    for (int i = 0; i < n_slots_; ++i) {
        const Slot & s = slots_[(size_t) i];
        if (s.state == SlotState::Empty) return i;
        if (s.state != SlotState::Ready || s.pins > 0) continue;
        if (best < 0 || s.last_use < slots_[(size_t) best].last_use) best = i;
    }
    return best;
}

int MoeStreamedExpertCache::warm(const MoeHybridRoutingStats & usage) {
    if (!ready() || usage.n_layer != (int) storage_->layers.size() || usage.n_expert != cfg_.n_expert ||
        usage.empty()) {
        return 0;
    }
    struct Ranked { uint64_t count; int layer; int expert; };
    std::vector<Ranked> ranked;
    for (int il = 0; il < usage.n_layer; ++il) {
        const MoeHybridLayerStorage & st = storage_->layers[(size_t) il];
        if (st.n_streamed == 0) continue;
        for (int e = 0; e < usage.n_expert; ++e) {
            const uint64_t c = usage.count(il, e);
            if (c > 0 && st.is_streamed(e)) ranked.push_back({c, il, e});
        }
    }
    const size_t n = std::min(ranked.size(), (size_t) n_slots_);
    std::partial_sort(ranked.begin(), ranked.begin() + (ptrdiff_t) n, ranked.end(),
                      [](const Ranked & a, const Ranked & b) { return a.count > b.count; });
    uint64_t all = 0, kept = 0;
    for (size_t i = 0; i < ranked.size(); ++i) (i < n ? kept : all) += ranked[i].count;
    all += kept;
    std::fprintf(stderr,
                 "[moe-stream] warm start: %zu of %zu profiled streamed experts, %.1f%% of their routes\n",
                 n, ranked.size(), all ? 100.0 * (double) kept / (double) all : 0.0);
    std::lock_guard<std::mutex> lk(mu_);
    warm_.clear();
    for (size_t i = 0; i < n; ++i) warm_.push_back(key(ranked[i].layer, ranked[i].expert));
    warm_next_ = 0;
    warm_loads_ = warm_bytes_ = 0;
    warm_t0_ = Clock::now();
    // Warm slots count as used before anything a request touches, the most
    // used one last, so eviction takes the least used warm experts first.
    tick_ = std::max<uint64_t>(tick_, warm_.size());
    cv_.notify_all();
    return (int) warm_.size();
}

int MoeStreamedExpertCache::next_warm_locked() {
    while (warm_next_ < warm_.size()) {
        const size_t rank = warm_next_++;
        const uint64_t k = warm_[rank];
        if (slot_of_.count(k)) continue;  // a request loaded it already
        const int slot = evict_locked();
        if (slot < 0 || slots_[(size_t) slot].state != SlotState::Empty) {
            warm_next_ = warm_.size();  // the pool is full: never evict for it
            return -1;
        }
        Slot & s = slots_[(size_t) slot];
        s = Slot{};
        s.layer = (int32_t) (k >> 32);
        s.expert = (int32_t) (uint32_t) k;
        s.state = SlotState::Loading;
        s.warm = true;
        s.last_use = warm_.size() - rank;
        slot_of_[k] = slot;
        const ExpertRanges r = expert_ranges(storage_->layer_regions[(size_t) s.layer], s.expert);
        ++warm_loading_;
        ++warm_loads_;
        warm_bytes_ += r.size[0] + r.size[1] + r.size[2];
        return slot;
    }
    return -1;
}

int MoeStreamedExpertCache::lookup_or_load_locked(int layer, int expert, bool front, bool * hit) {
    const uint64_t k = key(layer, expert);
    auto it = slot_of_.find(k);
    if (it != slot_of_.end()) {
        const int slot = it->second;
        if (front && slots_[(size_t) slot].state == SlotState::Loading) {
            // Needed now: move a queued prefetch ahead of the others.
            auto q = std::find(jobs_.begin(), jobs_.end(), slot);
            if (q != jobs_.end()) {
                jobs_.erase(q);
                jobs_.push_front(slot);
            }
        }
        *hit = true;
        return slot;
    }
    *hit = false;
    const int slot = evict_locked();
    if (slot < 0) return -1;
    Slot & s = slots_[(size_t) slot];
    if (s.layer >= 0) slot_of_.erase(key(s.layer, s.expert));
    s = Slot{};
    s.layer = layer;
    s.expert = expert;
    s.state = SlotState::Loading;
    s.demand = front;
    s.prefetched = !front;
    s.last_use = ++tick_;
    slot_of_[k] = slot;
    const ExpertRanges r = expert_ranges(storage_->layer_regions[(size_t) layer], expert);
    advise_willneed(storage_->mmap_data, storage_->mmap_size, r);
    if (front) jobs_.push_front(slot); else jobs_.push_back(slot);
    stats_.loads += 1;
    stats_.bytes += r.size[0] + r.size[1] + r.size[2];
    if (!front) stats_.prefetched += 1;
    cv_.notify_all();
    return slot;
}

void MoeStreamedExpertCache::stage(int layer, const int32_t * selected, int n_routes) {
    if (!ready() || layer < 0 || (size_t) layer >= storage_->layers.size()) return;
    const MoeHybridLayerStorage & st = storage_->layers[(size_t) layer];
    std::lock_guard<std::mutex> lk(mu_);
    for (int slot : staged_) --slots_[(size_t) slot].pins;
    staged_.clear();
    staged_layer_ = layer;
    // At most half the slots, so an eval chunk always finds room next to them.
    const size_t max_staged = (size_t) std::max(1, n_slots_ / 2);
    for (int i = 0; i < n_routes && staged_.size() < max_staged; ++i) {
        if (!st.is_streamed(selected[i])) continue;
        bool hit = false;
        const int slot = lookup_or_load_locked(layer, selected[i], /*front=*/true, &hit);
        if (slot < 0) break;  // eval waits for room instead
        if (std::find(staged_.begin(), staged_.end(), slot) != staged_.end()) continue;
        ++slots_[(size_t) slot].pins;
        slots_[(size_t) slot].last_use = ++tick_;
        staged_.push_back(slot);
    }
}

void MoeStreamedExpertCache::prefetch(int layer, const int32_t * experts, int n) {
    if (!ready() || layer < 0 || (size_t) layer >= storage_->layers.size()) return;
    const MoeHybridLayerStorage & st = storage_->layers[(size_t) layer];
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<int32_t> & pred = predicted_[(size_t) layer];
    pred.clear();
    for (int i = 0; i < n; ++i) {
        if (!st.is_streamed(experts[i])) continue;
        pred.push_back(experts[i]);
        bool hit = false;
        const int slot = lookup_or_load_locked(layer, experts[i], /*front=*/false, &hit);
        if (slot < 0) break;
        if (hit) slots_[(size_t) slot].last_use = ++tick_;
    }
}

// ── Evaluation ──────────────────────────────────────────────────────────

bool MoeStreamedExpertCache::build_graph(Graph & g, int view, const MoeLayerDesc & desc,
                                         int n_routes, int n_tokens, std::string * err) {
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (64 + 32 * (size_t) n_tokens) +
                  ggml_graph_overhead_custom(2048, false);
    ip.no_alloc = true;
    g.ctx = ggml_init(ip);
    if (!g.ctx) {
        if (err) *err = "ggml_init failed for the streamed expert graph";
        return false;
    }
    g.inp = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, cfg_.n_embd, n_tokens);
    g.sel = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, n_routes, n_tokens);
    g.wts = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n_routes, n_tokens);
    ggml_set_input(g.inp);
    ggml_set_input(g.sel);
    ggml_set_input(g.wts);
    const PoolView & v = views_[(size_t) view];
    g.out = build_moe_routed_experts(g.ctx, cfg_, desc, v.gate, v.up, v.down, v.gate_up,
                                     g.inp, g.sel, g.wts, n_routes, n_tokens);
    if (!g.out) {
        if (err) *err = "failed to build the streamed expert graph";
        g.free();
        return false;
    }
    ggml_set_output(g.out);
    g.gf = ggml_new_graph_custom(g.ctx, 2048, false);
    ggml_build_forward_expand(g.gf, g.out);
    g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!g.alloc || !ggml_gallocr_alloc_graph(g.alloc, g.gf)) {
        if (err) *err = "failed to allocate the streamed expert graph";
        g.free();
        return false;
    }
    return true;
}

// Decode-sized graphs are built once per (view, routes, tokens) and reused.
MoeStreamedExpertCache::Graph * MoeStreamedExpertCache::graph_for(
        int view, const MoeLayerDesc & desc, int n_routes, int n_tokens, std::string * err) {
    const auto k = std::make_tuple(view, n_routes, n_tokens);
    auto it = graphs_.find(k);
    if (it != graphs_.end()) return &it->second;
    Graph g;
    if (!build_graph(g, view, desc, n_routes, n_tokens, err)) return nullptr;
    return &graphs_.emplace(k, g).first->second;
}

bool MoeStreamedExpertCache::pin_ready_locked(std::unique_lock<std::mutex> & lk, int layer,
                                              const int32_t * experts, size_t n,
                                              std::vector<int> & slots) {
    slots.clear();
    slots.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        bool hit = false;
        int slot;
        while ((slot = lookup_or_load_locked(layer, experts[i], /*front=*/true, &hit)) < 0) {
            cv_.wait(lk);  // every slot pinned or loading
        }
        Slot & s = slots_[(size_t) slot];
        ++s.pins;
        if (!s.demand) ++stats_.hits;
        if (s.prefetched) ++stats_.prefetch_hits;
        if (s.warm) ++stats_.warm_hits;
        s.demand = s.prefetched = s.warm = false;
        slots.push_back(slot);
    }
    const auto t0 = Clock::now();
    cv_.wait(lk, [&] {
        if (!failed_.empty()) return true;
        for (int slot : slots) {
            if (slots_[(size_t) slot].state != SlotState::Ready) return false;
        }
        return true;
    });
    stats_.wait_us += elapsed_us(t0, Clock::now());
    stats_.experts += slots.size();
    return failed_.empty();
}

void MoeStreamedExpertCache::release_staged(int layer) {
    std::lock_guard<std::mutex> lk(mu_);
    if (staged_layer_ != layer) return;
    for (int slot : staged_) --slots_[(size_t) slot].pins;
    staged_.clear();
    staged_layer_ = -1;
}

bool MoeStreamedExpertCache::acquire(int layer, const int32_t * selected, int n_routes,
                                     std::vector<int32_t> & slot_of_expert, std::string * err) {
    if (!ready() || layer < 0 || (size_t) layer >= storage_->layers.size() ||
        (int) slot_of_expert.size() != cfg_.n_expert) {
        if (err) *err = "streamed expert cache not ready for this layer";
        return false;
    }
    const MoeHybridLayerStorage & st = storage_->layers[(size_t) layer];
    std::vector<int32_t> uniq;
    std::vector<uint8_t> seen((size_t) cfg_.n_expert, 0);
    for (int i = 0; i < n_routes; ++i) {
        const int32_t e = selected[i];
        if (e < 0 || e >= cfg_.n_expert || !st.is_streamed(e) || seen[(size_t) e]) continue;
        seen[(size_t) e] = 1;
        uniq.push_back(e);
    }
    if (uniq.empty()) {
        release_staged(layer);
        return true;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        if ((int) (uniq.size() + acquired_.size()) > n_slots_) {
            if (err) *err = "streamed experts of one graph exceed the slot pool";
            return false;
        }
        std::vector<int32_t> & pred = predicted_[(size_t) layer];
        if (!pred.empty()) {
            stats_.predicted_of += uniq.size();
            for (int32_t e : uniq) {
                if (std::find(pred.begin(), pred.end(), e) != pred.end()) ++stats_.predicted_used;
            }
            pred.clear();
        }
    }
    std::vector<int> slots;
    bool ok = false;
    {
        std::unique_lock<std::mutex> lk(mu_);
        ok = pin_ready_locked(lk, layer, uniq.data(), uniq.size(), slots);
        acquired_.insert(acquired_.end(), slots.begin(), slots.end());
        if (!ok && err) *err = failed_;
    }
    release_staged(layer);
    for (size_t i = 0; i < uniq.size(); ++i) slot_of_expert[(size_t) uniq[i]] = slots[i];
    return ok;
}

void MoeStreamedExpertCache::release_acquired() {
    std::lock_guard<std::mutex> lk(mu_);
    for (int slot : acquired_) {
        --slots_[(size_t) slot].pins;
        slots_[(size_t) slot].last_use = ++tick_;
    }
    acquired_.clear();
    cv_.notify_all();
}

MoeStreamedExpertCache::Stacks MoeStreamedExpertCache::stacks(int layer) const {
    Stacks out;
    if (layer < 0 || (size_t) layer >= view_of_layer_.size() || view_of_layer_[(size_t) layer] < 0) {
        return out;
    }
    const PoolView & v = views_[(size_t) view_of_layer_[(size_t) layer]];
    out.gate = v.gate;
    out.up = v.up;
    out.down = v.down;
    out.gate_up = v.gate_up;
    return out;
}

bool MoeStreamedExpertCache::eval(int layer, const MoeLayerDesc & desc,
                                  const float * inp, const int32_t * selected,
                                  const float * weights, int n_used, int n_tokens,
                                  float * out, std::string * err) {
    if (!ready() || layer < 0 || (size_t) layer >= storage_->layers.size() ||
        view_of_layer_[(size_t) layer] < 0) {
        if (err) *err = "streamed expert cache not ready for this layer";
        return false;
    }
    const MoeHybridLayerStorage & st = storage_->layers[(size_t) layer];
    const int n_routes_all = n_used * n_tokens;

    // Streamed experts of this call, in first-route order.
    std::vector<int32_t> uniq;
    std::vector<uint8_t> seen((size_t) cfg_.n_expert, 0);
    for (int i = 0; i < n_routes_all; ++i) {
        const int32_t e = selected[i];
        if (!st.is_streamed(e) || seen[(size_t) e]) continue;
        seen[(size_t) e] = 1;
        uniq.push_back(e);
    }
    if (uniq.empty()) {
        release_staged(layer);
        return true;
    }

    {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<int32_t> & pred = predicted_[(size_t) layer];
        if (!pred.empty()) {
            stats_.predicted_of += uniq.size();
            for (int32_t e : uniq) {
                if (std::find(pred.begin(), pred.end(), e) != pred.end()) ++stats_.predicted_used;
            }
            pred.clear();
        }
    }

    const int chunk_cap = std::max(1, n_slots_ / 2);
    std::vector<int> slot_of_expert((size_t) cfg_.n_expert, -1);
    std::vector<float> part((size_t) cfg_.n_embd * (size_t) n_tokens);
    for (size_t c0 = 0; c0 < uniq.size(); c0 += (size_t) chunk_cap) {
        const size_t c1 = std::min(uniq.size(), c0 + (size_t) chunk_cap);
        std::vector<int> slots;
        {
            std::unique_lock<std::mutex> lk(mu_);
            pin_ready_locked(lk, layer, uniq.data() + c0, c1 - c0, slots);
            for (size_t i = c0; i < c1; ++i) slot_of_expert[(size_t) uniq[i]] = slots[i - c0];
        }
        // The staged loads have started and the first chunk holds its own
        // pins; later chunks must be able to use the staged slots' room.
        if (c0 == 0) release_staged(layer);
        const auto unpin = [&] {
            std::lock_guard<std::mutex> lk(mu_);
            for (int slot : slots) {
                --slots_[(size_t) slot].pins;
                slots_[(size_t) slot].last_use = ++tick_;
            }
        };
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!failed_.empty()) {
                if (err) *err = failed_;
                for (int slot : slots) --slots_[(size_t) slot].pins;
                return false;
            }
        }

        // Per token, this chunk's routes in route order, padded with a
        // zero-weight route to a slot the chunk pinned (always finite data).
        int n_routes = 0;
        for (int t = 0; t < n_tokens; ++t) {
            int n = 0;
            for (int k = 0; k < n_used; ++k) {
                const int32_t e = selected[t * n_used + k];
                if (e >= 0 && e < cfg_.n_expert && slot_of_expert[(size_t) e] >= 0) ++n;
            }
            n_routes = std::max(n_routes, n);
        }
        std::vector<int32_t> sel((size_t) n_routes * (size_t) n_tokens, slots[0]);
        std::vector<float> wts((size_t) n_routes * (size_t) n_tokens, 0.0f);
        for (int t = 0; t < n_tokens; ++t) {
            int n = 0;
            for (int k = 0; k < n_used; ++k) {
                const int32_t e = selected[t * n_used + k];
                if (e < 0 || e >= cfg_.n_expert || slot_of_expert[(size_t) e] < 0) continue;
                sel[(size_t) t * n_routes + n] = slot_of_expert[(size_t) e];
                wts[(size_t) t * n_routes + n] = weights[t * n_used + k];
                ++n;
            }
        }
        for (size_t i = c0; i < c1; ++i) slot_of_expert[(size_t) uniq[i]] = -1;

        const auto t0 = Clock::now();
        const int view = view_of_layer_[(size_t) layer];
        Graph scratch;
        Graph * g = n_tokens <= 8 ? graph_for(view, desc, n_routes, n_tokens, err)
                                  : (build_graph(scratch, view, desc, n_routes, n_tokens, err) ? &scratch : nullptr);
        bool ok = g != nullptr;
        if (ok) {
            ggml_backend_tensor_set(g->inp, inp, 0, sizeof(float) * (size_t) cfg_.n_embd * (size_t) n_tokens);
            ggml_backend_tensor_set(g->sel, sel.data(), 0, sizeof(int32_t) * sel.size());
            ggml_backend_tensor_set(g->wts, wts.data(), 0, sizeof(float) * wts.size());
            ok = ggml_backend_graph_compute(backend_, g->gf) == GGML_STATUS_SUCCESS;
            if (ok) {
                ggml_backend_tensor_get(g->out, part.data(), 0, sizeof(float) * part.size());
                for (size_t i = 0; i < part.size(); ++i) out[i] += part[i];
            } else if (err) {
                *err = "streamed expert graph compute failed";
            }
        }
        scratch.free();
        unpin();
        {
            std::lock_guard<std::mutex> lk(mu_);
            stats_.compute_us += elapsed_us(t0, Clock::now());
        }
        if (!ok) return false;
    }
    return true;
}


// ── Mailbox ─────────────────────────────────────────────────────────────

namespace {

constexpr size_t kMailboxLine = 64;

void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

uint32_t mailbox_load(const uint32_t * p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
void mailbox_store(uint32_t * p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }

}  // namespace

MoeStreamedMailbox::~MoeStreamedMailbox() {
    destroy();
}

bool MoeStreamedMailbox::init(MoeStreamedExpertCache * cache, int n_layers, int n_expert,
                              int n_expert_used, std::string * err) {
    destroy();
    if (!cache || n_layers <= 0 || n_expert <= 0 || n_expert_used <= 0) {
        if (err) *err = "invalid streamed mailbox geometry";
        return false;
    }
    const size_t ids_bytes = align_up(sizeof(int32_t) * (size_t) n_expert_used * kMaxTokens, kMailboxLine);
    const size_t row_bytes = align_up(sizeof(int32_t) * (size_t) n_expert * kMaxTokens, kMailboxLine);
    const size_t layer_bytes = 3 * kMailboxLine + 2 * ids_bytes + 2 * row_bytes;
    const size_t total = kMailboxLine + (size_t) n_layers * layer_bytes;
    // Mapped and coherent: the device polls and writes these words while the
    // resolver thread does, with no copy or cache flush between them.
#if defined(LUCE_BACKEND_HIP) || defined(GGML_USE_HIP)
    const hipError_t rc = hipHostMalloc(&base_, total,
        hipHostMallocMapped | hipHostMallocPortable | hipHostMallocCoherent);
#else
    const cudaError_t rc = cudaHostAlloc(&base_, total, cudaHostAllocMapped | cudaHostAllocPortable);
#endif
    if (rc != cudaSuccess) {
        base_ = nullptr;
        if (err) *err = "failed to allocate the mapped streamed mailbox";
        return false;
    }
    std::memset(base_, 0, total);
    cache_ = cache;
    n_expert_ = n_expert;
    n_expert_used_ = n_expert_used;
    auto * at = static_cast<uint8_t *>(base_);
    step_ = reinterpret_cast<uint32_t *>(at);
    at += kMailboxLine;
    channels_.assign((size_t) n_layers, Channel{});
    for (Channel & c : channels_) {
        c.step = step_;
        c.posted = reinterpret_cast<uint32_t *>(at);
        c.answered = reinterpret_cast<uint32_t *>(at + kMailboxLine);
        c.ids = reinterpret_cast<int32_t *>(at + 2 * kMailboxLine);
        c.lut = reinterpret_cast<int32_t *>(at + 2 * kMailboxLine + ids_bytes);
        c.valid = reinterpret_cast<float *>(at + 2 * kMailboxLine + ids_bytes + row_bytes);
        c.predicted = reinterpret_cast<uint32_t *>(at + 2 * kMailboxLine + ids_bytes + 2 * row_bytes);
        c.predicted_ids = reinterpret_cast<int32_t *>(at + 3 * kMailboxLine + ids_bytes + 2 * row_bytes);
        at += layer_bytes;
    }
    stopping_ = false;
    thread_ = std::thread(&MoeStreamedMailbox::resolver_main, this);
    return true;
}

void MoeStreamedMailbox::destroy() {
    if (thread_.joinable()) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stopping_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }
    if (base_) cudaFreeHost(base_);
    base_ = nullptr;
    step_ = nullptr;
    channels_.clear();
    jobs_.clear();
    pending_ = launch_done_ = stopping_ = false;
    error_.clear();
    cache_ = nullptr;
}

const MoeStreamedMailbox::Channel * MoeStreamedMailbox::channel(int layer) const {
    return base_ && layer >= 0 && (size_t) layer < channels_.size()
        ? &channels_[(size_t) layer] : nullptr;
}

void MoeStreamedMailbox::begin(const std::vector<Job> & jobs, int32_t invalid_route) {
    std::lock_guard<std::mutex> lk(mu_);
    uint32_t next = *step_ + 1;
    if (next == 0) next = 1;  // 0 is the initial value of every flag
    mailbox_store(step_, next);
    jobs_ = jobs;
    invalid_route_ = invalid_route;
    error_.clear();
    launch_done_ = false;
    pending_ = true;
    cv_.notify_all();
}

bool MoeStreamedMailbox::end(std::string * err) {
    std::unique_lock<std::mutex> lk(mu_);
    launch_done_ = true;
    cv_.wait(lk, [&] { return !pending_; });
    const bool ok = error_.empty();
    if (!ok && err) *err = error_;
    lk.unlock();
    cache_->release_acquired();
    return ok;
}

void MoeStreamedMailbox::resolver_main() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
        cv_.wait(lk, [&] { return stopping_ || pending_; });
        if (stopping_) break;
        const std::vector<Job> jobs = jobs_;
        const uint32_t step = mailbox_load(step_);
        lk.unlock();
        std::string error;
        for (size_t j = 0; j < jobs.size(); ++j) {
            const Job & job = jobs[j];
            const Channel & c = channels_[(size_t) job.layer];
            // A post that never comes means the launch skipped the layer
            // (it failed); stop once end() says the launch is over.
            bool posted = false;
            for (uint32_t spin = 0;; ++spin) {
                if (mailbox_load(c.posted) == step) { posted = true; break; }
                if ((spin & 1023) == 1023) {
                    std::lock_guard<std::mutex> g(mu_);
                    if (launch_done_ || stopping_) break;
                }
                cpu_relax();
            }
            if (!posted) {
                if (error.empty()) error = "layer " + std::to_string(job.layer) + " never posted its routes";
                break;
            }
            // The next layer's prediction was posted ahead of these routes:
            // queue its loads behind this layer's before blocking on them.
            if (j + 1 < jobs.size() && jobs[j + 1].layer == job.layer + 1) {
                const Channel & next = channels_[(size_t) jobs[j + 1].layer];
                if (mailbox_load(next.predicted) == step) {
                    cache_->prefetch(jobs[j + 1].layer, next.predicted_ids,
                                     std::min(jobs[j + 1].n_routes, n_expert_used_ * kMaxTokens));
                }
            }
            std::string layer_error;
            if (!answer(job, step, &layer_error) && error.empty()) {
                error = "layer " + std::to_string(job.layer) + ": " + layer_error;
            }
        }
        lk.lock();
        if (error_.empty()) error_ = error;
        pending_ = false;
        cv_.notify_all();
    }
}

// Writes the layer's lookup rows and publishes them. Always publishes, so the
// device never waits on a failed layer; the rows then drop its streamed routes
// and end() reports the failure.
bool MoeStreamedMailbox::answer(const Job & job, uint32_t step, std::string * err) {
    const Channel & c = channels_[(size_t) job.layer];
    const int n_routes = std::min(job.n_routes, n_expert_used_ * kMaxTokens);
    const int n_tokens = std::max(1, std::min(job.n_tokens, kMaxTokens));
    slot_of_.assign((size_t) n_expert_, -1);
    const bool ok = cache_->acquire(job.layer, c.ids, n_routes, slot_of_, err);
    for (int e = 0; e < n_expert_; ++e) {
        const int32_t slot = ok ? slot_of_[(size_t) e] : -1;
        c.lut[e] = slot >= 0 ? slot : invalid_route_;
        c.valid[e] = slot >= 0 ? 1.0f : 0.0f;
    }
    for (int t = 1; t < n_tokens; ++t) {
        std::memcpy(c.lut + (size_t) t * n_expert_, c.lut, sizeof(int32_t) * (size_t) n_expert_);
        std::memcpy(c.valid + (size_t) t * n_expert_, c.valid, sizeof(float) * (size_t) n_expert_);
    }
    mailbox_store(c.answered, step);
    return ok;
}

}  // namespace luce::common
