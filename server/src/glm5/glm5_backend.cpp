// Glm5Backend implementation.

#include "glm5_backend.h"
#include "glm5_graph.h"

#include "common/dynamic_backend.h"
#include "common/thinking_budget.h"
#include "internal.h"
#include "luce.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <thread>
#include <utility>

namespace luce::common {

namespace {

constexpr size_t kSchedNodes = 1 << 16;

// Contiguous layer ranges sized by weight bytes; at least one layer per device.
std::vector<int> split_layers(const std::vector<size_t> & layer_bytes,
                              const std::vector<double> & weights) {
    const int n_layer = (int) layer_bytes.size();
    const int n_dev = (int) weights.size();
    std::vector<int> device(n_layer, 0);
    if (n_dev <= 1 || n_layer == 0) return device;

    const double total_w = std::accumulate(weights.begin(), weights.end(), 0.0);
    const double total_b = (double) std::accumulate(layer_bytes.begin(), layer_bytes.end(), (size_t) 0);
    int d = 0;
    double acc = 0.0;
    double target = total_b * weights[0] / total_w;
    for (int il = 0; il < n_layer; ++il) {
        // next device once its share is reached (or layers run short)
        const int layers_left = n_layer - il;
        const int devices_left = n_dev - d - 1;
        if (d < n_dev - 1 && il > 0 &&
            (acc + 0.5 * (double) layer_bytes[il] > target || layers_left <= devices_left)) {
            ++d;
            target += total_b * weights[d] / total_w;
        }
        device[il] = d;
        acc += (double) layer_bytes[il];
    }
    return device;
}

}  // namespace

Glm5Backend::Glm5Backend(Glm5BackendConfig cfg) : cfg_(std::move(cfg)) {}

Glm5Backend::~Glm5Backend() { shutdown(); }

bool Glm5Backend::plan_layers() {
    const int n_layer = read_glm5_layer_count(cfg_.model_path);
    if (n_layer <= 0) {
        std::fprintf(stderr, "[glm5] cannot read %s: %s\n", cfg_.model_path.c_str(), luce_last_error());
        return false;
    }
    const int n_dev = (int) backends_.size();
    if (n_dev <= 1) {
        layer_device_.assign(n_layer, 0);
        return true;
    }
    const std::vector<size_t> bytes = read_glm5_layer_bytes(cfg_.model_path);
    std::vector<double> weights(n_dev, 1.0);
    if ((int) cfg_.device.layer_split_weights.size() == n_dev) {
        weights = cfg_.device.layer_split_weights;
    }
    layer_device_ = split_layers(bytes, weights);
    return (int) layer_device_.size() == n_layer;
}

bool Glm5Backend::load_weights() {
    // Embedding on the host.
    if (!load_glm5_model(cfg_.model_path, backends_, layer_device_, w_, cpu_)) {
        std::fprintf(stderr, "[glm5] model load failed: %s\n", luce_last_error());
        return false;
    }
    w_.hc_fused = !cfg_.cpu_only;
    return true;
}

bool Glm5Backend::init() {
    const int threads = cfg_.cpu_threads > 0 ? cfg_.cpu_threads
                                             : (int) std::max(1u, std::thread::hardware_concurrency());
    if (cfg_.cpu_only) {
        backends_.push_back(ggml_backend_cpu_init());
        if (backends_.back()) ggml_backend_cpu_set_n_threads(backends_.back(), threads);
    } else {
        std::vector<int> gpus = cfg_.device.layer_split_gpus;
        if (gpus.empty()) gpus.push_back(cfg_.device.gpu);
        for (size_t i = 0; i < gpus.size(); ++i) {
            std::string err;
            const PlacementBackend kind = cfg_.device.layer_split_gpus.empty()
                ? cfg_.device.backend : cfg_.device.layer_split_backend(i);
            ggml_backend_t b = init_placement_backend(kind, gpus[i], &err);
            if (!b) {
                std::fprintf(stderr, "[glm5] %s\n", err.c_str());
                return false;
            }
            backends_.push_back(b);
        }
        cpu_ = ggml_backend_cpu_init();
        if (cpu_) ggml_backend_cpu_set_n_threads(cpu_, threads);
    }
    if (backends_.empty() || !backends_[0]) {
        std::fprintf(stderr, "[glm5] backend init failed\n");
        return false;
    }

    if (!plan_layers() || !load_weights()) return false;

    // The 512-wide latent FA kernel needs > 2 heads (GLM-5.3-Flash has 64).
    if (!cfg_.cpu_only && w_.n_mla_layers() > 0 && w_.n_head < 4) {
        std::fprintf(stderr, "[glm5] %d attention heads: the GPU MLA path needs at least 4\n",
                     w_.n_head);
        return false;
    }

    if (!create_glm5_cache(w_, backends_, cfg_.device.max_ctx, cache_)) {
        std::fprintf(stderr, "[glm5] %s\n", luce_last_error());
        return false;
    }

    // The scheduler wants the CPU backend last.
    std::vector<ggml_backend_t> sched_backends = backends_;
    if (cpu_) sched_backends.push_back(cpu_);
    sched_ = ggml_backend_sched_new(sched_backends.data(), nullptr, (int) sched_backends.size(),
                                    kSchedNodes, false, false);
    if (!sched_) {
        std::fprintf(stderr, "[glm5] ggml_backend_sched_new failed\n");
        return false;
    }

    std::printf("[glm5] loaded %s: %d layers (%d KDA, %d MLA), %d experts top-%d, vocab %d, "
                "max_ctx %d, %zu device(s)\n",
                cfg_.model_path.c_str(), w_.n_layer, w_.n_kda_layers(), w_.n_mla_layers(),
                w_.n_expert, w_.n_expert_used, w_.n_vocab, cache_.max_ctx, backends_.size());
    if (backends_.size() > 1) {
        for (size_t d = 0; d < backends_.size(); ++d) {
            int first = -1, last = -1;
            for (int il = 0; il < w_.n_layer; ++il) {
                if (layer_device_[il] != (int) d) continue;
                if (first < 0) first = il;
                last = il;
            }
            std::printf("[glm5]   device %zu (%s): layers %d-%d\n", d,
                        ggml_backend_name(backends_[d]), first, last);
        }
    }
    std::fflush(stdout);
    return true;
}

void Glm5Backend::print_ready_banner() const {
    std::printf("[glm5-daemon] ready layers=%d kda=%d mla=%d experts=%d/%d ctx=%d devices=%zu\n",
                w_.n_layer, w_.n_kda_layers(), w_.n_mla_layers(), w_.n_expert_used, w_.n_expert,
                cache_.max_ctx, backends_.size());
    std::fflush(stdout);
}

// ── Park / unpark ────────────────────────────────────────────────────────

bool Glm5Backend::park(ParkTarget target) {
    if (target != ParkTarget::TargetModel && target != ParkTarget::All) return false;
    if (!parked_) {
        free_glm5_model(w_);
        parked_ = true;
        std::printf("[glm5] target parked\n");
        std::fflush(stdout);
    }
    return true;
}

bool Glm5Backend::unpark(ParkTarget target) {
    if (target != ParkTarget::TargetModel && target != ParkTarget::All) return false;
    if (parked_) {
        if (!load_weights()) return false;
        parked_ = false;
        std::printf("[glm5] target unparked\n");
        std::fflush(stdout);
    }
    return true;
}

// ── Forward ──────────────────────────────────────────────────────────────

void Glm5Backend::reset_sequence() {
    clear_glm5_cache(cache_);
    last_logits_.clear();
}

bool Glm5Backend::forward(const int32_t * tokens, int n_tokens, int kv_start, bool all_logits,
                          std::vector<float> & logits) {
    if (parked_) return false;
    if (kv_start + n_tokens > cache_.max_ctx) {
        std::fprintf(stderr, "[glm5] step past max_ctx (%d + %d > %d)\n",
                     kv_start, n_tokens, cache_.max_ctx);
        return false;
    }
    Glm5StepGraph g;
    if (!build_glm5_step_graph(w_, cache_, n_tokens, kv_start, all_logits, g)) {
        std::fprintf(stderr, "[glm5] graph build failed (n=%d, kv_start=%d)\n", n_tokens, kv_start);
        return false;
    }
    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, g.gf)) {
        std::fprintf(stderr, "[glm5] graph alloc failed (n=%d, kv_start=%d)\n", n_tokens, kv_start);
        g.free();
        return false;
    }
    set_glm5_step_inputs(w_, g, tokens);
    const ggml_status st = ggml_backend_sched_graph_compute(sched_, g.gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[glm5] graph compute failed (status=%d, n=%d, kv_start=%d)\n",
                     (int) st, n_tokens, kv_start);
        g.free();
        return false;
    }
    const size_t n = (size_t) w_.n_vocab * (all_logits ? n_tokens : 1);
    logits.resize(n);
    ggml_backend_tensor_get(g.logits, logits.data(), 0, n * sizeof(float));
    g.free();
    cache_.cur_pos = kv_start + n_tokens;
    return true;
}

bool Glm5Backend::prefill(const std::vector<int32_t> & prompt, int from, int to,
                          const DaemonIO & io) {
    const int chunk = std::max(1, cfg_.chunk);
    for (int start = from; start < to; start += chunk) {
        if (io.is_cancelled()) return true;
        const int n = std::min(chunk, to - start);
        std::vector<float> logits;
        if (!forward(prompt.data() + start, n, start, /*all_logits=*/false, logits)) return false;
        last_logits_ = std::move(logits);
    }
    return true;
}

// ── Generation ───────────────────────────────────────────────────────────

bool Glm5Backend::is_stop_token(int32_t t) const {
    return std::find(w_.stop_tokens.begin(), w_.stop_tokens.end(), t) != w_.stop_tokens.end();
}

int32_t Glm5Backend::pick_token(const std::vector<float> & logits,
                                const std::vector<int32_t> & history) {
    if (sampler_.needs_logit_processing()) {
        return sample_logits(logits.data(), w_.n_vocab, sampler_, history, rng_);
    }
    return (int32_t) (std::max_element(logits.begin(), logits.begin() + w_.n_vocab) - logits.begin());
}

GenerateResult Glm5Backend::continue_generation(const GenerateRequest & req, const DaemonIO & io,
                                                int restored) {
    GenerateResult result;
    result.restored_prefix_tokens = restored;
    ThinkingBudget budget(req.budget_hook, req.n_gen);
    DaemonIO out_io = io.with_token_callback(req.on_token);
    sampler_ = req.sampler;
    if (req.do_sample) rng_.seed(request_rng_seed(sampler_));

    const int P = (int) req.prompt.size();
    if (P == 0) {
        result.fail(GenerateErrorCode::PrefillFailed, "empty prompt");
        out_io.emit(-1);
        return result;
    }
    if (P > cache_.max_ctx) {
        result.fail(GenerateErrorCode::ContextOverflow);
        out_io.emit(-1);
        return result;
    }

    int pos = restored;
    // Stop at the snapshot position: KDA state cannot rewind.
    if (req.snap_slot >= 0 && req.snap_pos > pos && req.snap_pos <= P) {
        if (!prefill(req.prompt, pos, req.snap_pos, out_io)) {
            result.fail(GenerateErrorCode::PrefillFailed);
            out_io.emit(-1);
            return result;
        }
        pos = req.snap_pos;
        if (!out_io.is_cancelled() && snapshot_save(req.snap_slot)) {
            std::printf("[snap] inline slot=%d cur_pos=%d\n", req.snap_slot, pos);
            std::fflush(stdout);
        }
    }
    if (!prefill(req.prompt, pos, P, out_io)) {
        result.fail(GenerateErrorCode::PrefillFailed);
        out_io.emit(-1);
        return result;
    }
    if (out_io.is_cancelled()) {
        out_io.emit(-1);
        result.succeed();
        return result;
    }
    if (last_logits_.empty()) {
        result.fail(GenerateErrorCode::DecodeSeedMissing);
        out_io.emit(-1);
        return result;
    }

    std::vector<float> logits = last_logits_;
    while (budget.remaining() > 0) {
        int32_t next = pick_token(logits, result.tokens);
        next = budget.apply(next).token;
        result.budget_forced_close = budget.forced_close();
        result.tokens.push_back(next);
        out_io.emit(next);
        if (out_io.is_cancelled()) break;
        if (!req.ignore_eos && is_stop_token(next)) break;
        if (budget.remaining() <= 0) break;
        if (cache_.cur_pos >= cache_.max_ctx) break;
        if (!forward(&next, 1, cache_.cur_pos, false, logits)) {
            result.fail(GenerateErrorCode::DecodeFailed);
            out_io.emit(-1);
            return result;
        }
        last_logits_ = logits;
    }

    out_io.emit(-1);
    result.succeed();
    return result;
}

GenerateResult Glm5Backend::generate_impl(const GenerateRequest & req, const DaemonIO & io) {
    if (parked_) {
        GenerateResult result;
        result.fail(GenerateErrorCode::ModelParked);
        io.with_token_callback(req.on_token).emit(-1);
        return result;
    }
    reset_sequence();
    return continue_generation(req, io, 0);
}

GenerateResult Glm5Backend::restore_and_generate_impl(int slot, const GenerateRequest & req,
                                                      const DaemonIO & io) {
    if (slot < 0 || slot >= kSnapshotSlots || !snapshots_[slot].used) {
        GenerateResult result;
        result.fail(GenerateErrorCode::InvalidSnapshotSlot);
        io.with_token_callback(req.on_token).emit(-1);
        return result;
    }
    const Snapshot & snap = snapshots_[slot];
    if (snap.cur_pos > (int) req.prompt.size() || parked_) {
        return generate_impl(req, io);
    }

    reset_sequence();
    const auto tensors = snapshot_tensors(snap.cur_pos);
    for (size_t i = 0; i < tensors.size(); ++i) {
        const size_t bytes = tensors[i].first->nb[1] * tensors[i].second;
        ggml_backend_tensor_set(tensors[i].first, snap.data[i].data(), 0, bytes);
    }
    cache_.cur_pos = snap.cur_pos;
    last_logits_ = snap.logits;
    return continue_generation(req, io, snap.cur_pos);
}

// ── Snapshots ────────────────────────────────────────────────────────────

std::vector<std::pair<ggml_tensor *, size_t>> Glm5Backend::snapshot_tensors(int pos) const {
    std::vector<std::pair<ggml_tensor *, size_t>> out;
    for (const Glm5LayerCache & C : cache_.layers) {
        if (C.conv) {
            out.push_back({C.conv, (size_t) C.conv->ne[1]});
            out.push_back({C.ssm, (size_t) (C.ssm->ne[1] * C.ssm->ne[2])});
        } else {
            out.push_back({C.latent, (size_t) pos});
            out.push_back({C.idx_k, (size_t) pos});
            out.push_back({C.idx_g, (size_t) pos});
            out.push_back({C.pooled, (size_t) (pos / w_.idx_kpool)});
        }
    }
    return out;
}

bool Glm5Backend::snapshot_save(int slot) {
    if (slot < 0 || slot >= kSnapshotSlots || last_logits_.empty()) return false;
    Snapshot & snap = snapshots_[slot];
    snap = Snapshot{};
    const auto tensors = snapshot_tensors(cache_.cur_pos);
    snap.data.resize(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        const size_t bytes = tensors[i].first->nb[1] * tensors[i].second;
        snap.data[i].resize(bytes);
        ggml_backend_tensor_get(tensors[i].first, snap.data[i].data(), 0, bytes);
    }
    snap.cur_pos = cache_.cur_pos;
    snap.logits = last_logits_;
    snap.used = true;
    return true;
}

void Glm5Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= kSnapshotSlots) return;
    snapshots_[slot] = Snapshot{};
}

bool Glm5Backend::snapshot_used(int slot) const {
    return slot >= 0 && slot < kSnapshotSlots && snapshots_[slot].used;
}

int Glm5Backend::snapshot_cur_pos(int slot) const {
    return snapshot_used(slot) ? snapshots_[slot].cur_pos : 0;
}

size_t Glm5Backend::snapshot_bytes_estimate(int tokens) const {
    size_t bytes = (size_t) w_.n_vocab * sizeof(float);
    tokens = std::clamp(tokens, 0, cache_.max_ctx);
    for (const auto & t : snapshot_tensors(tokens)) bytes += t.first->nb[1] * t.second;
    return bytes;
}

// ── PFlash (not supported) ───────────────────────────────────────────────

bool Glm5Backend::handle_compress(const std::string & /*line*/, const DaemonIO & io) {
    std::fprintf(stderr, "[glm5] PFlash compression is not supported for glm5-next\n");
    io.emit(-1);
    return false;
}

// ── Shutdown ─────────────────────────────────────────────────────────────

void Glm5Backend::shutdown() {
    for (Snapshot & s : snapshots_) s = Snapshot{};
    if (sched_) {
        ggml_backend_sched_free(sched_);
        sched_ = nullptr;
    }
    free_glm5_cache(cache_);
    free_glm5_model(w_);
    for (ggml_backend_t b : backends_) {
        if (b) ggml_backend_free(b);
    }
    backends_.clear();
    if (cpu_) {
        ggml_backend_free(cpu_);
        cpu_ = nullptr;
    }
}

}  // namespace luce::common
