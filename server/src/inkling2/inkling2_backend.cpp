// Inkling2Backend — ModelBackend implementation for Inkling 2 / Inkling 2 Mini.

#include "inkling2_backend.h"
#include "inkling2_2dsd_draft.h"
#include "inkling2_2dsd_executor.h"
#include "inkling2_2dsd_graph.h"

#include "common/peer_access.h"
#include "common/sampler.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <utility>

namespace luce::common {

namespace {

static ggml_backend_t init_gpu_backend(int gpu_id) {
    ggml_backend_t b = ggml_backend_cuda_init(gpu_id);
    if (!b) {
        std::fprintf(stderr, "[inkling2] failed to initialize GPU backend %d\n", gpu_id);
    }
    return b;
}

static void free_snapshot(Inkling2Snapshot & s) {
    if (s.buf) { ggml_backend_buffer_free(s.buf); s.buf = nullptr; }
    if (s.ctx) { ggml_free(s.ctx); s.ctx = nullptr; }
    s.k_cache.clear();
    s.v_cache.clear();
    s.conv_state.clear();
    s.dspark_k.clear();
    s.dspark_v.clear();
    s.last_taps = nullptr;
    s.logits.clear();
    s.cur_pos = 0;
    s.dspark_ctx_pos = 0;
    s.used = false;
}

} // namespace

Inkling2Backend::Inkling2Backend(Inkling2BackendConfig cfg)
    : cfg_(std::move(cfg)) {}

Inkling2Backend::~Inkling2Backend() {
    shutdown();
}

std::string Inkling2Backend::resolve_dspark_path() const {
    if (cfg_.verify_width == 1) return {};
    if (cfg_.draft_path.has_value()) {
        const std::string & p = *cfg_.draft_path;
        if (p == "0" || p == "off" || p == "none") return {};
        return p;
    }
    if (const char * env = std::getenv("LUCE_INKLING2_DSPARK")) {
        if (std::strcmp(env, "0") == 0 || std::strcmp(env, "off") == 0) return {};
        if (*env) return std::string(env);
    }
    // Auto-discover `mtp.safetensors` alongside the GGUF file.
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::path(cfg_.model_path).parent_path();
    const fs::path cand = dir / "mtp.safetensors";
    if (fs::is_regular_file(cand, ec)) {
        return cand.string();
    }
    return {};
}

bool Inkling2Backend::load_all() {
    if (!backend_) {
        backend_ = init_gpu_backend(cfg_.device.gpu);
        if (!backend_) return false;
    }
    if (cfg_.expert_device.has_value() && !expert_backend_) {
        expert_backend_ = init_gpu_backend(cfg_.expert_device->gpu);
        if (expert_backend_) {
            g_peer_access_opt_in = true;
            enable_peer_access_pair(cfg_.device.gpu, cfg_.expert_device->gpu);
        }
    }
    if (!snap_backend_) {
        snap_backend_ = ggml_backend_cpu_init();
    }

    std::string err;
    if (!load_inkling2_gguf(cfg_.model_path, backend_, expert_backend_, weights_, err)) {
        std::fprintf(stderr, "[inkling2] %s\n", err.c_str());
        return false;
    }

    // Load bundled DSpark speculator (`mtp.safetensors`) or 2DSD drafter directory if found/requested.
    const std::string dspark_path = resolve_dspark_path();
    if (!dspark_path.empty()) {
        std::string derr;
        if (is_inkling2_2dsd_drafter_path(dspark_path)) {
            if (!load_inkling2_2dsd_drafter(dspark_path, backend_, weights_, derr)) {
                std::fprintf(stderr, "[inkling2] warning: 2DSD drafter load failed (%s); continuing in AR mode\n", derr.c_str());
            }
        } else if (!load_inkling2_dspark_safetensors(dspark_path, backend_, weights_, derr)) {
            std::fprintf(stderr, "[inkling2] warning: DSpark load failed (%s); continuing in AR mode\n", derr.c_str());
        }
    }

    // Resolve chunk size and KV cache type.
    chunk_ = cfg_.chunk > 0 ? cfg_.chunk : 512;
    ggml_type kv_type = GGML_TYPE_Q8_0;
    if (cfg_.cache_type_k != GGML_TYPE_COUNT) {
        kv_type = cfg_.cache_type_k;
    } else if (const char * env_kv = std::getenv("LUCE_INKLING2_KV_TYPE")) {
        if (std::strcmp(env_kv, "f16") == 0 || std::strcmp(env_kv, "F16") == 0) kv_type = GGML_TYPE_F16;
    }

    // Determine max_ctx (reuse frozen first-load layout on reinstate so host-RAM
    // prefix snapshots remain valid; otherwise auto-fit when <= 0 or fit_ctx).
    int max_ctx = fitted_max_ctx_;
    if (max_ctx <= 0) {
        max_ctx = cfg_.device.max_ctx;
        if (max_ctx <= 0) {
            max_ctx = weights_.context_length > 0 ? std::min(weights_.context_length, 65536) : 32768;
        }
        if (cfg_.device.fit_ctx) {
            // Fit context comfortably to available memory (with Split-KV when expert_backend is present).
            size_t tfree = 0, ttotal = 0;
            ggml_backend_cuda_get_device_memory(cfg_.device.gpu, &tfree, &ttotal);
            size_t efree = 0, etotal = 0;
            if (expert_backend_) {
                ggml_backend_cuda_get_device_memory(cfg_.expert_device->gpu, &efree, &etotal);
            }
            const size_t n_global = (weights_.draft_mode == Inkling2DraftMode::TwoDSD)
                ? (weights_.global_slots.size() + 2) // +2 extra coda depth planes for L23
                : weights_.global_slots.size();
            const size_t kv_row_b = ggml_row_size(kv_type, weights_.head_dim);
            const size_t draft_kv_b = (weights_.draft_mode == Inkling2DraftMode::TwoDSD)
                ? ((size_t) weights_.twodsd.n_layer * 2 * weights_.twodsd.n_head_kv * weights_.twodsd.head_dim * 2)
                : (weights_.dspark.loaded ? (size_t) 3 * 2 * weights_.n_head_kv * weights_.head_dim * 2 : 0);
            const size_t bytes_per_tok = n_global * 2 * weights_.n_head_kv * kv_row_b + draft_kv_b;
            const size_t reserve = (size_t) 4 << 30;
            const size_t usable = (tfree > reserve ? tfree - reserve : 0)
                                + (expert_backend_ && efree > reserve ? efree - reserve : 0);
            if (bytes_per_tok > 0) {
                const int fit = (int) std::min<size_t>((size_t) max_ctx, usable / bytes_per_tok);
                max_ctx = std::max(4096, (fit / 256) * 256);
            }
        }
        fitted_max_ctx_ = max_ctx;
    }

    if (!create_inkling2_cache(backend_, expert_backend_, weights_, max_ctx, chunk_, kv_type, cache_, err,
                               fitted_remote_split_pos_)) {
        std::fprintf(stderr, "[inkling2] %s\n", err.c_str());
        return false;
    }
    if (fitted_remote_split_pos_ < 0) {
        fitted_remote_split_pos_ = cache_.remote_split_pos;
    }

    return true;
}

void Inkling2Backend::release_device_state() {
    last_logits_.clear();
    free_inkling2_cache(cache_);
    free_inkling2_weights(weights_);
    if (backend_) ggml_backend_cuda_trim_pool(backend_);
    if (expert_backend_) ggml_backend_cuda_trim_pool(expert_backend_);
}

void Inkling2Backend::release_all() {
    for (auto & s : snapshots_) free_snapshot(s);
    release_device_state();
}

bool Inkling2Backend::init() {
    if (!load_all()) {
        release_all();
        return false;
    }
    return true;
}

void Inkling2Backend::shutdown() {
    release_all();
    if (snap_backend_)   { ggml_backend_free(snap_backend_); snap_backend_ = nullptr; }
    if (expert_backend_) { ggml_backend_free(expert_backend_); expert_backend_ = nullptr; }
    if (backend_)        { ggml_backend_free(backend_); backend_ = nullptr; }
}

void Inkling2Backend::print_ready_banner() const {
    const auto & w = weights_;
    std::string spec_str = "off";
    if (w.draft_mode == Inkling2DraftMode::TwoDSD && w.twodsd.loaded) {
        const auto tdcfg = inkling2_2dsd_config_from_env(w.n_passes, w.twodsd.block_size);
        const char * sname = (tdcfg.stage == Inkling2TwoDSDStage::Stage1) ? "stage1"
                           : (tdcfg.stage == Inkling2TwoDSDStage::Stage2a) ? "stage2a" : "stage2b";
        char buf[96];
        std::snprintf(buf, sizeof(buf), "2dsd(%s,k=%d,conf=%.2f)", sname, tdcfg.k, tdcfg.draft_confidence);
        spec_str = buf;
    } else if (w.dspark.loaded) {
        spec_str = "on(block=8)";
    }
    std::printf(
        "[inkling2-daemon] ready arch=%s phys=%d eff=%d (core=[%d,%d)x%d) "
        "experts=%d/%d kv=%s ctx=%d chunk=%d dspark=%s dual_gpu=%s\n",
        w.is_loop ? "inkling2_loop" : "inkling2",
        w.n_layer_phys, w.n_layer_eff, w.core_start, w.core_end, w.n_passes,
        w.n_expert_used, w.n_expert, ggml_type_name(cache_.kv_type),
        cache_.max_ctx, chunk_,
        spec_str.c_str(),
        (w.expert_backend && w.split_exp > 0) ? "on" : "off");
    std::fflush(stdout);
}

bool Inkling2Backend::park(ParkTarget target) {
    if (target != ParkTarget::TargetModel && target != ParkTarget::All) return false;
    if (released_) return false;
    if (parked_) return true;
    release_all();
    parked_ = true;
    return true;
}

bool Inkling2Backend::unpark(ParkTarget target) {
    if (target != ParkTarget::TargetModel && target != ParkTarget::All) return false;
    if (!parked_) return true;
    if (!load_all()) {
        release_all();
        return false;
    }
    parked_ = false;
    return true;
}

ModelBackend::Residency Inkling2Backend::residency() const {
    Residency r;
    r.gpus.push_back(cfg_.device.gpu);
    if (cfg_.expert_device.has_value()) r.gpus.push_back(cfg_.expert_device->gpu);
    if (!backend_) {
        r.reason = "inkling2: not initialized";
        return r;
    }
    r.supported = true;
    auto buf_size = [](ggml_backend_buffer_t b) -> size_t {
        return b ? ggml_backend_buffer_get_size(b) : 0;
    };
    if (released_ || parked_) {
        r.released = r.gpus;
        r.bytes.assign(r.gpus.size(), 0);
        return r;
    }
    const size_t target_bytes =
        buf_size(weights_.buf) + buf_size(weights_.derived_buf) +
        buf_size(weights_.dspark.buf) + buf_size(weights_.twodsd.buf) +
        buf_size(cache_.buf) + buf_size(cache_.dspark_kv_buf);
    r.bytes.push_back(target_bytes);
    if (cfg_.expert_device.has_value()) {
        const size_t expert_bytes =
            buf_size(weights_.expert_buf) + buf_size(cache_.remote_buf);
        r.bytes.push_back(expert_bytes);
    }
    return r;
}

bool Inkling2Backend::release_gpus(const std::vector<int> & gpus, std::string & error) {
    if (!backend_) {
        error = "inkling2: not initialized";
        return false;
    }
    if (parked_) {
        error = "inkling2: parked";
        return false;
    }
    auto touches = [&](int gpu) {
        return gpus.empty() || std::find(gpus.begin(), gpus.end(), gpu) != gpus.end();
    };
    const bool hit_target = touches(cfg_.device.gpu);
    const bool hit_expert = cfg_.expert_device.has_value() && touches(cfg_.expert_device->gpu);
    if (!hit_target && !hit_expert) return true;
    if (released_) return true;
    release_device_state();
    released_ = true;
    return true;
}

bool Inkling2Backend::reinstate(std::string & error) {
    if (!released_) return true;
    if (!load_all()) {
        release_device_state();
        released_ = true;
        error = "inkling2: reinstate load failed";
        return false;
    }
    released_ = false;
    return true;
}

// ─── Prefix Snapshots (Host RAM) ────────────────────────────────────────
size_t Inkling2Backend::snapshot_bytes_estimate(int tokens) const {
    if (!cache_.buf) return 0;
    const int n_tok = std::clamp(tokens, 0, cache_.max_ctx);
    const size_t kv_row_b = ggml_row_size(cache_.kv_type, weights_.head_dim);
    size_t total = 0;
    for (int il = 0; il < weights_.n_layer_eff; ++il) {
        const bool is_swa = weights_.layers[weights_.phys_layer_for_slot(il)].is_swa;
        const int rows = is_swa ? cache_.swa_cap : n_tok;
        total += (size_t) rows * 2 * weights_.n_head_kv * kv_row_b;
        total += ggml_nbytes(cache_.conv_state[il]);
    }
    if (weights_.dspark.loaded) {
        total += (size_t) weights_.dspark.n_layer * 2 * n_tok * weights_.dspark.n_head_kv * weights_.dspark.head_dim * 2;
        total += (size_t) weights_.n_embd * 4 * sizeof(float);
    }
    return total;
}

bool Inkling2Backend::snapshot_save(int slot) {
    if (slot < 0 || slot >= kMaxSlots || parked_ || released_ || cache_.cur_pos <= 0) return false;
    // Skip snapshots when Split-KV remote pool or 2DSD 60-plane tree cache is active (v1)
    if (cache_.remote_backend || cache_.tree.active) return false;

    Inkling2Snapshot & s = snapshots_[slot];
    free_snapshot(s);

    const int pos = cache_.cur_pos;
    const int n_slots = weights_.n_layer_eff;
    const size_t n_tensors = (size_t) n_slots * 3 + (weights_.dspark.loaded ? (size_t) weights_.dspark.n_layer * 2 + 1 : 0) + 4;
    struct ggml_init_params gp = { n_tensors * ggml_tensor_overhead() + 4096, nullptr, true };
    s.ctx = ggml_init(gp);
    if (!s.ctx) return false;

    s.k_cache.resize(n_slots, nullptr);
    s.v_cache.resize(n_slots, nullptr);
    s.conv_state.resize(n_slots, nullptr);
    for (int il = 0; il < n_slots; ++il) {
        const bool is_swa = weights_.layers[weights_.phys_layer_for_slot(il)].is_swa;
        const int rows = is_swa ? cache_.swa_cap : pos;
        s.k_cache[il]    = ggml_new_tensor_3d(s.ctx, cache_.kv_type, weights_.head_dim, rows, weights_.n_head_kv);
        s.v_cache[il]    = ggml_new_tensor_3d(s.ctx, cache_.kv_type, weights_.head_dim, rows, weights_.n_head_kv);
        s.conv_state[il] = ggml_dup_tensor(s.ctx, cache_.conv_state[il]);
    }
    if (weights_.dspark.loaded) {
        const int dl = weights_.dspark.n_layer;
        s.dspark_k.resize(dl, nullptr);
        s.dspark_v.resize(dl, nullptr);
        for (int i = 0; i < dl; ++i) {
            s.dspark_k[i] = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F16, weights_.dspark.head_dim, pos, weights_.dspark.n_head_kv);
            s.dspark_v[i] = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F16, weights_.dspark.head_dim, pos, weights_.dspark.n_head_kv);
        }
        s.last_taps = ggml_dup_tensor(s.ctx, cache_.last_taps_dev);
    }

    s.buf = ggml_backend_alloc_ctx_tensors(s.ctx, snap_backend_);
    if (!s.buf) { free_snapshot(s); return false; }

    auto copy_kv_to_host = [](ggml_tensor * src, ggml_tensor * dst, int rows) {
        const size_t row_bytes = (size_t) rows * src->nb[1];
        for (int64_t h = 0; h < src->ne[2]; ++h) {
            ggml_backend_tensor_get(src, (char *) dst->data + (size_t) h * dst->nb[2], (size_t) h * src->nb[2], row_bytes);
        }
    };
    for (int il = 0; il < n_slots; ++il) {
        const bool is_swa = weights_.layers[weights_.phys_layer_for_slot(il)].is_swa;
        const int rows = is_swa ? cache_.swa_cap : pos;
        copy_kv_to_host(cache_.k_cache[il], s.k_cache[il], rows);
        copy_kv_to_host(cache_.v_cache[il], s.v_cache[il], rows);
        ggml_backend_tensor_get(cache_.conv_state[il], s.conv_state[il]->data, 0, ggml_nbytes(s.conv_state[il]));
    }
    if (weights_.dspark.loaded) {
        for (int i = 0; i < weights_.dspark.n_layer; ++i) {
            copy_kv_to_host(cache_.dspark_k[i], s.dspark_k[i], pos);
            copy_kv_to_host(cache_.dspark_v[i], s.dspark_v[i], pos);
        }
        ggml_backend_tensor_get(cache_.last_taps_dev, s.last_taps->data, 0, ggml_nbytes(s.last_taps));
    }

    s.cur_pos = pos;
    s.dspark_ctx_pos = cache_.dspark_ctx_pos;
    s.logits = last_logits_;
    s.used = true;
    return true;
}

void Inkling2Backend::snapshot_free(int slot) {
    if (slot >= 0 && slot < kMaxSlots) free_snapshot(snapshots_[slot]);
}

bool Inkling2Backend::snapshot_used(int slot) const {
    return slot >= 0 && slot < kMaxSlots && snapshots_[slot].used;
}

int Inkling2Backend::snapshot_cur_pos(int slot) const {
    return snapshot_used(slot) ? snapshots_[slot].cur_pos : 0;
}

GenerateResult Inkling2Backend::restore_and_generate_impl(int slot,
                                                          const GenerateRequest & req,
                                                          const DaemonIO & io) {
    if (!snapshot_used(slot) || parked_ || released_ || cache_.remote_backend) {
        return generate_impl(req, io);
    }
    const Inkling2Snapshot & s = snapshots_[slot];
    if (s.cur_pos > (int) req.prompt.size()) {
        return generate_impl(req, io);
    }

    auto copy_kv_to_dev = [](ggml_tensor * src, ggml_tensor * dst, int rows) {
        const size_t row_bytes = (size_t) rows * dst->nb[1];
        for (int64_t h = 0; h < dst->ne[2]; ++h) {
            ggml_backend_tensor_set(dst, (const char *) src->data + (size_t) h * src->nb[2], (size_t) h * dst->nb[2], row_bytes);
        }
    };
    const int n_slots = weights_.n_layer_eff;
    for (int il = 0; il < n_slots; ++il) {
        const bool is_swa = weights_.layers[weights_.phys_layer_for_slot(il)].is_swa;
        const int rows = is_swa ? cache_.swa_cap : s.cur_pos;
        copy_kv_to_dev(s.k_cache[il], cache_.k_cache[il], rows);
        copy_kv_to_dev(s.v_cache[il], cache_.v_cache[il], rows);
        ggml_backend_tensor_set(cache_.conv_state[il], s.conv_state[il]->data, 0, ggml_nbytes(s.conv_state[il]));
    }
    if (weights_.dspark.loaded && s.last_taps) {
        for (int i = 0; i < weights_.dspark.n_layer; ++i) {
            copy_kv_to_dev(s.dspark_k[i], cache_.dspark_k[i], s.cur_pos);
            copy_kv_to_dev(s.dspark_v[i], cache_.dspark_v[i], s.cur_pos);
        }
        ggml_backend_tensor_set(cache_.last_taps_dev, s.last_taps->data, 0, ggml_nbytes(s.last_taps));
    }
    cache_.cur_pos = s.cur_pos;
    cache_.dspark_ctx_pos = s.dspark_ctx_pos;
    last_logits_ = s.logits;

    return run(req, io, s.cur_pos);
}

GenerateResult Inkling2Backend::generate_impl(const GenerateRequest & req,
                                              const DaemonIO & io) {
    last_logits_.clear();
    if (!parked_ && !released_) reset_inkling2_cache(cache_);
    return run(req, io, 0);
}

// ─── Prefill + DSpark Speculative Decode Loop ───────────────────────────
GenerateResult Inkling2Backend::run(const GenerateRequest & req, const DaemonIO & raw_io, int restored) {
    GenerateResult result;
    result.restored_prefix_tokens = restored;
    const DaemonIO io = raw_io.with_token_callback(req.on_token);

    if (parked_ || released_) {
        result.fail(GenerateErrorCode::ModelParked,
                    released_ ? "inkling2 GPUs are released" : "inkling2 target is parked");
        return result;
    }
    if (req.prompt.empty()) {
        result.fail(GenerateErrorCode::PrefillFailed, "empty prompt");
        return result;
    }
    if ((int64_t) req.prompt.size() + std::max(0, req.n_gen) > cache_.max_ctx) {
        result.fail(GenerateErrorCode::ContextOverflow, "inkling2 request exceeds max_ctx");
        return result;
    }

    const int max_vw = weights_.dspark.loaded
        ? std::min(kInkling2MaxVerify, weights_.dspark.block_size)
        : 1;
    int vw = cfg_.verify_width;
    if (const char * evw = std::getenv("LUCE_INKLING2_VERIFY_WIDTH")) {
        const int v = std::atoi(evw);
        if (v >= 1 && v <= max_vw) vw = v;
    }
    const bool use_dspark = weights_.dspark.loaded && !req.force_ar_decode && vw != 1 && req.n_gen > 2;
    const bool is_dual = (weights_.expert_backend && weights_.split_exp > 0);
    // Shared cost-aware AdaptiveSpecWidth controller over all widths 2..max_vw
    // (seed-inclusive). Uses a hardware-agnostic linear shape seed scaled to
    // live measured step times per context range and execution mode (single vs
    // dual GPU), with reached-depth acceptance tracking across every depth.
    const bool fixed_vw = (vw >= 2 && vw <= max_vw);
    const bool adaptive_vw = use_dspark && !fixed_vw &&
        (std::getenv("LUCE_INKLING2_ADAPTIVE_VW") == nullptr ||
         std::atoi(std::getenv("LUCE_INKLING2_ADAPTIVE_VW")) != 0);
    if (adaptive_vw) {
        if (!spec_width_init_ || spec_width_.max_width() != max_vw) {
            spec_width_ = AdaptiveSpecWidth(max_vw, /*min_width=*/2, /*enabled=*/true);
            std::vector<float> shape((size_t) max_vw + 1, 0.0f);
            for (int w = 2; w <= max_vw; ++w) {
                shape[(size_t) w] = 1.0f + 0.08f * (float) (w - 1);
            }
            spec_width_.set_relative_costs(shape, AdaptiveSpecWidth::CostSeed::kShape);
            spec_width_.set_acceptance_model(AdaptiveSpecWidth::AcceptanceModel::kReachedDepths);
            spec_width_init_ = true;
        } else {
            spec_width_.carry_acceptance(32.0f);
        }
        (is_dual ? spec_costs_dual_ : spec_costs_single_).load(spec_width_, (int64_t) req.prompt.size());
    }

    const bool is_greedy = sample_logits_is_argmax(req.sampler);
    // ─── 2DSD Greedy Path (Stage 2b default with static draft_confidence pruning) ──
    const bool use_2dsd = (weights_.draft_mode == Inkling2DraftMode::TwoDSD) &&
                          weights_.twodsd.loaded && cache_.tree.active &&
                          !req.force_ar_decode && vw != 1 && is_greedy && req.n_gen > 2 && restored == 0;
    if (use_2dsd) {
        const auto t_pre0 = std::chrono::steady_clock::now();
        int pos_2d = 0;
        int32_t first_tok = -1;
        while (pos_2d < (int) req.prompt.size()) {
            if (io.is_cancelled()) {
                result.fail(GenerateErrorCode::Cancelled, "cancelled during 2dsd prefill");
                return result;
            }
            const int end = std::min((int) req.prompt.size(), pos_2d + chunk_);
            const int n   = end - pos_2d;
            const bool is_last = (end == (int) req.prompt.size());
            if (!inkling2_2dsd_prefill_chunk(backend_, weights_, cache_, req.prompt.data() + pos_2d, n, pos_2d, is_last, first_tok)) {
                result.fail(GenerateErrorCode::PrefillFailed, "inkling2 2dsd prefill failed");
                return result;
            }
            pos_2d = end;
            if (io.observer) io.observer("prefill_chunk", { pos_2d, (int) req.prompt.size() });
        }
        const auto t_pre1 = std::chrono::steady_clock::now();
        result.prefill_s = std::chrono::duration<double>(t_pre1 - t_pre0).count();

        if (req.n_gen <= 0) {
            result.succeed();
            return result;
        }

        ThinkingBudget budget(req.budget_hook, req.n_gen);
        bool cancelled = false;
        auto commit_2d = [&](int32_t & tok) -> bool {
            tok = budget.apply(tok).token;
            result.budget_forced_close = budget.forced_close();
            result.tokens.push_back(tok);
            io.emit(tok);
            if (io.is_cancelled()) {
                cancelled = true;
                return false;
            }
            return tok != weights_.eos_id && tok != weights_.eos_chat_id && (int) result.tokens.size() < req.n_gen;
        };

        class GpuTwoDSDRunner final : public Inkling2TwoDSDRunner {
        public:
            GpuTwoDSDRunner(ggml_backend_t b, const Inkling2Weights & w, Inkling2Cache & c, bool split)
                : b_(b), w_(w), c_(c), split_(split) {}
            int prefix_length() const override { return c_.tree.prefix_len; }
            int max_nodes() const override { return c_.tree.max_nodes; }
            int max_ctx() const override { return c_.max_ctx; }
            bool advance_nodes(const std::vector<Inkling2TwoDSDNodeDesc> & nodes,
                               uint32_t readout_mask,
                               std::vector<int> & out_rows,
                               std::vector<int32_t> & out_argmax) override {
                return inkling2_2dsd_advance_nodes(b_, w_, c_, nodes, readout_mask, split_, out_rows, out_argmax);
            }
            bool commit_nodes(const int * slots, int n) override {
                return inkling2_2dsd_commit_nodes(b_, w_, c_, slots, n);
            }
            bool draft_batch(const std::vector<Inkling2TwoDSDDraftReq> & reqs,
                             std::vector<Inkling2TwoDSDDraftOut> & out) override {
                return inkling2_2dsd_draft_batch(b_, w_, c_, reqs, out);
            }
        private:
            ggml_backend_t b_;
            const Inkling2Weights & w_;
            Inkling2Cache & c_;
            bool split_;
        };

        const auto td_cfg = inkling2_2dsd_config_from_env(weights_.n_passes, weights_.twodsd.block_size);
        GpuTwoDSDRunner runner(backend_, weights_, cache_, is_dual);
        Inkling2TwoDSDStats td_stats;
        std::string td_err;

        const auto t_dec0 = std::chrono::steady_clock::now();
        const bool ok_2d = inkling2_2dsd_run_greedy(runner, td_cfg, first_tok, req.n_gen, commit_2d, td_stats, td_err);
        const auto t_dec1 = std::chrono::steady_clock::now();

        if (!ok_2d) {
            result.fail(GenerateErrorCode::DecodeFailed, td_err);
            return result;
        }
        if (cancelled) {
            result.fail(GenerateErrorCode::Cancelled, "cancelled during 2dsd decode");
            return result;
        }
        result.spec_decode_ran = (td_stats.target_calls > 0);
        result.decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
        result.accept_rate = td_stats.draft_slots > 0 ? (double) td_stats.commit_draft / (double) td_stats.draft_slots : 0.0;
        const double s_count = td_stats.target_calls > 0
            ? (double) td_cfg.R * (double) std::max<size_t>(1, result.tokens.size()) / (double) td_stats.target_calls
            : 0.0;
        std::fprintf(stderr,
            "[inkling2-2dsd] decode: %zu tok in %.3f s (%.1f tok/s) | calls=%lld S_count=%.2f avg_cells=%.1f max_live=%d | "
            "commits(main=%lld frozen=%lld fork=%lld bonus=%lld) promo=%lld forks=%lld | "
            "draft=%.3f s batches=%lld slots=%lld cut=%lld (conf=%.2f)\n",
            result.tokens.size(), result.decode_s,
            result.decode_s > 0 ? result.tokens.size() / result.decode_s : 0.0,
            td_stats.target_calls, s_count,
            td_stats.target_calls > 0 ? (double) td_stats.target_cells / (double) td_stats.target_calls : 0.0,
            td_stats.max_live_nodes,
            td_stats.commit_main, td_stats.commit_frozen, td_stats.commit_fork, td_stats.commit_bonus,
            td_stats.promotions, td_stats.forks,
            td_stats.draft_s, td_stats.drafter_batches, td_stats.draft_slots, td_stats.draft_slots_cut,
            td_cfg.draft_confidence);
        result.succeed();
        return result;
    }
    // Exact GPU top-candidate fast path for standard stochastic sampling (temp > 0, no penalties/grammar):
    // Extracts <=256 candidates + exact full-vocab Z on GPU in ~20 us (cuts non-greedy CPU stall from 17 ms -> 0.02 ms!).
    const bool use_fast_gpu_sample = !is_greedy && req.sampler.temp > 0.0f &&
        req.sampler.rep_pen <= 1.0f && req.sampler.freq_pen == 0.0f &&
        req.sampler.pres_pen == 0.0f && req.sampler.constraint == nullptr;
    const float fast_temp = use_fast_gpu_sample ? req.sampler.temp : 0.0f;
    const int n_vocab = weights_.n_vocab_unpadded;

    int pos = restored;
    std::vector<float> logits = last_logits_;
    int32_t greedy_tok = -1;

    // ─── Prefill (3-Pass Wavefront Loop-Fused when >= 2 chunks, else chunked) ──
    const auto t_pre0 = std::chrono::steady_clock::now();
    const int rem_pre = (int) req.prompt.size() - pos;
    // Dynamically use chunk=1024 for >=2048-tok Wavefront Prefill (~3,600 tok/s on R9700!),
    // or chunk=512 for 1024..2047-tok Wavefront Prefill (~3,000 tok/s).
    int wf_chunk = (rem_pre >= 2048 && cache_.swa_cap >= 2048) ? std::max(chunk_, 1024) : chunk_;
    if (const char * ec = std::getenv("LUCE_INKLING2_WF_CHUNK")) {
        const int v = std::atoi(ec);
        if (v >= 128 && v <= cache_.swa_cap - weights_.swa_window) wf_chunk = v;
    }
    const bool use_wavefront = weights_.is_loop && weights_.n_passes == 3 &&
        !cache_.remote_backend && req.snap_slot < 0 &&
        (rem_pre >= wf_chunk * 2) &&
        (std::getenv("LUCE_INKLING2_WAVEFRONT_PREFILL") == nullptr ||
         std::atoi(std::getenv("LUCE_INKLING2_WAVEFRONT_PREFILL")) != 0);
    if (use_wavefront) {
        Inkling2ForwardOut r = inkling2_prefill_wavefront(
            backend_, weights_, cache_, req.prompt.data() + pos, rem_pre, pos, wf_chunk,
            /*need_logits=*/!is_greedy, /*capture_taps=*/weights_.dspark.loaded);
        if (!r.ok) {
            result.fail(GenerateErrorCode::PrefillFailed, "inkling2 wavefront prefill failed");
            return result;
        }
        pos += rem_pre;
        if (!r.greedy.empty()) greedy_tok = r.greedy.back();
        if (!r.logits.empty()) {
            logits.assign(r.logits.end() - n_vocab, r.logits.end());
            last_logits_ = logits;
        }
    }
    while (pos < (int) req.prompt.size()) {
        if (io.is_cancelled()) {
            result.fail(GenerateErrorCode::Cancelled, "cancelled during prefill");
            return result;
        }
        int end = std::min((int) req.prompt.size(), pos + chunk_);
        if (req.snap_slot >= 0 && req.snap_pos > pos && req.snap_pos < end) {
            end = req.snap_pos;
        }
        const int n = end - pos;
        const bool is_last_chunk = (end == (int) req.prompt.size());
        const bool need_l = is_last_chunk && !is_greedy;

        Inkling2ForwardOut r = inkling2_forward(
            backend_, weights_, cache_, req.prompt.data() + pos, n, pos,
            /*verify=*/false, /*need_logits=*/need_l || (req.snap_slot >= 0 && req.snap_pos == end),
            /*capture_taps=*/weights_.dspark.loaded);
        if (!r.ok) {
            result.fail(GenerateErrorCode::PrefillFailed, "inkling2 prefill forward failed");
            return result;
        }
        pos += n;
        if (!r.greedy.empty()) greedy_tok = r.greedy.back();
        if (!r.logits.empty()) {
            logits.assign(r.logits.end() - n_vocab, r.logits.end());
            last_logits_ = logits;
        }
        if (req.snap_slot >= 0 && req.snap_pos == pos) {
            snapshot_save(req.snap_slot);
        }
        if (io.observer) {
            io.observer("prefill_chunk", {pos, (int) req.prompt.size()});
        }
    }
    const auto t_pre1 = std::chrono::steady_clock::now();
    result.prefill_s = std::chrono::duration<double>(t_pre1 - t_pre0).count();

    if (req.n_gen <= 0) {
        result.succeed();
        return result;
    }

    std::mt19937_64 rng(request_rng_seed(req.sampler));
    std::vector<int32_t> history = req.prompt;
    auto sample_row = [&](const float * row) -> int32_t {
        return (int32_t) sample_logits(row, n_vocab, req.sampler, history, rng);
    };
    // Exact fast sampler over GPU-extracted top-256 candidates + exact full-vocab Z (auto-fallback to full row if needed!)
    std::vector<float> fallback_row;
    auto sample_row_fast = [&](const Inkling2ForwardOut & fr, int row_idx) -> int32_t {
        if (!fr.top_cands.empty()) {
            const int32_t * rc = fr.top_cands.data() + (size_t) row_idx * 516;
            const int cnt = rc[0];
            float gmax = 0.0f, Z = 0.0f;
            std::memcpy(&gmax, &rc[1], sizeof(float));
            std::memcpy(&Z,    &rc[2], sizeof(float));
            if (cnt > 0 && cnt <= 256 && Z > 0.0f && std::isfinite(Z)) {
                struct Cand { float v; int32_t id; };
                Cand cands[256];
                for (int j = 0; j < cnt; ++j) {
                    std::memcpy(&cands[j].v, &rc[4 + 2 * j], sizeof(float));
                    cands[j].id = rc[4 + 2 * j + 1];
                }
                std::sort(cands, cands + cnt, [](const Cand & a, const Cand & b) {
                    return a.v > b.v || (a.v == b.v && a.id < b.id);
                });
                const int k_lim = (req.sampler.top_k > 0 && req.sampler.top_k < cnt) ? req.sampler.top_k : cnt;
                const float inv_t = 1.0f / std::max(req.sampler.temp, 1e-6f);
                float probs[256];
                double cum = 0.0;
                int keep = 0;
                for (int j = 0; j < k_lim; ++j) {
                    const float p = std::exp((cands[j].v - gmax) * inv_t) / Z;
                    probs[j] = p;
                    cum += p;
                    keep = j + 1;
                    if (req.sampler.top_p > 0.0f && req.sampler.top_p < 1.0f && cum >= req.sampler.top_p) {
                        break;
                    }
                }
                const bool exact_ok =
                    (req.sampler.top_k > 0 && cnt >= req.sampler.top_k) ||
                    (req.sampler.top_p > 0.0f && req.sampler.top_p < 1.0f && cum >= req.sampler.top_p) ||
                    (cum >= 0.9995);
                if (exact_ok && keep > 0 && cum > 0.0) {
                    std::uniform_real_distribution<double> u01(0.0, cum);
                    double r = u01(rng);
                    double acc = 0.0;
                    for (int j = 0; j < keep; ++j) {
                        acc += probs[j];
                        if (r <= acc || j == keep - 1) return cands[j].id;
                    }
                }
            }
            // Rare fallback (flat high-temperature distribution): fetch only this single row
            if (fr.logits_dev) {
                fallback_row.resize((size_t) n_vocab);
                ggml_backend_tensor_get(fr.logits_dev, fallback_row.data(),
                    (size_t) row_idx * n_vocab * sizeof(float), (size_t) n_vocab * sizeof(float));
                return sample_row(fallback_row.data());
            }
        }
        return sample_row(fr.logits.data() + (size_t) row_idx * n_vocab);
    };
    ThinkingBudget budget(req.budget_hook, req.n_gen);

    bool cancelled = false;
    auto commit = [&](int32_t & tok) -> bool {
        tok = budget.apply(tok).token;
        result.budget_forced_close = budget.forced_close();
        result.tokens.push_back(tok);
        history.push_back(tok);
        io.emit(tok);
        if (io.is_cancelled()) {
            cancelled = true;
            return false;
        }
        return tok != weights_.eos_id && tok != weights_.eos_chat_id && (int) result.tokens.size() < req.n_gen;
    };

    const auto t_dec0 = std::chrono::steady_clock::now();
    int32_t next_tok = (is_greedy && greedy_tok >= 0) ? greedy_tok : sample_row(logits.data());
    bool more = commit(next_tok);

    long long total_drafts = 0;
    long long total_accepted = 0;
    long long spec_steps = 0;
    long long pos_hits[kInkling2MaxVerify] = {};
    long long pos_tries[kInkling2MaxVerify] = {};
    int vw_hist[kInkling2MaxVerify + 1] = {};
    double draft_s_total = 0.0;

    // Automatic Dual-GPU Thermal / Contention Guard:
    // Because target GPU holds all 128 experts, if Dual-GPU step latency degrades
    // sharply above its steady baseline for 2 consecutive steps (e.g. APU thermal
    // throttle or contention), seamlessly drop to target-only verify.
    bool allow_dual_split = is_dual;
    int dual_slow_streak = 0;
    double dual_fast_ms = 0.0;

    // ─── Decode Loop: DSpark Block-8 Speculative or Fast AR ─────────────
    while (more) {
        const int rem_gen = req.n_gen - (int) result.tokens.size();
        const int rem_ctx = cache_.max_ctx - pos - 1;
        const int step_max_vw = std::min({max_vw, rem_gen, rem_ctx + 1});
        const int cur_vw = !use_dspark ? 1
            : fixed_vw ? std::min(vw, step_max_vw)
            : adaptive_vw ? spec_width_.next_width_cost_aware({}, step_max_vw)
            : step_max_vw;
        const int k_draft = (use_dspark && cur_vw >= 2) ? (cur_vw - 1) : 0;

        if (k_draft >= 1) {
            // 1. Draft k_draft speculative tokens with DSpark in ONE HIP graph!
            const auto tstep0 = std::chrono::steady_clock::now();
            const auto td0 = tstep0;
            std::vector<int32_t> drafts;
            const bool draft_ok = inkling2_dspark_draft(
                backend_, weights_, cache_, next_tok, pos, k_draft, drafts,
                adaptive_vw ? (max_vw - 1) : 0);
            const auto td1 = std::chrono::steady_clock::now();
            draft_s_total += std::chrono::duration<double>(td1 - td0).count();

            if (draft_ok && !drafts.empty()) {
                result.spec_decode_ran = true;
                const int n_d = (int) drafts.size();
                if (io.observer) io.observer("draft", drafts);

                // 2. Verify [next_tok, draft_0, ..., draft_{n_d-1}] in ONE parallel target forward!
                std::vector<int32_t> verify_ids(1 + n_d);
                verify_ids[0] = next_tok;
                for (int i = 0; i < n_d; ++i) verify_ids[1 + i] = drafts[i];

                Inkling2ForwardOut vr = inkling2_forward(
                    backend_, weights_, cache_, verify_ids.data(), 1 + n_d, pos,
                    /*verify=*/true, /*need_logits=*/!is_greedy, /*capture_taps=*/true,
                    /*allow_split=*/allow_dual_split, /*fast_sample_temp=*/fast_temp);
                if (!vr.ok) {
                    result.fail(GenerateErrorCode::DecodeFailed, "inkling2 verify forward failed");
                    return result;
                }

                // 3. Exact rejection / acceptance check (greedy fast-path or GPU top-cand / full sampling)
                int n_accepted = 0;
                int32_t bonus_tok = -1;
                for (int i = 0; i <= n_d; ++i) {
                    int32_t target_tok = is_greedy
                        ? vr.greedy[i]
                        : sample_row_fast(vr, i);
                    if (i < n_d) {
                        int32_t emit_tok = target_tok;
                        more = commit(emit_tok);
                        if (!more) {
                            bonus_tok = emit_tok;
                            n_accepted = i; // committed through verify_ids[i]
                            break;
                        }
                        if (emit_tok == drafts[i]) {
                            ++n_accepted;
                            continue;
                        } else {
                            bonus_tok = emit_tok;
                            break;
                        }
                    } else {
                        // All n_d drafts accepted -> emit the bonus token at row n_d!
                        int32_t emit_tok = target_tok;
                        more = commit(emit_tok);
                        bonus_tok = emit_tok;
                    }
                }

                const int retained = n_accepted + 1; // verify_ids[0 .. n_accepted] are committed
                // 4. Exact O(1)-launch conv rollback + commit accepted taps to DSpark KV!
                if (!inkling2_verify_rollback(backend_, weights_, cache_, pos, retained)) {
                    result.fail(GenerateErrorCode::DecodeFailed, "inkling2 conv rollback failed");
                    return result;
                }
                ggml_tensor * taps_dev = cache_.verify_ws[1 + n_d].taps_out;
                if (taps_dev) {
                    inkling2_dspark_commit_taps(backend_, weights_, cache_, taps_dev, pos, retained);
                }

                for (int i = 0; i < n_d; ++i) {
                    pos_tries[i]++;
                    if (i < n_accepted) pos_hits[i]++;
                }

                // Shared AdaptiveSpecWidth + Dual-GPU Thermal Guard update:
                const double step_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tstep0).count();
                if (adaptive_vw) {
                    spec_width_.observe(retained, 1 + n_d, (float) step_ms);
                }
                if (allow_dual_split && weights_.target_has_all_experts && spec_steps >= 2 && step_ms < 120.0) {
                    if (dual_fast_ms <= 0.0 || step_ms < dual_fast_ms) {
                        dual_fast_ms = step_ms;
                    }
                    const double trip_ms = std::max(30.0, dual_fast_ms * 1.45);
                    if (step_ms > trip_ms) {
                        if (++dual_slow_streak >= 2) {
                            if (adaptive_vw) {
                                spec_costs_dual_.store(spec_width_, (int64_t) pos);
                                spec_costs_single_.load(spec_width_, (int64_t) pos);
                            }
                            allow_dual_split = false;
                        }
                    } else {
                        dual_slow_streak = 0;
                    }
                }

                pos += retained;
                next_tok = bonus_tok;
                total_drafts += n_d;
                total_accepted += n_accepted;
                ++spec_steps;
                if (1 + n_d <= kInkling2MaxVerify) ++vw_hist[1 + n_d];
                if (io.observer) io.observer("accept", {n_accepted, n_d});
                continue;
            }
        }

        // Autoregressive fallback / T=1 decode (zero PCIe hops on R9700!)
        Inkling2ForwardOut r = inkling2_forward(
            backend_, weights_, cache_, &next_tok, 1, pos,
            /*verify=*/false, /*need_logits=*/!is_greedy, /*capture_taps=*/weights_.dspark.loaded,
            /*allow_split=*/false, /*fast_sample_temp=*/fast_temp);
        if (!r.ok) {
            result.fail(GenerateErrorCode::DecodeFailed, "inkling2 AR decode failed");
            return result;
        }
        pos += 1;
        next_tok = is_greedy ? r.greedy[0] : sample_row_fast(r, 0);
        more = commit(next_tok);
    }

    if (adaptive_vw && spec_steps > 0) {
        (allow_dual_split ? spec_costs_dual_ : spec_costs_single_).store(
            spec_width_, (int64_t) req.prompt.size());
    }

    if (cancelled) {
        result.fail(GenerateErrorCode::Cancelled, "cancelled during decode");
        return result;
    }

    const auto t_dec1 = std::chrono::steady_clock::now();
    result.decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    result.accept_rate = total_drafts > 0 ? (double) total_accepted / (double) total_drafts : 0.0;
    if (spec_steps > 0) {
        std::fprintf(stderr,
            "[inkling2] decode: %zu tok in %.3f s (%.1f tok/s), dspark steps=%lld accepted=%lld/%lld (%.2f/step, %.1f%%), draft=%.3f s | vw=[2:%d 3:%d 4:%d 5:%d 6:%d 7:%d 8:%d] | pos_acc=[%.0f%% %.0f%% %.0f%% %.0f%% %.0f%% %.0f%% %.0f%%]\n",
            result.tokens.size(), result.decode_s,
            result.decode_s > 0 ? result.tokens.size() / result.decode_s : 0.0,
            spec_steps, total_accepted, total_drafts,
            (double) (total_accepted + spec_steps) / (double) spec_steps,
            100.0 * result.accept_rate, draft_s_total,
            vw_hist[2], vw_hist[3], vw_hist[4], vw_hist[5], vw_hist[6], vw_hist[7], vw_hist[8],
            pos_tries[0] ? 100.0 * pos_hits[0] / pos_tries[0] : 0.0,
            pos_tries[1] ? 100.0 * pos_hits[1] / pos_tries[1] : 0.0,
            pos_tries[2] ? 100.0 * pos_hits[2] / pos_tries[2] : 0.0,
            pos_tries[3] ? 100.0 * pos_hits[3] / pos_tries[3] : 0.0,
            pos_tries[4] ? 100.0 * pos_hits[4] / pos_tries[4] : 0.0,
            pos_tries[5] ? 100.0 * pos_hits[5] / pos_tries[5] : 0.0,
            pos_tries[6] ? 100.0 * pos_hits[6] / pos_tries[6] : 0.0);
    }
    result.succeed();
    return result;
}

} // namespace luce::common
