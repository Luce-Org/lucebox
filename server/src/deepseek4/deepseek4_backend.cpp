// DeepSeek4Backend implementation — AR-only decode, chunked prefill.

#include "deepseek4_backend.h"
#include "deepseek4_internal.h"
#include "common/sampler.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dflash::common {

namespace {
using Clock = std::chrono::steady_clock;

static double elapsed_s(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

static bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] && std::strcmp(value, "0") != 0;
}

static uint64_t layer_expert_bytes(const DeepSeek4Layer & layer, int n_expert) {
    if (n_expert <= 0) return 0;
    uint64_t bytes = 0;
    if (layer.ffn_gate_exps) bytes += ggml_nbytes(layer.ffn_gate_exps) / (uint64_t) n_expert;
    if (layer.ffn_up_exps) bytes += ggml_nbytes(layer.ffn_up_exps) / (uint64_t) n_expert;
    if (layer.ffn_down_exps) bytes += ggml_nbytes(layer.ffn_down_exps) / (uint64_t) n_expert;
    return bytes;
}

static uint64_t estimate_ds4_cache_bytes(const DeepSeek4Weights & w, int max_ctx) {
    size_t total_bytes = 0;
    const size_t head_dim = (size_t) w.head_dim;
    const size_t swa_size = (size_t) w.n_swa;

    for (int il = 0; il < w.n_layer; ++il) {
        total_bytes += swa_size * head_dim * sizeof(uint16_t);
        const uint32_t ratio = w.compress_ratios[(size_t) il];
        if (ratio == 0) continue;

        const size_t comp_cap = (size_t) (max_ctx / (int) ratio) + 16;
        total_bytes += comp_cap * head_dim * sizeof(uint16_t);

        const size_t window = (ratio == 4) ? 8 : ratio;
        total_bytes += window * head_dim * sizeof(float) * 2;

        if (ratio == 4) {
            const size_t index_comp_width = (size_t) w.n_indexer_head * (size_t) w.n_indexer_head_dim;
            total_bytes += comp_cap * index_comp_width * sizeof(uint16_t);
            total_bytes += window * index_comp_width * sizeof(float) * 2;
        }
    }

    total_bytes += (size_t) w.n_hc * (size_t) w.n_embd * sizeof(float);
    return total_bytes;
}

}  // namespace

DeepSeek4Backend::DeepSeek4Backend(const DeepSeek4BackendConfig & cfg)
    : cfg_(cfg) {}

DeepSeek4Backend::~DeepSeek4Backend() {
    shutdown();
}

bool DeepSeek4Backend::init() {
    backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!backend_) {
        std::fprintf(stderr, "[deepseek4] failed to create CUDA backend (gpu=%d)\n",
                     cfg_.device.gpu);
        return false;
    }

    snap_backend_ = ggml_backend_init_by_name("cpu", nullptr);

    if (env_flag_enabled("DFLASH_DS4_HYBRID")) {
        if (!init_hybrid_model()) {
            return false;
        }
    } else if (!load_deepseek4_gguf(cfg_.model_path, backend_, w_)) {
        std::fprintf(stderr, "[deepseek4] failed to load model: %s\n", cfg_.model_path);
        return false;
    }

    const int max_ctx = cfg_.max_ctx > 0 ? cfg_.max_ctx : 8192;
    if (!create_deepseek4_cache(backend_, w_, max_ctx, cache_)) {
        std::fprintf(stderr, "[deepseek4] failed to allocate KV cache (ctx=%d)\n", max_ctx);
        return false;
    }

    std::fprintf(stderr, "[deepseek4] initialized: %d layers, ctx=%d, %d experts (%d used)%s\n",
                 w_.n_layer, max_ctx, w_.n_expert, w_.n_expert_used,
                 moe_hybrid_ ? " [hybrid]" : "");
    return true;
}

bool DeepSeek4Backend::compute_uniform_hybrid_placement(const DeepSeek4Weights & w,
                                                       int max_ctx,
                                                       MoeHybridPlacement & out,
                                                       std::string * err) const {
    size_t gpu_free = 0;
    size_t gpu_total = 0;
    ggml_backend_cuda_get_device_memory(cfg_.device.gpu, &gpu_free, &gpu_total);
    if (gpu_total == 0) {
        if (err) *err = "could not query GPU memory";
        return false;
    }

    std::vector<uint64_t> layer_bytes((size_t) w.n_layer, 0);
    uint64_t total_expert_bytes = 0;
    uint64_t bytes_per_uniform_round = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        const uint64_t bytes = layer_expert_bytes(w.layers[(size_t) il], w.n_expert);
        layer_bytes[(size_t) il] = bytes;
        total_expert_bytes += bytes * (uint64_t) w.n_expert;
        bytes_per_uniform_round += bytes;
    }
    if (bytes_per_uniform_round == 0) {
        if (err) *err = "expert tensor metadata missing after partial load";
        return false;
    }

    const uint64_t core_bytes = gpu_total - gpu_free;
    const uint64_t kv_bytes = estimate_ds4_cache_bytes(w, max_ctx);
    const uint64_t warm_bytes = 256ULL * 1024 * 1024;
    const uint64_t safety_bytes = 512ULL * 1024 * 1024;

    uint64_t expert_budget = 0;
    if (gpu_total > core_bytes + kv_bytes + warm_bytes + safety_bytes) {
        expert_budget = gpu_total - core_bytes - kv_bytes - warm_bytes - safety_bytes;
    }
    if (expert_budget > total_expert_bytes) {
        expert_budget = total_expert_bytes;
    }
    if (const char * cap_env = std::getenv("DFLASH_EXPERT_BUDGET_MB")) {
        const uint64_t cap_bytes = (uint64_t) std::max(0, std::atoi(cap_env)) * 1024ULL * 1024ULL;
        if (cap_bytes > 0 && cap_bytes < expert_budget) {
            expert_budget = cap_bytes;
        }
    }
    if (expert_budget == 0) {
        if (err) *err = "no VRAM budget available for DS4 experts";
        return false;
    }

    const int hot_per_layer = std::min(w.n_expert, (int) (expert_budget / bytes_per_uniform_round));
    if (hot_per_layer <= 0) {
        if (err) *err = "expert budget is smaller than one uniform expert round";
        return false;
    }

    out = {};
    out.n_layer = w.n_layer;
    out.n_expert = w.n_expert;
    out.n_expert_used = w.n_expert_used;
    out.hot_counts.assign((size_t) w.n_layer, hot_per_layer);
    out.hot_expert_ids.resize((size_t) w.n_layer);
    out.total_hot = hot_per_layer * w.n_layer;
    for (int il = 0; il < w.n_layer; ++il) {
        auto & ids = out.hot_expert_ids[(size_t) il];
        ids.reserve((size_t) hot_per_layer);
        for (int ie = 0; ie < hot_per_layer; ++ie) {
            ids.push_back((int32_t) ie);
        }
    }

    std::fprintf(stderr,
                 "[deepseek4] hybrid placement: gpu_total=%.2f GiB core=%.2f GiB kv=%.2f GiB expert_budget=%.2f GiB hot/layer=%d\n",
                 gpu_total / 1024.0 / 1024.0 / 1024.0,
                 core_bytes / 1024.0 / 1024.0 / 1024.0,
                 kv_bytes / 1024.0 / 1024.0 / 1024.0,
                 expert_budget / 1024.0 / 1024.0 / 1024.0,
                 hot_per_layer);
    return true;
}

bool DeepSeek4Backend::init_hybrid_model() {
    TargetLoadPlan plan;
    plan.skip_expert_tensors = true;
    if (!load_deepseek4_gguf_partial(cfg_.model_path, backend_, plan, w_)) {
        std::fprintf(stderr, "[deepseek4] failed to partially load model for hybrid mode: %s\n",
                     cfg_.model_path);
        return false;
    }

    std::string err;
    const int max_ctx = cfg_.max_ctx > 0 ? cfg_.max_ctx : 8192;
    if (!compute_uniform_hybrid_placement(w_, max_ctx, moe_placement_, &err)) {
        std::fprintf(stderr, "[deepseek4] failed to compute hybrid placement: %s\n", err.c_str());
        return false;
    }

    if (moe_placement_.total_hot >= w_.n_layer * w_.n_expert) {
        free_deepseek4_weights(w_);
        if (!load_deepseek4_gguf(cfg_.model_path, backend_, w_)) {
            std::fprintf(stderr, "[deepseek4] failed to reload full model after placement: %s\n",
                         cfg_.model_path);
            return false;
        }
        return true;
    }

    auto hybrid = std::make_shared<MoeHybridStorage>();
    if (!build_deepseek4_moe_hybrid_storage_from_file(cfg_.model_path, backend_, w_, moe_placement_, *hybrid, &err)) {
        std::fprintf(stderr, "[deepseek4] failed to build hybrid expert storage: %s\n", err.c_str());
        return false;
    }

    moe_hybrid_ = std::move(hybrid);
    w_.moe_hybrid = true;
    const int total_cold = w_.n_layer * w_.n_expert - moe_placement_.total_hot;
    std::fprintf(stderr, "[deepseek4] hybrid experts ready: hot=%d cold=%d\n",
                 moe_placement_.total_hot, total_cold);
    return true;
}

void DeepSeek4Backend::print_ready_banner() const {
    std::printf("[deepseek4-daemon] ready layers=%d ctx=%d experts=%d/%d\n",
                w_.n_layer, cache_.max_ctx, w_.n_expert_used, w_.n_expert);
    std::fflush(stdout);
}

bool DeepSeek4Backend::park(const std::string & what) {
    (void)what;
    // TODO: Release GPU resources
    parked_ = true;
    return true;
}

bool DeepSeek4Backend::unpark(const std::string & what) {
    (void)what;
    parked_ = false;
    return true;
}

int DeepSeek4Backend::do_prefill(const std::vector<int32_t> & tokens,
                                  const DaemonIO & io,
                                  int kv_offset) {
    const int chunk = cfg_.chunk > 0 ? cfg_.chunk : 512;
    const int n_total = (int)tokens.size();
    int pos = kv_offset;

    for (int i = 0; i < n_total; i += chunk) {
        if (io.cancelled) return pos;

        const int n_tok = std::min(chunk, n_total - i);

        // Embed tokens
        std::vector<float> embed(w_.n_embd * n_tok);
        w_.embedder.embed(tokens.data() + i, n_tok, embed.data());

        // Run forward pass
        std::vector<float> logits;
        if (!deepseek4_step(backend_, w_, cache_, embed.data(), n_tok, pos, logits,
                            moe_hybrid_.get())) {
            std::fprintf(stderr, "[deepseek4] prefill step failed at pos=%d\n", pos);
            return -1;
        }
        pos += n_tok;
    }
    return pos;
}

bool DeepSeek4Backend::do_decode(int committed, int n_gen,
                                  std::vector<int32_t> & out_tokens,
                                  const DaemonIO & io,
                                  const BudgetHook & budget_hook,
                                  bool * forced_close_out) {
    if (forced_close_out) *forced_close_out = false;

    for (int generated = 0; generated < n_gen; generated++) {
        if (io.cancelled) break;

        // Budget hook: force-close if remaining budget hits threshold
        if (!budget_hook.close_token_ids.empty() &&
            (n_gen - generated) <= budget_hook.hard_limit_remaining) {
            // Inject close-tag tokens
            for (int32_t close_tok : budget_hook.close_token_ids) {
                out_tokens.push_back(close_tok);
                io.emit(close_tok);
                if (io.cancelled) break;
            }
            if (forced_close_out) *forced_close_out = true;
            break;
        }

        // Get last logits and sample
        std::vector<float> logits;
        {
            // For decode, we embed the last token and run one step
            int32_t last_tok = out_tokens.empty()
                ? -1  // Should not happen in normal flow
                : out_tokens.back();

            // First token of decode uses the last prefill logits
            if (generated == 0 && cache_.cur_pos > 0) {
                // Logits from the last prefill step are already computed
                // We need to sample from them — they should be in the last step's output
                // For now, run one more forward step with the last token
                std::vector<float> embed(w_.n_embd);
                // This is a placeholder — real decode seeds from prefill's last logits
                // TODO: Cache logits from prefill and sample directly
            }

            std::vector<float> embed(w_.n_embd);
            int32_t tok_to_eval = out_tokens.empty() ? 0 : out_tokens.back();
            w_.embedder.embed(&tok_to_eval, 1, embed.data());

            if (!deepseek4_step(backend_, w_, cache_, embed.data(), 1,
                                committed + generated, logits,
                                moe_hybrid_.get())) {
                std::fprintf(stderr, "[deepseek4] decode step failed\n");
                return false;
            }
        }

        // Sample (argmax for now)
        int32_t next_token = 0;
        {
            float max_val = logits[0];
            for (int i = 1; i < w_.n_vocab; i++) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    next_token = i;
                }
            }
        }
        out_tokens.push_back(next_token);
        io.emit(next_token);

        // Check EOS
        // TODO: proper EOS detection from tokenizer metadata
        if (next_token == 151643 || next_token == 151644) {  // common DS EOS/EOT
            break;
        }
    }
    return true;
}

GenerateResult DeepSeek4Backend::generate_impl(const GenerateRequest & req,
                                                const DaemonIO & io) {
    GenerateResult result;
    auto t0 = Clock::now();

    // Prefill
    int committed = do_prefill(req.prompt, io);
    if (committed < 0) {
        result.error = "prefill";
        return result;
    }
    result.prefill_s = elapsed_s(t0);

    if (req.n_gen <= 0) {
        result.ok = true;
        return result;
    }

    // Decode
    auto t1 = Clock::now();
    std::vector<int32_t> gen_tokens;
    gen_tokens.reserve(req.n_gen);

    bool forced_close = false;
    if (!do_decode(committed, req.n_gen, gen_tokens, io,
                   req.budget_hook, &forced_close)) {
        result.error = "decode";
        return result;
    }

    result.ok = true;
    result.tokens = std::move(gen_tokens);
    result.decode_s = elapsed_s(t1);
    result.budget_forced_close = forced_close;
    return result;
}

// ── Snapshots ───────────────────────────────────────────────────────────

bool DeepSeek4Backend::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    // TODO: Implement snapshot save (copy KV cache + HC state to CPU)
    return false;
}

void DeepSeek4Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return;
    free_deepseek4_snapshot(snapshots_[slot]);
}

bool DeepSeek4Backend::snapshot_used(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    return snapshots_[slot].ctx != nullptr;
}

int DeepSeek4Backend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return 0;
    return snapshots_[slot].cur_pos;
}

GenerateResult DeepSeek4Backend::restore_and_generate_impl(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    // TODO: Implement snapshot restore + generate
    (void)slot;
    return generate_impl(req, io);
}

bool DeepSeek4Backend::handle_compress(const std::string & line,
                                        const DaemonIO & io) {
    (void)line; (void)io;
    std::fprintf(stderr, "[deepseek4] compress not yet supported\n");
    return false;
}

void DeepSeek4Backend::free_drafter() {
    // No drafter in AR-only mode
}

void DeepSeek4Backend::shutdown() {
    for (int i = 0; i < PREFIX_SLOTS; i++) {
        free_deepseek4_snapshot(snapshots_[i]);
    }
    free_deepseek4_cache(cache_);
    moe_hybrid_.reset();
    moe_placement_ = {};
    free_deepseek4_weights(w_);
    if (snap_backend_) { ggml_backend_free(snap_backend_); snap_backend_ = nullptr; }
    if (backend_) { ggml_backend_free(backend_); backend_ = nullptr; }
}

}  // namespace dflash::common
