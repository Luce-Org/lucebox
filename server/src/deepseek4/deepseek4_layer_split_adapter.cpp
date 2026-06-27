// DeepSeek4 layer-split adapter implementation.
//
// Splits DS4 layers across CUDA and AMD Halo GPUs. The first shard handles
// embedding + layers [0, split), the second handles layers [split, 43) + output.
// HC state is transferred between shards at the boundary.

#include "deepseek4_layer_split_adapter.h"
#include "deepseek4_internal.h"
#include "common/layer_split_runtime.h"
#include "common/gguf_inspect.h"

#include "ggml-cuda.h"
#if defined(GGML_USE_CUDA)
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dflash::common {

namespace {

static int env_int(const char * name, int fallback) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) return fallback;
    return std::atoi(value);
}

} // namespace

DeepSeek4LayerSplitAdapter::DeepSeek4LayerSplitAdapter(
        const DeepSeek4LayerSplitAdapterConfig & cfg)
    : cfg_(cfg) {
    snapshots_.resize(PREFIX_SLOTS);
}

DeepSeek4LayerSplitAdapter::~DeepSeek4LayerSplitAdapter() {
    shutdown();
}

int DeepSeek4LayerSplitAdapter::estimate_cuda_layers_from_free_bytes(
        size_t free_bytes) {
    const uint64_t fixed_overhead = 2ULL * 1024 * 1024 * 1024;  // 2 GiB for KV cache + overhead
    // Estimate per-layer size from the model file. DS4 has 43 layers + embed/output.
    // Use file size if available, otherwise use conservative 1.9 GiB estimate.
    uint64_t per_layer_bytes = 1900ULL * 1024 * 1024;  // ~1.9 GiB default

    if ((uint64_t) free_bytes <= fixed_overhead) {
        return 1;
    }

    const uint64_t available = (uint64_t) free_bytes - fixed_overhead;
    const int max_layers = (int) (available / per_layer_bytes);
    return std::max(1, std::min(max_layers, 42));
}

size_t DeepSeek4LayerSplitAdapter::hc_state_elements(
        const DeepSeek4Weights & weights) {
    return (size_t) weights.n_hc * (size_t) weights.n_embd;
}

int DeepSeek4LayerSplitAdapter::compute_auto_split_layers() const {
    // Check env override first
    int override_layers = env_int("DFLASH_DS4_CUDA_LAYERS", -1);
    if (override_layers > 0) {
        return override_layers;
    }

    size_t free_bytes = 0, total_bytes = 0;
#if defined(GGML_USE_CUDA)
    int gpu = cfg_.device.gpu;
    if (gpu < 0) gpu = 0;
    const cudaError_t set_device_err = cudaSetDevice(gpu);
    const cudaError_t mem_info_err =
        set_device_err == cudaSuccess
            ? cudaMemGetInfo(&free_bytes, &total_bytes)
            : set_device_err;
    if (mem_info_err != cudaSuccess) {
        free_bytes = 20ULL * 1024 * 1024 * 1024;
        total_bytes = 24ULL * 1024 * 1024 * 1024;
    }
#else
    // Fallback: assume 24GB GPU, 80% usable
    free_bytes = 20ULL * 1024 * 1024 * 1024;
    total_bytes = 24ULL * 1024 * 1024 * 1024;
#endif

    // DS4 layer memory estimation:
    // Per layer (for IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8 quantization):
    //   MoE experts: 256 experts × 3 matrices × (4096 × 2048) in IQ2_XXS ~ 1.5GB
    //   Attention (MLA): Q/KV projections + output in Q8 ~ 200MB
    //   Shared expert + HC + router + compressor ~ 100MB
    //   Total per layer ~ 1.9 GiB
    //
    // Plus fixed costs:
    //   Embedding: ~500MB
    //   Output head: ~500MB
    //   KV cache per layer: ~2MB at 8K ctx
    //   Safety margin: 1GB

    if ((uint64_t) free_bytes <= 2ULL * 1024 * 1024 * 1024) {
        std::fprintf(stderr, "[deepseek4-split] insufficient CUDA memory (%.1f GiB free)\n",
                     (double)free_bytes / (1024.0 * 1024.0 * 1024.0));
        return 1;  // Minimum 1 layer on CUDA
    }

    const int max_layers = estimate_cuda_layers_from_free_bytes(free_bytes);

    std::fprintf(stderr, "[deepseek4-split] auto-split: CUDA free=%.1f GiB, "
                 "estimated %d layers on CUDA, %d on Halo\n",
                 (double)free_bytes / (1024.0 * 1024.0 * 1024.0),
                 max_layers, 43 - max_layers);
    return max_layers;
}

bool DeepSeek4LayerSplitAdapter::init() {
    if (!cfg_.target_path) {
        std::fprintf(stderr, "[deepseek4-split] target_path is null\n");
        return false;
    }

    // Determine layer split configuration
    DevicePlacement device = cfg_.device;
    if (!device.is_layer_split()) {
        // Auto-compute split: CUDA gets first N layers, Halo gets the rest
        int cuda_layers = compute_auto_split_layers();
        int local_gpu = device.gpu >= 0 ? device.gpu : 0;
        device.layer_split_gpus = {local_gpu, 0};  // GPU 0 = local CUDA, remote GPU 0 on Halo
        device.layer_split_weights = {(double)cuda_layers, (double)(43 - cuda_layers)};
    }

    // When remote_target_shard is configured, use mixed (IPC) path:
    // only the first shard is local, the rest run via IPC daemon on Halo.
    if (cfg_.remote_target_shard.enabled()) {
        return init_mixed_target_split_full(device);
    }

    // Multi-GPU local path (multiple CUDA GPUs available)
    LayerSplitRuntimeInit runtime_cfg;
    runtime_cfg.target_path = cfg_.target_path;
    runtime_cfg.device = &device;
    runtime_cfg.log_prefix = "deepseek4-split";

    if (!init_layer_split_runtime(runtime_cfg, shards_, snapshot_backends_)) {
        std::fprintf(stderr, "[deepseek4-split] init_layer_split_runtime failed\n");
        return false;
    }

    // Load weights and create cache for each shard
    const int max_ctx = device.max_ctx > 0 ? device.max_ctx : 8192;
    for (size_t i = 0; i < shards_.size(); ++i) {
        auto & shard = shards_[i];
        const bool is_last = (i == shards_.size() - 1);

        TargetLoadPlan plan;
        plan.layer_begin = shard.layer_begin;
        plan.layer_end = shard.layer_end;
        plan.load_output = is_last;

        if (!load_deepseek4_gguf_partial(cfg_.target_path, shard.backend, plan, shard.weights)) {
            std::fprintf(stderr, "[deepseek4-split] failed to load shard %zu (layers %d-%d)\n",
                         i, shard.layer_begin, shard.layer_end);
            return false;
        }

        if (!create_deepseek4_cache(shard.backend, shard.weights, max_ctx, shard.cache)) {
            std::fprintf(stderr, "[deepseek4-split] failed to create cache for shard %zu\n", i);
            return false;
        }

        std::fprintf(stderr, "[deepseek4-split] shard %zu: gpu=%d layers=[%d,%d) %s\n",
                     i, shard.gpu, shard.layer_begin, shard.layer_end,
                     is_last ? "(+output)" : "");
    }

    // Initialize HC state (4 streams × n_embd)
    if (!shards_.empty()) {
        hc_state_.resize(hc_state_elements(shards_[0].weights), 0.0f);
    }

    std::fprintf(stderr, "[deepseek4-split] initialized with %zu local shards\n", shards_.size());
    return true;
}

bool DeepSeek4LayerSplitAdapter::init_mixed_target_split_full(const DevicePlacement & device) {
    // Mixed target split: local CUDA shard + remote Halo shard via IPC daemon.
    // Only the first shard runs locally; remaining layers handled by remote daemon.

    const auto info = inspect_gguf_model_info(cfg_.target_path);
    const int n_layer = info.n_layer;
    if (n_layer <= 0) {
        std::fprintf(stderr, "[deepseek4-split] failed to inspect target layer count\n");
        return false;
    }

    const auto ranges = compute_layer_ranges(
        n_layer, (int)device.layer_split_gpus.size(),
        device.layer_split_weights);
    if (ranges.size() != device.layer_split_gpus.size()) {
        std::fprintf(stderr, "[deepseek4-split] bad layer split\n");
        return false;
    }

    // Initialize only the local (first) shard
    shards_.resize(1);
    auto & local_shard = shards_[0];
    local_shard.gpu = device.layer_split_gpus[0];
    local_shard.layer_begin = ranges[0].begin;
    local_shard.layer_end = ranges[0].end;
    local_shard.backend = ggml_backend_cuda_init(local_shard.gpu);
    if (!local_shard.backend) {
        std::fprintf(stderr, "[deepseek4-split] local backend init failed gpu=%d\n",
                     local_shard.gpu);
        return false;
    }

    // Load weights for local shard
    const int max_ctx = device.max_ctx > 0 ? device.max_ctx : 8192;
    TargetLoadPlan plan;
    plan.layer_begin = local_shard.layer_begin;
    plan.layer_end = local_shard.layer_end;
    plan.load_output = false;  // Output head on remote shard

    if (!load_deepseek4_gguf_partial(cfg_.target_path, local_shard.backend, plan, local_shard.weights)) {
        std::fprintf(stderr, "[deepseek4-split] failed to load local shard layers [%d,%d)\n",
                     local_shard.layer_begin, local_shard.layer_end);
        return false;
    }

    if (!create_deepseek4_cache(local_shard.backend, local_shard.weights, max_ctx, local_shard.cache)) {
        std::fprintf(stderr, "[deepseek4-split] failed to create local cache\n");
        return false;
    }

    std::fprintf(stderr, "[deepseek4-split] local shard: gpu=%d layers=[%d,%d)\n",
                 local_shard.gpu, local_shard.layer_begin, local_shard.layer_end);

    // Initialize HC state
    hc_state_.resize(hc_state_elements(local_shard.weights), 0.0f);

    // Launch remote IPC daemon for remaining layers
    std::vector<int> remote_gpus;
    std::vector<int> remote_layer_begins;
    std::vector<int> remote_layer_ends;
    for (size_t i = 1; i < device.layer_split_gpus.size(); ++i) {
        remote_gpus.push_back(device.layer_split_gpus[i]);
        remote_layer_begins.push_back(ranges[i].begin);
        remote_layer_ends.push_back(ranges[i].end);
    }

    TargetShardIpcLaunchConfig launch_cfg;
    launch_cfg.mode = BackendIpcMode::DeepSeek4TargetShard;
    launch_cfg.bin = cfg_.remote_target_shard.ipc_bin;
    launch_cfg.target_path = cfg_.target_path;
    launch_cfg.gpus = remote_gpus;
    launch_cfg.layer_begins = remote_layer_begins;
    launch_cfg.layer_ends = remote_layer_ends;
    launch_cfg.max_ctx = max_ctx;
    launch_cfg.hidden = local_shard.weights.n_embd;
    launch_cfg.vocab = local_shard.weights.n_vocab;
    launch_cfg.max_tokens = std::max(1, max_ctx);
    launch_cfg.work_dir = cfg_.remote_target_shard.work_dir;

    if (!remote_target_shard_.start(launch_cfg)) {
        std::fprintf(stderr, "[deepseek4-split] failed to start remote target shard layers=[%d,%d)\n",
                     remote_layer_begins.front(), remote_layer_ends.back());
        return false;
    }

    std::fprintf(stderr, "[deepseek4-split] remote shard: layers=[%d,%d) via IPC daemon\n",
                 remote_layer_begins.front(), remote_layer_ends.back());

    // Snapshot backends for local shard only
    auto shard_metas = layer_split_shard_metas(shards_);
    if (!init_layer_split_snapshot_backends(shard_metas, snapshot_backends_, "deepseek4-split")) {
        return false;
    }

    std::fprintf(stderr, "[deepseek4-split] initialized mixed split: %d local + %d remote layers\n",
                 local_shard.layer_end - local_shard.layer_begin,
                 remote_layer_ends.back() - remote_layer_begins.front());
    return true;
}

bool DeepSeek4LayerSplitAdapter::init_mixed_target_split() {
    // Legacy — unused now, mixed init is done via init_mixed_target_split_full()
    return false;
}

void DeepSeek4LayerSplitAdapter::begin_request(const GenerateRequest & req) {
    (void)req;
}

void DeepSeek4LayerSplitAdapter::reset_request_state() {
    cur_pos_ = 0;
    last_tok_ = -1;
    const size_t hc_size = hc_state_.size();
    std::fill(hc_state_.begin(), hc_state_.end(), 0.0f);

    for (auto & shard : shards_) {
        shard.cache.cur_pos = 0;
        for (auto & lc : shard.cache.layers) {
            lc.n_comp = 0;
            lc.n_index_comp = 0;
        }
    }

    if (remote_target_shard_.active()) {
        remote_target_shard_.reset_request_state();
    }

    (void)hc_size;
}

bool DeepSeek4LayerSplitAdapter::run_forward(
        const std::vector<int32_t> & tokens,
        int base_pos,
        int & last_tok,
        std::vector<float> * logits_out) {
    if (shards_.empty()) return false;

    const int n_tokens = (int)tokens.size();
    const int n_embd = shards_[0].weights.n_embd;
    const int n_hc = shards_[0].weights.n_hc;

    // Embed tokens on first shard
    auto & first_shard = shards_[0];
    std::vector<float> embed((size_t)n_embd * n_tokens);
    if (!first_shard.weights.embedder.embed(tokens.data(), n_tokens, embed.data())) {
        std::fprintf(stderr, "[deepseek4-split] embedding failed on first shard\n");
        return false;
    }

    // Initialize HC state from embedding for new sequences
    if (base_pos == 0 && cur_pos_ == 0) {
        for (int t = 0; t < n_tokens; ++t) {
            for (int h = 0; h < n_hc; ++h) {
                std::memcpy(hc_state_.data() + (size_t)h * n_embd,
                           embed.data() + (size_t)t * n_embd,
                           (size_t)n_embd * sizeof(float));
            }
        }
    }

    // If using mixed target split (remote Halo shard), delegate to that path
    if (use_mixed_target_split()) {
        return run_mixed_forward(tokens, base_pos, last_tok, logits_out);
    }

    // Local multi-GPU path: run each shard's layers sequentially
    // Pass HC state between shards at boundaries
    for (size_t si = 0; si < shards_.size(); ++si) {
        auto & shard = shards_[si];
        const bool is_last = (si == shards_.size() - 1);
        std::vector<float> * shard_logits = is_last ? logits_out : nullptr;

        if (!deepseek4_step_layer_range(
                shard.backend, shard.weights, shard.cache,
                hc_state_, embed.data(), n_tokens, base_pos,
                shard.layer_begin, shard.layer_end,
                shard_logits, tokens.data(), nullptr)) {
            std::fprintf(stderr, "[deepseek4-split] forward failed on shard %zu\n", si);
            return false;
        }
    }

    cur_pos_ = base_pos + n_tokens;
    last_tok = tokens.back();
    last_tok_ = last_tok;

    if (logits_out && !logits_out->empty()) {
        prefill_last_logits_ = *logits_out;
    }

    return true;
}

bool DeepSeek4LayerSplitAdapter::run_mixed_forward(
        const std::vector<int32_t> & tokens,
        int base_pos,
        int & last_tok,
        std::vector<float> * logits_out) {
    if (shards_.empty() || !remote_target_shard_.active()) return false;

    const int n_tokens = (int)tokens.size();
    const int n_embd = shards_[0].weights.n_embd;

    // Run local shard (first N layers on CUDA)
    auto & local_shard = shards_[0];
    std::vector<float> embed((size_t)n_embd * n_tokens);
    if (!local_shard.weights.embedder.embed(tokens.data(), n_tokens, embed.data())) {
        std::fprintf(stderr, "[deepseek4-split] embedding failed on local shard\n");
        return false;
    }

    // Run only the local shard's layer range, getting hidden state output
    std::vector<float> hidden_out;
    std::vector<float> hc_state;  // unused for now
    if (!deepseek4_step_layer_range(local_shard.backend, local_shard.weights,
                                     local_shard.cache, hc_state,
                                     embed.data(), n_tokens, base_pos,
                                     local_shard.layer_begin, local_shard.layer_end,
                                     &hidden_out, tokens.data(), nullptr)) {
        std::fprintf(stderr, "[deepseek4-split] local shard forward failed\n");
        return false;
    }

    // Send boundary activation (hidden state) to remote shard
    TargetShardForwardRequest req;
    req.base_pos = base_pos;
    req.n_tokens = n_tokens;
    req.boundary_activation = &hidden_out;
    req.token_ids = &tokens;
    req.want_logits = (logits_out != nullptr);

    TargetShardForwardResponse resp;
    std::vector<float> remote_logits;
    if (logits_out) resp.logits_out = &remote_logits;

    if (!remote_target_shard_.forward(req, resp)) {
        std::fprintf(stderr, "[deepseek4-split] remote shard forward failed\n");
        return false;
    }

    cur_pos_ = base_pos + n_tokens;
    last_tok = resp.last_tok >= 0 ? resp.last_tok : tokens.back();
    last_tok_ = last_tok;

    if (logits_out && !remote_logits.empty()) {
        *logits_out = std::move(remote_logits);
        prefill_last_logits_ = *logits_out;
    }

    return true;
}

bool DeepSeek4LayerSplitAdapter::prefill(
        const std::vector<int32_t> & prompt,
        int base_pos,
        int & last_tok) {
    const int chunk_size = cfg_.chunk > 0 ? cfg_.chunk : 512;
    const int n_prompt = (int)prompt.size();

    for (int offset = 0; offset < n_prompt; offset += chunk_size) {
        const int chunk_end = std::min(offset + chunk_size, n_prompt);
        std::vector<int32_t> chunk(prompt.begin() + offset,
                                    prompt.begin() + chunk_end);

        std::vector<float> * logits_out =
            (chunk_end >= n_prompt) ? &prefill_last_logits_ : nullptr;

        if (!run_forward(chunk, base_pos + offset, last_tok, logits_out)) {
            return false;
        }
    }
    return true;
}

bool DeepSeek4LayerSplitAdapter::decode_ar(
        int last_tok_in,
        int committed,
        int n_gen,
        std::vector<int32_t> & out_tokens,
        const DaemonIO & io) {
    if (shards_.empty()) return false;

    const int vocab = shards_[0].weights.n_vocab;
    auto is_eos = [](int tok) -> bool {
        // DS4 EOS tokens (from model card / tokenizer)
        return tok == 128001 || tok == 128008 || tok == 128009;
    };

    LayerSplitForwardStep forward_one = [this](
            const std::vector<int32_t> & tokens,
            int committed_pos,
            int & next_tok,
            std::vector<float> * logits_out) -> bool {
        return run_forward(tokens, committed_pos, next_tok, logits_out);
    };

    return run_layer_split_ar_decode(
        last_tok_in, committed, n_gen, vocab,
        prefill_last_logits_, sampler_, sampler_rng_,
        forward_one, is_eos, out_tokens, io);
}

bool DeepSeek4LayerSplitAdapter::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    auto & snap = snapshots_[slot];
    snap.cur_pos = cur_pos_;
    snap.last_tok = last_tok_;
    snap.hc_state = hc_state_;
    snap.used = true;
    return true;
}

void DeepSeek4LayerSplitAdapter::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return;
    snapshots_[slot].used = false;
    snapshots_[slot].hc_state.clear();
}

bool DeepSeek4LayerSplitAdapter::snapshot_used(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    return snapshots_[slot].used;
}

int DeepSeek4LayerSplitAdapter::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return 0;
    return snapshots_[slot].cur_pos;
}

bool DeepSeek4LayerSplitAdapter::snapshot_restore(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS || !snapshots_[slot].used) return false;
    const auto & snap = snapshots_[slot];
    cur_pos_ = snap.cur_pos;
    last_tok_ = snap.last_tok;
    hc_state_ = snap.hc_state;
    return true;
}

void DeepSeek4LayerSplitAdapter::shutdown() {
    if (remote_target_shard_.active()) {
        remote_target_shard_.reset_request_state();
    }
    for (auto & shard : shards_) {
        free_deepseek4_cache(shard.cache);
        free_deepseek4_weights(shard.weights);
        if (shard.backend) {
            ggml_backend_free(shard.backend);
            shard.backend = nullptr;
        }
    }
    shards_.clear();
    free_layer_split_snapshot_backends(
        layer_split_shard_metas(shards_), snapshot_backends_);
}

}  // namespace dflash::common
