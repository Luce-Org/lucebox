#include "bailingmoe3_backend.h"

#include <cstdio>
#include <utility>

namespace dflash::common {
namespace {

Qwen35Config make_qwen_runtime_config(BailingMoe3Config cfg) {
    Qwen35Config runtime;
    runtime.target_path = std::move(cfg.model_path);
    runtime.device = cfg.device;
    runtime.stream_fd = cfg.stream_fd;
    // The Ling baseline uses the ordinary contiguous F16/Q4 KV cache and the
    // proven single-sequence AR loop. No DFlash draft or paged serving yet.
    // Its compressed MLA head is 576-wide, whose CUDA kernel contract uses a
    // 256-row K/V span and an explicit visibility mask even for decode.
    runtime.kq_stride_pad = 256;
    runtime.paged_attention = false;
    runtime.max_concurrency = 1;
    return runtime;
}

}  // namespace

BailingMoe3Backend::BailingMoe3Backend(BailingMoe3Config cfg)
    : Qwen35Backend(make_qwen_runtime_config(std::move(cfg))) {}

bool BailingMoe3Backend::load_target_model(ggml_backend_t backend,
                                           TargetWeights & out) {
    return load_bailingmoe3_gguf(cfg_.target_path, backend, out);
}

void BailingMoe3Backend::print_ready_banner() const {
    const TargetWeights & weights = target_weights();
    std::printf(
        "[bailingmoe3-daemon] ready layers=%d kda=%d mla=%d "
        "experts=%d/%d groups=%d/%d ctx=%d\n",
        weights.n_layer,
        weights.n_layer - weights.n_layer / weights.full_attention_interval,
        weights.n_layer / weights.full_attention_interval,
        weights.n_expert_used, weights.n_expert,
        weights.n_expert_groups_used, weights.n_expert_groups,
        cfg_.device.max_ctx);
    std::fflush(stdout);
}

}  // namespace dflash::common
