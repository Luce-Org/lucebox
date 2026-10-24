#include "pipeline.h"

#include "blas.h"
#include "kernels.h"
#include "text_encoder.h"
#include "weights.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <future>
#include <iterator>
#include <list>
#include <string>
#include <thread>
#include <vector>

namespace luce::moss_image {

namespace {

constexpr float   kEps         = 1e-6f;
constexpr size_t  kTextCache   = 4;    // prompts whose text rows are kept (stable-diffusion.cpp keeps 4 too)
constexpr int     kSkipLayer   = 22;   // the unconditional pass skips it, as stable-diffusion.cpp runs MOSS-Image
constexpr int     kGroups      = 32;   // VAE GroupNorm groups
constexpr int     kMaxStepRows = 16;   // linear_bf16_rows takes up to 16 steps at once

// Standby: load_weights() leaves the other model this much of the GPU to grow into, and a request waits up to
// kFreeWait for the memory the other model gives back as it stops. The driver clears memory before it is free for
// another process, which takes a moment for gigabytes: the memory given back is waited for up to kClearWait.
constexpr size_t kLoadHeadroom = size_t(1) << 30;
constexpr size_t kFreeSlack    = size_t(64) << 20;
constexpr auto   kFreeWait     = std::chrono::seconds(10);
constexpr auto   kClearWait    = std::chrono::seconds(2);

// A device buffer that only grows: work buffers are sized for the largest request so far. Grown only while the
// stream is idle.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { (void) hipFree(ptr_); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer & operator=(const DeviceBuffer &) = delete;

    bool reserve(const size_t bytes) {
        if (bytes <= bytes_) {
            return true;
        }
        (void) hipFree(ptr_);
        ptr_   = nullptr;
        bytes_ = 0;
        if (hipMalloc(&ptr_, bytes) != hipSuccess) {
            ptr_ = nullptr;
            return false;
        }
        bytes_ = bytes;
        return true;
    }

    // Frees the memory; the next reserve() allocates again.
    void release() {
        (void) hipFree(ptr_);
        ptr_   = nullptr;
        bytes_ = 0;
    }

    template <typename T> T * as() const { return static_cast<T *>(ptr_); }
    size_t bytes() const { return bytes_; }

private:
    void * ptr_   = nullptr;
    size_t bytes_ = 0;
};

// Pinned host memory for the per-request uploads (RoPE tables, timestep features).
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    ~PinnedBuffer() { (void) hipHostFree(ptr_); }
    PinnedBuffer(const PinnedBuffer &) = delete;
    PinnedBuffer & operator=(const PinnedBuffer &) = delete;

    float * reserve(const size_t count) {
        if (count > count_) {
            (void) hipHostFree(ptr_);
            ptr_   = nullptr;
            count_ = 0;
            if (hipHostMalloc(reinterpret_cast<void **>(&ptr_), count * sizeof(float)) != hipSuccess) {
                ptr_ = nullptr;
                return nullptr;
            }
            count_ = count;
        }
        return ptr_;
    }

private:
    float * ptr_   = nullptr;
    size_t  count_ = 0;
};

// A prompt's text rows of the DiT input: the encoder's hidden states through the text embedder.
struct TextRows {
    std::string  prompt;
    bool         valid = false;
    int64_t      n     = 0;
    DeviceBuffer rows;  // [n][kHidden] f32
};

} // namespace

struct Pipeline::Impl {
    hipStream_t stream = nullptr;
    Weights     weights;
    TextEncoder encoder;
    Blas        blas;

    // The text rows of the last kTextCache prompts and negative prompts, the most recent first: a request that
    // repeats one (another seed or size) skips the text encoder.
    std::list<TextRows> text_cache;
    const TextRows *    cond   = nullptr;  // this request's prompt rows, in text_cache
    const TextRows *    uncond = nullptr;  // and its negative prompt's, with CFG

    // Text embedder work buffers.
    DeviceBuffer t_states, t_bf16, t_mm, t_q, t_k, t_v, t_att, t_pe;
    // Timestep path, every step of a request at once.
    DeviceBuffer time_in, time_hidden, time_emb, block_mod, head_mod;
    // DiT.
    DeviceBuffer x, xq, xs, q, k, v, att, sums, hq, hs, pe, state_bf16, img, head_in, pred, state, old_den, old_dd;
    DeviceBuffer split;  // the int8 GEMMs' split-K workspace (gemm_i8_split_bytes): this stream's own
    // VAE.
    DeviceBuffer maps[4], nhwc[2], gn_scratch, scores, probs, rgb;
    PinnedBuffer host_pe, host_time;
    hipEvent_t   events[4] = {};
    bool         warming_up = false;  // the warm-up images log nothing
    std::string  tuning_path;         // where the tuned GEMM kernels are kept; empty: nowhere
    int          warmup_width = 1024, warmup_height = 1024;

    // Standby: the weights' copies onto the GPU run on copy_stream, and ready[] marks each part's arrival.
    bool        standby      = false;
    bool        needs_warmup = false;  // no tuning file yet: the first request tunes the GEMM kernels first
    bool        streaming    = false;  // this request's weights are still arriving
    hipStream_t copy_stream  = nullptr;
    hipEvent_t  ready[Weights::kParts] = {};

    ~Impl() {
        for (hipEvent_t e : events) {
            if (e != nullptr) {
                (void) hipEventDestroy(e);
            }
        }
        for (hipEvent_t e : ready) {
            if (e != nullptr) {
                (void) hipEventDestroy(e);
            }
        }
        if (copy_stream != nullptr) {
            (void) hipStreamDestroy(copy_stream);
        }
        if (stream != nullptr) {
            (void) hipStreamDestroy(stream);
        }
    }

    // Every work buffer above: release_memory() frees them, the text cache stays.
    std::vector<DeviceBuffer *> work_buffers() {
        std::vector<DeviceBuffer *> all = { &t_states, &t_bf16, &t_mm, &t_q, &t_k, &t_v, &t_att, &t_pe, &time_in,
                                            &time_hidden, &time_emb, &block_mod, &head_mod, &x, &xq, &xs, &q, &k, &v,
                                            &att, &sums, &hq, &hs, &pe, &split, &state_bf16, &img, &head_in, &pred,
                                            &state, &old_den, &old_dd, &gn_scratch, &scores, &probs, &rgb };
        for (DeviceBuffer & map : maps) {
            all.push_back(&map);
        }
        for (DeviceBuffer & map : nhwc) {
            all.push_back(&map);
        }
        return all;
    }

    // Work on `stream` waits for part `part` of the weights (Weights::upload) while they are arriving.
    void wait_part(const int part) {
        if (streaming) {
            (void) hipStreamWaitEvent(stream, ready[part], 0);
        }
    }

    bool   wait_for_free(size_t bytes, std::string & error);
    size_t free_bytes();
    void   wait_cleared(size_t free_target);

    bool text(const std::string & prompt, const TextRows *& found, std::string & error);
    bool reserve(int64_t n_img, int64_t rows, int n_seq, int steps, int64_t latent_hw, std::string & error,
                 size_t * shortfall = nullptr);
    bool dit_layer(const DitLayer & layer, const float * mod, int64_t active, int64_t rows_c, int64_t rows_u);
    bool sample(const ImageRequest & request, int lh, int lw, int64_t rows_c, int64_t rows_u, std::string & error);
    void resnet(const VaeResBlock & block, const float * in, int ic, int oc, int w, int h, float * tmp, float * tmp2,
                float * out);
    bool attention(const VaeAttention & block, const float * in, int w, int h, float * out);
    bool decode(int lh, int lw, std::string & error);
    void save_tuning();
};

// ---------------------------------------------------------------------------------------------------------------
// Text

bool Pipeline::Impl::text(const std::string & prompt, const TextRows *& found, std::string & error) {
    for (auto it = text_cache.begin(); it != text_cache.end(); ++it) {
        if (it->valid && it->prompt == prompt) {
            text_cache.splice(text_cache.begin(), text_cache, it);
            found = &text_cache.front();
            return true;
        }
    }
    // A new prompt takes the least recently used entry, never the one the request's other prompt just found.
    if (text_cache.size() < kTextCache) {
        text_cache.emplace_front();
    } else {
        text_cache.splice(text_cache.begin(), text_cache, std::prev(text_cache.end()));
    }
    TextRows & out = text_cache.front();
    out.valid      = false;
    TextHidden hidden;
    if (!encoder.encode(encoder.tokenize(prompt), hidden, error)) {
        return false;
    }
    const int64_t n = hidden.n_tokens;
    if (!t_bf16.reserve(n * kFfn * sizeof(uint16_t)) || !t_mm.reserve(n * 2 * kFfn * sizeof(float)) ||
        !t_q.reserve(n * kHidden * sizeof(float)) || !t_k.reserve(n * kHidden * sizeof(uint16_t)) ||
        !t_v.reserve(n * kHidden * sizeof(uint16_t)) || !t_att.reserve(n * kHidden * sizeof(float)) ||
        !t_pe.reserve(n * kRopePairs * 2 * sizeof(float)) || !out.rows.reserve(n * kHidden * sizeof(float)) ||
        (hidden.host && !t_states.reserve(n * kTextHidden * sizeof(float)))) {
        error = "out of GPU memory for the text embedder";
        return false;
    }
    const float * states = hidden.data;
    if (hidden.host) {
        // An encoder on another GPU hands its rows over in host memory.
        (void) hipMemcpyAsync(t_states.as<float>(), hidden.data, n * kTextHidden * sizeof(float),
                              hipMemcpyHostToDevice, stream);
        states = t_states.as<float>();
    }
    float * pe = host_pe.reserve(size_t(n) * kRopePairs * 2);
    if (pe == nullptr) {
        error = "out of pinned host memory";
        return false;
    }
    rope_table(0, 0, int(n), pe);  // the text rows: positions (0, i, 0, 0)
    (void) hipMemcpyAsync(t_pe.as<float>(), pe, n * kRopePairs * 2 * sizeof(float), hipMemcpyHostToDevice, stream);

    const DitWeights & w  = weights.dit;
    uint16_t *         b  = t_bf16.as<uint16_t>();
    float *            mm = t_mm.as<float>();
    float *            tx = out.rows.as<float>();
    bool               ok = true;
    wait_part(0);
    f32_to_bf16(states, n * kTextHidden, b, stream);
    ok = ok && blas.gemm(GemmType::bf16_f32, w.text_proj.w, kTextHidden, b, kTextHidden, tx, kHidden, kHidden, n,
                         kTextHidden);
    add_bias(tx, w.text_proj.b, n, kHidden, tx, stream);
    for (const TextBlock & block : w.text_blocks) {
        rms_norm_mul_bf16(tx, block.norm1, kEps, n, kHidden, b, stream);
        ok = ok && blas.gemm(GemmType::bf16_f32, block.qkv.w, kHidden, b, kHidden, mm, 3 * kHidden, 3 * kHidden, n,
                             kHidden);
        text_qkv_prologue(mm, block.qkv.b, block.q_norm, block.k_norm, kEps, t_pe.as<float>(), n, kHeads,
                          t_q.as<float>(), t_k.as<uint16_t>(), t_v.as<uint16_t>(), stream);
        attention_d128(t_q.as<float>(), t_k.as<uint16_t>(), t_v.as<uint16_t>(), n, 0, n, kHeads,
                       1.0f / std::sqrt(float(kHeadDim)), t_att.as<float>(), stream);
        f32_to_bf16(t_att.as<float>(), n * kHidden, b, stream);
        ok = ok && blas.gemm(GemmType::bf16_f32, block.proj.w, kHidden, b, kHidden, mm, kHidden, kHidden, n, kHidden);
        add_bias_residual(mm, block.proj.b, n, kHidden, tx, stream);
        rms_norm_mul_bf16(tx, block.norm2, kEps, n, kHidden, b, stream);
        ok = ok && blas.gemm(GemmType::bf16_f32, block.up_gate.w, kHidden, b, kHidden, mm, 2 * kFfn, 2 * kFfn, n,
                             kHidden);
        text_situ_bf16(mm, block.up_gate.b, block.up_gate.b + kFfn, n, kFfn, b, stream);
        ok = ok && blas.gemm(GemmType::bf16_f32, block.down.w, kFfn, b, kFfn, mm, kHidden, kHidden, n, kFfn);
        add_bias_residual(mm, block.down.b, n, kHidden, tx, stream);
    }
    // The encoder reuses its output buffer on the next prompt: let this one's readers finish first.
    const hipError_t status = hipStreamSynchronize(stream);
    if (!ok || status != hipSuccess) {
        error = ok ? std::string("text embedder: ") + hipGetErrorString(status) : "text embedder: a GEMM failed";
        return false;
    }
    out.prompt = prompt;
    out.n      = n;
    out.valid  = true;
    found      = &out;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Buffers

// With `shortfall`, nothing is allocated: it gets the bytes the allocations would add.
bool Pipeline::Impl::reserve(const int64_t n_img, const int64_t rows, const int n_seq, const int steps,
                             const int64_t latent_hw, std::string & error, size_t * shortfall) {
    const auto grow = [shortfall](DeviceBuffer & buffer, const size_t bytes) {
        if (shortfall == nullptr) {
            return buffer.reserve(bytes);
        }
        *shortfall += bytes > buffer.bytes() ? bytes - buffer.bytes() : 0;
        return true;
    };
    bool ok = grow(time_in, size_t(steps) * kTimeFreqs * sizeof(float)) &&
              grow(time_hidden, size_t(steps) * kHidden * sizeof(float)) &&
              grow(time_emb, size_t(steps) * kHidden * sizeof(float)) &&
              grow(block_mod, size_t(steps) * 6 * kHidden * sizeof(float)) &&
              grow(head_mod, size_t(steps) * 2 * kHidden * sizeof(float));
    ok = ok && grow(x, rows * kHidden * sizeof(float)) && grow(xq, rows * kHidden) &&
         grow(xs, rows * sizeof(float)) && grow(q, rows * kHidden * sizeof(float)) &&
         grow(k, rows * kHidden * sizeof(uint16_t)) && grow(v, rows * kHidden * sizeof(uint16_t)) &&
         grow(att, rows * kHidden * sizeof(float)) && grow(sums, rows * 2 * kFfn * sizeof(int32_t)) &&
         grow(hq, rows * kFfn) && grow(hs, rows * sizeof(float)) &&
         grow(pe, rows * kRopePairs * 2 * sizeof(float)) &&
         grow(state_bf16, n_img * kLatentChannels * sizeof(uint16_t)) &&
         grow(img, n_img * kHidden * sizeof(float)) && grow(head_in, n_seq * n_img * kHidden * sizeof(uint16_t)) &&
         grow(pred, n_seq * n_img * kLatentChannels * sizeof(float)) &&
         grow(state, n_img * kLatentChannels * sizeof(float)) &&
         grow(old_den, n_img * kLatentChannels * sizeof(float)) &&
         grow(old_dd, n_img * kLatentChannels * sizeof(float));
    const bool new_split = shortfall == nullptr && split.bytes() == 0;  // a new split-K workspace starts zeroed
    ok = ok && grow(split, gemm_i8_split_bytes()) &&
         (!new_split || hipMemsetAsync(split.as<void>(), 0, split.bytes(), stream) == hipSuccess);
    // VAE decoder: the largest map is 512 channels at 4x4 or 256 channels at 8x8 the latent's pixels, so 16384
    // values per latent pixel; the mid-block attention scores are latent_hw^2.
    size_t gn = 0;
    for (const auto & [c, scale] : { std::pair{ 512, 1 }, { 512, 4 }, { 512, 16 }, { 256, 16 }, { 256, 64 },
                                   { 128, 64 } }) {
        gn = std::max(gn, group_norm_scratch_bytes(c, latent_hw * scale, kGroups));
    }
    for (DeviceBuffer & map : maps) {
        ok = ok && grow(map, latent_hw * 16384 * sizeof(float));
    }
    for (DeviceBuffer & map : nhwc) {
        ok = ok && grow(map, latent_hw * 16384 * sizeof(uint16_t));
    }
    ok = ok && grow(gn_scratch, gn) && grow(scores, latent_hw * latent_hw * sizeof(float)) &&
         grow(probs, latent_hw * latent_hw * sizeof(uint16_t)) && grow(rgb, latent_hw * 64 * 3);
    if (!ok) {
        error = "out of GPU memory for this image size";
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// DiT

// One DiT block over the first `active` rows of x: the conditional sequence (rows_c rows), then the unconditional
// one (rows_u rows) unless this block skips it.
bool Pipeline::Impl::dit_layer(const DitLayer & layer, const float * mod, const int64_t active, const int64_t rows_c,
                               const int64_t rows_u) {
    const float * shift1 = mod;
    const float * scale1 = mod + kHidden;
    const float * gate1  = mod + 2 * kHidden;
    const float * shift2 = mod + 3 * kHidden;
    const float * scale2 = mod + 4 * kHidden;
    const float * gate2  = mod + 5 * kHidden;
    float *       xf     = x.as<float>();

    norm_quantize_i8(xf, active, kHidden, layer.norm1, scale1, shift1, kEps, xq.as<int8_t>(), xs.as<float>(), stream);
    gemm_i8_qkv_attention(layer.qkv.w, layer.qkv.scale, xq.as<int8_t>(), xs.as<float>(), active, kHidden, kHeads,
                          layer.q_norm, layer.k_norm, kEps, pe.as<float>(), q.as<float>(), k.as<uint16_t>(),
                          v.as<uint16_t>(), split.as<int32_t>(), stream);
    const float attn_scale = 1.0f / std::sqrt(float(kHeadDim));
    attention_d128(q.as<float>(), k.as<uint16_t>(), v.as<uint16_t>(), active, 0, rows_c, kHeads, attn_scale,
                   att.as<float>(), stream);
    if (active > rows_c) {
        attention_d128(q.as<float>(), k.as<uint16_t>(), v.as<uint16_t>(), active, rows_c, rows_u, kHeads, attn_scale,
                       att.as<float>(), stream);
    }
    quantize_i8(att.as<float>(), active, kHidden, xq.as<int8_t>(), xs.as<float>(), stream);
    gemm_i8_gated_residual(layer.o.w, layer.o.scale, xq.as<int8_t>(), xs.as<float>(), active, kHidden, kHidden, gate1,
                           xf, split.as<int32_t>(), stream);
    norm_quantize_i8(xf, active, kHidden, layer.norm2, scale2, shift2, kEps, xq.as<int8_t>(), xs.as<float>(), stream);
    gemm_i8_i32(layer.up_gate.w, xq.as<int8_t>(), active, kHidden, 2 * kFfn, sums.as<int32_t>(), stream);
    situ_quantize_i8(sums.as<int32_t>(), xs.as<float>(), layer.up_gate.scale, active, kFfn, hq.as<int8_t>(),
                     hs.as<float>(), stream);
    gemm_i8_gated_residual(layer.down.w, layer.down.scale, hq.as<int8_t>(), hs.as<float>(), active, kFfn, kHidden,
                           gate2, xf, split.as<int32_t>(), stream);
    return true;
}

// The sampling loop: the latent state after the last step, in `state`.
bool Pipeline::Impl::sample(const ImageRequest & request, const int lh, const int lw, const int64_t rows_c,
                            const int64_t rows_u, std::string & error) {
    const DitWeights & w      = weights.dit;
    const int64_t      n_img  = int64_t(lh) * lw;
    const int64_t      rows   = rows_c + rows_u;
    const int          n_seq  = rows_u > 0 ? 2 : 1;
    const uint64_t     seed   = uint64_t(request.seed);
    const Schedule     sched  = request.scheduler == Scheduler::simple ? Schedule::Simple : Schedule::Discrete;
    const Sampler      method = request.sample_method == SampleMethod::euler ? Sampler::Euler : Sampler::ErSde;

    const std::vector<float>       sigmas = flow_sigmas(sched, request.steps, request.flow_shift);
    const std::vector<SamplerStep> plan   = plan_steps(method, sigmas, request.cfg_scale, request.eta);
    const int                      steps  = int(plan.size());  // Euler skips a step whose sigma does not change

    // The timestep path of every step at once: the modulation of step s is row s.
    float * features = host_time.reserve(size_t(steps) * kTimeFreqs);
    float * table    = host_pe.reserve(size_t(rows) * kRopePairs * 2);
    if (features == nullptr || table == nullptr) {
        error = "out of pinned host memory";
        return false;
    }
    for (int s = 0; s < steps; ++s) {
        time_features(plan[s].sigma, features + size_t(s) * kTimeFreqs);
    }
    (void) hipMemcpyAsync(time_in.as<float>(), features, size_t(steps) * kTimeFreqs * sizeof(float),
                          hipMemcpyHostToDevice, stream);
    wait_part(0);
    for (int s0 = 0; s0 < steps; s0 += kMaxStepRows) {
        const int n = std::min(kMaxStepRows, steps - s0);
        linear_bf16_rows(w.time_in.w, w.time_in.b, time_in.as<float>() + size_t(s0) * kTimeFreqs, n, kHidden,
                         kTimeFreqs, true, time_hidden.as<float>() + size_t(s0) * kHidden, stream);
        linear_bf16_rows(w.time_out.w, w.time_out.b, time_hidden.as<float>() + size_t(s0) * kHidden, n, kHidden,
                         kHidden, false, time_emb.as<float>() + size_t(s0) * kHidden, stream);
        linear_bf16_rows(w.block_modulation.w, w.block_modulation.b, time_emb.as<float>() + size_t(s0) * kHidden, n,
                         6 * kHidden, kHidden, false, block_mod.as<float>() + size_t(s0) * 6 * kHidden, stream);
        linear_bf16_rows(w.head_modulation.w, w.head_modulation.b, time_emb.as<float>() + size_t(s0) * kHidden, n,
                         2 * kHidden, kHidden, false, head_mod.as<float>() + size_t(s0) * 2 * kHidden, stream);
    }

    // RoPE tables of both sequences, stacked like their rows.
    rope_table(lh, lw, int(rows_c - n_img), table);
    if (rows_u > 0) {
        rope_table(lh, lw, int(rows_u - n_img), table + rows_c * kRopePairs * 2);
    }
    (void) hipMemcpyAsync(pe.as<float>(), table, rows * kRopePairs * 2 * sizeof(float), hipMemcpyHostToDevice,
                          stream);

    init_noise(seed, n_img, kLatentChannels, state.as<float>(), stream);
    bool ok = true;
    for (int s = 0; s < steps; ++s) {
        f32_to_bf16(state.as<float>(), n_img * kLatentChannels, state_bf16.as<uint16_t>(), stream);
        ok = ok && blas.gemm(GemmType::bf16_f32, w.video_embedder.w, kLatentChannels, state_bf16.as<uint16_t>(),
                             kLatentChannels, img.as<float>(), kHidden, kHidden, n_img, kLatentChannels);
        assemble_tokens(img.as<float>(), w.video_embedder.b, n_img, cond->rows.as<float>(), cond->n,
                        rows_u > 0 ? uncond->rows.as<float>() : nullptr, rows_u > 0 ? uncond->n : 0, kHidden,
                        x.as<float>(), stream);
        const float * mod = block_mod.as<float>() + size_t(s) * 6 * kHidden;
        for (int l = 0; l < kLayers; ++l) {
            if (s == 0) {
                wait_part(1 + l);
            }
            ok = ok && dit_layer(w.layers[l], mod, l == kSkipLayer ? rows_c : rows, rows_c, rows_u);
        }
        final_head_bf16(x.as<float>(), kHidden, n_img, n_seq, rows_c, w.final_norm,
                        head_mod.as<float>() + size_t(s) * 2 * kHidden, kEps, head_in.as<uint16_t>(), stream);
        ok = ok && blas.gemm(GemmType::bf16_f32, w.final_linear.w, kHidden, head_in.as<uint16_t>(), kHidden,
                             pred.as<float>(), kLatentChannels, kLatentChannels, n_seq * n_img, kHidden);
        sampler_step(plan[s], pred.as<float>(), n_seq == 2 ? pred.as<float>() + n_img * kLatentChannels : nullptr,
                     seed, n_img, kLatentChannels, state.as<float>(), old_den.as<float>(), old_dd.as<float>(), stream);
    }
    if (!ok) {
        error = "DiT: a GEMM failed";
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// VAE decoder

// GroupNorm and SiLU, 3x3 convolution, twice, plus the shortcut (a 1x1 convolution where the channels change).
// in [ic][hw] -> out [oc][hw]; tmp and tmp2 are scratch maps.
void Pipeline::Impl::resnet(const VaeResBlock & block, const float * in, const int ic, const int oc, const int w,
                            const int h, float * tmp, float * tmp2, float * out) {
    const int64_t hw  = int64_t(w) * h;
    uint16_t *    n16 = nhwc[0].as<uint16_t>();
    group_norm_nhwc(in, ic, hw, kGroups, kEps, block.norm1.w, block.norm1.b, true, gn_scratch.as<void>(), n16, stream);
    conv2d(n16, w, h, ic, false, block.conv1.w, 3, 3, oc, 1, 1, block.conv1.b, nullptr, tmp, stream);
    group_norm_nhwc(tmp, oc, hw, kGroups, kEps, block.norm2.w, block.norm2.b, true, gn_scratch.as<void>(), n16, stream);
    if (block.nin.w == nullptr) {
        conv2d(n16, w, h, oc, false, block.conv2.w, 3, 3, oc, 1, 1, block.conv2.b, in, out, stream);
        return;
    }
    conv2d(n16, w, h, oc, false, block.conv2.w, 3, 3, oc, 1, 1, block.conv2.b, nullptr, tmp2, stream);
    uint16_t * in16 = nhwc[1].as<uint16_t>();
    to_nhwc_f16(in, ic, hw, in16, stream);
    conv2d(in16, w, h, ic, false, block.nin.w, 1, 1, oc, 0, 0, block.nin.b, tmp2, out, stream);
}

// The mid-block self-attention, one 512-wide head over every pixel: in [512][hw] -> out [512][hw].
bool Pipeline::Impl::attention(const VaeAttention & block, const float * in, const int w, const int h, float * out) {
    constexpr int C   = 512;
    const int64_t hw  = int64_t(w) * h;
    uint16_t *    n16 = nhwc[0].as<uint16_t>();
    float *       qkv = maps[2].as<float>();  // [3 * C][hw]
    group_norm_nhwc(in, C, hw, kGroups, kEps, block.norm.w, block.norm.b, false, gn_scratch.as<void>(), n16, stream);
    conv2d(n16, w, h, C, false, block.qkv.w, 1, 1, 3 * C, 0, 0, block.qkv.b, nullptr, qkv, stream);
    uint16_t * qk = nhwc[1].as<uint16_t>();  // [hw][2 * C]: q, then k
    uint16_t * vt = n16;                     // [C][hw]
    to_nhwc_f16(qkv, 2 * C, hw, qk, stream);
    f32_to_f16(qkv + 2 * C * hw, C * hw, vt, stream);
    bool ok = blas.gemm(GemmType::f16_f32, qk + C, 2 * C, qk, 2 * C, scores.as<float>(), hw, hw, hw, C);
    softmax_f16(scores.as<float>(), hw, hw, 1.0f / std::sqrt(float(C)), probs.as<uint16_t>(), stream);
    float * o = maps[3].as<float>();  // [hw][C]
    ok = ok && blas.gemm(GemmType::f16_f32, vt, hw, probs.as<uint16_t>(), hw, o, C, C, hw, hw);
    f32_to_f16(o, hw * C, qk, stream);
    conv2d(qk, w, h, C, false, block.proj.w, 1, 1, C, 0, 0, block.proj.b, in, out, stream);
    return ok;
}

// The latent state through the VAE decoder into rgb [16 lh][16 lw][3] uint8.
bool Pipeline::Impl::decode(const int lh, const int lw, std::string & error) {
    const VaeDecoder & d = weights.vae;
    int                h = 2 * lh, w = 2 * lw;  // latent pixels
    int64_t            hw = int64_t(h) * w;
    float *            a  = maps[0].as<float>();
    float *            b  = maps[1].as<float>();
    float *            t1 = maps[2].as<float>();
    float *            t2 = maps[3].as<float>();
    uint16_t *         n16 = nhwc[0].as<uint16_t>();

    wait_part(Weights::kParts - 1);
    unpack_latent(state.as<float>(), d.latents_mean, d.latents_std, lh, lw, a, stream);
    to_nhwc_f16(a, 32, hw, n16, stream);
    conv2d(n16, w, h, 32, false, d.post_quant.w, 1, 1, 32, 0, 0, d.post_quant.b, nullptr, b, stream);
    to_nhwc_f16(b, 32, hw, n16, stream);
    conv2d(n16, w, h, 32, false, d.conv_in.w, 3, 3, 512, 1, 1, d.conv_in.b, nullptr, a, stream);

    resnet(d.mid_block1, a, 512, 512, w, h, t1, t2, b);
    bool ok = attention(d.mid_attn, b, w, h, a);
    resnet(d.mid_block2, a, 512, 512, w, h, t1, t2, b);
    int channels = 512;
    for (int level = 3; level >= 0; --level) {
        const int out_channels = level >= 2 ? 512 : level == 1 ? 256 : 128;
        for (const VaeResBlock & block : d.up_blocks[level]) {
            resnet(block, b, channels, out_channels, w, h, t1, t2, a);
            std::swap(a, b);
            channels = out_channels;
        }
        if (d.upsample[level].w != nullptr) {
            to_nhwc_f16(b, channels, hw, n16, stream);
            h *= 2;
            w *= 2;
            hw *= 4;
            conv2d(n16, w, h, channels, true, d.upsample[level].w, 3, 3, channels, 1, 1, d.upsample[level].b, nullptr,
                   a, stream);
            std::swap(a, b);
        }
    }
    group_norm_nhwc(b, channels, hw, kGroups, kEps, d.norm_out.w, d.norm_out.b, true, gn_scratch.as<void>(), n16,
                    stream);
    conv2d(n16, w, h, channels, false, d.conv_out.w, 3, 3, 3, 1, 1, d.conv_out.b, nullptr, a, stream);
    to_rgb8(a, hw, rgb.as<uint8_t>(), stream);
    if (!ok) {
        error = "VAE: a GEMM failed";
    }
    return ok;
}

// Kernels the warm-up tuned go into the tuning file. A file that cannot be written is said once.
void Pipeline::Impl::save_tuning() {
    if (!tuning_path.empty() && blas.tuning_changed() && !blas.save_tuning(tuning_path)) {
        std::fprintf(stderr, "[moss-image] cannot write the GEMM tuning file %s\n", tuning_path.c_str());
        tuning_path.clear();
    }
}

// Standby: the memory another process gives back as it stops (the other model's) can take a moment to be free.
bool Pipeline::Impl::wait_for_free(const size_t bytes, std::string & error) {
    const auto deadline = std::chrono::steady_clock::now() + kFreeWait;
    for (;;) {
        size_t free = 0, total = 0;
        if (hipMemGetInfo(&free, &total) != hipSuccess) {
            error = "cannot read the GPU's free memory";
            return false;
        }
        if (free >= bytes + kFreeSlack) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            error = "out of GPU memory: the image needs " + std::to_string(bytes >> 20) + " MiB more, " +
                    std::to_string(free >> 20) + " MiB are free";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

// The GPU's free memory, for every process.
size_t Pipeline::Impl::free_bytes() {
    size_t free = 0, total = 0;
    return hipMemGetInfo(&free, &total) == hipSuccess ? free : 0;
}

// After a free: until the GPU has `free_target` bytes free again, so that the other model has them at once.
void Pipeline::Impl::wait_cleared(const size_t free_target) {
    const auto deadline = std::chrono::steady_clock::now() + kClearWait;
    while (free_bytes() + kFreeSlack < free_target && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

// ---------------------------------------------------------------------------------------------------------------

Pipeline::Pipeline() : impl_(std::make_unique<Impl>()) {}

Pipeline::~Pipeline() = default;

bool Pipeline::load(const PipelineOptions & options, std::string & error) {
    Impl & m      = *impl_;
    int    device = 0;
    if (hipGetDevice(&device) != hipSuccess || !device_supported(device)) {
        error = "MOSS-Image needs an RDNA4 GPU (gfx1200 or gfx1201)";
        return false;
    }
    if (hipStreamCreateWithFlags(&m.stream, hipStreamNonBlocking) != hipSuccess || !m.blas.init(m.stream)) {
        error = "cannot create the HIP stream or the hipBLASLt handle";
        return false;
    }
    for (hipEvent_t & e : m.events) {
        if (hipEventCreate(&e) != hipSuccess) {
            error = "cannot create HIP events";
            return false;
        }
    }
    m.standby       = options.standby;
    m.warmup_width  = options.warmup_width;
    m.warmup_height = options.warmup_height;
    if (m.standby) {
        bool ok = hipStreamCreateWithFlags(&m.copy_stream, hipStreamNonBlocking) == hipSuccess;
        for (hipEvent_t & e : m.ready) {
            ok = ok && hipEventCreateWithFlags(&e, hipEventDisableTiming) == hipSuccess;
        }
        if (!ok) {
            error = "cannot create the HIP stream and events of the weights' copies";
            return false;
        }
    }
    using clock        = std::chrono::steady_clock;
    const auto seconds = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    // GEMM kernels tuned by an earlier run need no warm-up image: they are found again, and their first products run,
    // beside the uploads on a thread and stream of their own.
    m.tuning_path      = options.tuning_path;
    const size_t tuned = m.tuning_path.empty() ? 0 : m.blas.load_tuning(m.tuning_path);
    // The two files load at once: the text encoder through ggml on its own thread, the model through our uploads
    // (into host memory in standby). The encoder can run on another GPU (an APU's, beside the R9700), which leaves
    // the model's GPU only the model to load.
    int model_gpu = 0;
    if (hipGetDevice(&model_gpu) != hipSuccess) {
        error = "no GPU";
        return false;
    }
    const int   encoder_gpu = options.text_encoder_gpu < 0 ? model_gpu : options.text_encoder_gpu;
    const auto  t0          = clock::now();
    std::string encoder_error;
    double      encoder_seconds = 0.0;
    auto        encoder         = std::async(std::launch::async, [&] {
        const bool ok   = m.encoder.load(options.text_encoder_path, encoder_gpu, model_gpu, encoder_error);
        encoder_seconds = seconds(t0, clock::now());
        return ok;
    });
    auto        kernels         = std::async(std::launch::async, [&] {
        hipStream_t stream = nullptr;
        if (tuned == 0 || hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) {
            return size_t(0);
        }
        const size_t ran = m.blas.preload(stream);
        (void) hipStreamDestroy(stream);
        return ran;
    });
    const bool   model_ok      = m.standby ? m.weights.read(options.model_path, error)
                                           : m.weights.load(options.model_path, m.stream, error);
    const double model_seconds = seconds(t0, clock::now());
    const bool   encoder_ok    = encoder.get();
    const size_t preloaded     = kernels.get();
    if (!model_ok) {
        return false;
    }
    if (!encoder_ok) {
        error = encoder_error;
        return false;
    }
    const auto        t1    = clock::now();
    const std::string where = encoder_gpu == model_gpu ? "" : " on GPU " + std::to_string(encoder_gpu);
    const char *      model = m.standby ? "model into host memory" : "model";
    if (tuned > 0 && preloaded == tuned) {
        // The encoder's first prompt loads its kernels, which the first request then finds loaded.
        TextHidden hidden;
        if (!m.encoder.encode(m.encoder.tokenize("a photo of a lighthouse"), hidden, error)) {
            return false;
        }
        std::fprintf(stderr,
                     "[moss-image] loaded in %.1f s: %s %.1f s and text encoder%s %.1f s at once, with the %zu "
                     "GEMM kernels tuned in %s, first prompt %.2f s\n",
                     seconds(t0, clock::now()), model, model_seconds, where.c_str(), encoder_seconds, preloaded,
                     m.tuning_path.c_str(), seconds(t1, clock::now()));
        return true;
    }
    if (m.standby) {
        // The GPU is the other model's until a request comes: the first one tunes the kernels before its image.
        m.needs_warmup = true;
        std::fprintf(stderr,
                     "[moss-image] loaded in %.1f s: %s %.1f s and text encoder%s %.1f s at once; the first request "
                     "tunes the GEMM kernels\n",
                     seconds(t0, clock::now()), model, model_seconds, where.c_str(), encoder_seconds);
        return true;
    }
    if (!warm_up(error)) {
        return false;
    }
    std::fprintf(stderr,
                 "[moss-image] loaded in %.1f s: model %.1f s and text encoder%s %.1f s at once, warm-up %.1f s\n",
                 seconds(t0, clock::now()), model_seconds, where.c_str(), encoder_seconds, seconds(t1, clock::now()));
    return true;
}

// The only GEMM tuning: every shape of a request at the warm-up size, before the first request (only those not found
// again, after a partial preload), kept in the tuning file for the next run.
bool Pipeline::warm_up(std::string & error) {
    Impl &       m = *impl_;
    ImageRequest warmup;
    warmup.prompt = "a photo of a lighthouse";
    warmup.width  = m.warmup_width;
    warmup.height = m.warmup_height;
    warmup.steps  = 1;
    warmup.seed   = 0;
    RgbImage image;
    m.warming_up = true;
    m.blas.set_tuning(true);
    bool ok = true;
    for (const float cfg : { 2.5f, 1.0f }) {
        warmup.cfg_scale = cfg;
        ok               = ok && generate(warmup, image, error);
    }
    m.blas.set_tuning(false);
    m.warming_up = false;
    if (!ok) {
        error = "warm-up: " + error;
        return false;
    }
    m.save_tuning();
    return true;
}

bool Pipeline::generate(const ImageRequest & request, RgbImage & image, std::string & error) {
    Impl & m = *impl_;
    if (request.width % 16 != 0 || request.height % 16 != 0 ||
        std::min(request.width, request.height) < kMinImageSide ||
        int64_t(request.width) * request.height > kMaxImagePixels) {
        error = "width and height must be multiples of 16, at least " + std::to_string(kMinImageSide) +
                ", and at most " + std::to_string(kMaxImagePixels) + " pixels together";
        return false;
    }
    if (request.steps < 1 || request.steps > kMaxImageSteps) {
        error = "steps must be 1 to " + std::to_string(kMaxImageSteps);
        return false;
    }
    if (!(request.cfg_scale >= 0.0f) || !std::isfinite(request.cfg_scale) || !(request.flow_shift > 0.0f) ||
        !std::isfinite(request.flow_shift) || !(request.eta >= 0.0f) || !std::isfinite(request.eta) ||
        request.seed < 0) {
        error = "flow_shift must be positive; cfg_scale, eta and the seed must not be negative";
        return false;
    }
    if (m.streaming) {
        // A request failed while its weights were arriving: they finish first.
        (void) hipStreamSynchronize(m.copy_stream);
        m.streaming = false;
    }
    if (m.needs_warmup && !m.warming_up) {
        const auto t0 = std::chrono::steady_clock::now();
        if (!warm_up(error)) {
            return false;
        }
        m.needs_warmup = false;
        std::fprintf(stderr, "[moss-image] tuned the GEMM kernels in %.1f s, before the first image\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    const bool    cfg   = request.cfg_scale != 1.0f;
    const int     lh    = request.height / 16;
    const int     lw    = request.width / 16;
    const int64_t n_img = int64_t(lh) * lw;

    (void) hipEventRecord(m.events[0], m.stream);
    const bool streamed = !m.weights.on_device();
    if (streamed) {
        // Standby, with the weights in host memory: they arrive on their own stream while the text encoder and the
        // first step run, each part waited for where it is first used. The warm-up times kernels, so it waits for all.
        if (!m.wait_for_free(m.weights.host_bytes(), error) ||
            !m.weights.upload(m.copy_stream, m.ready, nullptr, error)) {
            return false;
        }
        m.streaming = !m.warming_up;
        if (m.warming_up && hipStreamSynchronize(m.copy_stream) != hipSuccess) {
            error = "copying the weights to the GPU failed";
            return false;
        }
    }
    if (!m.text(request.prompt, m.cond, error) || (cfg && !m.text(request.negative_prompt, m.uncond, error))) {
        return false;
    }
    const int64_t rows_c = n_img + m.cond->n;
    const int64_t rows_u = cfg ? n_img + m.uncond->n : 0;
    if (m.standby) {
        size_t need = 0;
        m.reserve(n_img, rows_c + rows_u, cfg ? 2 : 1, request.steps, 4 * n_img, error, &need);
        if (need > 0 && !m.wait_for_free(need, error)) {
            return false;
        }
    }
    if (!m.reserve(n_img, rows_c + rows_u, cfg ? 2 : 1, request.steps, 4 * n_img, error)) {
        return false;
    }
    (void) hipEventRecord(m.events[1], m.stream);
    if (!m.sample(request, lh, lw, rows_c, rows_u, error)) {
        return false;
    }
    (void) hipEventRecord(m.events[2], m.stream);
    if (!m.decode(lh, lw, error)) {
        return false;
    }
    image.width  = request.width;
    image.height = request.height;
    image.rgb.resize(size_t(request.width) * request.height * 3);
    (void) hipMemcpyAsync(image.rgb.data(), m.rgb.as<uint8_t>(), image.rgb.size(), hipMemcpyDeviceToHost, m.stream);
    (void) hipEventRecord(m.events[3], m.stream);
    hipError_t status = hipStreamSynchronize(m.stream);
    if (status == hipSuccess) {
        status = hipGetLastError();
    }
    if (status != hipSuccess) {
        error = std::string("GPU error: ") + hipGetErrorString(status);
        return false;
    }
    m.streaming = false;  // the stream waited for every part
    if (m.warming_up) {
        return true;
    }
    float text_ms = 0.0f, sample_ms = 0.0f, decode_ms = 0.0f;
    (void) hipEventElapsedTime(&text_ms, m.events[0], m.events[1]);
    (void) hipEventElapsedTime(&sample_ms, m.events[1], m.events[2]);
    (void) hipEventElapsedTime(&decode_ms, m.events[2], m.events[3]);
    std::fprintf(stderr, "[moss-image] %dx%d, %d steps, cfg %.2f: text %.0f ms, sampling %.0f ms, decode %.0f ms%s\n",
                 request.width, request.height, request.steps, request.cfg_scale, text_ms, sample_ms, decode_ms,
                 streamed ? ", weights copied in meanwhile" : "");
    return true;
}

MemoryState Pipeline::memory() {
    Impl &      m = *impl_;
    MemoryState state;
    state.standby        = m.standby;
    state.weights_on_gpu = m.weights.on_device();
    state.gpu_bytes      = m.weights.device_bytes();
    for (const DeviceBuffer * buffer : m.work_buffers()) {
        state.gpu_bytes += buffer->bytes();
    }
    for (const TextRows & rows : m.text_cache) {
        state.gpu_bytes += rows.rows.bytes();
    }
    state.host_bytes = m.weights.host_bytes();
    size_t total     = 0;
    (void) hipMemGetInfo(&state.gpu_free_bytes, &total);
    return state;
}

bool Pipeline::release_memory(const bool keep_weights, std::string & error) {
    Impl & m = *impl_;
    if (!m.standby) {
        return Generator::release_memory(keep_weights, error);
    }
    if (hipStreamSynchronize(m.stream) != hipSuccess || hipStreamSynchronize(m.copy_stream) != hipSuccess) {
        error = "the GPU failed";
        return false;
    }
    m.streaming     = false;
    const auto   t0 = std::chrono::steady_clock::now();
    const size_t before = m.free_bytes();
    size_t       freed  = 0;
    for (DeviceBuffer * buffer : m.work_buffers()) {
        freed += buffer->bytes();
        buffer->release();
    }
    if (!keep_weights) {
        freed += m.weights.device_bytes();
        m.weights.release();
    }
    if (freed > 0) {
        m.wait_cleared(before + freed);
        std::fprintf(stderr, "[moss-image] gave %zu MiB of the GPU back in %.0f ms\n", freed >> 20,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return true;
}

bool Pipeline::load_weights(const std::function<bool()> & stopped, std::string & error) {
    Impl & m = *impl_;
    if (!m.standby) {
        return Generator::load_weights(stopped, error);
    }
    if (m.weights.on_device()) {
        return true;
    }
    if (stopped()) {
        return false;
    }
    const size_t bytes = m.weights.host_bytes();
    const size_t free  = m.free_bytes();
    if (free < bytes + kLoadHeadroom) {
        error = "no room for the weights: " + std::to_string(free >> 20) + " MiB free, " +
                std::to_string((bytes + kLoadHeadroom) >> 20) + " MiB wanted";
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!m.weights.upload(m.copy_stream, m.ready, &stopped, error)) {
        m.wait_cleared(free);  // stopped: the copy started is given back
        return false;
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "[moss-image] weights on the GPU in %.2f s (%.1f GB/s)\n", seconds, bytes / seconds / 1e9);
    return true;
}

} // namespace luce::moss_image
