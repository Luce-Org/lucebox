// Host API of the MOSS-Image GPU kernels (RDNA4, gfx1200 / gfx1201).
//
// Every launcher takes raw device pointers and a stream and only enqueues work. f16 and bf16 values are stored as
// uint16_t. Matrices are row-major. The kernels repeat the float operations of the stable-diffusion.cpp build they
// were measured in (same order, same rounding), so the port's images match it.
//
// Int8 linears use the convrot format: an activation row of k values is quantized to int8 after a 256-wide Hadamard
// rotation, with one scale per row (xq [rows][k] int8, xs [rows] f32).
//
// Batched CFG: the conditional and the unconditional sequence are stacked in one activation, conditional rows first
// (rows 0 .. n_cond - 1, then n_cond .. n_cond + n_uncond - 1). Each sequence is its n_img image tokens followed by
// its text tokens. Row kernels and GEMMs run on all rows at once; attention runs once per sequence.

#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <vector>

namespace luce::moss_image {

// Attention heads are 128 wide (kHeadDim in weights.h); a RoPE table row holds the (cos, sin) of their 64 rotated pairs.
constexpr int kRopePairs = 64;

// True when the kernels run on the device: RDNA4 (gfx1200, gfx1201).
bool device_supported(int device);

// ---------------------------------------------------------------------------------------------------------------
// Host math

// RoPE table of one sequence, n_img = h * w image tokens then n_text text tokens: pe [n_img + n_text][64][2] holds
// (cos, sin) of every rotated pair, positions (0, n_text, row - (h - 1) / 2, col - (w - 1) / 2) for image tokens and
// (0, i, 0, 0) for text token i, four axes of 16 frequencies at theta 10000. The text rows do not depend on h and w.
void rope_table(int h, int w, int n_text, float * pe);

// The timestep features of the model at sigma: t = 1000 sigma, out[i] = cos(t / 10000^(i / 128)), out[128 + i] the sin.
void time_features(float sigma, float * out);

enum class Schedule { Simple, Discrete };
enum class Sampler { Euler, ErSde };

// The flow-matching sigmas of a schedule (shift applied), steps + 1 values ending in 0.
std::vector<float> flow_sigmas(Schedule schedule, int steps, float flow_shift);

// One sampler step, computed on the host from the sigmas (see plan_steps) and applied by sampler_step on the GPU.
// den = pred * -sigma + x, with pred = uncond + cfg_scale * (cond - uncond) when there is an unconditional pass.
struct SamplerStep {
    float sigma      = 0.0f;
    float cfg_scale  = 1.0f;
    bool  last       = false;  // x = den
    bool  euler      = false;  // x += (x - den) / sigma * (sigma_next - sigma)
    float sigma_next = 0.0f;
    // er_sde: x = c1 x + c2 den, + coeff_d dd with dd = (den - old_den) / denom_d, + coeff_u (dd - old_dd) / denom_u,
    // + noise_scale * noise(noise_offset); old_den = den and old_dd = dd for the next step.
    float    c1 = 0.0f, c2 = 0.0f;
    bool     use_d = false, use_u = false, noise = false;
    float    denom_d = 0.0f, coeff_d = 0.0f;
    float    denom_u = 0.0f, coeff_u = 0.0f;
    float    noise_scale  = 0.0f;
    uint32_t noise_offset = 0;
};

// The steps of a sampler over sigmas (from flow_sigmas), with stable-diffusion.cpp's scalar arithmetic. The initial
// noise uses Philox offset 0; er_sde (eta = noise strength) draws its step noise at offsets 1, 2, ... in order.
std::vector<SamplerStep> plan_steps(Sampler sampler, const std::vector<float> & sigmas, float cfg_scale, float eta);

// ---------------------------------------------------------------------------------------------------------------
// Sampler (state x [tokens][channels] f32, token-major; the noise of element (t, c) is Philox index c * tokens + t)

// x = Philox normal noise at offset 0 (the initial latent, sigma 1).
void init_noise(uint64_t seed, int64_t tokens, int64_t channels, float * x, hipStream_t stream);

// One step on x in place. pred [tokens][channels] is the model output (conditional), pred_uncond the unconditional
// one or null. old_den and old_dd are er_sde's history buffers (same size as x, unused otherwise).
void sampler_step(const SamplerStep & step, const float * pred, const float * pred_uncond, uint64_t seed,
                  int64_t tokens, int64_t channels, float * x, float * old_den, float * old_dd, hipStream_t stream);

// ---------------------------------------------------------------------------------------------------------------
// Elementwise glue shared by the text blocks, the DiT and the VAE

// dst [n] = x rounded to bf16 (to nearest even): the activation of a bf16 linear.
void f32_to_bf16(const float * x, int64_t n, uint16_t * dst, hipStream_t stream);

// dst [n] = x rounded to f16.
void f32_to_f16(const float * x, int64_t n, uint16_t * dst, hipStream_t stream);

// dst [rows][n] = mm + bias[col]: a linear's bias, as a separate add. dst may be mm.
void add_bias(const float * mm, const float * bias, int64_t rows, int64_t n, float * dst, hipStream_t stream);

// x [rows][n] += mm + bias[col]: a linear's bias, then the residual add.
void add_bias_residual(const float * mm, const float * bias, int64_t rows, int64_t n, float * x, hipStream_t stream);

// dst [rows][n] = x w [rows][k] x [n][k] bf16 transposed, + bias [n], then SiLU when silu: the one-row linears of the
// timestep path (ggml's mmvf arithmetic per row), for up to 16 rows (steps) at once. k is even.
void linear_bf16_rows(const uint16_t * w, const float * bias, const float * x, int rows, int64_t n, int64_t k,
                      bool silu, float * dst, hipStream_t stream);

// ---------------------------------------------------------------------------------------------------------------
// Text blocks (bf16 linears on hipBLASLt in between)

// dst [rows][dim] bf16 = rms_norm(x) * w (eps), x [rows][dim] f32: a norm feeding a bf16 linear. dim >= 1024.
void rms_norm_mul_bf16(const float * x, const float * w, float eps, int64_t rows, int64_t dim, uint16_t * dst,
                       hipStream_t stream);

// The attention inputs from a text block's qkv linear, raw [rows][3 * heads * 128] without its bias: q and k get the
// bias, rms_norm * q_w / k_w (eps) and RoPE (pe [rows][64][2], see rope_table), v gets the bias. q f32, k and v f16,
// each [heads][rows][128] (the layout attention_d128 reads).
void text_qkv_prologue(const float * raw, const float * bias, const float * q_w, const float * k_w, float eps,
                       const float * pe, int64_t rows, int heads, float * q, uint16_t * k, uint16_t * v,
                       hipStream_t stream);

// The text FFN gate: mm [rows][2 * nc] holds the up linear (first half) and the gate linear (second half) without
// their biases; dst [rows][nc] bf16 = situ(gate + bias_gate, up + bias_up) with the caps 4 and 25, the down linear's
// input.
void text_situ_bf16(const float * mm, const float * bias_up, const float * bias_gate, int64_t rows, int64_t nc,
                    uint16_t * dst, hipStream_t stream);

// ---------------------------------------------------------------------------------------------------------------
// DiT

// The DiT input rows: for the conditional sequence (and the unconditional one when text_u is not null), n_img image
// rows = img + img_bias (the video embedder's output without its bias, shared by both) followed by the sequence's
// text rows (text_c [n_text_c][dim], text_u [n_text_u][dim]). x [n_img + n_text_c (+ n_img + n_text_u)][dim].
void assemble_tokens(const float * img, const float * img_bias, int64_t n_img, const float * text_c, int64_t n_text_c,
                     const float * text_u, int64_t n_text_u, int64_t dim, float * x, hipStream_t stream);

// Modulated RMS norm quantized for an int8 linear: rms_norm(x) * (norm_w * (1 + scale)) + shift (eps), x [rows][k]
// f32 -> xq [rows][k], xs [rows]. scale and shift are the step's modulation [k]. k is 2048, 3072 or 4096.
void norm_quantize_i8(const float * x, int64_t rows, int64_t k, const float * norm_w, const float * scale,
                      const float * shift, float eps, int8_t * xq, float * xs, hipStream_t stream);

// Convrot quantization without a norm: x [rows][k] f32 -> xq [rows][k], xs [rows]. k is 2048, 3072 or 4096.
void quantize_i8(const float * x, int64_t rows, int64_t k, int8_t * xq, float * xs, hipStream_t stream);

// The workspace `split` of the two int8 GEMMs below, which split the k loop of a partial last token block across
// blocks: zeroed before its first use, and left zero by every launch. Launches that can run at once (on two streams)
// need one each; nullptr runs them without.
size_t gemm_i8_split_bytes();

// The int8 qkv linear fused with the attention prologue: w [3 * heads * 128][k] int8 with w_scales [3 * heads * 128]
// times xq [rows][k] (xs [rows]), then q and k through rms_norm * q_w / k_w (eps) and RoPE (pe [rows][64][2]), v cast.
// q f32, k and v f16, each [heads][rows][128]. heads * 128 % 256 == 0, k % 64 == 0.
void gemm_i8_qkv_attention(const int8_t * w, const float * w_scales, const int8_t * xq, const float * xs, int64_t rows,
                           int64_t k, int heads, const float * q_w, const float * k_w, float eps, const float * pe,
                           float * q, uint16_t * k_out, uint16_t * v, int32_t * split, hipStream_t stream);

// Self-attention with head size 128, no mask, of the sequence in rows first .. first + n - 1 of q, k and v
// ([heads][rows][128] as written by the prologues). dst [rows][heads * 128] f32, written at rows first .. first + n - 1.
// scale is 1 / sqrt(128).
void attention_d128(const float * q, const uint16_t * k, const uint16_t * v, int64_t rows, int64_t first, int64_t n,
                    int heads, float scale, float * dst, hipStream_t stream);

// An int8 linear with a gated residual: x [rows][n] += (w [n][k] (w_scales [n]) times xq [rows][k] (xs [rows])) * gate
// [n]. n % 256 == 0, k % 64 == 0.
void gemm_i8_gated_residual(const int8_t * w, const float * w_scales, const int8_t * xq, const float * xs, int64_t rows,
                            int64_t k, int64_t n, const float * gate, float * x, int32_t * split, hipStream_t stream);

// The int32 sums of an int8 linear: w [n][k] int8 times xq [rows][k], sums [rows][n] (rows past rows not written).
// n % 256 == 0, k % 64 == 0.
void gemm_i8_i32(const int8_t * w, const int8_t * xq, int64_t rows, int64_t k, int64_t n, int32_t * sums,
                 hipStream_t stream);

// The FFN gate on the int32 sums of the up/gate linear and its convrot quantization for the down linear: sums
// [rows][2 * nc] (any row stride padding past rows is ignored) with the activation scales xs [rows] and w_scales
// [2 * nc]; up is the first half and the gate the second. dst [rows][nc] int8, dst_scales [rows]. nc is 4096 or 8192.
void situ_quantize_i8(const int32_t * sums, const float * xs, const float * w_scales, int64_t rows, int64_t nc,
                      int8_t * dst, float * dst_scales, hipStream_t stream);

// The output head's input: the image rows of each sequence (rows 0 .. n_img - 1 and, when n_seq is 2, uncond_first ..
// uncond_first + n_img - 1 of x [*][dim]) through rms_norm * norm_w (eps), then y + y * scale + shift with the head
// modulation [2 * dim] (shift, then scale), rounded to bf16: dst [n_seq * n_img][dim]. dim >= 1024.
void final_head_bf16(const float * x, int64_t dim, int64_t n_img, int n_seq, int64_t uncond_first, const float * norm_w,
                     const float * head_mod, float eps, uint16_t * dst, hipStream_t stream);

// ---------------------------------------------------------------------------------------------------------------
// VAE decoder (feature maps [C][H * W] f32, N = 1)

// The DiT state as the VAE's latent: state [h * w][128] (token-major, channel c of token y * w + x) times std[c] plus
// mean[c], unpatchified 2 x 2: latent [32][2h][2w] f32, latent[c][2y + py][2x + px] = state channel 4c + 2py + px.
void unpack_latent(const float * state, const float * mean, const float * std, int h, int w, float * latent,
                   hipStream_t stream);

// x [C][HW] f32 -> y [HW][C] f16: a convolution's channels-last input.
void to_nhwc_f16(const float * x, int C, int64_t HW, uint16_t * y, hipStream_t stream);

// Scratch bytes group_norm_nhwc needs for a map of C channels and HW pixels in groups groups.
size_t group_norm_scratch_bytes(int C, int64_t HW, int groups);

// GroupNorm with the per-channel affine, then SiLU when silu: x [C][HW] f32 -> y [HW][C] f16, a convolution's input.
// C % 64 == 0, HW % 4 == 0.
void group_norm_nhwc(const float * x, int C, int64_t HW, int groups, float eps, const float * gamma, const float * beta,
                     bool silu, void * scratch, uint16_t * y, hipStream_t stream);

// Stride-1 2D convolution: x [IH * IW][IC] f16 channels-last (with up2x the source of a nearest 2x upscale,
// [IH / 2 * IW / 2][IC]), w [OC][KH * KW][IC] f16, zero padding px, py; y [OC][OH * OW] f32 = conv + bias [OC] (if
// not null) + res [OC][OH * OW] (if not null). IC % 32 == 0; OC <= 16 (without up2x) runs the small-OC kernel.
void conv2d(const uint16_t * x, int IW, int IH, int IC, bool up2x, const uint16_t * w, int KW, int KH, int OC, int px,
            int py, const float * bias, const float * res, float * y, hipStream_t stream);

// Row softmax of s [rows][cols] f32 scaled by scale (ggml's soft_max arithmetic), rounded to f16: p [rows][cols].
void softmax_f16(const float * s, int64_t rows, int64_t cols, float scale, uint16_t * p, hipStream_t stream);

// The decoded image x [3][H * W] f32 in [-1, 1] -> rgb [H * W][3] uint8.
void to_rgb8(const float * x, int64_t HW, uint8_t * rgb, hipStream_t stream);

} // namespace luce::moss_image
