// MOSS-Image sampling: the RoPE and timestep tables, the flow sigmas, and the sampler steps (Euler, er_sde) with their
// Philox noise.
//
// stable-diffusion.cpp samples on the CPU in float, one rounded operation per term (its build has no FMA). This file
// repeats those operations in the same order with contraction off, on the host and on the device, so the sigmas, the
// step coefficients, the noise and the steps come out the same. HIP's __fadd_rn and __fmul_rn would not do: they are
// plain operators compiled where contraction is on, so they fuse once inlined.

#include "kernels.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>

#pragma clang fp contract(off)

namespace luce::moss_image {
namespace {

constexpr int kBlock     = 256;
constexpr int kTimesteps = 1000;  // the flow schedule's timesteps, t = 0 .. 999
constexpr int kAxisPairs = 16;    // RoPE pairs per position axis (4 axes of 32 dims)

int blocks(const int64_t n) {
    return int((n + kBlock - 1) / kBlock);
}

// Philox 4x32-10 of draw `offset`, element `index` (torch's CUDA randn, as PhiloxRNG::randn computes it), then
// Box-Muller of the first two words; the log, the square root and the sin run in double there too.
__device__ float philox_normal(const uint64_t seed, const uint32_t offset, const uint32_t index) {
    uint32_t c0 = offset, c1 = 0, c2 = index, c3 = 0;
    uint32_t k0 = uint32_t(seed), k1 = uint32_t(seed >> 32);
#pragma unroll
    for (int round = 0; round < 10; ++round) {
        const uint32_t hi0 = __umulhi(c0, 0xD2511F53u);
        const uint32_t lo0 = c0 * 0xD2511F53u;
        const uint32_t hi2 = __umulhi(c2, 0xCD9E8D57u);
        const uint32_t lo2 = c2 * 0xCD9E8D57u;
        c0                 = hi2 ^ c1 ^ k0;
        c1                 = lo2;
        c2                 = hi0 ^ c3 ^ k1;
        c3                 = lo0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    constexpr float inv     = 2.3283064e-10f;
    constexpr float inv_2pi = 2.3283064e-10f * 6.2831855f;
    const float     u       = float(c0) * inv + inv / 2;
    const float     v       = float(c1) * inv_2pi + inv_2pi / 2;
    const float     s       = float(sqrt(-2.0 * log(double(u))));
    return float(double(s) * sin(double(v)));
}

__global__ void init_noise_kernel(const uint64_t seed, const int64_t tokens, const int64_t channels, float * x) {
    const int64_t i = int64_t(blockIdx.x) * kBlock + threadIdx.x;
    if (i >= tokens * channels) {
        return;
    }
    const int64_t t = i / channels;
    const int64_t c = i - t * channels;
    x[i]            = philox_normal(seed, 0, uint32_t(c * tokens + t));
}

__global__ void sampler_step_kernel(const SamplerStep step, const float * pred, const float * pred_uncond,
                                    const uint64_t seed, const int64_t tokens, const int64_t channels, float * x,
                                    float * old_den, float * old_dd) {
    const int64_t i = int64_t(blockIdx.x) * kBlock + threadIdx.x;
    if (i >= tokens * channels) {
        return;
    }
    float p = pred[i];
    if (pred_uncond != nullptr) {
        const float u = pred_uncond[i];
        p             = u + (p - u) * step.cfg_scale;
    }
    const float xi  = x[i];
    const float den = p * -step.sigma + xi;
    if (step.last) {
        x[i] = den;
        return;
    }
    if (step.euler) {
        const float d = (xi - den) / step.sigma;
        x[i]          = xi + d * (step.sigma_next - step.sigma);
        return;
    }
    float y = xi * step.c1 + den * step.c2;
    if (step.use_d) {
        const float dd = (den - old_den[i]) / step.denom_d;
        y              = y + dd * step.coeff_d;
        if (step.use_u) {
            const float du = (dd - old_dd[i]) / step.denom_u;
            y              = y + du * step.coeff_u;
        }
        old_dd[i] = dd;
    }
    if (step.noise) {
        const int64_t t = i / channels;
        const int64_t c = i - t * channels;
        const float   n = philox_normal(seed, step.noise_offset, uint32_t(c * tokens + t));
        y               = y + n * step.noise_scale;
    }
    x[i]       = y;
    old_den[i] = den;
}

// DiscreteFlowDenoiser::t_to_sigma: time_snr_shift(shift, (t + 1) / 1000).
float t_to_sigma(float t, const float shift) {
    t = t + 1;
    t = t / 1000.f;
    if (shift == 1.0f) {
        return t;
    }
    return shift * t / (1 + (shift - 1) * t);
}

// er_sde's flow sigma clamp, lambda, alpha and noise scaler.
float er_flow_sigma(float sigma) {
    sigma = std::max(sigma, 1e-6f);
    return std::min(sigma, 1.0f - 1e-4f);
}

float er_lambda(const float sigma) {
    const float s = er_flow_sigma(sigma);
    return s / std::max(1.0f - s, 1e-6f);
}

float er_alpha(const float sigma) {
    return 1.0f - er_flow_sigma(sigma);
}

float er_scaler(float x) {
    x = std::max(x, 0.0f);
    return x * (std::exp(std::pow(x, 0.3f)) + 10.0f);
}

std::vector<SamplerStep> plan_er_sde(std::vector<float> sigmas, const float cfg_scale, const float eta) {
    constexpr int   kMaxStage  = 3;
    constexpr int   kPoints    = 200;
    constexpr float kPointsF   = 200.0f;
    const int       n          = int(sigmas.size()) - 1;
    for (int i = 0; i < n; ++i) {
        if (sigmas[i] > 1.0f) {
            sigmas[i] = er_flow_sigma(sigmas[i]);
        }
    }
    std::vector<float> lambdas(sigmas.size());
    for (size_t i = 0; i < sigmas.size(); ++i) {
        lambdas[i] = er_lambda(sigmas[i]);
    }

    std::vector<SamplerStep> steps;
    bool                     have_old_dd  = false;
    uint32_t                 noise_offset = 1;  // offset 0 is the initial noise
    for (int i = 0; i < n; ++i) {
        SamplerStep s;
        s.sigma      = sigmas[i];
        s.cfg_scale  = cfg_scale;
        s.sigma_next = sigmas[i + 1];
        if (sigmas[i + 1] == 0.0f) {
            s.last = true;
            steps.push_back(s);
            continue;
        }
        const int   stage    = std::min(kMaxStage, i + 1);
        const float lambda_s = lambdas[i];
        const float lambda_t = lambdas[i + 1];
        const float alpha_s  = er_alpha(sigmas[i]);
        const float alpha_t  = er_alpha(sigmas[i + 1]);
        const float scaled_s = er_scaler(lambda_s);
        const float scaled_t = er_scaler(lambda_t);
        const float r_alpha  = alpha_s > 0.0f ? alpha_t / alpha_s : 0.0f;
        const float r        = scaled_s > 0.0f ? scaled_t / scaled_s : 0.0f;
        s.c1                 = r_alpha * r;
        s.c2                 = alpha_t * (1.0f - r);

        if (stage >= 2 && i > 0) {
            const float dt        = lambda_t - lambda_s;
            const float step_size = -dt / kPointsF;
            const bool  stage_u   = stage >= 3 && have_old_dd;
            float       sum       = 0.0f;
            float       sum_u     = 0.0f;
            for (int p = 0; p < kPoints; ++p) {
                const float lambda_pos = lambda_t + p * step_size;
                const float scaled_pos = er_scaler(lambda_pos);
                if (scaled_pos <= 0.0f) {
                    continue;
                }
                sum += 1.0f / scaled_pos;
                if (stage_u) {
                    sum_u += (lambda_pos - lambda_s) / scaled_pos;
                }
            }
            sum *= step_size;

            const float denom_d = lambda_s - lambdas[i - 1];
            if (std::fabs(denom_d) > 1e-12f) {
                s.use_d   = true;
                s.denom_d = denom_d;
                s.coeff_d = alpha_t * (dt + sum * scaled_t);
                if (stage_u) {
                    const float denom_u = (lambda_s - lambdas[i - 2]) * 0.5f;
                    if (std::fabs(denom_u) > 1e-12f) {
                        sum_u *= step_size;
                        s.use_u   = true;
                        s.denom_u = denom_u;
                        s.coeff_u = alpha_t * (0.5f * dt * dt + sum_u * scaled_t);
                    }
                }
                have_old_dd = true;
            }
        }

        const float noise_sq = lambda_t * lambda_t - lambda_s * lambda_s * r * r;
        if (eta > 0.0f && noise_sq > 0.0f) {
            s.noise        = true;
            s.noise_scale  = alpha_t * std::sqrt(std::max(noise_sq, 0.0f));
            s.noise_offset = noise_offset++;
        }
        steps.push_back(s);
    }
    return steps;
}

} // namespace

void rope_table(const int h, const int w, const int n_text, float * pe) {
    // Rope::rope_frequencies(32, 10000): linspace(0, 30 / 32, 16) exponents.
    float       omega[kAxisPairs];
    const float end  = (32 * 1.f - 2) / 32;
    const float step = (end - 0.f) / (kAxisPairs - 1);
    for (int i = 0; i < kAxisPairs; ++i) {
        omega[i] = 1.0f / ::powf(10000.f, 0.f + i * step);
    }
    // The (cos, sin) of one axis at a position.
    auto axis = [&](const float pos, float * out) {
        for (int j = 0; j < kAxisPairs; ++j) {
            const float angle = pos * omega[j];
            out[2 * j]        = std::cos(angle);
            out[2 * j + 1]    = std::sin(angle);
        }
    };
    constexpr int      kAxis = 2 * kAxisPairs;
    std::vector<float> zero(kAxis), text_axis(kAxis), rows(size_t(h) * kAxis), cols(size_t(w) * kAxis);
    axis(0.f, zero.data());
    axis(float(n_text), text_axis.data());
    for (int y = 0; y < h; ++y) {
        axis(float(y) - float(h - 1) / 2.f, rows.data() + size_t(y) * kAxis);
    }
    for (int x = 0; x < w; ++x) {
        axis(float(x) - float(w - 1) / 2.f, cols.data() + size_t(x) * kAxis);
    }
    // Axes (0, text index, row, column): image tokens at (0, n_text, row, col), text token i at (0, i, 0, 0).
    float * dst = pe;
    for (int t = 0; t < h * w; ++t, dst += 4 * kAxis) {
        std::copy_n(zero.data(), kAxis, dst);
        std::copy_n(text_axis.data(), kAxis, dst + kAxis);
        std::copy_n(rows.data() + size_t(t / w) * kAxis, kAxis, dst + 2 * kAxis);
        std::copy_n(cols.data() + size_t(t % w) * kAxis, kAxis, dst + 3 * kAxis);
    }
    for (int i = 0; i < n_text; ++i, dst += 4 * kAxis) {
        std::copy_n(zero.data(), kAxis, dst);
        axis(float(i), dst + kAxis);
        std::copy_n(zero.data(), kAxis, dst + 2 * kAxis);
        std::copy_n(zero.data(), kAxis, dst + 3 * kAxis);
    }
}

void time_features(const float sigma, float * out) {
    const float t = sigma * 1000.f;  // DiscreteFlowDenoiser::sigma_to_t
    for (int i = 0; i < 128; ++i) {
        const double angle = t * std::pow(10000.0, -double(i) / double(128));
        out[i]             = float(std::cos(angle));
        out[i + 128]       = float(std::sin(angle));
    }
}

std::vector<float> flow_sigmas(const Schedule schedule, const int steps, const float flow_shift) {
    std::vector<float> sigmas;
    if (steps <= 0) {
        return sigmas;
    }
    const int t_max = kTimesteps - 1;
    if (schedule == Schedule::Discrete) {
        if (steps == 1) {
            sigmas.push_back(t_to_sigma(float(t_max), flow_shift));
        } else {
            const float step = float(t_max) / float(steps - 1);
            for (int i = 0; i < steps; ++i) {
                sigmas.push_back(t_to_sigma(t_max - step * i, flow_shift));
            }
        }
    } else {
        const float step_factor = float(kTimesteps) / float(steps);
        for (int i = 0; i < steps; ++i) {
            const int index = std::max(t_max - int(float(i) * step_factor), 0);
            sigmas.push_back(t_to_sigma(float(index), flow_shift));
        }
    }
    sigmas.push_back(0.0f);
    return sigmas;
}

std::vector<SamplerStep> plan_steps(const Sampler sampler, const std::vector<float> & sigmas, const float cfg_scale,
                                    const float eta) {
    if (sampler == Sampler::ErSde) {
        return plan_er_sde(sigmas, cfg_scale, eta);
    }
    std::vector<SamplerStep> steps;
    for (size_t i = 0; i + 1 < sigmas.size(); ++i) {
        if (sigmas[i] == sigmas[i + 1]) {
            continue;  // sample_euler skips the step, model call included
        }
        SamplerStep s;
        s.sigma      = sigmas[i];
        s.cfg_scale  = cfg_scale;
        s.euler      = true;
        s.sigma_next = sigmas[i + 1];
        steps.push_back(s);
    }
    return steps;
}

void init_noise(const uint64_t seed, const int64_t tokens, const int64_t channels, float * x, hipStream_t stream) {
    init_noise_kernel<<<blocks(tokens * channels), kBlock, 0, stream>>>(seed, tokens, channels, x);
}

void sampler_step(const SamplerStep & step, const float * pred, const float * pred_uncond, const uint64_t seed,
                  const int64_t tokens, const int64_t channels, float * x, float * old_den, float * old_dd,
                  hipStream_t stream) {
    sampler_step_kernel<<<blocks(tokens * channels), kBlock, 0, stream>>>(step, pred, pred_uncond, seed, tokens,
                                                                          channels, x, old_den, old_dd);
}

} // namespace luce::moss_image
