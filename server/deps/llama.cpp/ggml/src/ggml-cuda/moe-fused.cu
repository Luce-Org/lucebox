#include "moe-fused.cuh"
#include "ggml-cuda/vecdotq.cuh"
#include "ggml-cuda/dequantize.cuh"
#include "ggml-cuda/mmvq.cuh"

#include <cstring>

static __device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

// IQ2_XS: compute dot product of one weight row (256 values) with input vector
// Weight row starts at: (const char*)w_base + row * row_stride
// block_stride = nb[0] (74 bytes per block)
static __device__ float dot_iq2_xs(
        const char * w_row, const float * input) {
    float sum = 0.0f;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const block_iq2_xs * blk = (const block_iq2_xs *)(w_row + ib32 * sizeof(block_iq2_xs));
        const float d = (float)blk->d;
        for (int l = 0; l < 4; ++l) {
            const float db = d * (0.5f + ((blk->scales[ib32] >> 4*(l/2)) & 0xf)) * 0.25f;
            const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (blk->qs[4*ib32 + l] & 511));
            const uint8_t signs = ksigns_iq2xs[blk->qs[4*ib32 + l] >> 9];
            const int off = 32*ib32 + 8*l;
            for (int j = 0; j < 8; ++j) {
                sum += db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f) * input[off + j];
            }
        }
    }
    return sum;
}

// IQ3_XXS: compute dot product of one weight row (256 values) with input vector
static __device__ float dot_iq3_xxs(
        const char * w_row, const float * input) {
    float sum = 0.0f;
    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
        const block_iq3_xxs * blk = (const block_iq3_xxs *)(w_row + ib32 * sizeof(block_iq3_xxs));
        const float d = (float)blk->d;
        const uint8_t * qs = blk->qs;
        const uint16_t * gas = (const uint16_t *)(blk->qs + QK_K/4);
        const uint32_t aux32 = gas[0] | (gas[1] << 16);
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
            const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
            const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
            const int off = 8*l;
            for (int j = 0; j < 4; ++j) {
                sum += db * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f) * input[32*ib32 + off + j];
                sum += db * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f) * input[32*ib32 + off + 4 + j];
            }
        }
    }
    return sum;
}

template<int BLOCK_SIZE>
static __global__ void moe_fused_kernel(
    const float * __restrict__ input,
    const void * __restrict__ gate_w,
    const void * __restrict__ up_w,
    const void * __restrict__ down_w,
    const int32_t * __restrict__ expert_ids,
    const float * __restrict__ expert_wts,
    const void * __restrict__ sh_gate_w,
    const void * __restrict__ sh_up_w,
    const void * __restrict__ sh_down_w,
    const void * __restrict__ sh_gate_inp_w,
    float * __restrict__ output,
    const int n_embd,
    const int ff_dim,
    const int n_expert_used,
    const int n_expert_total,
    const int gate_type,
    const int up_type,
    const int down_type,
    const size_t gate_row_stride,
    const size_t up_row_stride,
    const size_t down_row_stride,
    const size_t gate_expert_stride,
    const size_t up_expert_stride,
    const size_t down_expert_stride,
    const size_t gate_block_stride,
    const size_t up_block_stride,
    const size_t down_block_stride) {

    extern __shared__ float gu_smem[];

    const int tid = threadIdx.x;
    const int nblocks_in = n_embd / QK_K;
    const int nblocks_ff = ff_dim / QK_K;

    for (int e = 0; e < n_expert_used; e++) {
        const int eidx = expert_ids[e];
        const char * g_exp = (const char *) gate_w + (size_t) eidx * gate_expert_stride;
        const char * u_exp = (const char *) up_w   + (size_t) eidx * up_expert_stride;

        for (int k = tid; k < ff_dim; k += BLOCK_SIZE) {
            float g_sum = 0.0f;
            float u_sum = 0.0f;

            const char * g_row = g_exp + k * gate_row_stride;
            const char * u_row = u_exp + k * up_row_stride;

            if (gate_type == 17) {
                for (int b = 0; b < nblocks_in; b++) {
                    const block_iq2_xs * g_blk = (const block_iq2_xs *)(g_row + b * sizeof(block_iq2_xs));
                    const float d = (float)g_blk->d;
                    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                        for (int l = 0; l < 4; ++l) {
                            const float db = d * (0.5f + ((g_blk->scales[ib32] >> 4*(l/2)) & 0xf)) * 0.25f;
                            const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (g_blk->qs[4*ib32 + l] & 511));
                            const uint8_t signs = ksigns_iq2xs[g_blk->qs[4*ib32 + l] >> 9];
                            const int off = b*QK_K + 32*ib32 + 8*l;
                            for (int j = 0; j < 8; ++j)
                                g_sum += db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f) * input[off + j];
                        }
                    }
                }
            } else if (gate_type == 18) {
                for (int b = 0; b < nblocks_in; b++) {
                    const block_iq3_xxs * g_blk = (const block_iq3_xxs *)(g_row + b * sizeof(block_iq3_xxs));
                    const float gd = (float)g_blk->d;
                    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                        const uint8_t * qs = g_blk->qs + 8*ib32;
                        const uint16_t * gas = (const uint16_t *)(g_blk->qs + QK_K/4) + 2*ib32;
                        const uint32_t aux32 = gas[0] | (gas[1] << 16);
                        const float gdb = gd * (0.5f + (aux32 >> 28)) * 0.5f;
                        for (int l = 0; l < 4; ++l) {
                            const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                            const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                            const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                            const int off = b*QK_K + 32*ib32 + 8*l;
                            for (int j = 0; j < 4; ++j) {
                                g_sum += gdb * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f) * input[off + j];
                                g_sum += gdb * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f) * input[off + 4 + j];
                            }
                        }
                    }
                }
            }

            if (up_type == 17) {
                for (int b = 0; b < nblocks_in; b++) {
                    const block_iq2_xs * u_blk = (const block_iq2_xs *)(u_row + b * sizeof(block_iq2_xs));
                    const float d = (float)u_blk->d;
                    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                        for (int l = 0; l < 4; ++l) {
                            const float db = d * (0.5f + ((u_blk->scales[ib32] >> 4*(l/2)) & 0xf)) * 0.25f;
                            const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (u_blk->qs[4*ib32 + l] & 511));
                            const uint8_t signs = ksigns_iq2xs[u_blk->qs[4*ib32 + l] >> 9];
                            const int off = b*QK_K + 32*ib32 + 8*l;
                            for (int j = 0; j < 8; ++j)
                                u_sum += db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f) * input[off + j];
                        }
                    }
                }
            } else if (up_type == 18) {
                for (int b = 0; b < nblocks_in; b++) {
                    const block_iq3_xxs * u_blk = (const block_iq3_xxs *)(u_row + b * sizeof(block_iq3_xxs));
                    const float ud = (float)u_blk->d;
                    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                        const uint8_t * qs = u_blk->qs + 8*ib32;
                        const uint16_t * gas = (const uint16_t *)(u_blk->qs + QK_K/4) + 2*ib32;
                        const uint32_t aux32 = gas[0] | (gas[1] << 16);
                        const float udb = ud * (0.5f + (aux32 >> 28)) * 0.5f;
                        for (int l = 0; l < 4; ++l) {
                            const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                            const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                            const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                            const int off = b*QK_K + 32*ib32 + 8*l;
                            for (int j = 0; j < 4; ++j) {
                                u_sum += udb * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f) * input[off + j];
                                u_sum += udb * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f) * input[off + 4 + j];
                            }
                        }
                    }
                }
            }

            gu_smem[e * ff_dim + k] = silu_f32(g_sum) * u_sum;
        }
    }
    __syncthreads();

    // Phase 2: down projection + weighted sum
    const char * d_base = (const char *) down_w;
    for (int n = tid; n < n_embd; n += BLOCK_SIZE) {
        float sum = 0.0f;

        for (int e = 0; e < n_expert_used; e++) {
            const int eidx = expert_ids[e];
            const char * d_exp = d_base + (size_t) eidx * down_expert_stride;
            const char * d_row = d_exp + n * down_row_stride;
            float d_sum = 0.0f;

            if (down_type == 17) {
                for (int b = 0; b < nblocks_ff; b++) {
                    const block_iq2_xs * d_blk = (const block_iq2_xs *)(d_row + b * sizeof(block_iq2_xs));
                    const float d = (float)d_blk->d;
                    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                        for (int l = 0; l < 4; ++l) {
                            const float db = d * (0.5f + ((d_blk->scales[ib32] >> 4*(l/2)) & 0xf)) * 0.25f;
                            const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (d_blk->qs[4*ib32 + l] & 511));
                            const uint8_t signs = ksigns_iq2xs[d_blk->qs[4*ib32 + l] >> 9];
                            const int off = b*QK_K + 32*ib32 + 8*l;
                            for (int j = 0; j < 8; ++j)
                                d_sum += db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f) * gu_smem[e * ff_dim + off + j];
                        }
                    }
                }
            } else if (down_type == 18) {
                for (int b = 0; b < nblocks_ff; b++) {
                    const block_iq3_xxs * d_blk = (const block_iq3_xxs *)(d_row + b * sizeof(block_iq3_xxs));
                    const float dd = (float)d_blk->d;
                    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                        const uint8_t * qs = d_blk->qs + 8*ib32;
                        const uint16_t * gas = (const uint16_t *)(d_blk->qs + QK_K/4) + 2*ib32;
                        const uint32_t aux32 = gas[0] | (gas[1] << 16);
                        const float ddb = dd * (0.5f + (aux32 >> 28)) * 0.5f;
                        for (int l = 0; l < 4; ++l) {
                            const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                            const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                            const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                            const int off = b*QK_K + 32*ib32 + 8*l;
                            for (int j = 0; j < 4; ++j) {
                                d_sum += ddb * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f) * gu_smem[e * ff_dim + off + j];
                                d_sum += ddb * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f) * gu_smem[e * ff_dim + off + 4 + j];
                            }
                        }
                    }
                }
            } else if (down_type == 23) {
                for (int b = 0; b < nblocks_ff; b++) {
                    const block_iq4_xs * d_blk = (const block_iq4_xs *)(d_row + b * sizeof(block_iq4_xs));
                    const float dd = (float)d_blk->d;
                    for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
                        const int ls = ((d_blk->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf)
                                     | (((d_blk->scales_h >> 2*ib32) & 3) << 4);
                        const float ddb = dd * (ls - 32);
                        const uint8_t * q4 = d_blk->qs + 16*ib32;
                        for (int l = 0; l < 4; ++l) {
                            const int off = b*QK_K + 32*ib32 + 8*l;
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+0] & 0xf] * gu_smem[e * ff_dim + off + 0];
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+1] & 0xf] * gu_smem[e * ff_dim + off + 1];
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+2] & 0xf] * gu_smem[e * ff_dim + off + 2];
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+3] & 0xf] * gu_smem[e * ff_dim + off + 3];
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+0] >>  4] * gu_smem[e * ff_dim + off + 4];
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+1] >>  4] * gu_smem[e * ff_dim + off + 5];
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+2] >>  4] * gu_smem[e * ff_dim + off + 6];
                            d_sum += ddb * kvalues_iq4nl[q4[4*l+3] >>  4] * gu_smem[e * ff_dim + off + 7];
                        }
                    }
                }
            }
            sum += expert_wts[e] * d_sum;
        }

        output[n] = sum;
    }
}

static __global__ void laguna_moe_combine_kernel(
    const char * __restrict__ experts,
    const char * __restrict__ weights,
    char * __restrict__ output,
    const int n_embd,
    const int n_used,
    const int n_tokens,
    const size_t experts_nb0,
    const size_t experts_nb1,
    const size_t experts_nb2,
    const size_t weights_nb0,
    const size_t weights_nb1,
    const size_t output_nb0,
    const size_t output_nb1,
    const float value_scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_embd * n_tokens;
    if (idx >= total) return;

    const int h = idx % n_embd;
    const int t = idx / n_embd;
    float sum = 0.0f;
    for (int e = 0; e < n_used; ++e) {
        const float v = *(const float *)(experts +
            (size_t)h * experts_nb0 +
            (size_t)e * experts_nb1 +
            (size_t)t * experts_nb2);
        const float w = *(const float *)(weights +
            (size_t)e * weights_nb0 +
            (size_t)t * weights_nb1);
        const float scaled = value_scale == 1.0f
            ? v : __fmul_rn(v, value_scale);
        const float prod = __fmul_rn(scaled, w);
        sum = (e == 0) ? prod : __fadd_rn(sum, prod);
    }
    *(float *)(output +
        (size_t)h * output_nb0 +
        (size_t)t * output_nb1) = sum;
}

static __global__ void ds4_peer_copy_f32_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}

// Align equal owner-local expert IDs across q-token warps.  The high 16 bits
// of every valid output encode the original route slot; invalid entries use
// the sign bit plus the original slot.  The dedicated MoE MMVQ kernel decodes
// this metadata and scatters its result back to the original route layout.
// A single thread is deliberate: this is at most a 4 x 6 assignment problem
// and runs once per owner/layer, outside the weight-streaming kernels.
static __global__ void ds4_align_moe_ids_kernel(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ aligned,
        int n_routes,
        int n_tokens,
        int ids_stride,
        int aligned_stride) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    constexpr int max_routes = 6;
    constexpr int max_tokens = 4;
    if (n_routes > max_routes || n_tokens > max_tokens) {
        for (int t = 0; t < n_tokens; ++t) {
            for (int r = 0; r < n_routes; ++r) {
                const int32_t id = ids[t * ids_stride + r];
                aligned[t * aligned_stride + r] = id >= 0
                    ? (int32_t) (0x5a000000u | ((uint32_t) r << 16) |
                                 ((uint32_t) id & 0xffffu))
                    : (int32_t) (0xda00ffffu | ((uint32_t) r << 16));
            }
        }
        return;
    }

    int route_for_slot[max_tokens][max_routes];
    for (int t = 0; t < max_tokens; ++t) {
        for (int s = 0; s < max_routes; ++s) {
            route_for_slot[t][s] = -1;
        }
    }
    for (int r = 0; r < n_routes; ++r) {
        route_for_slot[0][r] = r;
    }

    for (int t = 1; t < n_tokens; ++t) {
        bool used_slot[max_routes] = {};
        bool assigned_route[max_routes] = {};

        // Prefer assignments that preserve the most previous occurrences of
        // this expert in one slot.  This handles conflicts deterministically
        // when two earlier tokens placed different experts in the same slot.
        for (int pass = 0; pass < n_routes; ++pass) {
            int best_route = -1;
            int best_slot = -1;
            int best_count = 0;
            for (int r = 0; r < n_routes; ++r) {
                if (assigned_route[r]) continue;
                const int32_t id = ids[t * ids_stride + r];
                if (id < 0) continue;
                for (int s = 0; s < n_routes; ++s) {
                    if (used_slot[s]) continue;
                    int count = 0;
                    for (int p = 0; p < t; ++p) {
                        const int prev_route = route_for_slot[p][s];
                        if (prev_route >= 0 &&
                            ids[p * ids_stride + prev_route] == id) {
                            ++count;
                        }
                    }
                    if (count > best_count ||
                        (count == best_count && count > 0 &&
                         (best_route < 0 || r < best_route ||
                          (r == best_route && s < best_slot)))) {
                        best_route = r;
                        best_slot = s;
                        best_count = count;
                    }
                }
            }
            if (best_count == 0) break;
            route_for_slot[t][best_slot] = best_route;
            used_slot[best_slot] = true;
            assigned_route[best_route] = true;
        }

        // Keep unmatched routes in their original slot when possible, then
        // fill the remaining permutation slots in ascending order.
        for (int r = 0; r < n_routes; ++r) {
            if (assigned_route[r]) continue;
            int slot = !used_slot[r] ? r : -1;
            if (slot < 0) {
                for (int s = 0; s < n_routes; ++s) {
                    if (!used_slot[s]) {
                        slot = s;
                        break;
                    }
                }
            }
            route_for_slot[t][slot] = r;
            used_slot[slot] = true;
            assigned_route[r] = true;
        }
    }

    for (int t = 0; t < n_tokens; ++t) {
        for (int s = 0; s < n_routes; ++s) {
            const int original_route = route_for_slot[t][s];
            const int32_t id = ids[t * ids_stride + original_route];
            aligned[t * aligned_stride + s] = id >= 0
                ? (int32_t) (0x5a000000u |
                             ((uint32_t) original_route << 16) |
                             ((uint32_t) id & 0xffffu))
                : (int32_t) (0xda00ffffu |
                             ((uint32_t) original_route << 16));
        }
    }
}

static ggml_tensor make_contiguous_f32_tensor(
        float * data,
        int64_t ne0,
        int64_t ne1,
        int64_t ne2) {
    ggml_tensor tensor = {};
    tensor.type  = GGML_TYPE_F32;
    tensor.data  = data;
    tensor.ne[0] = ne0;
    tensor.ne[1] = ne1;
    tensor.ne[2] = ne2;
    tensor.ne[3] = 1;
    tensor.nb[0] = sizeof(float);
    tensor.nb[1] = tensor.nb[0] * ne0;
    tensor.nb[2] = tensor.nb[1] * ne1;
    tensor.nb[3] = tensor.nb[2] * ne2;
    return tensor;
}

static void ggml_cuda_op_ds4_moe_owner(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    const ggml_tensor * input      = dst->src[0];  // [n_embd, n_tokens]
    const ggml_tensor * gate_up_w  = dst->src[1];  // [n_embd, 2*n_ff, n_expert]
    const ggml_tensor * down_w     = dst->src[2];  // [n_ff, n_embd, n_expert]
    const ggml_tensor * expert_ids = dst->src[3];  // [n_used, n_tokens]
    const ggml_tensor * weights    = dst->src[4];  // [n_used, n_tokens]

    const int n_embd   = (int) input->ne[0];
    const int n_tokens = (int) input->ne[1];
    const int n_used   = (int) expert_ids->ne[0];
    const int n_ff     = ggml_get_op_params_i32(dst, 1);
    const float clamp  = ggml_get_op_params_f32(dst, 2);
    const float down_scale = ggml_get_op_params_f32(dst, 3);

    GGML_ASSERT(gate_up_w->ne[0] == n_embd);
    GGML_ASSERT(gate_up_w->ne[1] == 2 * n_ff);
    GGML_ASSERT(down_w->ne[0] == n_ff);
    GGML_ASSERT(down_w->ne[1] == n_embd);
    GGML_ASSERT(expert_ids->ne[1] == n_tokens);
    GGML_ASSERT(weights->ne[0] == n_used && weights->ne[1] == n_tokens);
    GGML_ASSERT(dst->ne[0] == n_embd && dst->ne[1] == n_tokens);

    // The checkpoint concatenates gate rows followed by up rows inside each
    // expert. Keep the original expert stride while viewing each half.
    ggml_tensor gate_w = *gate_up_w;
    ggml_tensor up_w   = *gate_up_w;
    gate_w.ne[1] = n_ff;
    up_w.ne[1]   = n_ff;
    up_w.data = (char *) gate_up_w->data + (size_t) n_ff * gate_up_w->nb[1];

    // MUL_MAT_ID expects token columns in dimension 2. The input is physically
    // identical; only the descriptor changes from [D,T] to [D,1,T].
    ggml_tensor input_3d = *input;
    input_3d.ne[1] = 1;
    input_3d.ne[2] = n_tokens;
    input_3d.ne[3] = 1;
    input_3d.nb[1] = input->nb[1];
    input_3d.nb[2] = input->nb[1];
    input_3d.nb[3] = input->nb[1] * n_tokens;
    // This descriptor is stack-local. Disable q8 memoization for it.
    input_3d.buffer = nullptr;

    ggml_cuda_pool_alloc<float> gu_alloc(
        ctx.pool(), (size_t) n_ff * n_used * n_tokens);
    ggml_tensor gu = make_contiguous_f32_tensor(
        gu_alloc.ptr, n_ff, n_used, n_tokens);

    ggml_cuda_mm_fusion_args_host gate_up_fusion{};
    gate_up_fusion.gate = &gate_w;
    gate_up_fusion.glu_op = clamp > 1.0e-6f
        ? GGML_GLU_OP_SWIGLU_DS4 : GGML_GLU_OP_SWIGLU;
    gate_up_fusion.glu_param0 = clamp;
    ggml_cuda_mul_mat_vec_q(
        ctx, &up_w, &input_3d, expert_ids, &gu, &gate_up_fusion);

    ggml_cuda_pool_alloc<float> experts_alloc(
        ctx.pool(), (size_t) n_embd * n_used * n_tokens);
    ggml_tensor experts = make_contiguous_f32_tensor(
        experts_alloc.ptr, n_embd, n_used, n_tokens);
    ggml_cuda_mul_mat_vec_q(
        ctx, down_w, &gu, expert_ids, &experts, nullptr);

    const int total = n_embd * n_tokens;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    laguna_moe_combine_kernel<<<grid, block, 0, ctx.stream()>>>(
        (const char *) experts.data,
        (const char *) weights->data,
        (char *) dst->data,
        n_embd, n_used, n_tokens,
        experts.nb[0], experts.nb[1], experts.nb[2],
        weights->nb[0], weights->nb[1],
        dst->nb[0], dst->nb[1],
        down_scale);
}

static void ggml_cuda_op_ds4_moe_owner_split(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    const ggml_tensor * input      = dst->src[0];  // [n_embd, n_tokens]
    const ggml_tensor * gate_w     = dst->src[1];  // [n_embd, n_ff, n_expert]
    const ggml_tensor * up_w       = dst->src[2];  // [n_embd, n_ff, n_expert]
    const ggml_tensor * down_w     = dst->src[3];  // [n_ff, n_embd, n_expert]
    const ggml_tensor * expert_ids = dst->src[4];  // [n_used, n_tokens]
    const ggml_tensor * weights    = dst->src[5];  // [n_used, n_tokens]

    const int n_embd   = (int) input->ne[0];
    const int n_tokens = (int) input->ne[1];
    const int n_used   = (int) expert_ids->ne[0];
    const int n_ff     = ggml_get_op_params_i32(dst, 1);
    const float clamp      = ggml_get_op_params_f32(dst, 2);
    const float gate_scale = ggml_get_op_params_f32(dst, 3);
    const float up_scale   = ggml_get_op_params_f32(dst, 4);
    const float down_scale = ggml_get_op_params_f32(dst, 5);

    GGML_ASSERT(gate_w->ne[0] == n_embd && gate_w->ne[1] == n_ff);
    GGML_ASSERT(up_w->ne[0] == n_embd && up_w->ne[1] == n_ff);
    GGML_ASSERT(down_w->ne[0] == n_ff && down_w->ne[1] == n_embd);
    GGML_ASSERT(expert_ids->ne[1] == n_tokens);
    GGML_ASSERT(weights->ne[0] == n_used && weights->ne[1] == n_tokens);
    GGML_ASSERT(dst->ne[0] == n_embd && dst->ne[1] == n_tokens);

    // MUL_MAT_ID consumes token columns in dimension 2. This descriptor is
    // stack-local, so it deliberately does not participate in q8 memoization.
    ggml_tensor input_3d = *input;
    input_3d.ne[1] = 1;
    input_3d.ne[2] = n_tokens;
    input_3d.ne[3] = 1;
    input_3d.nb[1] = input->nb[1];
    input_3d.nb[2] = input->nb[1];
    input_3d.nb[3] = input->nb[1] * n_tokens;
    input_3d.buffer = nullptr;

    ggml_cuda_pool_alloc<float> gu_alloc(
        ctx.pool(), (size_t) n_ff * n_used * n_tokens);
    ggml_tensor gu = make_contiguous_f32_tensor(
        gu_alloc.ptr, n_ff, n_used, n_tokens);

    ggml_cuda_mm_fusion_args_host gate_up_fusion{};
    gate_up_fusion.gate = gate_w;
    gate_up_fusion.glu_op = clamp > 1.0e-6f
        ? GGML_GLU_OP_SWIGLU_DS4 : GGML_GLU_OP_SWIGLU;
    gate_up_fusion.glu_param0 = clamp;
    gate_up_fusion.gate_value_scale = gate_scale;
    gate_up_fusion.x_value_scale = up_scale;
    ggml_cuda_mul_mat_vec_q(
        ctx, up_w, &input_3d, expert_ids, &gu, &gate_up_fusion);

    ggml_cuda_pool_alloc<float> experts_alloc(
        ctx.pool(), (size_t) n_embd * n_used * n_tokens);
    ggml_tensor experts = make_contiguous_f32_tensor(
        experts_alloc.ptr, n_embd, n_used, n_tokens);
    ggml_cuda_mul_mat_vec_q(
        ctx, down_w, &gu, expert_ids, &experts, nullptr);

    const int total = n_embd * n_tokens;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    laguna_moe_combine_kernel<<<grid, block, 0, ctx.stream()>>>(
        (const char *) experts.data,
        (const char *) weights->data,
        (char *) dst->data,
        n_embd, n_used, n_tokens,
        experts.nb[0], experts.nb[1], experts.nb[2],
        weights->nb[0], weights->nb[1],
        dst->nb[0], dst->nb[1],
        down_scale);
}

void ggml_cuda_op_moe_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int mode = ggml_get_op_params_i32(dst, 0);
    if (mode == -5) {
        const ggml_tensor * ids = dst->src[0];
        GGML_ASSERT(ids && ids->type == GGML_TYPE_I32);
        GGML_ASSERT(dst->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(ids) && ggml_is_contiguous(dst));
        ds4_align_moe_ids_kernel<<<1, 1, 0, ctx.stream()>>>(
            (const int32_t *) ids->data,
            (int32_t *) dst->data,
            (int) ids->ne[0],
            (int) ids->ne[1],
            (int) (ids->nb[1] / sizeof(int32_t)),
            (int) (dst->nb[1] / sizeof(int32_t)));
        return;
    }
    if (mode == -3) {
        GGML_ASSERT(ggml_is_contiguous(dst));

        void * event_context = nullptr;
        memcpy(&event_context, &dst->op_params[2], sizeof(event_context));
        GGML_ASSERT(event_context);

        const int transfer_mode = ggml_get_op_params_i32(dst, 6);
        if (transfer_mode == 3) {
            // Whole-step heterogeneous graph path: the Strix producer wrote
            // this activation into coherent mapped staging before publishing
            // the scheduler event.  The scheduler's late event wait precedes
            // this node, so only a local R9700 destination write is needed.
            void * staging_data = nullptr;
            memcpy(&staging_data, &dst->op_params[8], sizeof(staging_data));
            GGML_ASSERT(staging_data);
            const int64_t n = ggml_nelements(dst);
            const int block = 256;
            const int grid = (int) ((n + block - 1) / block);
            ds4_peer_copy_f32_kernel<<<grid, block, 0, ctx.stream()>>>(
                (const float *) staging_data, (float *) dst->data, n);
            return;
        }

        if (transfer_mode != 0) {
            // Diagnostic path: the scheduler prefilled dst with a synchronous
            // peer copy. Leave the graph node as a dependency-only no-op.
            return;
        }

        void * source_data = nullptr;
        memcpy(&source_data, &dst->op_params[4], sizeof(source_data));
        GGML_ASSERT(source_data);

        if (ggml_get_op_params_i32(dst, 7) == 0) {
            // Same-split control: capture the wait at its exact position in
            // the main GPU graph. On the current ROCm runtime this preserves
            // correctness but realizes almost no cross-device overlap.
            CUDA_CHECK(cudaStreamWaitEvent(
                ctx.stream(), (cudaEvent_t) event_context, 0));
        }

        // Direct peer reads are required here. On this R9700 + gfx1151 pair,
        // embedding hipMemcpyPeerAsync in the captured consumer graph returned
        // stale/corrupt values even with a host-synchronized producer event.
        // The small contiguous activation is only 16-64 KiB, and the peer-read
        // kernel is exact with both eager execution and HIP graph replay.
        const int64_t n = ggml_nelements(dst);
        const int block = 256;
        const int grid = (int) ((n + block - 1) / block);
        ds4_peer_copy_f32_kernel<<<grid, block, 0, ctx.stream()>>>(
            (const float *) source_data, (float *) dst->data, n);
        return;
    }
    if (mode == -2) {
        ggml_cuda_op_ds4_moe_owner(ctx, dst);
        return;
    }
    if (mode == -4) {
        ggml_cuda_op_ds4_moe_owner_split(ctx, dst);
        return;
    }
    if (mode == -1) {
        const ggml_tensor * experts = dst->src[0];  // [n_embd, n_used, n_tokens]
        const ggml_tensor * weights = dst->src[1];  // [n_used, n_tokens]
        const int n_embd   = (int) experts->ne[0];
        const int n_used   = (int) experts->ne[1];
        const int n_tokens = (int) experts->ne[2];
        const int total = n_embd * n_tokens;

        const int block = 256;
        const int grid = (total + block - 1) / block;
        laguna_moe_combine_kernel<<<grid, block, 0, ctx.stream()>>>(
            (const char *) experts->data,
            (const char *) weights->data,
            (char *) dst->data,
            n_embd, n_used, n_tokens,
            experts->nb[0], experts->nb[1], experts->nb[2],
            weights->nb[0], weights->nb[1],
            dst->nb[0], dst->nb[1],
            1.0f);
        return;
    }

    const ggml_tensor * input        = dst->src[0];
    const ggml_tensor * gate_w       = dst->src[1];
    const ggml_tensor * up_w         = dst->src[2];
    const ggml_tensor * down_w       = dst->src[3];
    const ggml_tensor * expert_ids   = dst->src[4];
    const ggml_tensor * expert_wts   = dst->src[5];

    const int n_embd         = (int) gate_w->ne[0];
    const int ff_dim         = (int) gate_w->ne[1];
    const int n_expert_used  = (int) expert_ids->ne[0];

    const size_t gate_row_stride = gate_w->nb[1];
    const size_t up_row_stride   = up_w->nb[1];
    const size_t down_row_stride = down_w->nb[1];
    const size_t gate_expert_stride = gate_w->nb[2];
    const size_t up_expert_stride   = up_w->nb[2];
    const size_t down_expert_stride = down_w->nb[2];

    const int BLOCK_SIZE = 256;
    const int smem_size = n_expert_used * ff_dim * sizeof(float);

    moe_fused_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE, smem_size, ctx.stream()>>>(
        (const float *) input->data,
        gate_w->data,
        up_w->data,
        down_w->data,
        (const int32_t *) expert_ids->data,
        (const float *) expert_wts->data,
        dst->src[6] ? dst->src[6]->data : nullptr,
        dst->src[7] ? dst->src[7]->data : nullptr,
        dst->src[8] ? dst->src[8]->data : nullptr,
        dst->src[9] ? dst->src[9]->data : nullptr,
        (float *) dst->data,
        n_embd,
        ff_dim,
        n_expert_used,
        (int) gate_w->ne[2],
        (int) gate_w->type,
        (int) up_w->type,
        (int) down_w->type,
        gate_row_stride,
        up_row_stride,
        down_row_stride,
        gate_expert_stride,
        up_expert_stride,
        down_expert_stride,
        gate_w->nb[0],
        up_w->nb[0],
        down_w->nb[0]);
}
