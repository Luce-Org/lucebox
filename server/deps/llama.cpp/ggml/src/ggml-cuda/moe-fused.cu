#include "moe-fused.cuh"
#include "ggml-cuda/vecdotq.cuh"
#include "ggml-cuda/dequantize.cuh"

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
    const size_t output_nb1) {
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
        const float prod = __fmul_rn(v, w);
        sum = (e == 0) ? prod : __fadd_rn(sum, prod);
    }
    *(float *)(output +
        (size_t)h * output_nb0 +
        (size_t)t * output_nb1) = sum;
}

void ggml_cuda_op_moe_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 0) == -1) {
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
            dst->nb[0], dst->nb[1]);
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
