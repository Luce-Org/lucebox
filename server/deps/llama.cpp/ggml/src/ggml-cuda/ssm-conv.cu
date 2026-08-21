#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * __restrict__ dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

// dflash27b_ggml: tree-mode ssm_conv kernel. For each new-token t, walks up
// the parent chain K-1 times via parent_ids[] to find the (K-1) ancestor slots
// in the conv input, then convolves with the kernel weights. Virtual-slot
// encoding: a non-negative parent index `p` maps to sx slot (K-1 + p). A
// parent index of -1 means "before the block" — i.e., the old conv state.
// Each successive walk beyond -1 decrements by 1, so virtual slot -k maps to
// sx slot (K-1 - k), which indexes into the old state region [0, K-1). This
// matches SGLang's causal_conv1d_triton HAS_EAGLE_TREE_CUSTOM_ATTN_MASK path.
template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_tree_f32(
        const float * __restrict__ src0,        // sx: [K-1+n_t, d_inner, n_s]
        const float * __restrict__ src1,        // c:  [K, d_inner]
        const int * __restrict__   parent_ids,  // [n_t, n_s]
        const int src0_nb0, const int src0_nb1, const int src0_nb2,
        const int src1_nb1,
        float * __restrict__ dst,               // [d_inner, n_t, n_s]
        const int dst_nb0, const int dst_nb1, const int dst_nb2,
        const int64_t n_t) {
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;  // sequence
    const int bidy = blockIdx.y;  // d_inner / split_d_inner

    const float * x_block = (const float *) ((const char *) src0
        + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1
        + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst
        + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    // Load kernel weights into registers.
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    const int * parent_ids_seq = parent_ids + bidx * n_t;

    for (int64_t i = 0; i < n_t; i++) {
        // Walk the parent chain K-1 times to fill the conv window.
        // ancestor_virtual[k] gives the "virtual slot" for kernel position k,
        // where the most recent slot is at k=K-1 (= token i itself) and older
        // slots are at k=K-2, K-3, ..., 0.
        //
        // ancestor_virtual[K-1] = i
        // ancestor_virtual[K-2] = parent_of(i) (or i-1 decay for negative)
        // ancestor_virtual[k  ] = parent_of(ancestor_virtual[k+1])
        int ancestors[d_conv];
        ancestors[d_conv - 1] = (int)i;
#pragma unroll
        for (int k = (int)d_conv - 2; k >= 0; k--) {
            int prev = ancestors[k + 1];
            int next;
            if (prev >= 0) {
                next = parent_ids_seq[prev];  // -1 if parent is before block
            } else {
                next = prev - 1;  // keep decaying through old state slots
            }
            ancestors[k] = next;
        }

        float sumf = 0.0f;
#pragma unroll
        for (size_t k = 0; k < d_conv; k++) {
            // Map virtual slot → sx slot: sx_slot = (K-1) + ancestors[k].
            const int sx_slot = (int)(d_conv - 1) + ancestors[k];
            const float x_val = x_block[tid * stride_x + sx_slot];
            sumf += x_val * w[k];
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_tree_f32_cuda(const float * src0, const float * src1, const int * parent_ids,
                                   const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                   const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                                   const int dst_nb2, const int64_t nc, const int64_t nr,
                                   const int64_t n_t, const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        ssm_conv_tree_f32<apply_silu, threads, kNC><<<blocks, threads, 0, stream>>>(
            src0, src1, parent_ids, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
            dst, dst_nb0, dst_nb1, dst_nb2, n_t);
    };

    switch (nc) {
        case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
        case 4: launch_kernel(std::integral_constant<int, 4>{}); break;
        case 5: launch_kernel(std::integral_constant<int, 5>{}); break;
        case 9: launch_kernel(std::integral_constant<int, 9>{}); break;
        default: GGML_ABORT("Tree ssm_conv only supports kernel sizes 3, 4, 5, 9.");
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            ssm_conv_f32<apply_silu, threads, kNC><<<blocks, threads, 0, stream>>>(src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
                                                                       dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
        case 4: launch_kernel(std::integral_constant<int, 4>{}); break;
        case 5: launch_kernel(std::integral_constant<int, 5>{}); break;
        case 9: launch_kernel(std::integral_constant<int, 9>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9 right now.");
    }
}

template <int d_conv>
static __global__ void ssm_conv_specla_hld_f32(
        const float * __restrict__ x,          // [d_inner, n_t]
        const float * __restrict__ weight,     // [d_conv, d_inner]
        float * __restrict__ state,            // [d_conv-1, d_inner]
        const int * __restrict__ meta,
        const int64_t * __restrict__ factor_ptrs,
        float * __restrict__ packed,
        int d_inner,
        int n_t,
        int n_layers,
        int layer,
        int pending_bank,
        int n_chains,
        int wave) {
    const int wave_chain = blockIdx.x;
    const int channel = blockIdx.y * blockDim.x + threadIdx.x;
    if (channel >= d_inner) return;

    const int order_off    = meta[6];
    const int offsets_off  = meta[7];
    const int parent_off   = meta[8];
    const int boundary_off = meta[9];
    const int wave_off     = meta[10];
    int chain = 0;
    while (chain < n_chains && meta[wave_off + chain] < wave) ++chain;
    chain += wave_chain;
    if (chain >= n_chains || meta[wave_off + chain] != wave) return;

    float window[d_conv - 1];
    const float * pending_x = (const float *)(uintptr_t)
        factor_ptrs[pending_bank*4 + 3];
    float * current_x = (float *)(uintptr_t)
        factor_ptrs[(1 - pending_bank)*4 + 3];
    const int parent_boundary = meta[parent_off + chain];
    if (parent_boundary < 0) {
#pragma unroll
        for (int j = 0; j < d_conv - 1; ++j) {
            window[j] = state[(size_t)channel * (d_conv - 1) + j];
        }
        // Delayed commit: compact accepted inputs from the preceding verify
        // are consumed before the current root chain. Only this committed
        // window is written to durable state.
        const int pending_count = meta[5];
        for (int t = 0; t < pending_count; ++t) {
#pragma unroll
            for (int j = 0; j < d_conv - 2; ++j) window[j] = window[j + 1];
            window[d_conv - 2] = pending_x[
                (size_t)channel + (size_t)d_inner*(layer + (size_t)n_layers*t)];
        }
#pragma unroll
        for (int j = 0; j < d_conv - 1; ++j) {
            state[(size_t)channel * (d_conv - 1) + j] = window[j];
        }
    } else {
        const size_t boundary_base =
            ((size_t)n_t + (size_t)parent_boundary*(d_conv - 1))*d_inner;
#pragma unroll
        for (int j = 0; j < d_conv - 1; ++j) {
            window[j] = packed[boundary_base + (size_t)j*d_inner + channel];
        }
    }

    const int begin = meta[offsets_off + chain];
    const int end   = meta[offsets_off + chain + 1];
    for (int p = begin; p < end; ++p) {
        const int node = meta[order_off + p];
        const float x_val = x[(size_t)node*d_inner + channel];
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv - 1; ++j) {
            sum += window[j] * weight[(size_t)channel*d_conv + j];
        }
        sum += x_val * weight[(size_t)channel*d_conv + d_conv - 1];
        packed[(size_t)node*d_inner + channel] =
            ggml_cuda_op_silu_single(sum);
        current_x[(size_t)channel +
            (size_t)d_inner*(layer + (size_t)n_layers*node)] = x_val;

#pragma unroll
        for (int j = 0; j < d_conv - 2; ++j) window[j] = window[j + 1];
        window[d_conv - 2] = x_val;
        const int boundary = meta[boundary_off + node];
        if (boundary >= 0) {
            const size_t boundary_base =
                ((size_t)n_t + (size_t)boundary*(d_conv - 1))*d_inner;
#pragma unroll
            for (int j = 0; j < d_conv - 1; ++j) {
                packed[boundary_base + (size_t)j*d_inner + channel] = window[j];
            }
        }
    }
}

static void ssm_conv_specla_hld_cuda(ggml_backend_cuda_context & ctx,
                                      ggml_tensor * dst) {
    ggml_tensor * x         = dst->src[0];
    ggml_tensor * weight    = dst->src[1];
    ggml_tensor * state     = dst->src[2];
    ggml_tensor * hld       = dst->src[3];
    ggml_tensor * factor_ptrs = dst->src[4];
    const int d_conv  = (int)weight->ne[0];
    const int d_inner = (int)x->ne[0];
    const int n_t     = (int)x->ne[1];
    const int n_chains = ggml_get_op_params_i32(dst, 2);
    const int n_waves  = ggml_get_op_params_i32(dst, 3);
    const int n_layers = ggml_get_op_params_i32(dst, 4);
    const int layer = ggml_get_op_params_i32(dst, 5);
    const int pending_bank = ggml_get_op_params_i32(dst, 6);
    const int max_parallel_chains = ggml_get_op_params_i32(dst, 7);
    GGML_ASSERT(x->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32);
    GGML_ASSERT(state->type == GGML_TYPE_F32 && factor_ptrs->type == GGML_TYPE_I64);
    GGML_ASSERT(hld->type == GGML_TYPE_I32 && n_chains > 0 && n_waves > 0);

    const dim3 block(128);
    const dim3 grid((unsigned)max_parallel_chains,
                    (unsigned)((d_inner + 127)/128), 1);
    auto launch = [&](auto DC, int wave) {
        constexpr int kDC = decltype(DC)::value;
        ssm_conv_specla_hld_f32<kDC><<<grid, block, 0, ctx.stream()>>>(
            (const float *)x->data, (const float *)weight->data,
            (float *)state->data, (const int *)hld->data,
            (const int64_t *)factor_ptrs->data, (float *)dst->data,
            d_inner, n_t, n_layers, layer, pending_bank, n_chains, wave);
    };
    for (int wave = 0; wave < n_waves; ++wave) {
        switch (d_conv) {
            case 3: launch(std::integral_constant<int, 3>{}, wave); break;
            case 4: launch(std::integral_constant<int, 4>{}, wave); break;
            case 5: launch(std::integral_constant<int, 5>{}, wave); break;
            case 9: launch(std::integral_constant<int, 9>{}, wave); break;
            default: GGML_ABORT("SpecLA ssm_conv supports kernel sizes 3, 4, 5, 9.");
        }
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst) {
    if (ggml_get_op_params_i32(dst, 0) == 1) {
        GGML_ASSERT(silu_dst == nullptr);
        ssm_conv_specla_hld_cuda(ctx, dst);
        return;
    }
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    // dflash27b_ggml: optional src[2] = parent_ids (i32) enables tree mode
    const struct ggml_tensor * parent_ids = dst->src[2];
    const bool fuse_silu = silu_dst != nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);

    if (parent_ids != nullptr) {
        GGML_ASSERT(parent_ids->type == GGML_TYPE_I32);
        const int * parent_ids_d = (const int *) parent_ids->data;
        if (fuse_silu) {
            ssm_conv_tree_f32_cuda<true>(src0_d, src1_d, parent_ids_d,
                src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1],
                dst_d, out->nb[0], out->nb[1], out->nb[2],
                nc, nr, n_t, n_s, stream);
        } else {
            ssm_conv_tree_f32_cuda<false>(src0_d, src1_d, parent_ids_d,
                src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1],
                dst_d, out->nb[0], out->nb[1], out->nb[2],
                nc, nr, n_t, n_s, stream);
        }
        return;
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}
