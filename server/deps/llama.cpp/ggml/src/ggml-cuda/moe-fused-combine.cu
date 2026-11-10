#include "common.cuh"
#include "moe-fused-combine.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Fused post-MMID route reduction and shared-expert add. The down projection
// remains a separate operation and materializes down_e before this kernel.
// Computes dst[t, d] = (shared_out ? shared_out[t, d] : 0) + sum_{e=0}^{n_used-1} (down_e[t, e, d] * weights[t, e])
// Uses vectorized float4 128-bit memory transactions and sequential non-FMA FP32 accumulation
// to preserve the legacy route-order reduction.

static __global__ void moe_fused_combine_shared_kernel_f32(
        const float4 * __restrict__ down_e,
        const float  * __restrict__ weights,
        const float4 * __restrict__ shared_out,
        const float  * __restrict__ shared_logit,
        float4       * __restrict__ output,
        const int n_embd_vec4,
        const int n_used,
        const int n_tokens,
        const size_t down_nb1,
        const size_t down_nb2,
        const size_t weights_nb1,
        const size_t shared_nb1,
        const size_t logit_nb1,
        const size_t out_nb1) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_embd_vec4 * n_tokens;
    if (idx >= total) return;

    const int h4 = idx % n_embd_vec4;
    const int t  = idx / n_embd_vec4;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    for (int e = 0; e < n_used; ++e) {
        const float  w = weights[e + t * weights_nb1];
        // Expert-major owners encode routes assigned to the other device as
        // weight zero and ID -1. Do not read those MMID output lanes: a masked
        // lane is semantically zero, and stale NaN/Inf multiplied by zero would
        // otherwise poison the reduction.
        if (w == 0.0f) {
            continue;
        }
        const float4 v = down_e[h4 + e * down_nb1 + t * down_nb2];

        const float p0 = __fmul_rn(v.x, w);
        const float p1 = __fmul_rn(v.y, w);
        const float p2 = __fmul_rn(v.z, w);
        const float p3 = __fmul_rn(v.w, w);

        // Start from +0 and add every active route, matching the legacy
        // sum_rows reduction even for a first product of -0.0f.
        sum0 = __fadd_rn(sum0, p0);
        sum1 = __fadd_rn(sum1, p1);
        sum2 = __fadd_rn(sum2, p2);
        sum3 = __fadd_rn(sum3, p3);
    }

    if (shared_out != nullptr) {
        float4 sh = shared_out[h4 + t * shared_nb1];
        if (shared_logit != nullptr) {
            // The shared expert's sigmoid gate as unary.cu's op_sigmoid, times the output as binbcast.cu's op_mul.
            const float g = 1.0f / (1.0f + expf(-shared_logit[t * logit_nb1]));
            sh = make_float4(__fmul_rn(sh.x, g), __fmul_rn(sh.y, g), __fmul_rn(sh.z, g), __fmul_rn(sh.w, g));
        }
        sum0 = __fadd_rn(sh.x, sum0);
        sum1 = __fadd_rn(sh.y, sum1);
        sum2 = __fadd_rn(sh.z, sum2);
        sum3 = __fadd_rn(sh.w, sum3);
    }

    output[h4 + t * out_nb1] = make_float4(sum0, sum1, sum2, sum3);
}

void ggml_cuda_op_ds4_moe_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                  const ggml_tensor * shared_ungated, const ggml_tensor * shared_logit) {
    const ggml_tensor * down_e     = dst->src[0];
    const ggml_tensor * weights    = dst->src[1];
    const ggml_tensor * shared_out = shared_logit ? shared_ungated : dst->src[2];
    if (shared_logit) {
        GGML_ASSERT(shared_out != nullptr && shared_logit->type == GGML_TYPE_F32 && shared_logit->ne[0] == 1 &&
                    shared_logit->ne[1] == down_e->ne[2]);
    }

    GGML_ASSERT(down_e->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int n_embd   = (int) down_e->ne[0];
    const int n_used   = (int) down_e->ne[1];
    const int n_tokens = (int) down_e->ne[2];

    static const bool trace_enabled = []() {
        const char * raw = std::getenv("LUCE_MOE_FUSED_COMBINE_TRACE");
        return raw && *raw && std::strcmp(raw, "0") != 0;
    }();
    if (trace_enabled) {
        static std::atomic<unsigned long long> launch_count{0};
        int device = -1;
        CUDA_CHECK(cudaGetDevice(&device));
        const unsigned long long launch =
            launch_count.fetch_add(1, std::memory_order_relaxed) + 1;
        std::fprintf(stderr,
            "[moe-fused-combine] launch=%llu device=%d tokens=%d routes=%d shared=%d\n",
            launch, device, n_tokens, n_used, shared_out != nullptr ? 1 : 0);
    }

    GGML_ASSERT(n_embd % 4 == 0 && "n_embd must be a multiple of 4 for float4 vectorization");
    GGML_ASSERT(reinterpret_cast<uintptr_t>(down_e->data) % 16 == 0 && "down_e->data must be 16-byte aligned");
    GGML_ASSERT(reinterpret_cast<uintptr_t>(dst->data) % 16 == 0 && "dst->data must be 16-byte aligned");
    GGML_ASSERT(down_e->nb[0] == sizeof(float) && "down_e must be contiguous in dimension 0");
    GGML_ASSERT(down_e->nb[1] % sizeof(float4) == 0 && "down_e->nb[1] must be divisible by sizeof(float4)");
    GGML_ASSERT(down_e->nb[2] % sizeof(float4) == 0 && "down_e->nb[2] must be divisible by sizeof(float4)");
    GGML_ASSERT(dst->nb[0] == sizeof(float) && "dst must be contiguous in dimension 0");
    GGML_ASSERT(dst->nb[1] % sizeof(float4) == 0 && "dst->nb[1] must be divisible by sizeof(float4)");
    if (shared_out != nullptr) {
        GGML_ASSERT(reinterpret_cast<uintptr_t>(shared_out->data) % 16 == 0 && "shared_out->data must be 16-byte aligned");
        GGML_ASSERT(shared_out->nb[0] == sizeof(float) && "shared_out must be contiguous in dimension 0");
        GGML_ASSERT(shared_out->nb[1] % sizeof(float4) == 0 && "shared_out->nb[1] must be divisible by sizeof(float4)");
    }

    const int n_embd_vec4 = n_embd / 4;
    const int total_threads = n_embd_vec4 * n_tokens;

    const int block_size = 256;
    const int grid_size = (total_threads + block_size - 1) / block_size;

    const size_t down_nb1 = down_e->nb[1] / sizeof(float4);
    const size_t down_nb2 = down_e->nb[2] / sizeof(float4);
    const size_t weights_nb1 = weights->nb[1] / sizeof(float);
    const size_t shared_nb1 = shared_out ? (shared_out->nb[1] / sizeof(float4)) : 0;
    const size_t logit_nb1 = shared_logit ? (shared_logit->nb[1] / sizeof(float)) : 0;
    const size_t out_nb1 = dst->nb[1] / sizeof(float4);

    cudaStream_t stream = ctx.stream();

    moe_fused_combine_shared_kernel_f32<<<grid_size, block_size, 0, stream>>>(
        (const float4 *) down_e->data,
        (const float *)  weights->data,
        shared_out ? (const float4 *) shared_out->data : nullptr,
        shared_logit ? (const float *) shared_logit->data : nullptr,
        (float4 *) dst->data,
        n_embd_vec4,
        n_used,
        n_tokens,
        down_nb1,
        down_nb2,
        weights_nb1,
        shared_nb1,
        logit_nb1,
        out_nb1
    );
    CUDA_CHECK(cudaGetLastError());
}

// The renormalized weights of each token's top-k routes, [k, T] from the router probabilities [n_expert, T] and the
// picks [k, T]: GET_ROWS, SUM_ROWS, CLAMP and DIV in one launch with their arithmetic. One thread per token; the row
// sum follows reduce_rows_f32 on a k-wide row (lane l holds element l, then block_reduce's WARP_SIZE-lane xor
// butterfly; the block's second stage adds zeros only), so k is at most WARP_SIZE.
//
// The graph allocator may give the DIV output the memory of the probabilities or the picks (in the unfused graph
// GET_ROWS is their last reader, and it runs before the DIV is written). one_block: every token in one block, each
// thread reading its picks before a block barrier and writing after it, so an output over the inputs is safe.
// Otherwise (more tokens than one block holds) the caller has checked that the output overlaps neither input.
template <bool one_block>
static __global__ void __launch_bounds__(one_block ? MOE_ROUTE_NORM_MAX_TOKENS : 64) moe_route_weights_norm_kernel(
        const char * __restrict__ probs, const size_t probs_nb1, const size_t probs_nb2,
        const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
        const int k, const int n_tokens, const float clamp_min, const float clamp_max, float * __restrict__ dst) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = t < n_tokens;
    if (!one_block && !active) return;
    float w[WARP_SIZE];
    float v[WARP_SIZE];
#pragma unroll
    for (int l = 0; l < WARP_SIZE; ++l) {
        w[l] = 0.0f;
        if (active && l < k) {
            const int32_t e = *(const int32_t *) (ids + (size_t) l * ids_nb0 + (size_t) t * ids_nb1);
            w[l] = *(const float *) (probs + (size_t) e * probs_nb1 + (size_t) t * probs_nb2);
        }
        v[l] = w[l];
    }
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        float n[WARP_SIZE];
#pragma unroll
        for (int l = 0; l < WARP_SIZE; ++l) n[l] = v[l] + v[l ^ off];
#pragma unroll
        for (int l = 0; l < WARP_SIZE; ++l) v[l] = n[l];
    }
    const float sum = fminf(fmaxf(v[0], clamp_min), clamp_max);   // clamp.cu's op_clamp
    if constexpr (one_block) __syncthreads();                       // every read before any write
    if (!active) return;
    for (int i = 0; i < k; ++i) {
        dst[(size_t) t * k + i] = w[i] / sum;                       // binbcast.cu's op_div
    }
}

static bool moe_route_bytes_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

bool ggml_cuda_moe_route_weights_norm_ok(const ggml_tensor * probs3, const ggml_tensor * ids, const ggml_tensor * dst) {
    return ids->ne[1] <= MOE_ROUTE_NORM_MAX_TOKENS ||
           (!moe_route_bytes_overlap(dst, probs3) && !moe_route_bytes_overlap(dst, ids));
}

void ggml_cuda_moe_route_weights_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * probs3,
                                      const ggml_tensor * ids, float clamp_min, float clamp_max, ggml_tensor * dst) {
    const int k = (int) ids->ne[0];
    const int n_tokens = (int) ids->ne[1];
    GGML_ASSERT(k <= WARP_SIZE && probs3->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_nelements(dst) == (int64_t) k * n_tokens);
    GGML_ASSERT(ggml_cuda_moe_route_weights_norm_ok(probs3, ids, dst));
    if (n_tokens <= MOE_ROUTE_NORM_MAX_TOKENS) {
        const int block = (n_tokens + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
        moe_route_weights_norm_kernel<true><<<1, block, 0, ctx.stream()>>>(
            (const char *) probs3->data, probs3->nb[1], probs3->nb[2], (const char *) ids->data, ids->nb[0], ids->nb[1],
            k, n_tokens, clamp_min, clamp_max, (float *) dst->data);
    } else {
        const int block = 64;
        moe_route_weights_norm_kernel<false><<<(n_tokens + block - 1) / block, block, 0, ctx.stream()>>>(
            (const char *) probs3->data, probs3->nb[1], probs3->nb[2], (const char *) ids->data, ids->nb[0], ids->nb[1],
            k, n_tokens, clamp_min, clamp_max, (float *) dst->data);
    }
    CUDA_CHECK(cudaGetLastError());
}

#if defined(GGML_USE_HIP)
#    define MOE_ROUTER_SHFL_XOR(v, mask) __shfl_xor((v), (mask))
#else
#    define MOE_ROUTER_SHFL_XOR(v, mask) __shfl_xor_sync(0xffffffffu, (v), (mask))
#endif

template <typename T>
static __device__ __forceinline__ void moe_router_swap(T & a, T & b) {
    const T t = a;
    a = b;
    b = t;
}

// One block of NE threads per token row (NE = n_expert): thread c owns column c through the softmax and the sort.
template <int NE>
static __global__ void moe_router_topk_kernel(const ggml_cuda_moe_router_topk_args a) {
    const int col = (int) threadIdx.x;
    const int t   = (int) blockIdx.x;

    __shared__ float buf_iw[WARP_SIZE];
    __shared__ float vals[NE];
    __shared__ int   idx_row[NE];
    __shared__ float key_row[NE];

    // soft_max_f32<true, NE, NE>: no mask, no sinks, one column per thread, values cached in shared memory.
    const float val = a.logits[(int64_t) t * a.logits_row + col] * a.scale + 0.0f;
    vals[col] = val;
    float max_val = -INFINITY;
    max_val = max(max_val, val);
    max_val = block_reduce<block_reduce_method::MAX, NE>(max_val, buf_iw);
    __syncthreads();   // buf_iw is reused by the sum below
    float tmp = 0.0f;
    {
        const float e = expf(vals[col] - max_val);
        tmp += e;
        vals[col] = e;
    }
    tmp = block_reduce<block_reduce_method::SUM, NE>(tmp, buf_iw);
    const float inv_sum = 1.0f / tmp;
    const float prob = vals[col] * inv_sum;

    // k_argsort_f32_i32<GGML_SORT_ORDER_DESC> with ncols = ncols_pad = NE: the same bitonic network and swaps.
    idx_row[col] = col;
    key_row[col] = prob;
    __syncthreads();
    for (int k = 2; k <= NE; k *= 2) {
        int j = k / 2;
        for (; j >= warpSize; j /= 2) {
            const int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (idx_row[col] >= NE || (idx_row[ixj] < NE && key_row[col] < key_row[ixj])) {
                        moe_router_swap(idx_row[col], idx_row[ixj]);
                        moe_router_swap(key_row[col], key_row[ixj]);
                    }
                } else {
                    if (idx_row[ixj] >= NE || (idx_row[col] < NE && key_row[col] > key_row[ixj])) {
                        moe_router_swap(idx_row[col], idx_row[ixj]);
                        moe_router_swap(key_row[col], key_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }
        if (j > 0) {
            int   my_idx = idx_row[col];
            float my_val = key_row[col];
            for (; j > 0; j /= 2) {
                const float p_val = MOE_ROUTER_SHFL_XOR(my_val, j);
                const int   p_idx = MOE_ROUTER_SHFL_XOR(my_idx, j);
                const bool  low      = (col & j) == 0;
                const int   low_idx  = low ? my_idx : p_idx;
                const int   high_idx = low ? p_idx : my_idx;
                const float low_val  = low ? my_val : p_val;
                const float high_val = low ? p_val : my_val;
                bool swap;
                if ((col & k) == 0) {
                    swap = low_idx >= NE || (high_idx < NE && low_val < high_val);
                } else {
                    swap = high_idx >= NE || (low_idx < NE && low_val > high_val);
                }
                if (swap) {
                    my_idx = p_idx;
                    my_val = p_val;
                }
            }
            idx_row[col] = my_idx;
            key_row[col] = my_val;
            __syncthreads();
        }
    }

    // The top k: the sorted rows' first k columns, the picks, their table entries (GET_ROWS of a [1, NE] I32 table).
    if (col < a.k) {
        const int e = idx_row[col];
        if (a.sorted) a.sorted[(int64_t) t * a.sorted_row + col] = e;
        if (a.picks)  a.picks[(int64_t) t * a.k + col] = e;
        if (a.out0)   a.out0[(int64_t) t * a.k + col] = a.lut0[e];
        if (a.out1)   a.out1[(int64_t) t * a.k + col] = a.lut1[e];
    }
    // moe_route_weights_norm_kernel: the gathered probabilities (each pick's key is its probability, moved with it
    // through the sort), the WARP_SIZE-lane butterfly row sum, CLAMP and DIV.
    if (a.weights && col == 0) {
        float w[WARP_SIZE];
        float v[WARP_SIZE];
#pragma unroll
        for (int l = 0; l < WARP_SIZE; ++l) {
            w[l] = l < a.k ? key_row[l] : 0.0f;
            v[l] = w[l];
        }
#pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            float n[WARP_SIZE];
#pragma unroll
            for (int l = 0; l < WARP_SIZE; ++l) n[l] = v[l] + v[l ^ off];
#pragma unroll
            for (int l = 0; l < WARP_SIZE; ++l) v[l] = n[l];
        }
        const float sum = fminf(fmaxf(v[0], a.clamp_min), a.clamp_max);
        for (int i = 0; i < a.k; ++i) {
            a.weights[(int64_t) t * a.k + i] = w[i] / sum;
        }
    }
#if defined(GGML_USE_HIP)
    if (a.post && t == 0 && col == 0) {
        __hip_atomic_fetch_add(a.post, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
#endif
}

uint32_t * ggml_cuda_wake_flags() {
    static uint32_t * flags = [] {
        uint32_t * p = nullptr;
#if defined(GGML_USE_HIP)
        CUDA_CHECK(hipHostMalloc((void **) &p, 2 * GGML_MOE_FUSED_WAKE_SLOTS * sizeof(uint32_t),
                                 hipHostMallocCoherent | hipHostMallocMapped | hipHostMallocPortable));
#else
        CUDA_CHECK(cudaHostAlloc((void **) &p, 2 * GGML_MOE_FUSED_WAKE_SLOTS * sizeof(uint32_t),
                                 cudaHostAllocPortable | cudaHostAllocMapped));
#endif
        memset(p, 0, 2 * GGML_MOE_FUSED_WAKE_SLOTS * sizeof(uint32_t));
        return p;
    }();
    return flags;
}

uint32_t * ggml_cuda_wake_post_word(const ggml_tensor * weights) {
    const int slot = weights ? ggml_get_op_params_i32(weights, 0) : 0;
    return slot >= 1 && slot <= GGML_MOE_FUSED_WAKE_SLOTS ? ggml_cuda_wake_flags() + (slot - 1) : nullptr;
}

extern "C" void ggml_backend_cuda_div_wake_post(ggml_tensor * div, int slot) {
    GGML_ASSERT(div != nullptr && div->op == GGML_OP_DIV && slot >= 1 && slot <= GGML_MOE_FUSED_WAKE_SLOTS);
    ggml_set_op_params_i32(div, 0, slot);
}

static __global__ void k_wake_post(uint32_t * post) {
#if defined(GGML_USE_HIP)
    __hip_atomic_fetch_add(post, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    atomicAdd_system(post, 1u);
#endif
}

void ggml_cuda_wake_post(ggml_backend_cuda_context & ctx, const ggml_tensor * weights) {
    uint32_t * post = ggml_cuda_wake_post_word(weights);
    if (post) {
        k_wake_post<<<1, 1, 0, ctx.stream()>>>(post);
        CUDA_CHECK(cudaGetLastError());
    }
}

bool ggml_cuda_moe_router_topk_supported(int n_expert, int k) {
    return k >= 1 && k <= WARP_SIZE && k <= n_expert &&
           (n_expert == 32 || n_expert == 64 || n_expert == 128 || n_expert == 256 || n_expert == 512 || n_expert == 1024);
}

void ggml_cuda_moe_router_topk(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_router_topk_args & a) {
    GGML_ASSERT(ggml_cuda_moe_router_topk_supported(a.n_expert, a.k));
    const dim3 grid((unsigned) a.n_tokens, 1, 1);
    cudaStream_t stream = ctx.stream();
    switch (a.n_expert) {
        case 32:   moe_router_topk_kernel<32>  <<<grid, 32,   0, stream>>>(a); break;
        case 64:   moe_router_topk_kernel<64>  <<<grid, 64,   0, stream>>>(a); break;
        case 128:  moe_router_topk_kernel<128> <<<grid, 128,  0, stream>>>(a); break;
        case 256:  moe_router_topk_kernel<256> <<<grid, 256,  0, stream>>>(a); break;
        case 512:  moe_router_topk_kernel<512> <<<grid, 512,  0, stream>>>(a); break;
        case 1024: moe_router_topk_kernel<1024><<<grid, 1024, 0, stream>>>(a); break;
        default: GGML_ABORT("moe router: unsupported n_expert");
    }
    CUDA_CHECK(cudaGetLastError());
}
