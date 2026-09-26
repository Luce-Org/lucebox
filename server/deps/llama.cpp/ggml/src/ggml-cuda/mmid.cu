#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    // token slots per warp lane group, padded to a power of 2 so a warp divides evenly
    constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                // Negative IDs are masked owner routes. They belong to no
                // expert and must not reserve holes in the compact arrays.
                nex_prev += expert_used >= 0 && expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used >= 0 && expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    // The write-back loop below reads store[] slots written by other lanes in
    // the loop above; the shuffle reduce does not order shared-memory writes
    // across lanes (upstream race fix 73a43d1f).
    __syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
__launch_bounds__(256, 1)
static __global__ void mm_ids_helper_fast(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1,
        int32_t * __restrict__ ids_dst,
        int32_t * __restrict__ expert_bounds,
        const int n_experts,
        const int n_tokens,
        const int n_expert_used_var,
        const int nchannels_y,
        const int si1,
        const int sis1) {

    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    __shared__ int smem_hist[256];
    __shared__ int smem_curr[256];

    const int tid = threadIdx.x;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;
    const int nwarps = 256 / warp_size;

    // 1. Initialize LDS
    if (tid < n_experts) {
        smem_hist[tid] = 0;
    }
    __syncthreads();

    // 2. Cooperative parallel histogram
    const int64_t total_items = (int64_t) n_tokens * n_expert_used;
    for (int64_t idx = tid; idx < total_items; idx += 256) {
        const int it = idx / n_expert_used;
        const int iex = idx % n_expert_used;
        const int exp_id = ids[it * si1 + iex];
        if (exp_id >= 0 && exp_id < n_experts) {
            atomicAdd(&smem_hist[exp_id], 1);
        }
    }
    __syncthreads();

    // 3. Exclusive prefix sum over at most 256 experts. A short serial scan
    // avoids partial-warp shuffle masks and is negligible next to route
    // histogram/scatter and the following matrix multiplies.
    if (tid == 0) {
        int offset = 0;
        for (int expert = 0; expert < n_experts; ++expert) {
            smem_curr[expert] = offset;
            expert_bounds[expert] = offset;
            offset += smem_hist[expert];
        }
        expert_bounds[n_experts] = offset;
    }
    __syncthreads();

    // 4. Deterministic token scatter in token order
    for (int it0 = 0; it0 < n_tokens; it0 += nwarps) {
        const int it = it0 + warp_id;
        if (it < n_tokens) {
            for (int iex = lane_id; iex < n_expert_used; iex += warp_size) {
                const int exp_id = ids[it * si1 + iex];
                if (exp_id >= 0 && exp_id < n_experts) {
                    const int pos = atomicAdd(&smem_curr[exp_id], 1);
                    ids_src1[pos] = it * sis1 + (iex % nchannels_y);
                    ids_dst[pos]  = it * n_expert_used + iex;
                }
            }
        }
        __syncthreads();
    }
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT((int64_t) n_tokens * n_expert_used_var <= 0x7FFFFFFFLL &&
                "MUL_MAT_ID route count exceeds compact int32 indexing");

    if (n_experts <= 256 && !write_inverse) {
        mm_ids_helper_fast<n_expert_used_template><<<1, 256, 0, stream>>>
            (ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used_var, nchannels_y, si1, sis1);
        return;
    }

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
}

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    switch (n_expert_used) {
        case  2:
            launch_mm_ids_helper< 2>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  4:
            launch_mm_ids_helper< 4>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  6:
            launch_mm_ids_helper< 6>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case  8:
            launch_mm_ids_helper< 8>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 10:
            launch_mm_ids_helper<10>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 16:
            launch_mm_ids_helper<16>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        case 32:
            launch_mm_ids_helper<32>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
        default:
            launch_mm_ids_helper< 0>(ids, ids_src1, ids_dst, expert_bounds, n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);
            break;
    }
}

static constexpr int MM_IDS_ROUTE_TILE = 1024;
static constexpr int MM_IDS_ROUTE_BITS = 10;
static constexpr int MM_IDS_ROUTE_EXPERTS = 512;
static constexpr int MM_IDS_ROUTE_USED = 10;

static __global__ void mm_ids_route_sort(
        const int32_t * ids, uint32_t * sorted, int32_t * offsets, int32_t * starts,
        int routes, int chunks, int ids_stride) {
    __shared__ uint32_t keys[MM_IDS_ROUTE_TILE];
    __shared__ int first[MM_IDS_ROUTE_EXPERTS];
    __shared__ int counts[MM_IDS_ROUTE_EXPERTS];
    const int chunk=blockIdx.x;
    for (int i=threadIdx.x;i<MM_IDS_ROUTE_EXPERTS;i+=blockDim.x) {
        first[i]=0;counts[i]=0;
    }
    for (int i=threadIdx.x;i<MM_IDS_ROUTE_TILE;i+=blockDim.x) {
        const int route=chunk*MM_IDS_ROUTE_TILE+i;
        int expert=-1;
        if (route<routes) expert=ids[(route/MM_IDS_ROUTE_USED)*ids_stride+route%MM_IDS_ROUTE_USED];
        keys[i]=expert>=0 && expert<MM_IDS_ROUTE_EXPERTS ?
            (uint32_t(expert)<<MM_IDS_ROUTE_BITS)|uint32_t(i) : UINT32_MAX;
    }
    __syncthreads();
    for (int span=2;span<=MM_IDS_ROUTE_TILE;span*=2) {
        for (int distance=span/2;distance>0;distance/=2) {
            for (int i=threadIdx.x;i<MM_IDS_ROUTE_TILE;i+=blockDim.x) {
                const int partner=i^distance;
                if (partner>i) {
                    const uint32_t a=keys[i],b=keys[partner];
                    if (((i&span)==0 && a>b) || ((i&span)!=0 && a<b)) {
                        keys[i]=b;keys[partner]=a;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i=threadIdx.x;i<MM_IDS_ROUTE_TILE;i+=blockDim.x) {
        if (keys[i]!=UINT32_MAX) {
            const int expert=keys[i]>>MM_IDS_ROUTE_BITS;
            if (i==0 || (keys[i-1]>>MM_IDS_ROUTE_BITS)!=uint32_t(expert)) first[expert]=i;
        }
        sorted[chunk*MM_IDS_ROUTE_TILE+i]=keys[i];
    }
    __syncthreads();
    for (int i=threadIdx.x;i<MM_IDS_ROUTE_TILE;i+=blockDim.x) {
        if (keys[i]!=UINT32_MAX) {
            const int expert=keys[i]>>MM_IDS_ROUTE_BITS;
            if (i+1==MM_IDS_ROUTE_TILE || (keys[i+1]>>MM_IDS_ROUTE_BITS)!=uint32_t(expert)) {
                counts[expert]=i+1-first[expert];
            }
        }
    }
    __syncthreads();
    for (int i=threadIdx.x;i<MM_IDS_ROUTE_EXPERTS;i+=blockDim.x) {
        offsets[chunk*MM_IDS_ROUTE_EXPERTS+i]=counts[i];
        starts[chunk*MM_IDS_ROUTE_EXPERTS+i]=first[i];
    }
    (void)chunks;
}

static __global__ void mm_ids_route_prefix(int32_t * offsets, int32_t * bounds, int chunks) {
    __shared__ int totals[MM_IDS_ROUTE_EXPERTS];
    const int expert=threadIdx.x;
    int total=0;
    for (int chunk=0;chunk<chunks;++chunk) total+=offsets[chunk*MM_IDS_ROUTE_EXPERTS+expert];
    totals[expert]=total;
    __syncthreads();
    int prefix=0;
    for (int previous=0;previous<expert;++previous) prefix+=totals[previous];
    bounds[expert]=prefix;
    if (expert==MM_IDS_ROUTE_EXPERTS-1) bounds[MM_IDS_ROUTE_EXPERTS]=prefix+total;
    for (int chunk=0;chunk<chunks;++chunk) {
        const int index=chunk*MM_IDS_ROUTE_EXPERTS+expert;
        const int count=offsets[index];
        offsets[index]=prefix;
        prefix+=count;
    }
}

static __global__ void mm_ids_route_scatter(
        const uint32_t * sorted, const int32_t * offsets, const int32_t * starts,
        int32_t * src_map, int32_t * dst_map, int chunks, int channels, int token_stride, bool inverse) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=chunks*MM_IDS_ROUTE_TILE) return;
    const uint32_t key=sorted[index];
    if (key==UINT32_MAX) return;
    const int chunk=index/MM_IDS_ROUTE_TILE;
    const int expert=key>>MM_IDS_ROUTE_BITS;
    const int local=index%MM_IDS_ROUTE_TILE;
    const int route=chunk*MM_IDS_ROUTE_TILE+(key&(MM_IDS_ROUTE_TILE-1));
    const int target=offsets[chunk*MM_IDS_ROUTE_EXPERTS+expert]+local-starts[chunk*MM_IDS_ROUTE_EXPERTS+expert];
    dst_map[target]=route;
    if (inverse) src_map[route]=target;
    else src_map[target]=(route/MM_IDS_ROUTE_USED)*token_stride+(route%MM_IDS_ROUTE_USED)%channels;
}

bool ggml_cuda_launch_mm_ids_bounded(
        ggml_backend_cuda_context & ctx, const int32_t * ids, int32_t * src_map,
        int32_t * dst_map, int32_t * bounds, int experts, int tokens, int used,
        int channels, int ids_stride, int token_stride, bool inverse, cudaStream_t stream) {
    const int device=ggml_cuda_get_device();
    if (!GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[device].cc) ||
            experts!=MM_IDS_ROUTE_EXPERTS || used!=MM_IDS_ROUTE_USED ||
            tokens<=4096 || tokens>32768 || (channels!=1 && channels!=used) ||
            ids_stride<used || token_stride<channels) return false;
    const int routes=tokens*used;
    const int chunks=(routes+MM_IDS_ROUTE_TILE-1)/MM_IDS_ROUTE_TILE;
    ggml_cuda_pool_alloc<uint32_t> sorted(ctx.pool(device),size_t(chunks)*MM_IDS_ROUTE_TILE);
    ggml_cuda_pool_alloc<int32_t> offsets(ctx.pool(device),size_t(chunks)*experts);
    ggml_cuda_pool_alloc<int32_t> starts(ctx.pool(device),size_t(chunks)*experts);
    mm_ids_route_sort<<<chunks,256,0,stream>>>(ids,sorted.get(),offsets.get(),starts.get(),routes,chunks,ids_stride);
    mm_ids_route_prefix<<<1,MM_IDS_ROUTE_EXPERTS,0,stream>>>(offsets.get(),bounds,chunks);
    mm_ids_route_scatter<<<(chunks*MM_IDS_ROUTE_TILE+255)/256,256,0,stream>>>
        (sorted.get(),offsets.get(),starts.get(),src_map,dst_map,chunks,channels,token_stride,inverse);
    CUDA_CHECK(cudaGetLastError());
    return true;
}
