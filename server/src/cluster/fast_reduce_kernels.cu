// fast_reduce_kernels.cu - the device half of the lean decode-path reduction.
//
// Three tiny kernels per reduction, all on the backend's own stream so they sit
// in the graph exactly where the collective used to:
//
//   set_flag    raise this rank's publish flag once the payload has been
//               copied out. The copy itself is a DMA, enqueued ahead of this
//               on the same stream, so stream order is what publishes it.
//   wait_flags  spin until every peer's flag has arrived.
//   add_peers   add the peers' partials, by then copied into device memory.
//
// WHY THE PAYLOAD IS NOT MOVED BY A KERNEL. It was, and it cost 33 ms per
// reduction against RCCL's 117 us -- 250 times worse than the thing it was
// meant to replace. Pinned host memory is mapped uncached for the GPU, so a
// kernel reading or writing bulk data there crawls; the copy engine does not
// care, because it is not reading through the GPU's caches. So the payload goes
// by hipMemcpyAsync and only the flags -- four bytes, three times per
// reduction -- are touched by a kernel at zero-copy speed.
//
// WHY THE FLAGS ARE PINNED AND NOT MANAGED. gfx1151 reports XNACK disabled, so
// the GPU cannot take a page fault, and a managed page the CPU has touched
// faults the moment a kernel reads it. Pinned pages never migrate, so they need
// no fault to be reached -- and the round trip through one measures 0.79 us
// (server/test/cluster_flag_latency.cu), which is what makes this worth doing.
//
// Every spin is bounded and counted. A rank that dies must surface as a visible
// failure, not as a GPU that never returns.

#include <hip/hip_runtime.h>

#include <cstdint>

namespace luce::cluster {

namespace {
constexpr int kBlock = 256;

// Ordering costs cache, and on this part it costs all of it. A system-scope
// release writes the L2 back and a system-scope acquire invalidates it, so a
// flag raised or spun on with the strong form evicts the weights the next layer
// is about to read. Four of them per reduction, ninety-seven reductions per
// step: measured, that alone was 34 ms per reduction against RCCL's 117 us,
// while the same path with the announcements removed ran the step in 32.5 ms.
//
// So each flag gets the weakest ordering that is still correct:
//
//   relaxed system    for anything the host reads or writes. The payload is
//                     ordered by the stream -- the copy is a separate
//                     operation that completes before the kernel starts -- so
//                     no release is needed to publish it.
//   one acquire       after a spin succeeds, not once per iteration. The
//                     invalidate has to happen; it does not have to happen a
//                     million times.
//   agent scope       for slot_done, which only this GPU's own later kernels
//                     read. Telling the whole system about it buys nothing.
__device__ __forceinline__ uint32_t sys_load_relaxed(const uint32_t * p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void sys_store_relaxed(uint32_t * p, uint32_t v) {
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void sys_acquire(const uint32_t * p) {
    (void) __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint32_t dev_load(const uint32_t * p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void dev_store(uint32_t * p, uint32_t v) {
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
}  // namespace

__global__ void fast_reduce_set_flag_kernel(uint32_t * pub_flag,
                                            uint32_t * slot_done,
                                            uint32_t * progress,
                                            uint32_t seq,
                                            uint32_t slot_span,
                                            uint64_t spin_limit) {
    if (threadIdx.x || blockIdx.x) return;
    // Do not overwrite a slot the progress thread has not finished with. The
    // GPU can be a step's worth of reductions ahead, so the ring is defended
    // rather than assumed wide enough.
    if (seq > slot_span) {
        uint64_t spins = 0;
        while (dev_load(slot_done) + slot_span < seq) {
            if (++spins > spin_limit) break;
        }
    }
    sys_store_relaxed(progress, seq * 4u + 1u);
    sys_store_relaxed(pub_flag, seq);
}

__global__ void fast_reduce_wait_flags_kernel(uint32_t * const * __restrict__ peer_flags,
                                              uint32_t * timed_out,
                                              uint32_t * progress,
                                              uint32_t seq,
                                              int n_peers,
                                              uint64_t spin_limit) {
    if (threadIdx.x || blockIdx.x) return;
    sys_store_relaxed(progress, seq * 4u + 2u);
    for (int p = 0; p < n_peers; ++p) {
        uint64_t spins = 0;
        while (sys_load_relaxed(peer_flags[p]) != seq) {
            if (++spins > spin_limit) {
                sys_store_relaxed(timed_out, seq);
                return;
            }
        }
    }
    // One invalidate, now that every peer has arrived, so the copies queued
    // behind this read what the NIC wrote and not a cached line.
    sys_acquire(peer_flags[0]);
}

// The peers' partials are in device memory by now -- the copy engine moved them
// there after the wait -- so this is an ordinary cached add.
__global__ void fast_reduce_add_kernel(float * __restrict__ dst,
                                       const float * __restrict__ scratch,
                                       int n_peers,
                                       int n,
                                       int stride,
                                       uint32_t * slot_done,
                                       uint32_t * progress,
                                       uint32_t seq) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int k = i; k < n; k += blockDim.x * gridDim.x) {
        float acc = dst[k];
        for (int p = 0; p < n_peers; ++p) {
            acc += scratch[(size_t) p * (size_t) stride + (size_t) k];
        }
        dst[k] = acc;
    }
    // The slot is free once its data has been consumed, not once it arrived.
    // Saying so from one thread is safe only because the whole grid's writes
    // are complete at the kernel boundary -- which is why the flag is raised
    // here and not by a fence inside the loop.
    //
    // A null slot_done means the caller wants the arithmetic without the
    // announcement: a system-scope release writes the L2 back, and the probe
    // needs to price that separately from the copies.
    if (i == 0 && slot_done) {
        dev_store(slot_done, seq);
        sys_store_relaxed(progress, seq * 4u + 3u);
    }
}

// ── Zero-copy path for discrete GPUs (LUCE_CLUSTER_FAST_REDUCE_ZC=1) ──────
// The partial goes straight into the pinned row the NIC sends from, and the
// peer's row is added straight out of pinned memory: no copy-engine hops.
// bf16 payloads round both operands so every rank computes the same sum.
__device__ __forceinline__ uint16_t fr_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ float fr_bf16_to_f32(uint16_t h) {
    return __uint_as_float(((uint32_t) h) << 16);
}

// LUCE_CLUSTER_FAST_REDUCE_KPUB: the grid that wrote an exchange row publishes
// its flag itself. Each block orders its writes at device scope and counts in;
// the last makes every block's writes visible system-wide with ONE release and
// stores the flag. Measured against a hipStreamWriteValue32 after the kernel:
// publish + host ack 9-11 us instead of 16-18 on gfx1201, 3.4 instead of 6.3 on
// gfx1151. A null flag publishes nothing. counter is this device's, zeroed, and
// left zeroed for the next grid on the same stream.
__device__ __forceinline__ void fr_publish_last_block(uint32_t * flag, uint32_t value, uint32_t * counter) {
    if (!flag) return;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t prev = atomicAdd(counter, 1u);
        if (prev == gridDim.x - 1) {
            *counter = 0;
            __threadfence_system();
            __hip_atomic_store(flag, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

__global__ void fast_reduce_pack_kernel(const float * __restrict__ src, void * dst, int n, int bf16,
                                        uint32_t * flag, uint32_t value, uint32_t * counter) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += blockDim.x * gridDim.x) {
        if (bf16) ((uint16_t *) dst)[k] = fr_f32_to_bf16(src[k]);
        else      ((float *) dst)[k] = src[k];
    }
    fr_publish_last_block(flag, value, counter);
}

__global__ void fast_reduce_add_zc_kernel(float * __restrict__ dst, const void * peer, int n, int bf16) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += blockDim.x * gridDim.x) {
        if (bf16) {
            const float own = fr_bf16_to_f32(fr_f32_to_bf16(dst[k]));
            dst[k] = own + fr_bf16_to_f32(((const uint16_t *) peer)[k]);
        } else {
            dst[k] = dst[k] + ((const float *) peer)[k];
        }
    }
}

// Hybrid exchange, iGPU side. own = (cold + hot) + shared in f32, the order
// the dGPU's graph adds them in, then the wire row; the final sum is
// bf16(own) + peer exactly as fast_reduce_add_zc_kernel forms it.
__global__ void fast_reduce_hyb_combine_kernel(const float * __restrict__ cold, const float * __restrict__ hot,
                                               const float * __restrict__ shared, void * own_row, int n, int bf16,
                                               uint32_t * flag, uint32_t value, uint32_t * counter) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += blockDim.x * gridDim.x) {
        const float own = shared ? (cold[k] + hot[k]) + shared[k] : cold[k] + hot[k];
        if (bf16) ((uint16_t *) own_row)[k] = fr_f32_to_bf16(own);
        else      ((float *) own_row)[k] = own;
    }
    fr_publish_last_block(flag, value, counter);
}

__global__ void fast_reduce_hyb_final_kernel(const void * own_row, const void * peer_row, float * __restrict__ dst,
                                             int n, int bf16, uint32_t * flag, uint32_t value, uint32_t * counter) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += blockDim.x * gridDim.x) {
        if (bf16) dst[k] = fr_bf16_to_f32(((const uint16_t *) own_row)[k]) + fr_bf16_to_f32(((const uint16_t *) peer_row)[k]);
        else      dst[k] = ((const float *) own_row)[k] + ((const float *) peer_row)[k];
    }
    fr_publish_last_block(flag, value, counter);
}

void fast_reduce_launch_hyb_combine_pub(const float * cold, const float * hot, const float * shared, void * own_row,
                                        int n, int bf16, uint32_t * flag, uint32_t value, uint32_t * counter,
                                        hipStream_t stream) {
    int grid = (n + kBlock - 1) / kBlock; if (grid < 1) grid = 1; if (grid > 128) grid = 128;
    hipLaunchKernelGGL(fast_reduce_hyb_combine_kernel, dim3(grid), dim3(kBlock), 0, stream, cold, hot, shared, own_row, n, bf16,
                       flag, value, counter);
}

void fast_reduce_launch_hyb_combine(const float * cold, const float * hot, const float * shared, void * own_row,
                                    int n, int bf16, hipStream_t stream) {
    fast_reduce_launch_hyb_combine_pub(cold, hot, shared, own_row, n, bf16, nullptr, 0, nullptr, stream);
}

void fast_reduce_launch_hyb_final_pub(const void * own_row, const void * peer_row, float * dst, int n, int bf16,
                                      uint32_t * flag, uint32_t value, uint32_t * counter, hipStream_t stream) {
    int grid = (n + kBlock - 1) / kBlock; if (grid < 1) grid = 1; if (grid > 128) grid = 128;
    hipLaunchKernelGGL(fast_reduce_hyb_final_kernel, dim3(grid), dim3(kBlock), 0, stream, own_row, peer_row, dst, n, bf16,
                       flag, value, counter);
}

void fast_reduce_launch_hyb_final(const void * own_row, const void * peer_row, float * dst, int n, int bf16,
                                  hipStream_t stream) {
    fast_reduce_launch_hyb_final_pub(own_row, peer_row, dst, n, bf16, nullptr, 0, nullptr, stream);
}

// A word another device polls, written after every earlier write of this
// stream's kernels left this device (peer writes, then the flag on the same
// path, so the reader sees the data once it sees the flag).
__global__ void fast_reduce_flag_sys_kernel(uint32_t * flag, uint32_t value) {
    __threadfence_system();
    __hip_atomic_store(flag, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

void fast_reduce_launch_flag_sys(uint32_t * flag, uint32_t value, hipStream_t stream) {
    hipLaunchKernelGGL(fast_reduce_flag_sys_kernel, dim3(1), dim3(1), 0, stream, flag, value);
}

void fast_reduce_launch_pack_pub(const float * src, void * dst, int n, int bf16,
                                 uint32_t * flag, uint32_t value, uint32_t * counter, hipStream_t stream) {
    int grid = (n + kBlock - 1) / kBlock; if (grid < 1) grid = 1; if (grid > 128) grid = 128;
    hipLaunchKernelGGL(fast_reduce_pack_kernel, dim3(grid), dim3(kBlock), 0, stream, src, dst, n, bf16,
                       flag, value, counter);
}

void fast_reduce_launch_pack(const float * src, void * dst, int n, int bf16, hipStream_t stream) {
    fast_reduce_launch_pack_pub(src, dst, n, bf16, nullptr, 0, nullptr, stream);
}

void fast_reduce_launch_add_zc(float * dst, const void * peer, int n, int bf16, hipStream_t stream) {
    int grid = (n + kBlock - 1) / kBlock; if (grid < 1) grid = 1; if (grid > 128) grid = 128;
    hipLaunchKernelGGL(fast_reduce_add_zc_kernel, dim3(grid), dim3(kBlock), 0, stream, dst, peer, n, bf16);
}

void fast_reduce_launch_set_flag(uint32_t * pub_flag, uint32_t * slot_done,
                                 uint32_t * progress, uint32_t seq,
                                 uint32_t slot_span, uint64_t spin_limit,
                                 hipStream_t stream) {
    hipLaunchKernelGGL(fast_reduce_set_flag_kernel, dim3(1), dim3(1), 0, stream,
                       pub_flag, slot_done, progress, seq, slot_span, spin_limit);
}

void fast_reduce_launch_wait_flags(uint32_t * const * peer_flags,
                                   uint32_t * timed_out, uint32_t * progress,
                                   uint32_t seq, int n_peers,
                                   uint64_t spin_limit, hipStream_t stream) {
    hipLaunchKernelGGL(fast_reduce_wait_flags_kernel, dim3(1), dim3(1), 0, stream,
                       peer_flags, timed_out, progress, seq, n_peers, spin_limit);
}

void fast_reduce_launch_add(float * dst, const float * scratch, int n_peers,
                            int n, int stride, uint32_t * slot_done,
                            uint32_t * progress, uint32_t seq,
                            hipStream_t stream) {
    int grid = (n + kBlock - 1) / kBlock;
    if (grid < 1) grid = 1;
    if (grid > 256) grid = 256;
    hipLaunchKernelGGL(fast_reduce_add_kernel, dim3(grid), dim3(kBlock), 0, stream,
                       dst, scratch, n_peers, n, stride, slot_done, progress, seq);
}

}  // namespace luce::cluster
