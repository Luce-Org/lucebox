// Runtime decode for GGML_TYPE_Q3_1_ROCMFP3_MIX (105).
// A per-tensor registry supplies the per-expert codebook/mode that the ggml
// to_fp16 converter signature cannot carry; the deepseek4 loader registers each
// fused down-expert tensor after staging its sidecar side-data to device memory.
#include "rocmfp3_mix.cuh"
// For ggml_cuda_op_swiglu_ds4_single -- the fused path applies the EXACT function the
// standalone swiglu_ds4 kernel applies, not a re-derivation.
#include "unary.cuh"
#include "convert.cuh"
#include <mutex>
#include <vector>

// MSVC's host toolchain gives nvcc device code neither __builtin_memcpy nor
// __builtin_assume_aligned. memcpy lowers to the same register moves under
// every device compiler, and the alignment hint is an optimization-only
// contract that the callers' address arithmetic already guarantees -- so the
// MSVC fallback changes nothing except compiling. GCC/Clang (every HIP build,
// where the wide-load numbers were measured) keep the builtins.
#if defined(_MSC_VER) && !defined(__clang__)
#define MIX_MEMCPY(dst, src, n)   memcpy((dst), (src), (n))
#define MIX_ASSUME_ALIGNED(p, a)  (p)
#else
#define MIX_MEMCPY(dst, src, n)   __builtin_memcpy((dst), (src), (n))
#define MIX_ASSUME_ALIGNED(p, a)  __builtin_assume_aligned((p), (a))
#endif

#if defined(GGML_USE_HIP)
#ifndef cudaPointerAttributes
// Same local-mapping pattern topk-rows.cu already uses in this directory: the vendor shim
// does not map the pointer-attribute API, and these are needed to find which device owns an
// expert tensor. hipPointerAttribute_t is layout-compatible for the fields read here.
#define cudaPointerAttributes    hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaMemoryTypeDevice     hipMemoryTypeDevice
#define cudaMemoryTypeManaged    hipMemoryTypeManaged
#endif
#endif


#define MIX_QK 32
#define MIX_QS 12
#define MIX_BLOCK_BYTES 14
// Learned levels per codebook. A qtype-105 entry carries TWO codebooks (the 7s1c
// layout's 1-bit select picks one per 16-weight half-block), so an expert's table
// is 2 * MIX_K bf16 = 32 B.
#define MIX_K 8

namespace {
struct MixEntry {
    const void * base;
    size_t nb02;          // byte stride between experts
    int n_experts, out, in;
    const nv_bfloat16 * codebooks;  // n_experts * 2 * 8
    const uint8_t * modes;          // n_experts
    const uint8_t * rotations;      // n_experts (unused until p3 rotation lands)
    bool owns_device;     // true => this entry cudaMalloc'd the 3 buffers above
                          // (register_host) and must free them on erase/update
    int  device;          // device the side-data lives on; frees must happen in that context
};
std::mutex g_mix_mtx;
std::vector<MixEntry> g_mix_registry;

// Resolve the device that owns `p`. The mix side-data must be allocated on the SAME device as
// the expert tensor it describes: the kernel dereferences both together, and cudaMalloc uses
// the CURRENT device, which is not necessarily the one the model was loaded onto. Returns the
// current device when the pointer cannot be attributed (single-device hosts, or host memory),
// which preserves the previous behaviour exactly.
static int mix_device_of(const void * p) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    cudaPointerAttributes attr{};
    if (p && cudaPointerGetAttributes(&attr, p) == cudaSuccess) {
        // cudaMemoryTypeUnregistered/Host leave `device` meaningless; only trust device memory.
        if (attr.type == cudaMemoryTypeDevice && attr.device >= 0) {
            return attr.device;
        }
    }
    cudaGetLastError();   // clear the error a non-device pointer sets
    return dev;
}

// RAII device switch: restores the previous device even on the early-return error paths.
struct MixDeviceGuard {
    int prev = -1;
    bool active = false;
    explicit MixDeviceGuard(int dev) {
        if (cudaGetDevice(&prev) != cudaSuccess) return;
        if (dev == prev) return;
        if (cudaSetDevice(dev) == cudaSuccess) active = true;
    }
    ~MixDeviceGuard() { if (active) cudaSetDevice(prev); }
};

// Free an entry's device side-data if it owns it. Caller holds g_mix_mtx.
void mix_free_entry_device(MixEntry & e) {
    if (!e.owns_device) return;
    MixDeviceGuard guard(e.device);   // free where it was allocated
    if (e.codebooks) cudaFree((void *) e.codebooks);
    if (e.modes)     cudaFree((void *) e.modes);
    if (e.rotations) cudaFree((void *) e.rotations);
    e.codebooks = nullptr; e.modes = nullptr; e.rotations = nullptr;
    e.owns_device = false;
}

void mix_register_impl(const void * base, size_t nb02, int n_experts, int out, int in,
                       const nv_bfloat16 * codebooks, const uint8_t * modes,
                       const uint8_t * rotations, bool owns_device) {
    std::lock_guard<std::mutex> lk(g_mix_mtx);
    MixEntry ne{base, nb02, n_experts, out, in, codebooks, modes, rotations,
                owns_device, mix_device_of(base)};
    for (auto & e : g_mix_registry) {
        if (e.base == base) {  // update in place — free the old owned buffers first
            mix_free_entry_device(e);
            e = ne;
            return;
        }
    }
    g_mix_registry.push_back(ne);
}
}  // namespace

// Non-owning registration: codebooks/modes/rotations are device buffers whose
// lifetime the CALLER manages (unregister will not free them).
extern "C" void ggml_cuda_rocmfp3_mix_register(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks, const void * modes, const void * rotations) {
    mix_register_impl(base, nb02, n_experts, out, in,
                      (const nv_bfloat16 *) codebooks, (const uint8_t *) modes,
                      (const uint8_t *) rotations, /*owns_device=*/false);
}

// Host-side convenience for the deepseek4 loader: stage per-expert codebooks
// (bf16) and modes from host memory into device buffers, then register. The
// registry OWNS these buffers and frees them on unregister/update. rotations
// host array optional (nullptr => none rotated). On a cudaMalloc failure the
// already-allocated buffers are freed before propagating the error, so a failed
// registration leaks nothing.
extern "C" void ggml_cuda_rocmfp3_mix_register_host(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks_bf16_host, const uint8_t * modes_host,
        const uint8_t * rotations_host) {
    const size_t cb_bytes = (size_t) n_experts * 2 * 8 * sizeof(nv_bfloat16);
    void * cb_dev = nullptr; void * modes_dev = nullptr; void * rots_dev = nullptr;
    // Allocate where the WEIGHTS are. Without this the side-data lands on the current
    // device while the kernel runs on the model's device, and the first request
    // segfaults on a multi-GPU host.
    MixDeviceGuard guard(mix_device_of(base));
    cudaError_t err = cudaMalloc(&cb_dev, cb_bytes);
    if (err == cudaSuccess) err = cudaMemcpy(cb_dev, codebooks_bf16_host, cb_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMalloc(&modes_dev, (size_t) n_experts);
    if (err == cudaSuccess) err = cudaMemcpy(modes_dev, modes_host, (size_t) n_experts, cudaMemcpyHostToDevice);
    bool any_rotation = false;
    for (int i = 0; rotations_host && i < n_experts; ++i) {
        any_rotation = any_rotation || rotations_host[i] != 0;
    }
    if (err == cudaSuccess && any_rotation) {
        err = cudaMalloc(&rots_dev, (size_t) n_experts);
        if (err == cudaSuccess) err = cudaMemcpy(rots_dev, rotations_host, (size_t) n_experts, cudaMemcpyHostToDevice);
    }
    if (err != cudaSuccess) {
        if (cb_dev)    cudaFree(cb_dev);
        if (modes_dev) cudaFree(modes_dev);
        if (rots_dev)  cudaFree(rots_dev);
        CUDA_CHECK(err);  // report/abort exactly as before, but only after cleanup
        return;
    }
    mix_register_impl(base, nb02, n_experts, out, in, (const nv_bfloat16 *) cb_dev,
                      (const uint8_t *) modes_dev, (const uint8_t *) rots_dev,
                      /*owns_device=*/true);
}

extern "C" void ggml_cuda_rocmfp3_mix_unregister(const void * base) {
    std::lock_guard<std::mutex> lk(g_mix_mtx);
    for (size_t i = 0; i < g_mix_registry.size(); ++i) {
        if (g_mix_registry[i].base == base) {
            mix_free_entry_device(g_mix_registry[i]);
            g_mix_registry.erase(g_mix_registry.begin() + i);
            return;
        }
    }
}

static bool mix_lookup(const void * vx, MixEntry & out_e, int & out_expert) {
    std::lock_guard<std::mutex> lk(g_mix_mtx);
    const char * p = (const char *) vx;
    for (const auto & e : g_mix_registry) {
        const char * b = (const char *) e.base;
        if (p >= b && p < b + (size_t) e.n_experts * e.nb02) {
            out_e = e;
            out_expert = (int) (((size_t) (p - b)) / e.nb02);
            return true;
        }
    }
    return false;
}

__device__ __forceinline__ float mix_ue4m3(uint8_t e) {
    if (e > 0x7E) return 0.0f;
    int exp = e >> 3, mant = e & 7;
    if (exp == 0) return (float) mant * 0.0009765625f;  // 2^-10
    return ldexpf((float) (8 + mant), exp - 11);
}

__device__ __forceinline__ uint32_t mix_fp3_code(const uint8_t * qs, int i) {
    int bit = i * 3, byte = bit >> 3, shift = bit & 7;
    uint32_t v = qs[byte];
    if (byte + 1 < MIX_QS) v |= (uint32_t) qs[byte + 1] << 8;
    if (byte + 2 < MIX_QS) v |= (uint32_t) qs[byte + 2] << 16;
    return (v >> shift) & 7u;
}

__device__ __forceinline__ float mix_fp3_fixed(uint32_t code) {
    uint32_t m = code & 3u;
    int mag = (m == 3u) ? 4 : (int) m;
    return (code & 4u) ? -(float) mag : (float) mag;
}

// One thread per element of a single expert slice (k = out*in elements).
__global__ void dequantize_rocmfp3_mix_kernel(
        const uint8_t * __restrict__ data, const nv_bfloat16 * __restrict__ book,
        const uint8_t * __restrict__ mode_ptr, int in, int64_t k, half * __restrict__ y) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    // Same defect, same fix as the matvec path: `book` was read from global once per
    // ELEMENT for a 32 B workgroup-invariant table. The dense (non-MoE) fallback runs
    // through this kernel, so it must be fixed before quoting any dense adaptive number.
    __shared__ float s_lut[2 * MIX_K];
    if ((int) threadIdx.x < 2 * MIX_K) {
        s_lut[threadIdx.x] = __bfloat162float(book[threadIdx.x]);
    }
    __syncthreads();
    if (idx >= k) return;
    const int mode = (int) mode_ptr[0];
    const int nb  = in / MIX_QK;
    const int row = idx / in;
    const int col = idx % in;
    const int b   = row * nb + (col / MIX_QK);
    const int j   = col % MIX_QK;
    const int half = (j >= MIX_QK / 2) ? 1 : 0;
    const uint8_t * blk = data + (int64_t) b * MIX_BLOCK_BYTES;
    const uint8_t meta = blk[MIX_QS + half];
    const uint32_t code = mix_fp3_code(blk, j);
    float val;
    if (mode == 0) {
        val = mix_ue4m3(meta) * mix_fp3_fixed(code);
    } else {
        const float scale = mix_ue4m3(meta & 0x7F);
        const int bk = meta >> 7;
        val = scale * s_lut[bk * MIX_K + (int) code];
    }
    y[idx] = __float2half(val);
}

void dequantize_rocmfp3_mix_to_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    MixEntry e;
    int expert;
    if (!mix_lookup(vx, e, expert)) {
        GGML_ABORT("rocmfp3_mix: tensor slice %p not registered", vx);
    }
    const nv_bfloat16 * book = e.codebooks + (size_t) expert * 2 * 8;
    const uint8_t * mode_ptr = e.modes + expert;
    const int threads = 256;
    const int blocks = (int) ((k + threads - 1) / threads);
    // Portable launch: triple-chevron compiles under both nvcc and hipcc; the
    // hipLaunchKernelGGL macro is HIP-only and breaks the default CUDA build,
    // which still globs this *.cu file.
    dequantize_rocmfp3_mix_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
        (const uint8_t *) vx, book, mode_ptr, e.in, k, y);
}

// ---- fused quantized matvec (MMVQ-style decode) ----
// One warp per output row; lanes stride over the row's blocks, decode 32 weights
// each (bit-identical to dequantize_rocmfp3_mix_kernel), multiply by x, f32
// accumulate, warp-reduce. blockIdx.y selects the column (token). Reading the
// quantized blocks once avoids the ~10x f16 round-trip of the dequant fallback.
#define MIX_WARP 32
#define MIX_UNROLL 4

// Down-shift warp shuffle confined to a 32-lane logical group. width=MIX_WARP
// keeps the reduction self-contained on wave64 (GFX8/9, physical wave = 64) and
// is a no-op vs the default warp width on wave32 (gfx1151) / NVIDIA, so the
// reduced value — and thus the greedy output hash — is bit-identical there.
// HIP keeps the bare (mask-free) __shfl_down; modern CUDA only has the _sync
// form (and HIP's vendor shim doesn't cover __shfl_down_sync), so branch.
__device__ __forceinline__ float mix_warp_shfl_down(float v, int off) {
#if defined(__HIP_PLATFORM_AMD__)
    return __shfl_down(v, off, MIX_WARP);
#else
    return __shfl_down_sync(0xffffffffu, v, off, MIX_WARP);
#endif
}

// Accumulate one block's 32 terms directly into acc, in fixed j order, exactly
// as the un-refactored loop did (acc += s * w_j * x_j). Adding each term into
// the shared running acc — rather than forming a per-block partial sum first —
// preserves the flat left-fold summation order, so the f32 result is bit-for-bit
// identical to the original. (Correctness hashes the greedy output; a per-block
// tree reduction changes the rounding and flips tokens.) The block's byte loads
// do not depend on acc, so unrolling the caller over several blocks lets the
// compiler overlap their loads even though the acc-add chain stays serial.
__device__ __forceinline__ void mix_block_accum(
        const uint8_t * __restrict__ b, const float * __restrict__ xc, int col0,
        int mode, const float * __restrict__ lut, float & acc) {
    // Stage the whole 14-byte block into registers with one 2-byte-wide copy,
    // then decode the fp3 codes out of registers instead of re-reading the
    // packed qs region through narrow per-byte global loads (each of the 32
    // weights touches up to 3 of the 12 qs bytes). Blocks are always 2-byte
    // aligned (block stride 14 and row stride nb*14=896 are both even), so a
    // 2-byte assume is safe and lets the compiler fold the packed-weight reads
    // into ushort loads — halving the weight load-instruction count. The
    // decode arithmetic and the fixed j accumulation order are untouched, so
    // acc is bit-for-bit identical to the per-byte path (the correctness gate
    // hashes the greedy output; any reassociation flips a token).
    const uint8_t * ba = (const uint8_t *) MIX_ASSUME_ALIGNED(b, 2);
    uint8_t buf[MIX_BLOCK_BYTES];
    MIX_MEMCPY(buf, ba, MIX_BLOCK_BYTES);
    const uint8_t m0 = buf[MIX_QS + 0], m1 = buf[MIX_QS + 1];
    if (mode == 0) {
        const float s0 = mix_ue4m3(m0), s1 = mix_ue4m3(m1);
        #pragma unroll
        for (int j = 0; j < MIX_QK; ++j) {
            const float s = (j < MIX_QK/2) ? s0 : s1;
            acc += s * mix_fp3_fixed(mix_fp3_code(buf, j)) * xc[col0 + j];
        }
    } else {
        const float s0 = mix_ue4m3(m0 & 0x7F), s1 = mix_ue4m3(m1 & 0x7F);
        // `lut` is the expert's 2 * MIX_K codebook already widened to f32 and staged in
        // LDS by the caller. It was a global bf16 pointer dereferenced inside the unrolled
        // j loop below -- MIX_QK dependent 2-byte global loads per block for a 32 B
        // workgroup-invariant table, all lanes hitting the same bytes. Pure LSU issue and
        // dependent-load latency, never bandwidth.
        // Bit-exact: __bfloat162float is a widening with no rounding, so hoisting it out
        // changes no value and the ascending-j fold keeps its two roundings per term.
        const float * bk0 = lut + (m0 >> 7) * MIX_K;
        const float * bk1 = lut + (m1 >> 7) * MIX_K;
        #pragma unroll
        for (int j = 0; j < MIX_QK; ++j) {
            const float s = (j < MIX_QK/2) ? s0 : s1;
            const float * bk = (j < MIX_QK/2) ? bk0 : bk1;
            acc += s * bk[mix_fp3_code(buf, j)] * xc[col0 + j];
        }
    }
}

// The lane's block loop is unrolled by MIX_UNROLL into a SINGLE accumulator kept
// in the exact original block order (acc += dot(blk), stride MIX_WARP), so the
// f32 output is bit-for-bit identical to the un-unrolled path — required because
// the correctness gate hashes the greedy output, which a reassociated reduction
// would flip. This matvec is read-once (fixed DRAM volume) and latency-bound on
// the strided per-block weight loads; the only serial dependency is the cheap
// acc-add chain, while the MIX_UNROLL mix_block_dot() evaluations are mutually
// independent, so unrolling lets the compiler issue several blocks' weight loads
// before consuming them — exposing memory-level parallelism per lane without
// touching the summation order.
__global__ void mix_matvec_rocmfp3_kernel(
        const uint8_t * __restrict__ data, const nv_bfloat16 * __restrict__ book,
        const uint8_t * __restrict__ mode_ptr, const float * __restrict__ x,
        float * __restrict__ y, int in, int out,
        int64_t x_col_stride, int64_t y_col_stride) {
    const int warps_per_block = blockDim.x / MIX_WARP;
    const int row  = blockIdx.x * warps_per_block + (threadIdx.x / MIX_WARP);
    const int lane = threadIdx.x % MIX_WARP;
    const int col  = blockIdx.y;
    // Widen the 2 * MIX_K bf16 codebook to f32 in LDS once per workgroup rather than
    // re-reading it from global per weight. HOISTED ABOVE the early return:
    // __syncthreads() needs every thread of the workgroup, and `row >= out` retires
    // whole warps in the tail block.
    __shared__ float s_lut[2 * MIX_K];
    if ((int) threadIdx.x < 2 * MIX_K) {
        s_lut[threadIdx.x] = __bfloat162float(book[threadIdx.x]);
    }
    __syncthreads();
    if (row >= out) return;
    const int mode = (int) mode_ptr[0];
    const int nb   = in / MIX_QK;
    const uint8_t * rowbase = data + (int64_t) row * nb * MIX_BLOCK_BYTES;
    const float   * xc      = x + (int64_t) col * x_col_stride;
    // f32 accumulate of f32-dequantized weights * f32 activations. The dequant
    // is bit-exact vs the reference (validated in ~/p4-validate/hip). This is
    // slightly higher precision than the f16 dequant->cuBLAS fallback; on the
    // (underpowered, N=10) smoke suite the two land within a +/-2 greedy-flip
    // noise band, with every divergence a shared model-limited miss, a harness
    // answer-extraction artifact, or an HE formatting coin-flip -- no reasoning
    // regression. Kept f32 for simplicity/speed (no per-weight rounding ops).
    float acc = 0.0f;
    int blk = lane;
    // Main body: MIX_UNROLL blocks per iteration, each accumulated in stride
    // order into the single acc. The blocks' byte loads are independent (only
    // the acc-add chain is serial), so unrolling overlaps their loads while the
    // summation order stays identical. Guard keeps all 4 in range.
    for (; blk + 3 * MIX_WARP < nb; blk += MIX_UNROLL * MIX_WARP) {
        const int b0 = blk, b1 = blk + MIX_WARP;
        const int b2 = blk + 2 * MIX_WARP, b3 = blk + 3 * MIX_WARP;
        mix_block_accum(rowbase + (int64_t) b0 * MIX_BLOCK_BYTES, xc, b0 * MIX_QK, mode, s_lut, acc);
        mix_block_accum(rowbase + (int64_t) b1 * MIX_BLOCK_BYTES, xc, b1 * MIX_QK, mode, s_lut, acc);
        mix_block_accum(rowbase + (int64_t) b2 * MIX_BLOCK_BYTES, xc, b2 * MIX_QK, mode, s_lut, acc);
        mix_block_accum(rowbase + (int64_t) b3 * MIX_BLOCK_BYTES, xc, b3 * MIX_QK, mode, s_lut, acc);
    }
    // Remainder: fewer than MIX_UNROLL strided blocks left for this lane.
    for (; blk < nb; blk += MIX_WARP) {
        mix_block_accum(rowbase + (int64_t) blk * MIX_BLOCK_BYTES, xc, blk * MIX_QK, mode, s_lut, acc);
    }
    #pragma unroll
    for (int off = MIX_WARP/2; off > 0; off >>= 1) acc += mix_warp_shfl_down(acc, off);
    if (lane == 0) y[(int64_t) col * y_col_stride + row] = acc;
}

// ---- stream-sync-free fused MoE matvec (mul_mat_id) ----
// One warp per (output row, expert-slot, token). The expert index is read from
// the routing `ids` tensor ON DEVICE, so the whole qtype-105 mul_mat_id runs
// without the generic ggml_cuda_mul_mat_id fallback's host id-sort +
// cudaStreamSynchronize (which serialises decode AND blocks CUDA-graph capture
// of the FFN subgraph — the dominant cost of the wall-clock-timed ffn_compute).
// The per-output-row math is the SAME flat fold as mix_matvec_rocmfp3_kernel
// (identical mix_block_accum, identical summation order) for the resolved
// expert, so every output element is bit-for-bit identical to the per-expert
// slice path the fallback would take (the correctness gate hashes the greedy
// output, so any reassociation would flip a token).

// ---- fused DENSE 3-D-slice matvec (batched mul_mat over src0->ne[2]) ----
// The target's attention output projection reshapes attn_output_a to
// [group_dim, n_lora_o, n_out_group] and mul_mats it against a matching 3-D src1
// (deepseek4_graph.cpp:2122), so src1->ne[2] > 1 and the 2-D fused hook's
// `src1->ne[2] == 1` gate rejects it -- sending ~31% of the attention weight read to
// dequantize->cuBLAS, which reads MORE bytes than the f16 it replaces.
//
// This is the MoE kernel with one change: the slice index comes from blockIdx.y
// instead of a routing `ids[]` lookup. Everything else -- the data + slice*nb02
// stride, codebooks + slice*2*K, modes[slice], the two-rows-per-warp register
// blocking, the fixed ascending-j fold -- is unchanged, so each output element is
// bit-identical to the MoE path for the same weights.
__global__ void mix_matvec_rocmfp3_slice_kernel(
        const uint8_t * __restrict__ data, size_t nb02,
        const nv_bfloat16 * __restrict__ codebooks, const uint8_t * __restrict__ modes,
        const float * __restrict__ src1,
        float * __restrict__ dst, int in, int out, int ne11,
        int64_t src1_s1, int64_t src1_s2,     // element strides (float) over ne11, token
        int64_t dst_s1, int64_t dst_s2) {     // element strides (float) over slot, token
    const int warps_per_block = blockDim.x / MIX_WARP;
    const int warp  = blockIdx.x * warps_per_block + (threadIdx.x / MIX_WARP);
    const int row0  = warp * 2;                 // two output rows per warp
    const int lane  = threadIdx.x % MIX_WARP;
    const int slot  = blockIdx.y;               // DENSE: the slice index. It indexes
                                                // src1 AND dst as well as the weights --
                                                // zeroing it made every slice read slice-0
                                                // activations and overwrite slice-0 output.
    const int token = blockIdx.z;
    const bool two  = (row0 + 1) < out;         // false only for an odd-out tail warp
    // slot = blockIdx.y, token = blockIdx.z, so `expert` -- and hence the codebook,
    // mode and expert base -- is WORKGROUP-UNIFORM, which is what makes one LDS table
    // per block legal. Staged before the row0 early return so all threads sync.
    const int expert = slot;                    // DENSE: the weight slice is the grid slice,
                                                // not a routing lookup. This is the ONLY
                                                // change from the MoE kernel.
    __shared__ float s_lut[2 * MIX_K];
    if ((int) threadIdx.x < 2 * MIX_K) {
        s_lut[threadIdx.x] =
            __bfloat162float(codebooks[(int64_t) expert * 2 * MIX_K + threadIdx.x]);
    }
    __syncthreads();
    if (row0 >= out) return;
    const uint8_t     * edata   = data + (int64_t) expert * nb02;
    const int           mode    = (int) modes[expert];
    const int           nb      = in / MIX_QK;
    const uint8_t     * rowbase0 = edata + (int64_t) row0 * nb * MIX_BLOCK_BYTES;
    // For an odd tail warp (no row1) reuse row0's base so the loads stay in-bounds;
    // acc1 is simply never written. out=hidden is even here so `two` is uniformly
    // true across the whole warp (no divergence in the hot loop).
    const uint8_t     * rowbase1 = two ? edata + (int64_t) (row0 + 1) * nb * MIX_BLOCK_BYTES
                                       : rowbase0;
    // src1 is [in, ne11, ntok]; the get_rows-equivalent row for (slot, token)
    // is token*ne11 + slot%ne11 — i.e. token column + the slot%ne11 broadcast.
    const float * xcol = src1 + (int64_t) token * src1_s2 + (int64_t) (slot % ne11) * src1_s1;
    // Two output rows in one warp. Each row is folded by the SAME mix_block_accum
    // that the single-row path uses (byte-identical inlined body, same fixed j
    // order, same acc-add chain) so acc0/acc1 are bit-for-bit identical to the
    // single-row kernel's output for those rows. The two calls per block share the
    // same __restrict__ xcol + col0, so the compiler CSEs the strided activation
    // loads to one issue per element — halving activation LSU issue on this partly
    // load-instruction-bound matvec — WITHOUT reordering either row's summation.
    float acc0 = 0.0f, acc1 = 0.0f;
    int blk = lane;
    for (; blk + 3 * MIX_WARP < nb; blk += MIX_UNROLL * MIX_WARP) {
        const int b0 = blk, b1 = blk + MIX_WARP;
        const int b2 = blk + 2 * MIX_WARP, b3 = blk + 3 * MIX_WARP;
        mix_block_accum(rowbase0 + (int64_t) b0 * MIX_BLOCK_BYTES, xcol, b0 * MIX_QK, mode, s_lut, acc0);
        mix_block_accum(rowbase1 + (int64_t) b0 * MIX_BLOCK_BYTES, xcol, b0 * MIX_QK, mode, s_lut, acc1);
        mix_block_accum(rowbase0 + (int64_t) b1 * MIX_BLOCK_BYTES, xcol, b1 * MIX_QK, mode, s_lut, acc0);
        mix_block_accum(rowbase1 + (int64_t) b1 * MIX_BLOCK_BYTES, xcol, b1 * MIX_QK, mode, s_lut, acc1);
        mix_block_accum(rowbase0 + (int64_t) b2 * MIX_BLOCK_BYTES, xcol, b2 * MIX_QK, mode, s_lut, acc0);
        mix_block_accum(rowbase1 + (int64_t) b2 * MIX_BLOCK_BYTES, xcol, b2 * MIX_QK, mode, s_lut, acc1);
        mix_block_accum(rowbase0 + (int64_t) b3 * MIX_BLOCK_BYTES, xcol, b3 * MIX_QK, mode, s_lut, acc0);
        mix_block_accum(rowbase1 + (int64_t) b3 * MIX_BLOCK_BYTES, xcol, b3 * MIX_QK, mode, s_lut, acc1);
    }
    for (; blk < nb; blk += MIX_WARP) {
        mix_block_accum(rowbase0 + (int64_t) blk * MIX_BLOCK_BYTES, xcol, blk * MIX_QK, mode, s_lut, acc0);
        mix_block_accum(rowbase1 + (int64_t) blk * MIX_BLOCK_BYTES, xcol, blk * MIX_QK, mode, s_lut, acc1);
    }
    #pragma unroll
    for (int off = MIX_WARP/2; off > 0; off >>= 1) {
        acc0 += mix_warp_shfl_down(acc0, off);
        acc1 += mix_warp_shfl_down(acc1, off);
    }
    if (lane == 0) {
        dst[(int64_t) token * dst_s2 + (int64_t) slot * dst_s1 + row0] = acc0;
        if (two) dst[(int64_t) token * dst_s2 + (int64_t) slot * dst_s1 + row0 + 1] = acc1;
    }
}

// FUSE_GLU: see the twin in rocmfp2_mix.cu. Templated so both instantiations share ONE
// accumulation over ONE fixed block order -- each dot stays bit-identical to the unfused
// launch, so the fused result equals swiglu_ds4(unfused_gate, unfused_up) exactly.
template <bool FUSE_GLU>
__global__ void mix_matvec_rocmfp3_moe_kernel(
        const uint8_t * __restrict__ data, size_t nb02,
        const nv_bfloat16 * __restrict__ codebooks, const uint8_t * __restrict__ modes,
        const float * __restrict__ src1, const int32_t * __restrict__ ids,
        float * __restrict__ dst, int in, int out, int n_experts, int ne11,
        int64_t ids_s0, int64_t ids_s1,       // element strides (int32) over slot, token
        int64_t src1_s1, int64_t src1_s2,     // element strides (float) over ne11, token
        int64_t dst_s1, int64_t dst_s2,
        // FUSE_GLU only: the gate tensor's own registry entry. Not assumed equal to up's.
        const uint8_t * __restrict__ gdata, size_t gnb02,
        const nv_bfloat16 * __restrict__ gcodebooks, const uint8_t * __restrict__ gmodes,
        float glu_limit) {     // element strides (float) over slot, token
    const int warps_per_block = blockDim.x / MIX_WARP;
    const int warp  = blockIdx.x * warps_per_block + (threadIdx.x / MIX_WARP);
    const int row0  = warp * 2;                 // two output rows per warp
    const int lane  = threadIdx.x % MIX_WARP;
    const int slot  = blockIdx.y;
    const int token = blockIdx.z;
    const bool two  = (row0 + 1) < out;         // false only for an odd-out tail warp
    // slot = blockIdx.y, token = blockIdx.z, so `expert` -- and hence the codebook,
    // mode and expert base -- is WORKGROUP-UNIFORM, which is what makes one LDS table
    // per block legal. Staged before the row0 early return so all threads sync.
    const int expert = ids[(int64_t) token * ids_s1 + (int64_t) slot * ids_s0];
    // Routed ids come from the router's top-k, which is in-range by construction -- but this
    // kernel deliberately bypasses the generic host-side id sort that used to sit between the
    // routing tensor and the weights, so it must not turn a sentinel (-1), a padded batch
    // slot, or corrupted routing into an unchecked OOB read of codebooks/modes/data. Degrade
    // an invalid id to a ZERO contribution for this (token, slot): deterministic, and the
    // same thing a masked-out slot means. All threads still reach the barrier below.
    const bool bad_expert = expert < 0 || expert >= n_experts;
    __shared__ float s_lut[FUSE_GLU ? 4 * MIX_K : 2 * MIX_K];
    if (!bad_expert && (int) threadIdx.x < 2 * MIX_K) {
        s_lut[threadIdx.x] =
            __bfloat162float(codebooks[(int64_t) expert * 2 * MIX_K + threadIdx.x]);
    }
    if (FUSE_GLU) {
        const int gt = (int) threadIdx.x - 2 * MIX_K;
        if (!bad_expert && gt >= 0 && gt < 2 * MIX_K) {
            s_lut[2 * MIX_K + gt] =
                __bfloat162float(gcodebooks[(int64_t) expert * 2 * MIX_K + gt]);
        }
    }
    __syncthreads();
    if (row0 >= out) return;
    if (bad_expert) {
        if (lane == 0) {
            const int64_t o = (int64_t) token * dst_s2 + (int64_t) slot * dst_s1 + row0;
            dst[o] = 0.0f;
            if (row0 + 1 < out) dst[o + 1] = 0.0f;
        }
        return;
    }

    const uint8_t     * edata   = data + (int64_t) expert * nb02;
    const int           mode    = (int) modes[expert];
    const int           nb      = in / MIX_QK;
    const uint8_t     * rowbase0 = edata + (int64_t) row0 * nb * MIX_BLOCK_BYTES;
    // For an odd tail warp (no row1) reuse row0's base so the loads stay in-bounds;
    // acc1 is simply never written. out=hidden is even here so `two` is uniformly
    // true across the whole warp (no divergence in the hot loop).
    const uint8_t     * rowbase1 = two ? edata + (int64_t) (row0 + 1) * nb * MIX_BLOCK_BYTES
                                       : rowbase0;
    const uint8_t * gedata    = FUSE_GLU ? gdata + (int64_t) expert * gnb02 : nullptr;
    const int       gmode     = FUSE_GLU ? (int) gmodes[expert] : 0;
    const uint8_t * growbase0 = FUSE_GLU ? gedata + (int64_t) row0 * nb * MIX_BLOCK_BYTES : nullptr;
    const uint8_t * growbase1 = FUSE_GLU ? (two ? gedata + (int64_t) (row0 + 1) * nb * MIX_BLOCK_BYTES
                                                : growbase0) : nullptr;
    // src1 is [in, ne11, ntok]; the get_rows-equivalent row for (slot, token)
    // is token*ne11 + slot%ne11 — i.e. token column + the slot%ne11 broadcast.
    const float * xcol = src1 + (int64_t) token * src1_s2 + (int64_t) (slot % ne11) * src1_s1;
    // Two output rows in one warp. Each row is folded by the SAME mix_block_accum
    // that the single-row path uses (byte-identical inlined body, same fixed j
    // order, same acc-add chain) so acc0/acc1 are bit-for-bit identical to the
    // single-row kernel's output for those rows. The two calls per block share the
    // same __restrict__ xcol + col0, so the compiler CSEs the strided activation
    // loads to one issue per element — halving activation LSU issue on this partly
    // load-instruction-bound matvec — WITHOUT reordering either row's summation.
    float acc0 = 0.0f, acc1 = 0.0f;
    float gacc0 = 0.0f, gacc1 = 0.0f;   // same block order as acc*, so bit-identical per row
    int blk = lane;
    for (; blk + 3 * MIX_WARP < nb; blk += MIX_UNROLL * MIX_WARP) {
        const int b0 = blk, b1 = blk + MIX_WARP;
        const int b2 = blk + 2 * MIX_WARP, b3 = blk + 3 * MIX_WARP;
        mix_block_accum(rowbase0 + (int64_t) b0 * MIX_BLOCK_BYTES, xcol, b0 * MIX_QK, mode, s_lut, acc0);
        mix_block_accum(rowbase1 + (int64_t) b0 * MIX_BLOCK_BYTES, xcol, b0 * MIX_QK, mode, s_lut, acc1);
        if (FUSE_GLU) {
            mix_block_accum(growbase0 + (int64_t) b0 * MIX_BLOCK_BYTES, xcol, b0 * MIX_QK, gmode, s_lut + 2 * MIX_K, gacc0);
            mix_block_accum(growbase1 + (int64_t) b0 * MIX_BLOCK_BYTES, xcol, b0 * MIX_QK, gmode, s_lut + 2 * MIX_K, gacc1);
        }
        mix_block_accum(rowbase0 + (int64_t) b1 * MIX_BLOCK_BYTES, xcol, b1 * MIX_QK, mode, s_lut, acc0);
        mix_block_accum(rowbase1 + (int64_t) b1 * MIX_BLOCK_BYTES, xcol, b1 * MIX_QK, mode, s_lut, acc1);
        if (FUSE_GLU) {
            mix_block_accum(growbase0 + (int64_t) b1 * MIX_BLOCK_BYTES, xcol, b1 * MIX_QK, gmode, s_lut + 2 * MIX_K, gacc0);
            mix_block_accum(growbase1 + (int64_t) b1 * MIX_BLOCK_BYTES, xcol, b1 * MIX_QK, gmode, s_lut + 2 * MIX_K, gacc1);
        }
        mix_block_accum(rowbase0 + (int64_t) b2 * MIX_BLOCK_BYTES, xcol, b2 * MIX_QK, mode, s_lut, acc0);
        mix_block_accum(rowbase1 + (int64_t) b2 * MIX_BLOCK_BYTES, xcol, b2 * MIX_QK, mode, s_lut, acc1);
        if (FUSE_GLU) {
            mix_block_accum(growbase0 + (int64_t) b2 * MIX_BLOCK_BYTES, xcol, b2 * MIX_QK, gmode, s_lut + 2 * MIX_K, gacc0);
            mix_block_accum(growbase1 + (int64_t) b2 * MIX_BLOCK_BYTES, xcol, b2 * MIX_QK, gmode, s_lut + 2 * MIX_K, gacc1);
        }
        mix_block_accum(rowbase0 + (int64_t) b3 * MIX_BLOCK_BYTES, xcol, b3 * MIX_QK, mode, s_lut, acc0);
        mix_block_accum(rowbase1 + (int64_t) b3 * MIX_BLOCK_BYTES, xcol, b3 * MIX_QK, mode, s_lut, acc1);
        if (FUSE_GLU) {
            mix_block_accum(growbase0 + (int64_t) b3 * MIX_BLOCK_BYTES, xcol, b3 * MIX_QK, gmode, s_lut + 2 * MIX_K, gacc0);
            mix_block_accum(growbase1 + (int64_t) b3 * MIX_BLOCK_BYTES, xcol, b3 * MIX_QK, gmode, s_lut + 2 * MIX_K, gacc1);
        }
    }
    for (; blk < nb; blk += MIX_WARP) {
        mix_block_accum(rowbase0 + (int64_t) blk * MIX_BLOCK_BYTES, xcol, blk * MIX_QK, mode, s_lut, acc0);
        mix_block_accum(rowbase1 + (int64_t) blk * MIX_BLOCK_BYTES, xcol, blk * MIX_QK, mode, s_lut, acc1);
        if (FUSE_GLU) {
            mix_block_accum(growbase0 + (int64_t) blk * MIX_BLOCK_BYTES, xcol, blk * MIX_QK, gmode, s_lut + 2 * MIX_K, gacc0);
            mix_block_accum(growbase1 + (int64_t) blk * MIX_BLOCK_BYTES, xcol, blk * MIX_QK, gmode, s_lut + 2 * MIX_K, gacc1);
        }
    }
    #pragma unroll
    for (int off = MIX_WARP/2; off > 0; off >>= 1) {
        acc0 += mix_warp_shfl_down(acc0, off);
        acc1 += mix_warp_shfl_down(acc1, off);
        if (FUSE_GLU) {
            gacc0 += mix_warp_shfl_down(gacc0, off);
            gacc1 += mix_warp_shfl_down(gacc1, off);
        }
    }
    if (lane == 0) {
        const int64_t o = (int64_t) token * dst_s2 + (int64_t) slot * dst_s1 + row0;
        if (FUSE_GLU) {
            dst[o]              = ggml_cuda_op_swiglu_ds4_single(gacc0, acc0, glu_limit);
            if (two) dst[o + 1] = ggml_cuda_op_swiglu_ds4_single(gacc1, acc1, glu_limit);
        } else {
            dst[o]              = acc0;
            if (two) dst[o + 1] = acc1;
        }
    }
}

// Launch the sync-free MoE matvec for a qtype-105 mul_mat_id. Resolves the
// tensor's registry entry (base/nb02/codebooks/modes for ALL experts) from vx;
// the per-expert index is read on device. Returns false if the tensor is not
// registered (caller keeps the generic fallback). ne11 is the src1 broadcast
// dim (1 for decode). All *_s* are element strides (see kernel).
bool ggml_cuda_rocmfp3_mix_mul_mat_vec_3d(
        const void * vx, const float * src1, float * dst,
        int in, int out, int nslices, int ntokens,
        int64_t src1_token_stride, int64_t src1_slice_stride,
        int64_t dst_token_stride,  int64_t dst_slice_stride,
        cudaStream_t stream) {
    MixEntry e;
    int slice0;
    if (!mix_lookup(vx, e, slice0)) {
        return false;  // not registered -> caller keeps the dequant fallback
    }
    // The registry must cover every slice this launch will index via blockIdx.y.
    // Registering a 3-D dense tensor with n_experts < ne02 would silently read a
    // codebook belonging to another tensor, so refuse rather than corrupt.
    if (slice0 != 0 || e.n_experts < nslices) {
        return false;
    }
    const int warps_per_block = 2;
    const int threads = warps_per_block * MIX_WARP;
    const int rows_per_block = 2 * warps_per_block;
    dim3 grid((out + rows_per_block - 1) / rows_per_block, nslices, ntokens);
    // Stride mapping, and the reason this wrapper names its arguments by MEANING.
    // The kernel inherits the MoE indexing `src1 + token*src1_s2 + (slot%ne11)*src1_s1`
    // and `dst[token*dst_s2 + slot*dst_s1]`, i.e. **_s1 is the SLOT stride and _s2 the
    // TOKEN stride** -- the opposite of the ne[1]/ne[2] reading the names suggest. Passing
    // ggml's nb[1]/nb[2] straight through (and ne11=1) silently dropped the slice offset
    // entirely, because `slot % 1 == 0`, and produced completely wrong values that the
    // per-slice comparison test caught. ne11 must be >= nslices for `slot % ne11` to be
    // the identity on the slice index.
    mix_matvec_rocmfp3_slice_kernel<<<grid, dim3(threads), 0, stream>>>(
        (const uint8_t *) e.base, e.nb02, e.codebooks, e.modes,
        src1, dst, in, out, /*ne11=*/nslices,
        /*src1_s1=slot  */ src1_slice_stride, /*src1_s2=token*/ src1_token_stride,
        /*dst_s1 =slot  */ dst_slice_stride,  /*dst_s2 =token*/ dst_token_stride);
    return true;
}

bool ggml_cuda_rocmfp3_mix_mul_mat_id(
        const void * vx, const float * src1, const int32_t * ids, float * dst,
        int in, int out, int n_expert_used, int n_tokens, int ne11,
        int64_t ids_s0, int64_t ids_s1,
        int64_t src1_s1, int64_t src1_s2,
        int64_t dst_s1, int64_t dst_s2, cudaStream_t stream) {
    MixEntry e;
    int expert0;
    if (!mix_lookup(vx, e, expert0)) {
        return false;  // not registered -> caller falls back to sort + dequant
    }
    const int warps_per_block = 2;               // 64 threads (mirror the mmvq path)
    const int threads = warps_per_block * MIX_WARP;
    // Two output rows per warp (register-blocked activation reuse), so a workgroup
    // of `warps_per_block` warps covers 2*warps_per_block rows.
    const int rows_per_block = 2 * warps_per_block;
    dim3 grid((out + rows_per_block - 1) / rows_per_block, n_expert_used, n_tokens);
    mix_matvec_rocmfp3_moe_kernel<false><<<grid, dim3(threads), 0, stream>>>(
        (const uint8_t *) e.base, e.nb02, e.codebooks, e.modes,
        src1, ids, dst, in, out, e.n_experts, ne11,
        ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2,
        nullptr, 0, nullptr, nullptr, 0.0f);
    return true;
}

// Fused gate/up + SwiGLU for a qtype-105 expert pair. See the rocmfp2 twin: `vx_up` is the
// surviving mul_mat_id's src0 and `vx_gate` the collapsed one, because swiglu_ds4 applies silu
// to GATE and the operands are not interchangeable. Refuses unless BOTH halves are registered
// and shape-matched, so a half-registered pair keeps the correct unfused path instead of fusing
// a decoded tensor with an undecoded one -- which would load and emit fluent garbage.
bool ggml_cuda_rocmfp3_mix_mul_mat_id_glu(
        const void * vx_up, const void * vx_gate,
        const float * src1, const int32_t * ids, float * dst,
        int in, int out, int n_expert_used, int n_tokens, int ne11,
        int64_t ids_s0, int64_t ids_s1,
        int64_t src1_s1, int64_t src1_s2,
        int64_t dst_s1, int64_t dst_s2,
        float glu_limit, cudaStream_t stream) {
    MixEntry eu, eg;
    int expert0_u, expert0_g;
    if (!mix_lookup(vx_up, eu, expert0_u) || !mix_lookup(vx_gate, eg, expert0_g)) {
        return false;
    }
    if (eu.in != eg.in || eu.out != eg.out || eu.n_experts != eg.n_experts) {
        return false;
    }
    const int warps_per_block = 2;
    const int threads = warps_per_block * MIX_WARP;
    const int rows_per_block = 2 * warps_per_block;
    dim3 grid((out + rows_per_block - 1) / rows_per_block, n_expert_used, n_tokens);
    mix_matvec_rocmfp3_moe_kernel<true><<<grid, dim3(threads), 0, stream>>>(
        (const uint8_t *) eu.base, eu.nb02, eu.codebooks, eu.modes,
        src1, ids, dst, in, out, eu.n_experts, ne11,
        ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2,
        (const uint8_t *) eg.base, eg.nb02, eg.codebooks, eg.modes, glu_limit);
    return true;
}

bool ggml_cuda_rocmfp3_mix_registered(const void * vx) {
    MixEntry e;
    int expert;
    return mix_lookup(vx, e, expert);
}

bool ggml_cuda_rocmfp3_mix_mmq_info(
        const void * vx, const void ** codebooks, const uint8_t ** modes) {
    MixEntry e;
    int expert;
    if (!mix_lookup(vx, e, expert) || e.rotations != nullptr) {
        return false;
    }
    *codebooks = e.codebooks + (size_t) expert * 2 * MIX_K;
    *modes = e.modes + expert;
    return true;
}

bool ggml_cuda_rocmfp3_mix_mul_mat_vec(
        const void * vx, const float * x, float * y,
        int in, int out, int ncols,
        int64_t x_col_stride, int64_t y_col_stride, cudaStream_t stream) {
    MixEntry e;
    int expert;
    if (!mix_lookup(vx, e, expert)) {
        return false;  // not registered -> caller falls back to dequant->cuBLAS
    }
    // TODO(rotation): the current artifact is rotation-free (e.rotations all 0).
    // When block-Hadamard-rotated p3 experts land, fold H_32 here (per-expert
    // e.rotations[expert]) exactly as the dequant path will — same hook.
    const nv_bfloat16 * book = e.codebooks + (size_t) expert * 2 * 8;
    const uint8_t * mode_ptr = e.modes + expert;
    // Launch-config-only occupancy lever (bit-exact: one warp still owns one
    // row, per-row summation order unchanged). 2 warps/block (64 threads) is the
    // finest grouping that keeps a sibling warp per block for latency hiding
    // (1 warp/block regressed — attempt #1), while halving the workgroup size to
    // give the scheduler finer packing/tail balance on this BW-bound matvec.
    const int warps_per_block = 2;               // 64 threads
    const int threads = warps_per_block * MIX_WARP;
    dim3 grid((out + warps_per_block - 1) / warps_per_block, ncols, 1);
    mix_matvec_rocmfp3_kernel<<<grid, dim3(threads), 0, stream>>>(
        (const uint8_t *) vx, book, mode_ptr, x, y, in, out, x_col_stride, y_col_stride);
    return true;
}
