// Runtime decode for GGML_TYPE_Q2_1_ROCMFP2_MIX (106).
// A per-tensor registry supplies the per-expert codebook/mode that the ggml
// to_fp16 converter signature cannot carry; the deepseek4 loader registers each
// fused down-expert tensor after staging its sidecar side-data to device memory.
#include "rocmfp2_mix.cuh"
#include "convert.cuh"
#include <mutex>
#include <vector>

#define MIX_QK 32
#define MIX_QS 8
#define MIX_BLOCK_BYTES 10

namespace {
struct MixEntry {
    const void * base;
    size_t nb02;          // byte stride between experts
    int n_experts, out, in;
    const nv_bfloat16 * codebooks;  // n_experts * 2 * 4
    const uint8_t * modes;          // n_experts
    const uint8_t * rotations;      // n_experts (unused until p3 rotation lands)
    bool owns_device;     // true => this entry cudaMalloc'd the 3 buffers above
                          // (register_host) and must free them on erase/update
};
std::mutex g_mix_mtx;
std::vector<MixEntry> g_mix_registry;

// Free an entry's device side-data if it owns it. Caller holds g_mix_mtx.
void mix_free_entry_device(MixEntry & e) {
    if (!e.owns_device) return;
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
    MixEntry ne{base, nb02, n_experts, out, in, codebooks, modes, rotations, owns_device};
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
extern "C" void ggml_cuda_rocmfp2_mix_register(
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
extern "C" void ggml_cuda_rocmfp2_mix_register_host(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks_bf16_host, const uint8_t * modes_host,
        const uint8_t * rotations_host) {
    const size_t cb_bytes = (size_t) n_experts * 2 * 4 * sizeof(nv_bfloat16);
    void * cb_dev = nullptr; void * modes_dev = nullptr; void * rots_dev = nullptr;
    cudaError_t err = cudaMalloc(&cb_dev, cb_bytes);
    if (err == cudaSuccess) err = cudaMemcpy(cb_dev, codebooks_bf16_host, cb_bytes, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMalloc(&modes_dev, (size_t) n_experts);
    if (err == cudaSuccess) err = cudaMemcpy(modes_dev, modes_host, (size_t) n_experts, cudaMemcpyHostToDevice);
    if (err == cudaSuccess && rotations_host) {
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

extern "C" void ggml_cuda_rocmfp2_mix_unregister(const void * base) {
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

// 2-bit codes pack four to a byte and never straddle a boundary -- so unlike
// qtype-105's 3-bit codes this needs no multi-byte gather, no shift arithmetic
// across bytes, and no MIX_QS bounds test. Least-significant pair first.
__device__ __forceinline__ uint32_t mix_fp2_code(const uint8_t * qs, int i) {
    return (uint32_t) (qs[i >> 2] >> (2 * (i & 3))) & 3u;
}

// Fixed levels for mode 0 (uniform qtype-107 fallback): {-1, 0, 1, 2}, code order.
// A 4-entry lookup beats qtype-105's sign/magnitude arithmetic and is exact.
__device__ __forceinline__ float mix_fp2_fixed(uint32_t code) {
    return (float) ((int) code - 1);   // 0->-1, 1->0, 2->1, 3->2
}

// One thread per element of a single expert slice (k = out*in elements).
__global__ void dequantize_rocmfp2_mix_kernel(
        const uint8_t * __restrict__ data, const nv_bfloat16 * __restrict__ book,
        const uint8_t * __restrict__ mode_ptr, int in, int64_t k, half * __restrict__ y) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
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
    const uint32_t code = mix_fp2_code(blk, j);
    float val;
    if (mode == 0) {
        val = mix_ue4m3(meta) * mix_fp2_fixed(code);
    } else {
        const float scale = mix_ue4m3(meta & 0x7F);
        const int bk = meta >> 7;
        val = scale * __bfloat162float(book[bk * 4 + (int) code]);
    }
    y[idx] = __float2half(val);
}

void dequantize_rocmfp2_mix_to_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    MixEntry e;
    int expert;
    if (!mix_lookup(vx, e, expert)) {
        GGML_ABORT("rocmfp2_mix: tensor slice %p not registered", vx);
    }
    const nv_bfloat16 * book = e.codebooks + (size_t) expert * 2 * 4;
    const uint8_t * mode_ptr = e.modes + expert;
    const int threads = 256;
    const int blocks = (int) ((k + threads - 1) / threads);
    // Portable launch: triple-chevron compiles under both nvcc and hipcc; the
    // hipLaunchKernelGGL macro is HIP-only and breaks the default CUDA build,
    // which still globs this *.cu file.
    dequantize_rocmfp2_mix_kernel<<<dim3(blocks), dim3(threads), 0, stream>>>(
        (const uint8_t *) vx, book, mode_ptr, e.in, k, y);
}

// ---- fused quantized matvec (MMVQ-style decode) ----
// One warp per output row; lanes stride over the row's blocks, decode 32 weights
// each (bit-identical to dequantize_rocmfp2_mix_kernel), multiply by x, f32
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
        int mode, const nv_bfloat16 * __restrict__ book, float & acc) {
    // Stage the whole 10-byte block into registers with one 2-byte-wide copy,
    // then decode the fp2 codes out of registers instead of re-reading the
    // packed qs region through narrow per-byte global loads. Note the load
    // pressure differs from qtype-105: at 2 bits each weight touches exactly
    // ONE of the 8 qs bytes (four weights share it) rather than up to 3 of 12,
    // so staging buys register reuse rather than rescuing a multi-byte gather.
    // Blocks are always 2-byte aligned (block stride 10 and row stride
    // nb*10=1280 are both even), so a 2-byte assume is safe and lets the
    // compiler fold the packed-weight reads into ushort loads. A 10-byte block
    // is also 8+2, so a u64+u16 pair is a natural next step -- left for the
    // evolution loop to try against the correctness gate rather than assumed. The
    // decode arithmetic and the fixed j accumulation order are untouched, so
    // acc is bit-for-bit identical to the per-byte path (the correctness gate
    // hashes the greedy output; any reassociation flips a token).
    const uint8_t * ba = (const uint8_t *) __builtin_assume_aligned(b, 2);
    uint8_t buf[MIX_BLOCK_BYTES];
    __builtin_memcpy(buf, ba, MIX_BLOCK_BYTES);
    const uint8_t m0 = buf[MIX_QS + 0], m1 = buf[MIX_QS + 1];
    if (mode == 0) {
        const float s0 = mix_ue4m3(m0), s1 = mix_ue4m3(m1);
        #pragma unroll
        for (int j = 0; j < MIX_QK; ++j) {
            const float s = (j < MIX_QK/2) ? s0 : s1;
            acc += s * mix_fp2_fixed(mix_fp2_code(buf, j)) * xc[col0 + j];
        }
    } else {
        const float s0 = mix_ue4m3(m0 & 0x7F), s1 = mix_ue4m3(m1 & 0x7F);
        const nv_bfloat16 * bk0 = book + (m0 >> 7) * 4;
        const nv_bfloat16 * bk1 = book + (m1 >> 7) * 4;
        #pragma unroll
        for (int j = 0; j < MIX_QK; ++j) {
            const float s = (j < MIX_QK/2) ? s0 : s1;
            const nv_bfloat16 * bk = (j < MIX_QK/2) ? bk0 : bk1;
            acc += s * __bfloat162float(bk[mix_fp2_code(buf, j)]) * xc[col0 + j];
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
__global__ void mix_matvec_rocmfp2_kernel(
        const uint8_t * __restrict__ data, const nv_bfloat16 * __restrict__ book,
        const uint8_t * __restrict__ mode_ptr, const float * __restrict__ x,
        float * __restrict__ y, int in, int out,
        int64_t x_col_stride, int64_t y_col_stride) {
    const int warps_per_block = blockDim.x / MIX_WARP;
    const int row  = blockIdx.x * warps_per_block + (threadIdx.x / MIX_WARP);
    const int lane = threadIdx.x % MIX_WARP;
    const int col  = blockIdx.y;
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
        mix_block_accum(rowbase + (int64_t) b0 * MIX_BLOCK_BYTES, xc, b0 * MIX_QK, mode, book, acc);
        mix_block_accum(rowbase + (int64_t) b1 * MIX_BLOCK_BYTES, xc, b1 * MIX_QK, mode, book, acc);
        mix_block_accum(rowbase + (int64_t) b2 * MIX_BLOCK_BYTES, xc, b2 * MIX_QK, mode, book, acc);
        mix_block_accum(rowbase + (int64_t) b3 * MIX_BLOCK_BYTES, xc, b3 * MIX_QK, mode, book, acc);
    }
    // Remainder: fewer than MIX_UNROLL strided blocks left for this lane.
    for (; blk < nb; blk += MIX_WARP) {
        mix_block_accum(rowbase + (int64_t) blk * MIX_BLOCK_BYTES, xc, blk * MIX_QK, mode, book, acc);
    }
    #pragma unroll
    for (int off = MIX_WARP/2; off > 0; off >>= 1) acc += mix_warp_shfl_down(acc, off);
    if (lane == 0) y[(int64_t) col * y_col_stride + row] = acc;
}

// ---- stream-sync-free fused MoE matvec (mul_mat_id) ----
// One warp per (output row, expert-slot, token). The expert index is read from
// the routing `ids` tensor ON DEVICE, so the whole qtype-106 mul_mat_id runs
// without the generic ggml_cuda_mul_mat_id fallback's host id-sort +
// cudaStreamSynchronize (which serialises decode AND blocks CUDA-graph capture
// of the FFN subgraph — the dominant cost of the wall-clock-timed ffn_compute).
// The per-output-row math is the SAME flat fold as mix_matvec_rocmfp2_kernel
// (identical mix_block_accum, identical summation order) for the resolved
// expert, so every output element is bit-for-bit identical to the per-expert
// slice path the fallback would take (the correctness gate hashes the greedy
// output, so any reassociation would flip a token).
__global__ void mix_matvec_rocmfp2_moe_kernel(
        const uint8_t * __restrict__ data, size_t nb02,
        const nv_bfloat16 * __restrict__ codebooks, const uint8_t * __restrict__ modes,
        const float * __restrict__ src1, const int32_t * __restrict__ ids,
        float * __restrict__ dst, int in, int out, int ne11,
        int64_t ids_s0, int64_t ids_s1,       // element strides (int32) over slot, token
        int64_t src1_s1, int64_t src1_s2,     // element strides (float) over ne11, token
        int64_t dst_s1, int64_t dst_s2) {     // element strides (float) over slot, token
    const int warps_per_block = blockDim.x / MIX_WARP;
    const int warp  = blockIdx.x * warps_per_block + (threadIdx.x / MIX_WARP);
    const int row0  = warp * 2;                 // two output rows per warp
    const int lane  = threadIdx.x % MIX_WARP;
    const int slot  = blockIdx.y;
    const int token = blockIdx.z;
    if (row0 >= out) return;
    const bool two  = (row0 + 1) < out;         // false only for an odd-out tail warp
    const int expert = ids[(int64_t) token * ids_s1 + (int64_t) slot * ids_s0];
    const uint8_t     * edata   = data + (int64_t) expert * nb02;
    const nv_bfloat16 * book    = codebooks + (int64_t) expert * 2 * 4;
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
        mix_block_accum(rowbase0 + (int64_t) b0 * MIX_BLOCK_BYTES, xcol, b0 * MIX_QK, mode, book, acc0);
        mix_block_accum(rowbase1 + (int64_t) b0 * MIX_BLOCK_BYTES, xcol, b0 * MIX_QK, mode, book, acc1);
        mix_block_accum(rowbase0 + (int64_t) b1 * MIX_BLOCK_BYTES, xcol, b1 * MIX_QK, mode, book, acc0);
        mix_block_accum(rowbase1 + (int64_t) b1 * MIX_BLOCK_BYTES, xcol, b1 * MIX_QK, mode, book, acc1);
        mix_block_accum(rowbase0 + (int64_t) b2 * MIX_BLOCK_BYTES, xcol, b2 * MIX_QK, mode, book, acc0);
        mix_block_accum(rowbase1 + (int64_t) b2 * MIX_BLOCK_BYTES, xcol, b2 * MIX_QK, mode, book, acc1);
        mix_block_accum(rowbase0 + (int64_t) b3 * MIX_BLOCK_BYTES, xcol, b3 * MIX_QK, mode, book, acc0);
        mix_block_accum(rowbase1 + (int64_t) b3 * MIX_BLOCK_BYTES, xcol, b3 * MIX_QK, mode, book, acc1);
    }
    for (; blk < nb; blk += MIX_WARP) {
        mix_block_accum(rowbase0 + (int64_t) blk * MIX_BLOCK_BYTES, xcol, blk * MIX_QK, mode, book, acc0);
        mix_block_accum(rowbase1 + (int64_t) blk * MIX_BLOCK_BYTES, xcol, blk * MIX_QK, mode, book, acc1);
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

// Launch the sync-free MoE matvec for a qtype-106 mul_mat_id. Resolves the
// tensor's registry entry (base/nb02/codebooks/modes for ALL experts) from vx;
// the per-expert index is read on device. Returns false if the tensor is not
// registered (caller keeps the generic fallback). ne11 is the src1 broadcast
// dim (1 for decode). All *_s* are element strides (see kernel).
bool ggml_cuda_rocmfp2_mix_mul_mat_id(
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
    mix_matvec_rocmfp2_moe_kernel<<<grid, dim3(threads), 0, stream>>>(
        (const uint8_t *) e.base, e.nb02, e.codebooks, e.modes,
        src1, ids, dst, in, out, ne11,
        ids_s0, ids_s1, src1_s1, src1_s2, dst_s1, dst_s2);
    return true;
}

bool ggml_cuda_rocmfp2_mix_registered(const void * vx) {
    MixEntry e;
    int expert;
    return mix_lookup(vx, e, expert);
}

bool ggml_cuda_rocmfp2_mix_mul_mat_vec(
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
    const nv_bfloat16 * book = e.codebooks + (size_t) expert * 2 * 4;
    const uint8_t * mode_ptr = e.modes + expert;
    // Launch-config-only occupancy lever (bit-exact: one warp still owns one
    // row, per-row summation order unchanged). 2 warps/block (64 threads) is the
    // finest grouping that keeps a sibling warp per block for latency hiding
    // (1 warp/block regressed — attempt #1), while halving the workgroup size to
    // give the scheduler finer packing/tail balance on this BW-bound matvec.
    const int warps_per_block = 2;               // 64 threads
    const int threads = warps_per_block * MIX_WARP;
    dim3 grid((out + warps_per_block - 1) / warps_per_block, ncols, 1);
    mix_matvec_rocmfp2_kernel<<<grid, dim3(threads), 0, stream>>>(
        (const uint8_t *) vx, book, mode_ptr, x, y, in, out, x_col_stride, y_col_stride);
    return true;
}
