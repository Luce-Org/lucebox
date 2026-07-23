// Runtime decode for GGML_TYPE_Q3_1_ROCMFP3_MIX (105).
// A per-tensor registry supplies the per-expert codebook/mode that the ggml
// to_fp16 converter signature cannot carry; the deepseek4 loader registers each
// fused down-expert tensor after staging its sidecar side-data to device memory.
#include "rocmfp3_mix.cuh"
#include "convert.cuh"
#include <mutex>
#include <vector>

#define MIX_QK 32
#define MIX_QS 12
#define MIX_BLOCK_BYTES 14

namespace {
struct MixEntry {
    const void * base;
    size_t nb02;          // byte stride between experts
    int n_experts, out, in;
    const nv_bfloat16 * codebooks;  // n_experts * 2 * 8
    const uint8_t * modes;          // n_experts
    const uint8_t * rotations;      // n_experts (unused until p3 rotation lands)
};
std::mutex g_mix_mtx;
std::vector<MixEntry> g_mix_registry;
}  // namespace

extern "C" void ggml_cuda_rocmfp3_mix_register(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks, const void * modes, const void * rotations) {
    std::lock_guard<std::mutex> lk(g_mix_mtx);
    for (auto & e : g_mix_registry) {
        if (e.base == base) {  // update in place
            e = MixEntry{base, nb02, n_experts, out, in,
                         (const nv_bfloat16 *) codebooks,
                         (const uint8_t *) modes, (const uint8_t *) rotations};
            return;
        }
    }
    g_mix_registry.push_back(MixEntry{base, nb02, n_experts, out, in,
        (const nv_bfloat16 *) codebooks, (const uint8_t *) modes,
        (const uint8_t *) rotations});
}

// Host-side convenience for the deepseek4 loader: stage per-expert codebooks
// (bf16) and modes from host memory into device buffers (model-lifetime), then
// register. rotations host array optional (nullptr => none rotated).
extern "C" void ggml_cuda_rocmfp3_mix_register_host(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks_bf16_host, const uint8_t * modes_host,
        const uint8_t * rotations_host) {
    const size_t cb_bytes = (size_t) n_experts * 2 * 8 * sizeof(nv_bfloat16);
    void * cb_dev = nullptr; void * modes_dev = nullptr; void * rots_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&cb_dev, cb_bytes));
    CUDA_CHECK(cudaMemcpy(cb_dev, codebooks_bf16_host, cb_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&modes_dev, (size_t) n_experts));
    CUDA_CHECK(cudaMemcpy(modes_dev, modes_host, (size_t) n_experts, cudaMemcpyHostToDevice));
    if (rotations_host) {
        CUDA_CHECK(cudaMalloc(&rots_dev, (size_t) n_experts));
        CUDA_CHECK(cudaMemcpy(rots_dev, rotations_host, (size_t) n_experts, cudaMemcpyHostToDevice));
    }
    ggml_cuda_rocmfp3_mix_register(base, nb02, n_experts, out, in, cb_dev, modes_dev, rots_dev);
}

extern "C" void ggml_cuda_rocmfp3_mix_unregister(const void * base) {
    std::lock_guard<std::mutex> lk(g_mix_mtx);
    for (size_t i = 0; i < g_mix_registry.size(); ++i) {
        if (g_mix_registry[i].base == base) {
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
        val = scale * __bfloat162float(book[bk * 8 + (int) code]);
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
    hipLaunchKernelGGL(dequantize_rocmfp3_mix_kernel, dim3(blocks), dim3(threads), 0, stream,
        (const uint8_t *) vx, book, mode_ptr, e.in, k, y);
}
