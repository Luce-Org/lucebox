#include "qsa-ids.cuh"

#include <climits>

// One CTA per query; only selected ids are sorted, never scores. No warp-size or ISA assumptions.
static __global__ void qsa_decode_ids(
        const char * blocks, const size_t row_stride, const int32_t * positions, int32_t * ids,
        const int budget, const int ratio, const int sort_size) {
    extern __shared__ int32_t sorted[];
    const int tid = threadIdx.x;
    const int32_t * row = (const int32_t *) (blocks + (size_t) blockIdx.x * row_stride);
    for (int j = tid; j < sort_size; j += blockDim.x) {
        sorted[j] = j < budget ? row[j] : INT_MAX;
    }
    __syncthreads();
    for (int width = 2; width <= sort_size; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            for (int j = tid; j < sort_size; j += blockDim.x) {
                const int peer = j ^ stride;
                if (peer > j) {
                    const int32_t a = sorted[j], b = sorted[peer];
                    if ((j & width) == 0 ? a > b : a < b) {
                        sorted[j] = b;
                        sorted[peer] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    const int n_ids = budget * ratio + ratio - 1;
    int32_t * out = ids + (size_t) blockIdx.x * n_ids;
    const int64_t pos = positions[blockIdx.x];
    const int64_t br = ((pos + 1) / ratio) * ratio;
    for (int j = tid; j < n_ids; j += blockDim.x) {
        if (j < budget * ratio) {
            const int64_t first = (int64_t) ratio * sorted[j / ratio];
            out[j] = first + ratio - 1 <= pos ? (int32_t) (first + j % ratio) : -1;
        } else {
            const int64_t cell = br + j - budget * ratio;
            out[j] = cell <= pos ? (int32_t) cell : -1;
        }
    }
}

void ggml_cuda_op_qsa_decode_ids(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * blocks = dst->src[0];
    const int budget = (int) blocks->ne[0];
    const int ratio = ggml_get_op_params_i32(dst, 0);
    int sort_size = 1;
    while (sort_size < budget) {
        sort_size <<= 1;
    }
    qsa_decode_ids<<<blocks->ne[1], 256, sort_size * sizeof(int32_t), ctx.stream()>>>(
        (const char *) blocks->data, blocks->nb[1], (const int32_t *) dst->src[1]->data,
        (int32_t *) dst->data, budget, ratio, sort_size);
}
