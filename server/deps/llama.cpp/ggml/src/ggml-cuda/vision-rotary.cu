#include "vision-rotary.cuh"
#include "norm.cuh"
#include <cmath>
#include <limits>

#if defined(GGML_USE_HIP)
// PyTorch 3d3aa833db84eed6b7f5595cb5f162c2f78300a4, aten/src/ATen/native/cuda/:
// Pow.cuh uses global ::pow on Linux; UnaryFractionKernels.cu uses float 1/a;
// UnaryGeometric{Cos,Sin}Kernel.cu use global ::cos and ::sin. The five kernels
// preserve eager F32 materialization between power, reciprocal, angle and trig.
static __global__ void vision_rotary_power(float * output, float base) {
    static_assert(sizeof(decltype(::pow(0.0f, 0.0f))) == sizeof(float), "source pow overload changed");
    const int j = threadIdx.x;
    if (j < 16) {
        output[j] = ::pow(base, float(2*j)*(1.0f/32.0f));
    }
}

static __global__ void vision_rotary_inverse(const float * power, float * inverse) {
    const int j = threadIdx.x;
    if (j < 16) {
        inverse[j] = 1.0f/power[j];
    }
}

static __global__ void vision_rotary_angle(const float * inverse, float * angle, int count, int width) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < count) {
        const int row = i/32, j = i%32;
        angle[i] = float(j < 16 ? row/width : row%width)*inverse[j%16];
    }
}

static __global__ void vision_rotary_cos(const float * angle, float * cosine, int count) {
    static_assert(sizeof(decltype(::cos(0.0f))) == sizeof(float), "source cos overload changed");
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < count) {
        cosine[i] = ::cos(angle[i]);
    }
}

static __global__ void vision_rotary_sin(const float * angle, float * sine, int count) {
    static_assert(sizeof(decltype(::sin(0.0f))) == sizeof(float), "source sin overload changed");
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < count) {
        sine[i] = ::sin(angle[i]);
    }
}

namespace {
struct rotary_temporary {
    float * data = nullptr;
    cudaStream_t stream;

    rotary_temporary(size_t bytes, cudaStream_t stream) : stream(stream) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data), bytes));
    }
    ~rotary_temporary() {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(data));
    }
    rotary_temporary(const rotary_temporary &) = delete;
    rotary_temporary & operator=(const rotary_temporary &) = delete;
};
} // namespace

bool ggml_hip_vision_rotary(ggml_backend_cuda_context & ctx, int64_t height, int64_t width,
                          float * cosine, float * sine) {
    if (!ggml_hip_vision_norm_capable(ctx.device) || !cosine || !sine || cosine == sine ||
        height <= 0 || width <= 0 || height > 1152 || width > 1152) {
        return false;
    }
    const size_t h = static_cast<size_t>(height), w = static_cast<size_t>(width);
    constexpr size_t cap = 128ULL*1024*1024;
    constexpr size_t workspace = 76ULL*1024*1024;
    constexpr size_t fixed_bytes = 2*16*sizeof(float);
    constexpr size_t bytes_per_row = 3*32*sizeof(float);
    if (h > std::numeric_limits<size_t>::max()/w) {
        return false;
    }
    const size_t rows = h*w;
    if (rows > (cap - workspace - fixed_bytes)/bytes_per_row) {
        return false;
    }
    const size_t bytes = fixed_bytes + rows*bytes_per_row;
    const int count = static_cast<int>(rows*32);
    ggml_cuda_set_device(ctx.device);
    const auto stream = ctx.stream();
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        return false;
    }
    {
        rotary_temporary temporary(bytes, stream);
        float * power = temporary.data;
        float * inverse = power + 16;
        float * angle = inverse + 16;
        float * cos = angle + count;
        float * sin = cos + count;
        vision_rotary_power<<<1, 32, 0, stream>>>(power, 10000.0f);
        CUDA_CHECK(cudaGetLastError());
        vision_rotary_inverse<<<1, 32, 0, stream>>>(power, inverse);
        CUDA_CHECK(cudaGetLastError());
        vision_rotary_angle<<<(count + 255)/256, 256, 0, stream>>>(inverse, angle, count, int(width));
        CUDA_CHECK(cudaGetLastError());
        vision_rotary_cos<<<(count + 255)/256, 256, 0, stream>>>(angle, cos, count);
        CUDA_CHECK(cudaGetLastError());
        vision_rotary_sin<<<(count + 255)/256, 256, 0, stream>>>(angle, sin, count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(cosine, cos, count*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(sine, sin, count*sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    ++ctx.vision_rotary_launches;
    return true;
}
#endif
