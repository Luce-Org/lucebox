#pragma once

#include "ggml-backend.h"
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace dflash::vision {

struct VisionConfig {
    int layers = 32, dimension = 1024, heads = 16, intermediate = 2816;
    int patch_size = 14, downsample = 3, language_dimension = 4096, vocabulary = 129280;
    int max_image_tokens = 384;
    float rope_theta = 10000.0f, rms_epsilon = 1e-6f;
};
struct PatchGrid { int height, width; };
enum class Sentinel { Start, Pad, Newline, End };
struct VisionOutput {
    int rows = 0, columns = 0;
    std::vector<float> embeddings;
    // Optional raster [height * width, 1024] qualification output.
    std::vector<float> features;
};
// Callback receives row-major values with the GGML shape (fastest axis first).
// It is synchronous: retaining diagnostics is the caller's responsibility.
using StageObserver = std::function<void(const std::string &, const std::vector<int64_t> &,
                                         const std::vector<float> &)>;

// Owns weights and reusable graph allocator; borrows the backend, which must
// outlive this object. Sequential use only. No image decoding or token layout.
class VisionRuntime {
public:
    VisionRuntime();
    ~VisionRuntime();
    VisionRuntime(const VisionRuntime &) = delete;
    VisionRuntime & operator=(const VisionRuntime &) = delete;
    bool load(const std::string & path, ggml_backend_t backend,
              int language_dimension, int vocabulary, std::string & error);
    bool encode(const std::vector<float> & channel_major_patches, PatchGrid grid,
                VisionOutput & output, std::string & error, bool retain_features = false,
                const StageObserver & observer = {});
    bool sentinel(Sentinel identity, std::vector<float> & output, std::string & error) const;
    // Releases graph arena; backend-owned HIP Lt workspace remains retained.
    void release_scratch();
    const VisionConfig * config() const;
    size_t weight_bytes() const;
    // Conservative graph arena + external HIP workspace reservation.
    size_t scratch_bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The same geometry primitives used by the runtime and standalone unit tests.
namespace detail {
// HIP registry capability; zero on CPU/NVIDIA or an unavailable backend.
size_t hip_bias_workspace(ggml_backend_t);
size_t hip_bias_launches(ggml_backend_t);
bool hip_norm_capable(ggml_backend_t);
size_t hip_norm_launches(ggml_backend_t);
// F32 values exactly representing BF16 inputs; HIP source order needs at least 16 rows.
ggml_tensor * rms_norm(ggml_context *, ggml_tensor * input, float epsilon, ggml_backend_t);
ggml_tensor * linear(ggml_context *, ggml_tensor * weight, ggml_tensor * input, ggml_tensor * bias,
                     bool preserve_biased_product, ggml_backend_t backend = nullptr);
void rotary_tables(PatchGrid grid, std::vector<float> & cosine, std::vector<float> & sine);
ggml_tensor * rotate(ggml_context *, ggml_tensor *, ggml_tensor * cosine, ggml_tensor * sine);
ggml_tensor * attention(ggml_context *, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v);
ggml_tensor * unfold(ggml_context *, ggml_tensor *, PatchGrid, int channels);
}
} // namespace dflash::vision
