#pragma once

#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace luce::vision {

// The two checkpoints that carry this tower. They share every ViT and aligner
// operation; they differ in the language width the aligner projects into,
// the image span the decoder sees, and V4.1's lack of an image_pad row.
enum class VisionVariant {
    V4,    // DeepSeek-V4-Flash-Vision-Exp, projector schema 1
    V41,   // DeepSeek-V4.1-Flash, projector schema 2
};
struct VisionConfig {
    VisionVariant variant = VisionVariant::V4;
    int layers = 32, dimension = 1024, heads = 16, intermediate = 2816;
    int patch_size = 14, downsample = 3, language_dimension = 4096, vocabulary = 129280;
    int max_image_tokens = 384;
    float rope_theta = 10000.0f, rms_epsilon = 1e-6f;
    bool has_pad_sentinel = true;
    // Checkpoint commit the projector was exported from; empty when unknown.
    std::string source_revision;
};
// Image tokens an aligner grid of rows x columns costs in the decoder.
// V4: the row-pair layout with its padding and the three compression pads
// the source budget reserves. V4.1: [start] + (columns + newline) * rows + [end].
uint64_t image_block_tokens(VisionVariant variant, uint64_t rows, uint64_t columns);
// Why `projector` cannot serve this model, or "" when it can: the family must
// match (V4.1's projector is 5120 wide, V4 Vision's 4096), and so must the
// checkpoint when both files record it. An encoder is trained with its own
// checkpoint's language weights (DeepSeek's October update retrained both),
// so one from another checkpoint would load and quietly misread images.
std::string projector_mismatch(const VisionConfig & projector, bool model_is_v41,
                               const std::string & model_revision);
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
    // Reads the variant from the projector's schema version and refuses a
    // projector whose language width or vocabulary differs from the model's.
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
bool hip_rotary_capable(ggml_backend_t);
size_t hip_rotary_launches(ggml_backend_t);
bool hip_softmax_capable(ggml_backend_t);
size_t hip_softmax_launches(ggml_backend_t);
bool hip_av_capable(ggml_backend_t);
size_t hip_av_launches(ggml_backend_t);
// F32 values exactly representing BF16 inputs; HIP source order needs at least 16 rows.
ggml_tensor * rms_norm(ggml_context *, ggml_tensor * input, float epsilon, ggml_backend_t);
ggml_tensor * linear(ggml_context *, ggml_tensor * weight, ggml_tensor * input, ggml_tensor * bias,
                     bool preserve_biased_product, ggml_backend_t backend = nullptr);
void rotary_tables(PatchGrid grid, std::vector<float> & cosine, std::vector<float> & sine,
                   ggml_backend_t backend = nullptr);
ggml_tensor * rotate(ggml_context *, ggml_tensor *, ggml_tensor * cosine, ggml_tensor * sine);
ggml_tensor * attention(ggml_context *, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                        ggml_backend_t backend = nullptr);
// The same attention over query blocks of at most `block` rows: V4.1 images
// reach ~9.2K patches, whose full [N, N, heads] scores alone would take 5.4 GB.
// Each query row still sees every key, so the softmax is the full one; it is
// the source-order HIP kernel while N fits it (4096 keys), ggml's otherwise.
ggml_tensor * attention_blocked(ggml_context *, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                int64_t block, ggml_backend_t backend = nullptr);
// Query rows per block for N patches: the block's scores and probabilities
// stay under 512 MiB.
int64_t attention_query_block(int64_t patches);
ggml_tensor * unfold(ggml_context *, ggml_tensor *, PatchGrid, int channels);
}
} // namespace luce::vision
