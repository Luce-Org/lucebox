// MOSS-Image text to image on one RDNA4 GPU: the Qwen3.5-9B text encoder, the int8 DiT sampling loop and the FLUX.2
// VAE decoder, all on one stream; the text encoder can run on a second GPU instead (text_encoder_gpu).
//
// A request with CFG runs the conditional and the unconditional sequence as one batch of rows through every GEMM;
// only attention and the rotary tables stay per sequence. The text side of a prompt (encoder, projection and the two
// text blocks) does not depend on the step, so it runs once per prompt and the last prompt and the negative prompt
// stay cached. Buffers are sized for the largest request so far and reused.
//
// In standby the GPU is shared with another model between requests: the weights wait in pinned host memory, and the
// GPU holds them only while it has room (load_weights()) or a request runs. A request without them on the GPU starts
// its first denoising step while they arrive, each layer waiting for its own weights only.

#pragma once

#include "image_server.h"

#include <memory>
#include <string>

namespace luce::moss_image {

struct PipelineOptions {
    std::string model_path;         // the moss-image GGUF: DiT and VAE (scripts/convert_moss_image.py)
    std::string text_encoder_path;  // the Qwen3.5-9B GGUF
    std::string tuning_path;        // the GEMM kernels tuned by an earlier run (Blas::save_tuning); empty: none
    int         text_encoder_gpu = -1;  // the GPU the text encoder runs on; -1: the model's
    int         warmup_width  = 1024;
    int         warmup_height = 1024;
    bool        standby       = false;  // the weights wait in host memory (see above)
};

class Pipeline final : public Generator {
public:
    Pipeline();
    ~Pipeline() override;
    Pipeline(const Pipeline &) = delete;
    Pipeline & operator=(const Pipeline &) = delete;

    // Weights on the GPU, so that the first request pays no GEMM tuning: the GEMM kernels from the tuning file, loaded
    // beside the uploads, or when it has none, a warm-up image at the warm-up size that tunes them and writes the file.
    // In standby the weights go to host memory instead, and a missing tuning file is made before the first request.
    // False with `error` set.
    bool load(const PipelineOptions & options, std::string & error);

    bool generate(const ImageRequest & request, RgbImage & image, std::string & error) override;

    MemoryState memory() override;
    bool        release_memory(bool keep_weights, std::string & error) override;
    bool        load_weights(const std::function<bool()> & stopped, std::string & error) override;

private:
    // One image with and one without CFG at the warm-up size, which tune every GEMM a request runs.
    bool warm_up(std::string & error);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace luce::moss_image
