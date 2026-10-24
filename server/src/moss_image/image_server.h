// The MOSS-Image HTTP server: the OpenAI images API as stable-diffusion.cpp's sd-server answers it, so clients written
// for sd-server keep working.
//
//   GET  /v1/models
//   POST /v1/images/generations  {prompt, n, size "WxH", output_format "png", output_compression}
//   POST /sdapi/v1/txt2img       {prompt, negative_prompt, width, height, steps, cfg_scale, seed, batch_size,
//                                 sampler_name, scheduler}
//
// A server started with --standby also gives its GPU memory back between requests, for another model to use the GPU
// (409 when it cannot, with the reason):
//
//   GET  /memory                 {standby, weights "gpu" or "host", gpu_mib, host_mib, gpu_free_mib}
//   POST /memory/release         {keep_weights}: frees the work buffers, and the weights unless keep_weights
//   POST /memory/load            copies the weights back onto the GPU when it has room for them; a later release
//                                stops it, also while it waits for the GPU
//
// A prompt may carry sd-server's per-request options as JSON between <sd_cpp_extra_args> and
// </sd_cpp_extra_args>: seed, negative_prompt, width, height, batch_count and sample_params {sample_steps,
// sample_method, scheduler, eta, flow_shift, guidance {txt_cfg}}. Image i of a request uses seed + i; a negative
// seed draws a random one. One image is made at a time.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace luce::moss_image {

// The sizes the server takes: sides that are multiples of 16 and at least 256, at most 1024 x 1024 pixels in all;
// 1 to 100 steps.
constexpr int     kMinImageSide   = 256;
constexpr int64_t kMaxImagePixels = int64_t(1024) * 1024;
constexpr int     kMaxImageSteps  = 100;

enum class SampleMethod { euler, er_sde };
enum class Scheduler { discrete, simple };

struct ImageRequest {
    std::string  prompt;
    std::string  negative_prompt;
    int          width         = 1024;
    int          height        = 1024;
    int          steps         = 8;
    float        cfg_scale     = 1.0f;  // 1: no unconditional pass
    SampleMethod sample_method = SampleMethod::er_sde;
    Scheduler    scheduler     = Scheduler::simple;
    float        eta           = 1.0f;  // er_sde noise
    float        flow_shift    = 5.0f;
    int64_t      seed          = -1;    // resolved to a non-negative seed before generate
};

struct RgbImage {
    int                  width  = 0;
    int                  height = 0;
    std::vector<uint8_t> rgb;  // [height][width][3]
};

// Where a generator's weights are, and the memory it holds for them and its work buffers.
struct MemoryState {
    bool   standby        = false;  // it can give its GPU memory back
    bool   weights_on_gpu = true;
    size_t gpu_bytes      = 0;
    size_t host_bytes     = 0;  // pinned host memory: the weights' copy that waits there
    size_t gpu_free_bytes = 0;  // the GPU's free memory, for every process
};

// What the server drives: one image per call, calls never overlap.
class Generator {
public:
    virtual ~Generator() = default;
    // False with `error` set when the image cannot be made.
    virtual bool generate(const ImageRequest & request, RgbImage & image, std::string & error) = 0;

    // A generator in standby overrides these; the next request takes back whatever was given.
    virtual MemoryState memory() { return {}; }
    // Frees the GPU memory of the work buffers, and of the weights unless `keep_weights`; false with `error` set.
    virtual bool release_memory(bool /*keep_weights*/, std::string & error) {
        error = "the server holds its GPU memory: start it with --standby";
        return false;
    }
    // Copies the weights back onto the GPU when it has room for them; false with `error` set when it has not, false
    // with no error once `stopped()` is true, which it asks before the copies and between them.
    virtual bool load_weights(const std::function<bool()> & /*stopped*/, std::string & error) {
        error = "the server holds its GPU memory: start it with --standby";
        return false;
    }
};

struct ImageServerConfig {
    std::string  host     = "0.0.0.0";  // luce_server's defaults
    int          port     = 8080;
    std::string  model_id = "moss-image";
    ImageRequest defaults;  // what a request leaves unset
};

// Serves until the process is stopped; returns the exit code (non-zero when the port cannot be bound).
int run_image_server(const ImageServerConfig & config, Generator & generator);

} // namespace luce::moss_image
