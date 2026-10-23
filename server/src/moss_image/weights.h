// MOSS-Image weights on the device: the DiT and the FLUX.2 VAE decoder, read from the GGUF that
// scripts/convert_moss_image.py writes (general.architecture "moss-image").
//
// Large matrices keep their stored type: int8 with f32 per-row scales for the DiT layers, bf16 for the rest of the
// DiT, f16 for the VAE convolutions. Norm weights, biases and scales are f32 on the device (bf16 to f32 is exact).
// All of it lives in one device allocation, laid out in the order a request uses it: what the DiT needs before its
// first layer, the layers one after another, then the VAE decoder. A copy of that allocation can wait in pinned host
// memory instead, and go onto the device when needed, while the first denoising step already runs on what arrived.

#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace luce::moss_image {

constexpr int kHidden         = 3072;
constexpr int kHeads          = 24;
constexpr int kHeadDim        = 128;
constexpr int kFfn            = 8192;
constexpr int kLayers         = 42;
constexpr int kTextBlocks     = 2;
constexpr int kTextHidden     = 4096;  // Qwen3.5-9B hidden size
constexpr int kLatentChannels = 128;   // FLUX.2 latent: 32 channels at 1/8 scale, patchified 2x2
constexpr int kTimeFreqs      = 256;

// y = (x8 . w) * x_scale * scale[n]
struct I8Linear {
    const int8_t * w     = nullptr;  // [n][k]
    const float *  scale = nullptr;  // [n]
};

struct Bf16Linear {
    const uint16_t * w = nullptr;  // [n][k] bf16
    const float *    b = nullptr;  // [n]; null without a bias
};

struct DitLayer {
    I8Linear      qkv;      // [3 * 3072][3072]
    I8Linear      o;        // [3072][3072]
    I8Linear      up_gate;  // [2 * 8192][3072]: up rows, then gate rows
    I8Linear      down;     // [3072][8192]
    const float * norm1  = nullptr;  // [3072]
    const float * norm2  = nullptr;  // [3072]
    const float * q_norm = nullptr;  // [128]
    const float * k_norm = nullptr;  // [128]
};

struct TextBlock {
    Bf16Linear    qkv;      // [3 * 3072][3072]
    Bf16Linear    proj;     // [3072][3072]
    Bf16Linear    up_gate;  // [2 * 8192][3072]: ffn_up rows, then ffn_gate rows (joined at load)
    Bf16Linear    down;     // [3072][8192]
    const float * norm1  = nullptr;  // [3072]
    const float * norm2  = nullptr;  // [3072]
    const float * q_norm = nullptr;  // [128]
    const float * k_norm = nullptr;  // [128]
};

struct DitWeights {
    Bf16Linear    text_proj;                 // text_embedder.proj [3072][4096]
    TextBlock     text_blocks[kTextBlocks];  // text_embedder.blocks
    Bf16Linear    video_embedder;            // [3072][128]
    Bf16Linear    time_in;                   // time_embedding.0 [3072][256], SiLU after it
    Bf16Linear    time_out;                  // time_embedding.2 [3072][3072]
    Bf16Linear    block_modulation;          // shared_block_modulation [6 * 3072][3072]
    Bf16Linear    head_modulation;           // video_head_modulation [2 * 3072][3072]: shift, scale
    const float * final_norm = nullptr;      // final_norm_video [3072]
    Bf16Linear    final_linear;              // final_linear_video [128][3072], no bias
    DitLayer      layers[kLayers];
};

// Stride 1, padding (k - 1) / 2.
struct Conv2d {
    const uint16_t * w  = nullptr;  // f16 [oc][k * k][ic]
    const float *    b  = nullptr;  // [oc]
    int              oc = 0;
    int              ic = 0;
    int              k  = 0;
};

// 32 groups.
struct GroupNorm {
    const float * w        = nullptr;  // [channels]
    const float * b        = nullptr;  // [channels]
    int           channels = 0;
};

struct VaeResBlock {
    GroupNorm norm1;
    Conv2d    conv1;
    GroupNorm norm2;
    Conv2d    conv2;
    Conv2d    nin;  // 1x1 shortcut where the channel count changes; w is null elsewhere
};

struct VaeAttention {
    GroupNorm norm;
    Conv2d    qkv;   // the q, k and v 1x1 convolutions joined: [3 * 512][1][512]
    Conv2d    proj;  // proj_out [512][1][512]
};

struct VaeDecoder {
    const float * latents_mean = nullptr;  // [128], per patchified channel
    const float * latents_std  = nullptr;  // [128]
    Conv2d        post_quant;              // [32][1][32]
    Conv2d        conv_in;                 // [512][9][32]
    VaeResBlock   mid_block1;
    VaeAttention  mid_attn;
    VaeResBlock   mid_block2;
    VaeResBlock   up_blocks[4][3];         // [level][block]: 512, 512, 256, 128 channels; level 3 runs first
    Conv2d        upsample[4];             // nearest 2x then this conv after levels 3, 2 and 1; upsample[0].w is null
    GroupNorm     norm_out;                // [128]
    Conv2d        conv_out;                // [3][9][128]
};

class Weights {
public:
    Weights() = default;
    ~Weights();
    Weights(const Weights &) = delete;
    Weights & operator=(const Weights &) = delete;

    // The parts upload() records an event after: everything before the first DiT layer, each layer, the VAE decoder.
    static constexpr int kParts = kLayers + 2;

    // Reads the GGUF into the current device through pinned staging buffers, uploads on `stream`, and returns once
    // the uploads are done; false with `error` set.
    bool load(const std::string & path, hipStream_t stream, std::string & error);

    // Reads the GGUF into pinned host memory, laid out as on the device, and puts nothing on the device: upload()
    // copies it there, as often as release() takes it away. False with `error` set.
    bool read(const std::string & path, std::string & error);

    // Queues the copy read() made onto the current device on `stream`: ready[0] is recorded once everything before the
    // first DiT layer is there, ready[1 + l] once layer l is, ready[kLayers + 1] once the VAE decoder is, so that work
    // on another stream can wait for each part alone. With `stopped`, it returns once the copies are done, and stops
    // early once (*stopped)() is true: the device copy is then freed and the result is false with no error. A failed
    // copy frees it too, with `error` set.
    bool upload(hipStream_t stream, hipEvent_t * ready, const std::function<bool()> * stopped, std::string & error);

    // Frees the device copy; the work using it must be done.
    void release();

    bool   on_device() const { return device_ != nullptr; }
    size_t device_bytes() const { return device_ != nullptr ? bytes_ : 0; }
    size_t host_bytes() const { return host_ != nullptr ? bytes_ : 0; }

    DitWeights dit;
    VaeDecoder vae;

private:
    void *                                      device_ = nullptr;
    void *                                      host_   = nullptr;
    size_t                                      bytes_  = 0;
    size_t                                      part_end_[kParts] = {};  // byte offset where each part ends
    std::vector<std::function<void(uint8_t *)>> bind_;  // points the fields above into a device copy
};

} // namespace luce::moss_image
