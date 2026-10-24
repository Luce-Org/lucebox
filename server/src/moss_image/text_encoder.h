// MOSS-Image's text encoder: Qwen3.5-9B from its own GGUF, through the engine's qwen35 loader and graph without the
// lm_head. A prompt is wrapped in MOSS-Image's template,
//   "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n",
// cut to 4096 tokens, and every token's final RMS-normed hidden state conditions the DiT.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luce::moss_image {

struct TextHidden {
    const float * data     = nullptr;  // [n_tokens][4096] f32 on the model's GPU, or in host memory when the encoder
                                       // runs on another GPU (`host`); valid until the next encode
    int           n_tokens = 0;
    bool          host     = false;
};

class TextEncoder {
public:
    static constexpr int kMaxTokens = 4096;

    TextEncoder();
    ~TextEncoder();
    TextEncoder(const TextEncoder &) = delete;
    TextEncoder & operator=(const TextEncoder &) = delete;

    // Weights on GPU `gpu` and the tokenizer from the same GGUF; false with `error` set. On a GPU other than the
    // model's (`model_gpu`), encode() hands its rows over in host memory.
    bool load(const std::string & gguf_path, int gpu, int model_gpu, std::string & error);

    // The template around `prompt`, tokenized and cut to kMaxTokens.
    std::vector<int32_t> tokenize(const std::string & prompt) const;

    // Runs the encoder over `tokens` (1 .. kMaxTokens of them) and returns once the hidden states are in `out` (the
    // encoder's own stream is synchronized), with the calling thread's GPU as it found it; false with `error` set.
    bool encode(const std::vector<int32_t> & tokens, TextHidden & out, std::string & error);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace luce::moss_image
