// Vision encoder wrapping mtmd for native mmproj inference.

#pragma once

#include "common/vision_types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef DFLASH_HAVE_MMPROJ
#include "llama.h"
#include "mtmd.h"
#endif

namespace dflash::common {

#ifdef DFLASH_HAVE_MMPROJ

class VisionEncoder {
public:
    VisionEncoder();
    ~VisionEncoder();

    VisionEncoder(const VisionEncoder &) = delete;
    VisionEncoder & operator=(const VisionEncoder &) = delete;

    // Load mmproj + a CPU-only llama_model (vocab/metadata) for tokenization.
    bool init(const char * model_path,
              const char * mmproj_path,
              bool use_gpu,
              int n_threads);

    bool ready() const { return ctx_ != nullptr; }

    // Tokenize marked_text + raw image bytes into mtmd chunks.  Caller owns
    // the returned pointer and must free with mtmd_input_chunks_free().
    mtmd_input_chunks * tokenize(const std::string & marked_text,
                                 const std::vector<DecodedImage> & images) const;

    // Run the vision projector for an image/audio chunk.
    bool encode_chunk(const mtmd_input_chunk * chunk);

    // Embeddings from the last encode_chunk() call (not owned).
    float * output_embeddings() const;

    bool uses_mrope() const;
    bool uses_non_causal(const mtmd_input_chunk * chunk) const;
    void get_image_decoder_pos(const mtmd_image_tokens * image_tokens,
                               mtmd_decoder_pos * out_pos,
                               size_t n_tokens) const;

private:
    struct LlamaModelDeleter {
        void operator()(llama_model * m) const {
            if (m) llama_model_free(m);
        }
    };
    struct MtmdContextDeleter {
        void operator()(mtmd_context * c) const {
            if (c) mtmd_free(c);
        }
    };

    std::unique_ptr<llama_model, LlamaModelDeleter> model_;
    std::unique_ptr<mtmd_context, MtmdContextDeleter> ctx_;
};

#else  // !DFLASH_HAVE_MMPROJ

class VisionEncoder {
public:
    bool init(const char *, const char *, bool, int) { return false; }
    bool ready() const { return false; }
};

#endif  // DFLASH_HAVE_MMPROJ

}  // namespace dflash::common
