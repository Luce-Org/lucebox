// Vision encoder implementation (mtmd wrapper).

#include "vision_encoder.h"

#ifdef DFLASH_HAVE_MMPROJ

#include "mtmd.h"
#include "mtmd-helper.h"
#include "llama.h"

#include <cstdio>
#include <vector>

namespace dflash::common {

VisionEncoder::VisionEncoder() = default;
VisionEncoder::~VisionEncoder() = default;

bool VisionEncoder::init(const char * model_path,
                         const char * mmproj_path,
                         bool use_gpu,
                         int n_threads) {
    if (!model_path || !mmproj_path) {
        std::fprintf(stderr, "[vision] model_path and mmproj_path are required\n");
        return false;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // vocab/metadata only
    mparams.use_mmap       = true;

    llama_model * raw_model = llama_model_load_from_file(model_path, mparams);
    if (!raw_model) {
        std::fprintf(stderr, "[vision] failed to load llama_model from %s\n", model_path);
        return false;
    }
    model_.reset(raw_model);

    mtmd_context_params cparams = mtmd_context_params_default();
    cparams.use_gpu   = use_gpu;
    cparams.n_threads = (n_threads > 0) ? n_threads : 4;
    cparams.warmup    = false;
    if (const char * s = std::getenv("IMAGE_MIN_TOKENS")) {
        const int v = std::atoi(s);
        if (v > 0) cparams.image_min_tokens = v;
    }
    if (const char * s = std::getenv("IMAGE_MAX_TOKENS")) {
        const int v = std::atoi(s);
        if (v > 0) cparams.image_max_tokens = v;
    }

    mtmd_context * raw_ctx = mtmd_init_from_file(mmproj_path, model_.get(), cparams);
    if (!raw_ctx) {
        std::fprintf(stderr, "[vision] failed to load mmproj from %s\n", mmproj_path);
        model_.reset();
        return false;
    }
    ctx_.reset(raw_ctx);

    std::fprintf(stderr, "[vision] loaded mmproj %s (gpu=%s threads=%d)\n",
                 mmproj_path, use_gpu ? "on" : "off", cparams.n_threads);
    return true;
}

mtmd_input_chunks * VisionEncoder::tokenize(
        const std::string & marked_text,
        const std::vector<DecodedImage> & images) const {
    if (!ctx_) return nullptr;

    std::vector<mtmd::bitmap> bitmap_wrappers;
    bitmap_wrappers.reserve(images.size());
    std::vector<const mtmd_bitmap *> bitmap_ptrs;
    bitmap_ptrs.reserve(images.size());

    for (const auto & img : images) {
        mtmd_bitmap * bmp = mtmd_helper_bitmap_init_from_buf(
            ctx_.get(), img.bytes.data(), img.bytes.size());
        if (!bmp) {
            std::fprintf(stderr, "[vision] failed to decode image bytes\n");
            return nullptr;
        }
        bitmap_wrappers.emplace_back(bmp);
        bitmap_ptrs.push_back(bitmap_wrappers.back().ptr.get());
    }

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) return nullptr;

    mtmd_input_text text{};
    text.text          = marked_text.c_str();
    text.add_special   = false;
    text.parse_special = true;

    const int32_t rc = mtmd_tokenize(ctx_.get(), chunks, &text,
                                     bitmap_ptrs.data(), bitmap_ptrs.size());
    if (rc != 0) {
        std::fprintf(stderr, "[vision] mtmd_tokenize failed rc=%d\n", rc);
        mtmd_input_chunks_free(chunks);
        return nullptr;
    }
    return chunks;
}

bool VisionEncoder::encode_chunk(const mtmd_input_chunk * chunk) {
    if (!ctx_ || !chunk) return false;
    const int32_t rc = mtmd_encode_chunk(ctx_.get(), chunk);
    if (rc != 0) {
        std::fprintf(stderr, "[vision] mtmd_encode_chunk failed rc=%d\n", rc);
        return false;
    }
    return true;
}

float * VisionEncoder::output_embeddings() const {
    return ctx_ ? mtmd_get_output_embd(ctx_.get()) : nullptr;
}

bool VisionEncoder::uses_mrope() const {
    return ctx_ && mtmd_decode_use_mrope(ctx_.get());
}

bool VisionEncoder::uses_non_causal(const mtmd_input_chunk * chunk) const {
    return ctx_ && mtmd_decode_use_non_causal(ctx_.get(), chunk);
}

void VisionEncoder::get_image_decoder_pos(const mtmd_image_tokens * image_tokens,
                                          mtmd_decoder_pos * out_pos,
                                          size_t n_tokens) const {
    if (!image_tokens || !out_pos) return;
    for (size_t i = 0; i < n_tokens; i++) {
        out_pos[i] = mtmd_image_tokens_get_decoder_pos(image_tokens, i);
    }
}

}  // namespace dflash::common

#endif  // DFLASH_HAVE_MMPROJ
