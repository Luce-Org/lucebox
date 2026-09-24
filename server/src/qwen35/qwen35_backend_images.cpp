// Image input for Qwen35Backend: projector loading, request preparation, and
// encoding. The prefill and decode changes live next to the code they touch
// in qwen35_backend.cpp.
#include "qwen35_backend.h"

#include "common/image_prompt.h"
#include "common/vision/image_decode.h"
#include "qwen35_image_request.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <new>

namespace luce::common {

namespace {
constexpr size_t MAX_IMAGES_PER_REQUEST = common::MAX_REQUEST_IMAGES;
}

bool Qwen35Backend::load_vision() {
    if (cfg_.mmproj_path.empty()) return true;
    if (w_.image_pad_id < 0) {
        std::fprintf(stderr, "[vision] this model's vocabulary has no <|image_pad|> token\n");
        return false;
    }
    auto tower = std::make_unique<vision::Qwen35VisionTower>();
    std::string error;
    if (!tower->load(cfg_.mmproj_path, target_backend_, w_.n_embd, error)) {
        std::fprintf(stderr, "[vision] %s\n", error.c_str());
        return false;
    }
    // Request threads read these two without a lock, so they are written
    // once. A reload after unpark must bring back the same projector.
    if (!image_input_) {
        vision_config_ = tower->config();
        image_input_ = true;
    } else if (!tower->config().same_geometry(vision_config_)) {
        std::fprintf(stderr, "[vision] the projector file changed while the model was parked\n");
        return false;
    }
    vision_ = std::move(tower);
    std::printf("[vision] projector loaded: %d layers, %.0f MiB, up to %d tokens per image\n",
                vision_config_.layers, vision_->weight_bytes() / (1024.0 * 1024.0),
                vision_config_.max_image_tokens);
    return true;
}

std::string Qwen35Backend::image_placeholder() const {
    return image_input_ ? QWEN35_IMAGE_PLACEHOLDER : "";
}

ImagePrepareStatus Qwen35Backend::prepare_images(std::vector<int32_t> & tokens, std::vector<EncodedImage> images,
                                   uint64_t context_capacity, uint64_t output_reserve,
                                   ImagePromptHandle & payload, std::string & error) const {
    payload.reset();
    if (images.empty()) {
        // A pad typed into a text prompt would be embedded as an ordinary
        // token, with no image behind it.
        if (image_input_ && std::find(tokens.begin(), tokens.end(), w_.image_pad_id) != tokens.end()) {
            error = "image marker in a prompt without images";
            return ImagePrepareStatus::invalid;
        }
        return ImagePrepareStatus::ok;
    }
    if (!image_input_) { error = "this model was started without --mmproj"; return ImagePrepareStatus::invalid; }
    if (images.size() > MAX_IMAGES_PER_REQUEST) { error = "too many images in request"; return ImagePrepareStatus::invalid; }
    try {
        auto prompt = std::make_shared<Qwen35ImagePrompt>();
        prompt->owner = this;
        for (const EncodedImage & image : images) {
            auto decoded = vision::decode_image({image.bytes.data(), image.bytes.size()});
            if (!decoded) { error = decoded.status.message; return ImagePrepareStatus::invalid; }
            vision::Qwen35Pixels pixels;
            if (!vision::qwen35_vision_preprocess(vision_config_, decoded.image, pixels, error)) return ImagePrepareStatus::invalid;
            Qwen35ImageSlot slot;
            slot.columns = pixels.grid_columns;
            slot.rows = pixels.grid_rows;
            prompt->slots.push_back(slot);
            prompt->pixels.push_back(std::move(pixels));
        }
        const uint64_t limit = context_capacity > output_reserve ? context_capacity - output_reserve : 0;
        if (!qwen35_expand_image_tokens(tokens, w_.image_pad_id, prompt->slots, limit, error)) return ImagePrepareStatus::invalid;
        prompt->expanded_tokens = tokens;
        prompt->positions = qwen35_image_rope_positions((int) tokens.size(), prompt->slots);
        payload = std::move(prompt);
        return ImagePrepareStatus::ok;
    } catch (const std::bad_alloc &) {
        error = "image preparation allocation failed";
        return ImagePrepareStatus::invalid;
    }
}

bool Qwen35Backend::encode_images(const Qwen35ImagePrompt & prompt, Qwen35ImageRows & rows,
                                  std::string & error) {
    if (prompt.owner != this || !vision_) {
        error = "image binding does not belong to the loaded backend";
        return false;
    }
    rows.prompt = &prompt;
    bool ok = true;
    int tokens = 0;
    const auto start = std::chrono::steady_clock::now();
    try {
        rows.rows.resize(prompt.pixels.size());
        for (size_t i = 0; ok && i < prompt.pixels.size(); ++i) {
            ok = vision_->encode(prompt.pixels[i], rows.rows[i], error);
            tokens += prompt.pixels[i].tokens();
        }
    } catch (const std::bad_alloc &) {
        error = "image encoding allocation failed";
        ok = false;
    }
    ggml_backend_synchronize(target_backend_);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (ok) {
        std::printf("[vision] encoded %zu image(s), %d tokens, in %.0f ms\n", prompt.pixels.size(), tokens, ms);
        std::fflush(stdout);
    }
    // The attention scratch is large and only needed here; give it back
    // before prefill sizes its own graphs.
    vision_->release_scratch();
    return ok;
}

}  // namespace luce::common
