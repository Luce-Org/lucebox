#pragma once
#include "deepseek4_vision_preprocess.h"

namespace dflash::vision {

struct ImagePatchInput {
    ResizePlan plan;
    std::vector<std::uint16_t> patches_bf16;
};
struct PromptImage {
    ImagePatchInput input;
    ImageLayout layout;
};
struct ImageTokenizerContract {
    std::uint32_t vocabulary = 129280;
    std::int32_t marker = 129264;
};
struct ImagePromptLimits {
    std::uint64_t context_capacity = 131072;
    std::uint64_t output_reserve = 4096;
    std::uint64_t max_expanded_tokens = 131072;
};
constexpr std::uint64_t MAX_PREPARED_PROMPT_TOKENS = 1048576;
enum class ImagePromptError {
    None, InvalidContract, InvalidLimits, InvalidToken, ImageCount,
    MarkerCount, InvalidPlan, InvalidPatches, TokenLimit, ContextOverflow, AllocationFailed,
};
struct PreparedImagePrompt {
    ImagePromptError error = ImagePromptError::None;
    std::string message;
    std::vector<std::int32_t> tokens;
    std::vector<PromptImage> images;
    explicit operator bool() const { return error == ImagePromptError::None; }
};

// Consumes final rendered token IDs and already-preprocessed image inputs.
// Success owns independent copies of patches and layouts. Failure owns no
// partial tokens/images. No decoding, model execution, or cache policy occurs.
PreparedImagePrompt prepare_image_prompt(
    const std::vector<std::int32_t> & rendered_tokens,
    const std::vector<ImagePatchInput> & images,
    const ImagePromptLimits & limits = {},
    const ImageTokenizerContract & tokenizer = {});

} // namespace dflash::vision
