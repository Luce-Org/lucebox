#pragma once
#include "deepseek4/deepseek4_image_prompt.h"
#include "deepseek4/deepseek4_vision_decode.h"
#include "server/image_input.h"
#include "server/tokenizer.h"

// Qualification-only adapter. This is not HttpServer normalization or serving.
struct Composition {
    std::string error;
    dflash::vision::PreparedImagePrompt prepared;
    std::vector<dflash::vision::DecodedRgb> decoded;
    std::vector<int32_t> rendered_tokens;
    explicit operator bool() const { return error.empty(); }
};
Composition compose(const nlohmann::json & messages, dflash::common::Tokenizer & tokenizer,
                    const dflash::vision::ImagePromptLimits & limits = {},
                    const std::string & tools = "", const std::string & jinja = "");
