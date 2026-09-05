#include "image_input.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace dflash::common {
namespace {
int base64_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}
bool reserved_placeholder(std::string_view text) {
    return text.find(DS4_IMAGE_PLACEHOLDER) != std::string_view::npos;
}
void require(bool valid, const char * message) {
    if (!valid) throw std::invalid_argument(message);
}
void validate_message_structure(const nlohmann::json & messages) {
    std::vector<std::pair<const nlohmann::json *, size_t>> pending{{&messages, 0}};
    while (!pending.empty()) {
        const auto [value, depth] = pending.back();
        pending.pop_back();
        require(depth <= 64, "image message nesting exceeds 64 levels");
        if (value->is_string()) {
            require(!reserved_placeholder(value->get_ref<const std::string &>()),
                    "text contains the reserved image placeholder");
        } else if (value->is_structured()) {
            for (const auto & child : *value) pending.emplace_back(&child, depth + 1);
        }
    }
}
}

bool parse_image_data_url(std::string_view url, EncodedImage & image,
                          std::string & error, size_t max_bytes) {
    image = {};
    error.clear();
    try {
        EncodedImage parsed;
        constexpr std::string_view jpeg = "data:image/jpeg;base64,";
        constexpr std::string_view png = "data:image/png;base64,";
        if (url.substr(0, jpeg.size()) == jpeg) {
            parsed.mime_type = "image/jpeg";
            url.remove_prefix(jpeg.size());
        } else if (url.substr(0, png.size()) == png) {
            parsed.mime_type = "image/png";
            url.remove_prefix(png.size());
        } else {
            throw std::invalid_argument("image_url must be a JPEG or PNG base64 data URL");
        }
        require(!url.empty() && url.size() % 4 == 0, "invalid image base64 length");
        size_t padding = url.back() == '=' ? 1 : 0;
        if (url[url.size() - 2] == '=') ++padding;
        const size_t decoded_bytes = url.size() / 4 * 3 - padding;
        require(decoded_bytes > 0 && decoded_bytes <= max_bytes, "image exceeds encoded byte limit");
        for (size_t i = 0; i < url.size() - padding; ++i) {
            require(base64_value(url[i]) >= 0, "invalid image base64 alphabet or padding");
        }
        for (size_t i = url.size() - padding; i < url.size(); ++i) {
            require(url[i] == '=', "invalid image base64 padding");
        }
        if (padding == 1) require((base64_value(url[url.size() - 2]) & 3) == 0, "noncanonical image base64 padding");
        if (padding == 2) require((base64_value(url[url.size() - 3]) & 15) == 0, "noncanonical image base64 padding");
        parsed.bytes.reserve(decoded_bytes);
        for (size_t i = 0; i < url.size(); i += 4) {
            uint32_t value = 0;
            for (size_t j = 0; j < 4; ++j) {
                value = value << 6 | uint32_t(url[i+j] == '=' ? 0 : base64_value(url[i+j]));
            }
            for (int shift : {16, 8, 0}) {
                if (parsed.bytes.size() < decoded_bytes) parsed.bytes.push_back(uint8_t(value >> shift));
            }
        }
        constexpr std::array<uint8_t, 3> jpeg_signature{255, 216, 255};
        constexpr std::array<uint8_t, 8> png_signature{137, 80, 78, 71, 13, 10, 26, 10};
        if (parsed.mime_type == "image/jpeg") {
            require(parsed.bytes.size() >= jpeg_signature.size() &&
                    std::equal(jpeg_signature.begin(), jpeg_signature.end(), parsed.bytes.begin()),
                    "image bytes do not match JPEG media type");
        } else {
            require(parsed.bytes.size() >= png_signature.size() &&
                    std::equal(png_signature.begin(), png_signature.end(), parsed.bytes.begin()),
                    "image bytes do not match PNG media type");
        }
        image = std::move(parsed);
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}

bool extract_chat_images(const nlohmann::json & messages, nlohmann::json & normalized,
                         std::vector<EncodedImage> & images,
                         std::string & error, const ImageInputLimits & limits) {
    normalized = nullptr;
    images.clear();
    error.clear();
    try {
        require(messages.is_array(), "image messages must be an array");
        validate_message_structure(messages);
        nlohmann::json result = messages;
        std::vector<EncodedImage> collected;
        size_t total_bytes = 0;
        for (auto & message : result) {
            require(message.is_object(), "each image message must be an object");
            if (!message.contains("content")) continue;
            auto & content = message["content"];
            if (!content.is_array()) continue;
            std::string text_segment;
            for (auto & part : content) {
                require(part.is_object(), "each content part must be an object");
                const auto type = part.value("type", std::string());
                if (type == "text" || type == "input_text" || type == "output_text") {
                    text_segment += part.value("text", std::string());
                    continue;
                }
                require(type != "image" && type != "input_image", "use image_url content parts for images");
                if (type != "image_url") continue;
                require(!reserved_placeholder(text_segment), "text contains the reserved image placeholder");
                text_segment.clear();
                require(message.value("role", std::string("user")) == "user", "images are supported only in user messages");
                require(collected.size() < limits.image_count, "too many images in request");
                require(part.contains("image_url") && part["image_url"].is_object() &&
                        part["image_url"].contains("url") && part["image_url"]["url"].is_string(),
                        "image_url must contain a string url");
                const auto & descriptor = part["image_url"];
                if (descriptor.contains("detail")) {
                    require(descriptor["detail"].is_string(), "image detail must be auto, low, or high");
                    const auto detail = descriptor["detail"].get<std::string>();
                    require(detail == "auto" || detail == "low" || detail == "high", "image detail must be auto, low, or high");
                }
                require(total_bytes <= limits.request_bytes, "images exceed request byte limit");
                EncodedImage decoded;
                std::string decode_error;
                const size_t remaining = std::min(limits.image_bytes, limits.request_bytes - total_bytes);
                if (!parse_image_data_url(descriptor["url"].get_ref<const std::string &>(), decoded, decode_error, remaining)) {
                    error = std::move(decode_error);
                    return false;
                }
                total_bytes += decoded.bytes.size();
                collected.push_back(std::move(decoded));
                part = {{"type", "text"}, {"text", DS4_IMAGE_PLACEHOLDER}};
            }
            require(!reserved_placeholder(text_segment), "text contains the reserved image placeholder");
        }
        normalized = std::move(result);
        images = std::move(collected);
        return true;
    } catch (const nlohmann::json::exception &) {
        error = "invalid image content field type";
        return false;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}

static bool contains_image_content(const nlohmann::json & value) {
    std::vector<const nlohmann::json *> pending{&value};
    while (!pending.empty()) {
        const auto & current = *pending.back();
        pending.pop_back();
        if (current.is_object()) {
            auto type = current.find("type");
            if (type != current.end() && type->is_string() &&
                (*type == "image" || *type == "input_image" || *type == "image_url")) return true;
            for (const auto & item : current.items()) {
                if (item.key() == "image_url") return true;
                pending.push_back(&item.value());
            }
        } else if (current.is_array()) {
            for (const auto & item : current) pending.push_back(&item);
        }
    }
    return false;
}

bool prepare_request_images(const nlohmann::json & messages,
                            const ImageRequestPolicy & policy,
                            nlohmann::json & normalized,
                            std::vector<EncodedImage> & images,
                            std::string & error, const ImageInputLimits & limits) {
    normalized = nullptr;
    images.clear();
    error.clear();
    const bool has_images = contains_image_content(messages);
    if (has_images && !policy.chat_completions) {
        error = "image input is supported only through /v1/chat/completions image_url parts";
        return false;
    }
    if (has_images && !policy.image_capable) {
        error = "image input is unavailable for this backend or serving mode; configure a supported --mmproj projector";
        return false;
    }
    if (policy.chat_completions && (has_images || policy.reserve_placeholder)) {
        nlohmann::json prepared;
        std::vector<EncodedImage> extracted;
        if (!extract_chat_images(messages, prepared, extracted, error, limits)) return false;
        if (contains_image_content(prepared)) {
            error = "images must be user content-array image_url parts";
            return false;
        }
        normalized = std::move(prepared);
        images = std::move(extracted);
    } else {
        normalized = messages;
    }
    return true;
}

void redact_image_urls(nlohmann::json & value) {
    std::vector<nlohmann::json *> pending{&value};
    while (!pending.empty()) {
        auto * current = pending.back();
        pending.pop_back();
        if (current->is_object()) {
            for (auto & item : current->items()) {
                if (item.key() == "image_url") item.value() = "[image omitted]";
                else pending.push_back(&item.value());
            }
        } else if (current->is_array()) {
            for (auto & item : *current) pending.push_back(&item);
        }
    }
}
} // namespace dflash::common
