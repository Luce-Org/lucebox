// Vision input parsing implementation.

#include "vision_input.h"

#ifdef DFLASH_HAVE_MMPROJ
#include "mtmd.h"
#endif

#include <cctype>
#include <stdexcept>

namespace dflash::common {

namespace {

static const char * kBase64Chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static bool is_base64(unsigned char c) {
    return (std::isalnum(c) != 0) || (c == '+') || (c == '/');
}

static void append_image_marker(std::string & out) {
#ifdef DFLASH_HAVE_MMPROJ
    out += mtmd_default_marker();
#else
    out += "<__media__>";
#endif
}

static bool part_is_image_type(const std::string & type) {
    return type == "image_url" || type == "input_image" || type == "image";
}

static std::vector<uint8_t> decode_image_url_object(const json & image_url) {
    if (!image_url.is_object()) {
        throw std::runtime_error("image_url must be an object");
    }
    const std::string url = image_url.value("url", "");
    if (url.empty()) {
        throw std::runtime_error("image_url.url is required");
    }
    return decode_image_data_url(url);
}

static void append_part_text(std::string & out, const json & part) {
    const std::string type = part.value("type", "");
    if (type == "text" || type == "input_text" || type == "output_text") {
        out += part.value("text", "");
        return;
    }
    if (part_is_image_type(type)) {
        append_image_marker(out);
        return;
    }
}

static void process_content_array(MultimodalPrompt & mm, const json & content) {
    if (!content.is_array()) return;
    for (const auto & part : content) {
        if (!part.is_object()) continue;
        const std::string type = part.value("type", "");
        if (part_is_image_type(type)) {
            DecodedImage img;
            if (type == "input_image") {
                const std::string source = part.value("source", "");
                if (source == "base64" || part.contains("data")) {
                    const std::string data = part.value("data", "");
                    if (data.empty()) {
                        throw std::runtime_error("input_image data is empty");
                    }
                    img.bytes = base64_decode(data);
                } else if (part.contains("image_url")) {
                    img.bytes = decode_image_url_object(part["image_url"]);
                } else {
                    throw std::runtime_error("unsupported input_image source");
                }
            } else {
                img.bytes = decode_image_url_object(part.value("image_url", json::object()));
            }
            if (img.bytes.empty()) {
                throw std::runtime_error("decoded image is empty");
            }
            mm.images.push_back(std::move(img));
            append_image_marker(mm.marked_text);
            continue;
        }
        append_part_text(mm.marked_text, part);
    }
}

}  // namespace

std::vector<uint8_t> base64_decode(const std::string & encoded_string) {
    const int in_len = (int)encoded_string.size();
    int i = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::vector<uint8_t> ret;
    ret.reserve((size_t)in_len * 3 / 4);

    while (in_len - in_ > 0 && encoded_string[in_] != '=' && is_base64(encoded_string[in_])) {
        char_array_4[i++] = (unsigned char)encoded_string[in_++];
        if (i == 4) {
            for (i = 0; i < 4; i++) {
                const char * p = std::strchr(kBase64Chars, char_array_4[i]);
                if (!p) throw std::runtime_error("invalid base64");
                char_array_4[i] = (unsigned char)(p - kBase64Chars);
            }
            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
            for (i = 0; i < 3; i++) ret.push_back(char_array_3[i]);
            i = 0;
        }
    }

    if (i) {
        for (int j = i; j < 4; j++) char_array_4[j] = 0;
        for (int j = 0; j < 4; j++) {
            const char * p = std::strchr(kBase64Chars, char_array_4[j]);
            if (!p) throw std::runtime_error("invalid base64");
            char_array_4[j] = (unsigned char)(p - kBase64Chars);
        }
        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];
        for (int j = 0; j < i - 1; j++) ret.push_back(char_array_3[j]);
    }
    return ret;
}

std::vector<uint8_t> decode_image_data_url(const std::string & url) {
    const auto comma = url.find(',');
    if (comma == std::string::npos) {
        throw std::runtime_error("invalid image data URL");
    }
    const std::string header = url.substr(0, comma);
    if (header.rfind("data:image/", 0) != 0) {
        throw std::runtime_error("image url must be data:image/...;base64,...");
    }
    if (header.size() < 7 || header.substr(header.size() - 7) != ";base64") {
        throw std::runtime_error("image url must be base64 encoded");
    }
    return base64_decode(url.substr(comma + 1));
}

MultimodalPrompt extract_multimodal_from_messages(const json & messages) {
    MultimodalPrompt mm;
    if (!messages.is_array()) return mm;

    for (const auto & m : messages) {
        if (!m.is_object()) continue;
        if (m.contains("content") && m["content"].is_string()) {
            mm.marked_text += m["content"].get<std::string>();
        } else if (m.contains("content") && m["content"].is_array()) {
            process_content_array(mm, m["content"]);
        }
    }
    return mm;
}

bool messages_contain_images(const json & messages) {
    if (!messages.is_array()) return false;
    for (const auto & m : messages) {
        if (!m.is_object() || !m.contains("content") || !m["content"].is_array()) continue;
        for (const auto & part : m["content"]) {
            if (!part.is_object()) continue;
            const std::string type = part.value("type", "");
            if (part_is_image_type(type)) return true;
        }
    }
    return false;
}

}  // namespace dflash::common
