#include "server/image_input.h"

#include <iostream>
#include <stdexcept>

using namespace dflash::common;
using json = nlohmann::json;

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
static json image_part(const std::string & url) {
    return {{"type", "image_url"}, {"image_url", {{"url", url}}}};
}
static json text_part(const std::string & text) {
    return {{"type", "text"}, {"text", text}};
}

int main() {
    try {
        const std::string jpeg = "data:image/jpeg;base64,/9j/";
        const std::string png = "data:image/png;base64,iVBORw0KGgo=";
        EncodedImage image;
        std::string error;
        check(parse_image_data_url(jpeg, image, error), "valid JPEG transport rejected");
        check(image.mime_type == "image/jpeg" && image.bytes == std::vector<uint8_t>({255,216,255}), "JPEG bytes differ");
        check(parse_image_data_url(png, image, error), "valid PNG transport rejected");
        check(image.mime_type == "image/png" && image.bytes == std::vector<uint8_t>({137,80,78,71,13,10,26,10}), "PNG bytes differ");
        for (const auto & bad : std::vector<std::string>{
                "https://example.com/a.png", "file:///etc/passwd", "data:image/gif;base64,R0lG",
                "data:image/png,iVBORw0KGgo=", "data:image/png;base64,", "data:image/png;base64,AAA",
                "data:image/png;base64,AAAA====", "data:image/png;base64,AA=A", "data:image/png;base64,AA!A",
                "data:image/png;base64,iVBORw0KGgo=\n", "data:image/png;base64,iVBORw0KGgp=",
                "data:image/png;base64,AB==", "data:image/png;base64,/9j/"}) {
            check(!parse_image_data_url(bad, image, error), "invalid data URL accepted");
            check(image.bytes.empty() && image.mime_type.empty(), "failed parse retained image");
            check(!error.empty() && error.find(bad) == std::string::npos, "parse error leaks URL");
        }
        check(!parse_image_data_url(png, image, error, 7), "image byte cap ignored");
        check(parse_image_data_url(png, image, error, 8), "exact byte cap rejected");

        json messages = json::array({{{"role", "user"}, {"content", json::array({
            text_part("before"), image_part(jpeg), text_part("between"), image_part(png), text_part("after")})}}});
        json normalized;
        std::vector<EncodedImage> images;
        check(extract_chat_images(messages, normalized, images, error), "ordered images rejected");
        check(images.size() == 2 && images[0].mime_type == "image/jpeg" && images[1].mime_type == "image/png", "image order differs");
        std::string text;
        for (const auto & part : normalized[0]["content"]) text += part.at("text").get<std::string>();
        check(text == "before" + std::string(DS4_IMAGE_PLACEHOLDER) + "between" + DS4_IMAGE_PLACEHOLDER + "after", "text/image placement differs");
        check(messages[0]["content"][1]["image_url"]["url"] == jpeg, "input JSON mutated");
        check(normalized.dump().find("base64") == std::string::npos, "normalized messages retain bytes");

        const json plain = json::array({{{"role", "user"}, {"content", "hello"}},
                                       {{"role", "assistant"}, {"content", "hi"}, {"tool_calls", json::array()}}});
        check(extract_chat_images(plain, normalized, images, error) && normalized == plain && images.empty(), "text-only request changed");

        for (const auto & bad : std::vector<json>{
                json::array({{{"role", "user"}, {"content", DS4_IMAGE_PLACEHOLDER}}}),
                json::array({{{"role", "user"}, {"content", json::array({text_part("<｜deepseek_"), text_part("image｜>")})}}}),
                json::array({{{"role", "assistant"}, {"reasoning_content", DS4_IMAGE_PLACEHOLDER}, {"content", "hi"}}}),
                json::array({{{"type", "function_call_output"}, {"output", DS4_IMAGE_PLACEHOLDER}}}),
                json::array({{{"type", "function_call"}, {"arguments", DS4_IMAGE_PLACEHOLDER}}}),
                json::array({{{"role", "assistant"}, {"tool_calls", json::array({{{"function", {{"arguments", DS4_IMAGE_PLACEHOLDER}}}}})}}}),
                json::array({{{"role", "assistant"}, {"content", json::array({image_part(png)})}}}),
                json::array({{{"role", "user"}, {"content", json::array({{{"type", "image_url"}, {"image_url", 42}}})}}}),
                json::array({{{"role", "user"}, {"content", json::array({{{"type", "image_url"}, {"image_url", {{"url", 42}}}}})}}}),
                json::array({{{"role", "user"}, {"content", json::array({{{"type", "image_url"}}})}}})}) {
            check(!extract_chat_images(bad, normalized, images, error), "invalid image message accepted");
            check(images.empty() && normalized.is_null(), "failed extraction retains partial state");
        }
        ImageInputLimits limits;
        limits.image_count = 1;
        check(!extract_chat_images(messages, normalized, images, error, limits), "image count cap ignored");
        limits.image_count = 4;
        limits.request_bytes = 10;
        check(!extract_chat_images(messages, normalized, images, error, limits), "aggregate byte cap ignored");
        limits.request_bytes = 11;
        check(extract_chat_images(messages, normalized, images, error, limits), "exact aggregate cap rejected");

        json deep = plain;
        json * nested = &deep[0]["metadata"];
        for (int i = 0; i < 72; ++i) {
            *nested = json::object();
            nested = &(*nested)["nested"];
        }
        *nested = image_part(png);
        check(!extract_chat_images(deep, normalized, images, error), "deep metadata accepted before copy");
        check(error == "image message nesting exceeds 64 levels", "unexpected depth rejection");
        redact_image_urls(deep);
        check((*nested)["image_url"] == "[image omitted]", "deep redaction failed");

        json status = {{"messages", messages}, {"raw_body", {{"nested", image_part(png)}}}};
        redact_image_urls(status);
        check(status.dump().find("base64") == std::string::npos, "status leaks image data URL");
        check(status["messages"][0]["content"][0]["text"] == "before", "redaction alters ordinary text");
        std::cout << "PASS: strict bounded data URLs, ordered content, placeholder rejection, transactional failures, redaction\n";
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
