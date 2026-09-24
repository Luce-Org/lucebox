#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luce::common {

// Most images one request may carry, for every vision backend and the HTTP
// transport. Each image still has its own token and byte bounds.
inline constexpr size_t MAX_REQUEST_IMAGES = 16;

// Outcome of binding a request's images to its prompt. `busy` means the
// request is valid but the backend already holds as many image requests as it
// serves at once; the server answers 503 so the client retries.
enum class ImagePrepareStatus { ok, invalid, busy };

struct EncodedImage {
    std::string mime_type;
    std::vector<uint8_t> bytes;
};

class ImagePromptPayload {
public:
    virtual ~ImagePromptPayload() = default;
    virtual bool matches(const std::vector<int32_t> & tokens) const = 0;
};

using ImagePromptHandle = std::shared_ptr<const ImagePromptPayload>;

} // namespace luce::common
