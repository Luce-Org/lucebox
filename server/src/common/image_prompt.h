#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

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

} // namespace dflash::common
