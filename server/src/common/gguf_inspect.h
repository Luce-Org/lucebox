// GGUF model inspection utilities.
//
// Reads minimal metadata (architecture, layer count) from a GGUF file
// without loading weights. Useful for arch detection and layer-split planning.

#pragma once

#include <string>

namespace dflash::common {

struct GgufModelInfo {
    std::string arch;       // e.g. "qwen35", "laguna", "qwen3", "gemma4"
    int         n_layer = -1;
};

// Read architecture and layer count from a GGUF file.
// Returns info with arch="" and n_layer=-1 on failure.
GgufModelInfo inspect_gguf_model_info(const char * path);

}  // namespace dflash::common
