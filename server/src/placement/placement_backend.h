// Backend placement identifiers shared by C++ server/runtime code.

#pragma once

#include <string>

namespace dflash::common {

enum class PlacementBackend {
    Auto,
    Cuda,
    Hip,
};

inline const char * placement_backend_name(PlacementBackend backend) {
    switch (backend) {
        case PlacementBackend::Auto: return "auto";
        case PlacementBackend::Cuda: return "cuda";
        case PlacementBackend::Hip:  return "hip";
    }
    return "auto";
}

inline bool parse_placement_backend(const std::string & value,
                                    PlacementBackend & out) {
    if (value == "auto") {
        out = PlacementBackend::Auto;
        return true;
    }
    if (value == "cuda") {
        out = PlacementBackend::Cuda;
        return true;
    }
    if (value == "hip") {
        out = PlacementBackend::Hip;
        return true;
    }
    return false;
}

inline PlacementBackend compiled_placement_backend() {
#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
    return PlacementBackend::Hip;
#else
    return PlacementBackend::Cuda;
#endif
}

inline bool placement_backend_supported(PlacementBackend backend) {
    return backend == PlacementBackend::Auto ||
           backend == compiled_placement_backend();
}

}  // namespace dflash::common
