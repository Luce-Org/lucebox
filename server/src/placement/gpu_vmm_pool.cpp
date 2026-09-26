#include "placement/gpu_vmm_pool.h"

#include "ggml-backend.h"

#include <cstring>

namespace luce::common {

bool gpu_backend_uses_vmm_pool() {
    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        bool has_gpu = false;
        for (size_t d = 0; d < ggml_backend_reg_dev_count(reg); ++d) {
            if (ggml_backend_dev_type(ggml_backend_reg_dev_get(reg, d)) ==
                GGML_BACKEND_DEVICE_TYPE_GPU) {
                has_gpu = true;
                break;
            }
        }
        if (!has_gpu) continue;
        auto get_features = (ggml_backend_get_features_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (!get_features) return true;
        for (ggml_backend_feature * f = get_features(reg); f && f->name; ++f) {
            if (std::strcmp(f->name, "NO_VMM") == 0) return false;
        }
        return true;
    }
    return true;
}

}  // namespace luce::common
