#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <string>

namespace dflash::common {

struct BackendPrecisionPolicy {
    ggml_type   weight_type    = GGML_TYPE_BF16;
    ggml_type   compute_type   = GGML_TYPE_BF16;
    std::string backend_name;
    std::string device_name;
    std::string runtime_arch;
    int         device_id      = -1;
    int         cuda_sm        = 0;
    std::string reason;
};

BackendPrecisionPolicy select_drafter_precision_policy(ggml_backend_t backend);

const char * backend_precision_type_name(ggml_type type);

} // namespace dflash::common
