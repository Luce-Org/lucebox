#pragma once

#include "backend_factory.h"

namespace dflash::common::detail {

// Pure planning seam used by prepare_backend() after GGUF inspection and by
// model-free tests with supplied model facts.
class BackendPlanBuilder {
public:
    static BackendPreparation resolve(
        BackendArgs args,
        BackendAdmissionContext admission,
        GgufModelInfo model,
        PlacementBackend compiled_backend);
};

}  // namespace dflash::common::detail
