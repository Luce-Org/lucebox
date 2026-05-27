// Remote draft execution configuration for mixed-backend target/draft placement.

#pragma once

#include <string>

namespace dflash::common {

struct RemoteDraftConfig {
    std::string ipc_bin;
    std::string work_dir;
    int ring_cap = 0;

    bool enabled() const { return !ipc_bin.empty(); }
    bool has_aux_options() const { return !work_dir.empty() || ring_cap > 0; }
};

}  // namespace dflash::common
