#include "gguf_inspect.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace dflash::common {

GgufModelInfo inspect_gguf_model_info(const char * path) {
    GgufModelInfo info;

    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx = nullptr;
    gguf_context * gctx = gguf_init_from_file(path, gip);
    if (!gctx) return info;

    // Read architecture
    int64_t arch_id = gguf_find_key(gctx, "general.architecture");
    if (arch_id >= 0) {
        const char * v = gguf_get_val_str(gctx, arch_id);
        if (v) info.arch = v;
    }

    // Read layer count: <arch>.block_count
    if (!info.arch.empty()) {
        std::string key = info.arch + ".block_count";
        int64_t kid = gguf_find_key(gctx, key.c_str());
        if (kid >= 0) {
            info.n_layer = (int)gguf_get_val_u32(gctx, kid);
        }
    }

    gguf_free(gctx);
    return info;
}

}  // namespace dflash::common
