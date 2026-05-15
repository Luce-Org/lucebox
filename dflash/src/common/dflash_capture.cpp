#include "dflash_capture.h"

namespace dflash27b {

int target_capture_index(const int * capture_layer_ids,
                         int n_capture_layers,
                         int layer_idx) {
    if (!capture_layer_ids) return -1;
    for (int k = 0; k < n_capture_layers; k++) {
        if (capture_layer_ids[k] == layer_idx) return k;
    }
    return -1;
}

}  // namespace dflash27b
