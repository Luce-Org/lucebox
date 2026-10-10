#include "dflash_capture.h"

namespace luce::common {

int target_capture_index(const int * capture_layer_ids,
                         int n_capture_layers,
                         int layer_idx) {
    if (!capture_layer_ids) return -1;
    for (int k = 0; k < n_capture_layers; k++) {
        if (capture_layer_ids[k] == layer_idx) return k;
    }
    return -1;
}

bool adopt_drafter_capture_layers(const std::vector<int> & drafter_ids,
                                  int target_n_layer,
                                  int max_slots,
                                  int * capture_layer_ids,
                                  int & n_capture_layers) {
    const int n = (int)drafter_ids.size();
    if (n <= 0 || n > max_slots || !capture_layer_ids) return false;
    for (int id : drafter_ids) {
        if (id < 0 || id >= target_n_layer) return false;
    }
    for (int k = 0; k < n; k++) capture_layer_ids[k] = drafter_ids[(size_t)k];
    n_capture_layers = n;
    return true;
}

}  // namespace luce::common
