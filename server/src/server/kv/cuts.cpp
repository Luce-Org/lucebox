#include "server/kv/cuts.h"

#include <algorithm>

namespace luce::common {

int select_inline_snapshot_boundary(const std::vector<int> & boundaries,
                                    int restored_prefix_len,
                                    bool prefer_head,
                                    int reachable_from) {
    if (boundaries.empty()) return 0;
    const auto usable = [&](int cut) {
        return cut > restored_prefix_len && cut >= reachable_from;
    };
    // Save the system/tools head (first marker) before deepening into
    // conversation turns. A head the backend cannot save is skipped: it is
    // cheap to prefill, and waiting for it would keep the cache from ever
    // saving a deeper cut.
    if (prefer_head) {
        const int head = boundaries.front();
        if (usable(head)) return head;
    }
    return usable(boundaries.back()) ? boundaries.back() : 0;
}

std::vector<int> spaced_restore_points(const std::vector<int> & boundaries, int min_gap) {
    if (min_gap <= 0) return boundaries;
    std::vector<int> kept;
    for (int b : boundaries) {
        if (kept.empty() || b - kept.back() >= min_gap) kept.push_back(b);
    }
    return kept;
}

std::vector<int> prefix_restore_points(const std::vector<int32_t> & prompt,
                                       const ChatMarkers & markers,
                                       std::initializer_list<int> cuts,
                                       bool drop_last_boundary,
                                       int spacing) {
    std::vector<int> points = find_all_boundaries(prompt, markers);
    if (drop_last_boundary && !points.empty()) points.pop_back();
    points = spaced_restore_points(points, spacing);
    for (int cut : cuts) {
        if (cut > 0 && cut < (int) prompt.size()) points.push_back(cut);
    }
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    return points;
}

bool should_force_inline_snapshot_boundary(
        const std::vector<int> & boundaries,
        int prompt_len,
        int restored_prefix_len,
        bool prefer_tools_boundary,
        int forced_cut,
        int reachable_from) {
    const bool tools_pin_restored =
        prefer_tools_boundary && !boundaries.empty() &&
        restored_prefix_len >= boundaries.front();
    return !tools_pin_restored &&
           forced_cut > restored_prefix_len &&
           forced_cut >= reachable_from &&
           forced_cut <= prompt_len;
}

}  // namespace luce::common
