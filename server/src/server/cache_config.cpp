// Unified cache-capacity helpers.

#include "cache_config.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <limits>

namespace dflash::common {

static std::string trim_copy(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

static std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

bool parse_cache_size_bytes(const std::string & text, size_t & out_bytes) {
    std::string s = trim_copy(text);
    if (s.empty()) return false;

    errno = 0;
    char * end = nullptr;
    double value = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || errno == ERANGE || !std::isfinite(value) ||
        value < 0.0) {
        return false;
    }

    std::string suffix = lower_copy(trim_copy(end ? std::string(end) : ""));
    size_t mult = 1;
    if (suffix.empty() || suffix == "b") {
        mult = 1;
    } else if (suffix == "k" || suffix == "kb" || suffix == "kib") {
        mult = (size_t)1024;
    } else if (suffix == "m" || suffix == "mb" || suffix == "mib") {
        mult = (size_t)1024 * 1024;
    } else if (suffix == "g" || suffix == "gb" || suffix == "gib") {
        mult = (size_t)1024 * 1024 * 1024;
    } else if (suffix == "t" || suffix == "tb" || suffix == "tib") {
        mult = (size_t)1024 * 1024 * 1024 * 1024;
    } else {
        return false;
    }

    const double bytes = value * (double)mult;
    if (bytes > (double)std::numeric_limits<size_t>::max()) return false;
    out_bytes = (size_t)std::ceil(bytes);
    return true;
}

std::string format_cache_size(size_t bytes) {
    auto exact = [bytes](size_t unit) {
        return unit > 0 && bytes >= unit && (bytes % unit) == 0;
    };
    const size_t kib = (size_t)1024;
    const size_t mib = kib * 1024;
    const size_t gib = mib * 1024;
    const size_t tib = gib * 1024;
    if (bytes == 0) return "0";
    if (exact(tib)) return std::to_string(bytes / tib) + "TiB";
    if (exact(gib)) return std::to_string(bytes / gib) + "GiB";
    if (exact(mib)) return std::to_string(bytes / mib) + "MiB";
    if (exact(kib)) return std::to_string(bytes / kib) + "KiB";
    return std::to_string(bytes) + "B";
}

int cache_slots_for_ram_budget(size_t bytes, int max_slots) {
    if (bytes == 0 || max_slots <= 0) return 0;
    size_t slots = (bytes + kCacheRamSlotEstimateBytes - 1) /
                   kCacheRamSlotEstimateBytes;
    if (slots == 0) slots = 1;
    if (slots > (size_t)max_slots) slots = (size_t)max_slots;
    return (int)slots;
}

std::string cache_subdir(const std::string & root, const char * name) {
    if (root.empty()) return "";
    if (root.back() == '/') return root + name;
    return root + "/" + name;
}

}  // namespace dflash::common
