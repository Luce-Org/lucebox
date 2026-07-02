#include "expert_split_target_config.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>

namespace dflash::common {

namespace {

std::string trim_ascii(std::string text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace((unsigned char) text[begin])) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace((unsigned char) text[end - 1])) {
        --end;
    }
    return text.substr(begin, end - begin);
}

bool parse_u64_text(const std::string & text, uint64_t & out) {
    if (text.empty()) return false;
    errno = 0;
    char * end = nullptr;
    unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0') {
        return false;
    }
    out = (uint64_t) value;
    return true;
}

bool parse_capacity_token(const std::string & raw, uint64_t & out_bytes) {
    const std::string text = trim_ascii(raw);
    if (text.empty()) return false;

    size_t suffix_begin = text.size();
    while (suffix_begin > 0 &&
           std::isalpha((unsigned char) text[suffix_begin - 1])) {
        --suffix_begin;
    }
    const std::string number = trim_ascii(text.substr(0, suffix_begin));
    const std::string suffix = text.substr(suffix_begin);
    uint64_t value = 0;
    if (!parse_u64_text(number, value)) {
        return false;
    }

    uint64_t scale = 1;
    if (suffix.empty() || suffix == "b" || suffix == "B") {
        scale = 1;
    } else if (suffix == "k" || suffix == "K" || suffix == "kb" || suffix == "KB") {
        scale = 1024ULL;
    } else if (suffix == "m" || suffix == "M" || suffix == "mb" || suffix == "MB") {
        scale = 1024ULL * 1024ULL;
    } else if (suffix == "g" || suffix == "G" || suffix == "gb" || suffix == "GB") {
        scale = 1024ULL * 1024ULL * 1024ULL;
    } else {
        return false;
    }
    if (value > std::numeric_limits<uint64_t>::max() / scale) {
        return false;
    }
    out_bytes = value * scale;
    return true;
}

bool query_backend_device_total_memory(PlacementBackend backend,
                                       int device_id,
                                       uint64_t & total_bytes) {
    if (device_id < 0) return false;
    const char * reg_name = nullptr;
    switch (backend) {
        case PlacementBackend::Auto:
            reg_name = placement_backend_name(compiled_placement_backend());
            break;
        case PlacementBackend::Cuda:
        case PlacementBackend::Hip:
            reg_name = placement_backend_name(backend);
            break;
    }
    if (!reg_name) return false;

    ggml_backend_reg_t reg = ggml_backend_reg_by_name(reg_name);
    if (!reg) return false;
    const size_t n_dev = ggml_backend_reg_dev_count(reg);
    if (device_id < 0 || (size_t)device_id >= n_dev) return false;
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, (size_t)device_id);
    if (!dev) return false;
    size_t free_bytes = 0;
    size_t total = 0;
    ggml_backend_dev_memory(dev, &free_bytes, &total);
    (void) free_bytes;
    if (total == 0) return false;
    total_bytes = (uint64_t) total;
    return true;
}

bool specs_have_duplicate_device(const std::vector<ExpertSplitTargetSpec> & specs) {
    for (size_t i = 0; i < specs.size(); ++i) {
        for (size_t j = i + 1; j < specs.size(); ++j) {
            if (specs[i].backend == specs[j].backend &&
                specs[i].device_id == specs[j].device_id) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

bool parse_expert_split_target_list(const std::string & value,
                                    std::vector<ExpertSplitTargetSpec> & out,
                                    std::string * err) {
    out.clear();
    if (value.empty()) {
        if (err) *err = "expert target list is empty";
        return false;
    }

    size_t begin = 0;
    while (begin < value.size()) {
        const size_t end = value.find(',', begin);
        const std::string item = trim_ascii(value.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin));
        if (item.empty()) {
            if (err) *err = "expert target list contains an empty item";
            out.clear();
            return false;
        }

        ExpertSplitTargetSpec spec;
        if (item == "cpu") {
            spec.backend = PlacementBackend::Auto;
            spec.device_id = -1;
            spec.auto_capacity = false;
            spec.capacity_bytes = 0;
            spec.unlimited = true;
        } else {
            const size_t sep = item.find(':');
            if (sep == std::string::npos || sep == 0 || sep + 1 >= item.size()) {
                if (err) *err = "expert target item must look like backend:id or cpu";
                out.clear();
                return false;
            }
            if (!parse_placement_backend(item.substr(0, sep), spec.backend)) {
                if (err) *err = "unsupported expert target backend: " + item.substr(0, sep);
                out.clear();
                return false;
            }
            char * num_end = nullptr;
            long device_id = std::strtol(item.c_str() + sep + 1, &num_end, 10);
            if (num_end == item.c_str() + sep + 1 || *num_end != '\0' || device_id < 0) {
                if (err) *err = "invalid expert target device id: " + item;
                out.clear();
                return false;
            }
            spec.device_id = (int)device_id;
            spec.auto_capacity = true;
            spec.unlimited = false;
        }

        out.push_back(spec);
        if (end == std::string::npos) break;
        begin = end + 1;
    }

    if (out.empty()) {
        if (err) *err = "expert target list is empty";
        return false;
    }
    if (specs_have_duplicate_device(out)) {
        if (err) *err = "expert target list contains duplicate devices";
        out.clear();
        return false;
    }
    return true;
}

bool parse_expert_split_capacity_overrides(const std::string & value,
                                           std::vector<uint64_t> & out_bytes,
                                           std::string * err) {
    out_bytes.clear();
    if (value.empty()) return true;

    size_t begin = 0;
    while (begin < value.size()) {
        const size_t end = value.find(',', begin);
        const std::string item = trim_ascii(value.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin));
        if (item.empty()) {
            if (err) *err = "expert target capacity list contains an empty item";
            out_bytes.clear();
            return false;
        }
        uint64_t bytes = 0;
        if (item == "auto" || item == "AUTO") {
            out_bytes.push_back(0);
        } else if (!parse_capacity_token(item, bytes)) {
            if (err) *err = "invalid expert target capacity: " + item;
            out_bytes.clear();
            return false;
        } else {
            out_bytes.push_back(bytes);
        }
        if (end == std::string::npos) break;
        begin = end + 1;
    }

    return true;
}

bool build_expert_split_targets(const std::vector<ExpertSplitTargetSpec> & specs,
                                uint64_t primary_capacity_bytes,
                                std::vector<ExpertSplitTarget> & out,
                                std::string * err) {
    out.clear();
    if (specs.empty()) {
        if (err) *err = "no expert targets configured";
        return false;
    }

    out.reserve(specs.size());
    for (size_t i = 0; i < specs.size(); ++i) {
        const ExpertSplitTargetSpec & spec = specs[i];
        ExpertSplitTarget target;
        if (spec.unlimited) {
            target.name = "cpu";
            target.backend = "cpu";
            target.device_id = -1;
            target.unlimited = true;
            out.push_back(std::move(target));
            continue;
        }

        PlacementBackend backend =
            spec.backend == PlacementBackend::Auto
                ? compiled_placement_backend()
                : spec.backend;
        target.name = std::string(placement_backend_name(backend)) + ":" +
                      std::to_string(spec.device_id);
        target.backend = placement_backend_name(backend);
        target.device_id = spec.device_id;
        target.unlimited = false;

        uint64_t capacity_bytes = 0;
        if (i == 0 && primary_capacity_bytes > 0) {
            capacity_bytes = primary_capacity_bytes;
        } else if (!spec.auto_capacity) {
            capacity_bytes = spec.capacity_bytes;
        } else if (spec.capacity_bytes > 0) {
            capacity_bytes = spec.capacity_bytes;
        } else if (!query_backend_device_total_memory(backend, spec.device_id, capacity_bytes)) {
            if (err) {
                *err = "could not auto-discover capacity for expert target " +
                       target.name;
            }
            out.clear();
            return false;
        }

        target.capacity_bytes = capacity_bytes;
        out.push_back(std::move(target));
    }

    return true;
}

bool resolve_expert_split_targets_from_env(const char * targets_env_name,
                                           const char * caps_env_name,
                                           uint64_t primary_capacity_bytes,
                                           std::vector<ExpertSplitTarget> & out,
                                           std::string * err) {
    out.clear();
    if (!targets_env_name || !*targets_env_name) {
        if (err) *err = "expert target env name is empty";
        return false;
    }

    const char * targets_env = std::getenv(targets_env_name);
    if (!targets_env || !*targets_env) {
        return true;
    }

    std::vector<ExpertSplitTargetSpec> specs;
    if (!parse_expert_split_target_list(targets_env, specs, err)) {
        return false;
    }

    if (caps_env_name && *caps_env_name) {
        if (const char * caps_env = std::getenv(caps_env_name)) {
            std::vector<uint64_t> caps;
            if (!parse_expert_split_capacity_overrides(caps_env, caps, err)) {
                return false;
            }
            if (!caps.empty() && caps.size() != specs.size()) {
                if (err) {
                    *err = "expert target cap count must match expert target count";
                }
                return false;
            }
            for (size_t i = 0; i < caps.size(); ++i) {
                if (caps[i] == 0) continue;
                specs[i].auto_capacity = false;
                specs[i].capacity_bytes = caps[i];
            }
        }
    }

    return build_expert_split_targets(specs, primary_capacity_bytes, out, err);
}

bool validate_primary_expert_split_target(
        const std::vector<ExpertSplitTarget> & targets,
        PlacementBackend local_backend,
        int local_device_id,
        std::string * err) {
    if (targets.empty()) {
        if (err) *err = "expert split requires at least one target";
        return false;
    }
    const char * backend_name = placement_backend_name(local_backend);
    if (!backend_name) {
        if (err) *err = "could not resolve local expert backend";
        return false;
    }
    const ExpertSplitTarget & primary = targets.front();
    if (primary.backend != backend_name || primary.device_id != local_device_id) {
        if (err) {
            *err = "first expert target must match the local backend device (" +
                std::string(backend_name) + ":" + std::to_string(local_device_id) + ")";
        }
        return false;
    }
    return true;
}

}  // namespace dflash::common
