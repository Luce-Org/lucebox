#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace luce::common::detail {

// KFD's location_id contains the full PCI bus/device/function tuple. HIP's
// PCI address preserves the function (also used for GPU partition IDs),
// unlike hipDeviceProp_t, which exposes only domain, bus and device.
// The root is explicit so the mapping can be exercised without a GPU.
inline uint32_t kfd_gpu_id(const std::filesystem::path & nodes,
                           const char * pci_address) {
    unsigned pci_domain = 0, pci_bus = 0, pci_device = 0, pci_function = 0;
    if (std::sscanf(pci_address, "%x:%x:%x.%x", &pci_domain, &pci_bus,
                    &pci_device, &pci_function) != 4) {
        return 0;
    }
    const unsigned location = (pci_bus << 8) | (pci_device << 3) | pci_function;
    namespace fs = std::filesystem;
    std::error_code ec;
    for (auto node = fs::directory_iterator(nodes, ec);
         !ec && node != fs::directory_iterator(); node.increment(ec)) {
        std::ifstream props(node->path() / "properties");
        std::string key;
        unsigned long long value = 0;
        long long domain = -1, location_id = -1;
        while (props >> key >> value) {
            if (key == "domain") domain = (long long) value;
            if (key == "location_id") location_id = (long long) value;
        }
        if (domain != pci_domain || location_id != (long long) location) continue;
        uint32_t gpu_id = 0;
        std::ifstream id(node->path() / "gpu_id");
        if (id >> gpu_id && gpu_id) return gpu_id;
    }
    return 0;
}

// Narrows `items` to the ones a device visibility list names (comma-separated
// indices into `items`); an unset list keeps them all. ROCR_VISIBLE_DEVICES
// (`rocr`) set but empty hides every GPU, and HIP takes an empty list as unset
// and skips an index it already took. Anything but plain in-range indices (a
// UUID, -1, " 1", an index ROCr is given twice) leaves the result unknown.
template <typename Item>
inline bool kfd_apply_visible_list(const char * list, bool rocr, std::vector<Item> & items) {
    if (!list) return true;
    if (!*list) {
        if (rocr) items.clear();
        return true;
    }
    std::vector<Item> kept;
    std::vector<bool> taken(items.size(), false);
    const std::string text = list;
    for (size_t start = 0;;) {
        const size_t comma = text.find(',', start);
        const std::string token = text.substr(start, comma == std::string::npos ? comma : comma - start);
        // Only a plain decimal index: HIP stops at " 1", "+1" or "01".
        char * end = nullptr;
        const long index = std::strtol(token.c_str(), &end, 10);
        if (token.empty() || *end != '\0' || token != std::to_string(index) || index < 0 ||
            index >= (long) items.size() || (rocr && taken[(size_t) index])) {
            return false;
        }
        if (!taken[(size_t) index]) kept.push_back(items[(size_t) index]);
        taken[(size_t) index] = true;
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    items = std::move(kept);
    return true;
}

// A GPU as KFD describes it: its architecture ("gfx1201") and the size of its
// node's memory banks, which is a discrete GPU's VRAM and, on the Strix Halo,
// the GTT it maps rather than the carve.
struct KfdGpu {
    std::string arch;
    uint64_t memory_bytes = 0;
};

// The GPUs HIP enumerates, in its device order: the KFD nodes that hold a GPU
// whose render node (under `dri`) this process can open, in node order,
// narrowed by ROCR_VISIBLE_DEVICES and then by HIP_VISIBLE_DEVICES
// (`rocr_visible`, `hip_visible`). It reads sysfs and opens render nodes
// only, so the HIP runtime has not started when it returns. False when the
// order cannot be known: no KFD topology, a GPU node without a render node,
// or a visibility list that is not plain indices.
inline bool kfd_visible_gpus(const std::filesystem::path & nodes, const std::filesystem::path & dri,
                             const char * rocr_visible, const char * hip_visible,
                             std::vector<KfdGpu> & visible) {
    namespace fs = std::filesystem;
    visible.clear();
    std::vector<std::pair<long, KfdGpu>> gpus;
    std::error_code ec;
    for (auto node = fs::directory_iterator(nodes, ec);
         !ec && node != fs::directory_iterator(); node.increment(ec)) {
        const std::string name = node->path().filename().string();
        char * end = nullptr;
        const long index = std::strtol(name.c_str(), &end, 10);
        if (name.empty() || *end != '\0') continue;
        std::ifstream props(node->path() / "properties");
        std::string key;
        unsigned long long value = 0, version = 0;
        long long render_minor = -1;
        while (props >> key >> value) {
            if (key == "gfx_target_version") version = value;
            if (key == "drm_render_minor") render_minor = (long long) value;
        }
        // A CPU node has no GPU version and render minor 0; a node with a render
        // node but no version is a GPU whose architecture is unknown.
        if (version == 0) {
            if (render_minor > 0) return false;
            continue;
        }
        if (render_minor < 0) return false;
        // ROCm leaves out a GPU whose render node it cannot open (a container
        // given only some of /dev/dri) before it numbers the others.
        std::fstream render(dri / ("renderD" + std::to_string(render_minor)),
                            std::ios::in | std::ios::out);
        if (!render.is_open()) continue;
        // gfx_target_version is major * 10000 + minor * 100 + stepping, and
        // the name spells minor and stepping as one hex digit each.
        char arch[32];
        std::snprintf(arch, sizeof(arch), "gfx%llu%llx%llx", version / 10000,
                      (version / 100) % 100, version % 100);
        KfdGpu gpu{arch};
        std::error_code bank_ec;
        for (auto bank = fs::directory_iterator(node->path() / "mem_banks", bank_ec);
             !bank_ec && bank != fs::directory_iterator(); bank.increment(bank_ec)) {
            std::ifstream bank_props(bank->path() / "properties");
            while (bank_props >> key >> value) {
                if (key == "size_in_bytes") gpu.memory_bytes += value;
            }
        }
        gpus.emplace_back(index, std::move(gpu));
    }
    if (ec) return false;
    std::sort(gpus.begin(), gpus.end(),
              [](const auto & a, const auto & b) { return a.first < b.first; });
    for (auto & gpu : gpus) visible.push_back(std::move(gpu.second));
    if (!kfd_apply_visible_list(rocr_visible, true, visible) ||
        !kfd_apply_visible_list(hip_visible, false, visible)) {
        visible.clear();
        return false;
    }
    return true;
}

}  // namespace luce::common::detail
