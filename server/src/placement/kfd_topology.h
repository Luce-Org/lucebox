#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

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

}  // namespace luce::common::detail
