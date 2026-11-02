#include "CppUnitTestFramework.hpp"
#include "placement/kfd_topology.h"

#if defined(__linux__)
#include <cstdlib>

using namespace CppUnitTestFramework;

namespace {

struct KfdTopologyFixture : CommonFixture {
    using CommonFixture::CommonFixture;
};

}  // namespace

// Selection-policy tests supply GPU facts directly and cannot detect discovery
// aliasing: different PCI functions/partitions must retain distinct KFD IDs.
TEST_CASE(KfdTopologyFixture, test_kfd_mapping_preserves_full_pci_address) {
    namespace fs = std::filesystem;
    std::string path = (fs::temp_directory_path() / "luce-kfd-topology-XXXXXX").string();
    REQUIRE(mkdtemp(path.data()) != nullptr);
    const fs::path nodes = path;
    struct Cleanup {
        fs::path root;
        ~Cleanup() { std::error_code ec; fs::remove_all(root, ec); }
    } cleanup{nodes};
    auto add_node = [&](const char * name, unsigned domain, unsigned location, uint32_t id) {
        const fs::path node = nodes / name;
        fs::create_directory(node);
        std::ofstream(node / "properties") << "domain " << domain
            << "\nlocation_id " << location << '\n';
        std::ofstream(node / "gpu_id") << id << '\n';
    };
    auto lookup = [&](const char * address) {
        return luce::common::detail::kfd_gpu_id(nodes, address);
    };

    add_node("0", 0, 0, 0);          // CPU topology node is not a GPU.
    add_node("1", 0, 0x4100, 101);   // 0000:41:00.0
    add_node("2", 0, 0x4101, 202);   // 0000:41:00.1, same device, another function/partition.
    add_node("3", 1, 0x4101, 303);   // 0001:41:00.1, distinct PCI domain.
    add_node("4", 0, 0x411a, 404);   // 0000:41:03.2, nonzero PCI device as well.

    CHECK(lookup("0000:41:00.0") == 101);
    CHECK(lookup("0000:41:00.1") == 202);
    CHECK(lookup("0001:41:00.1") == 303);
    CHECK(lookup("0000:41:03.2") == 404);
    CHECK(lookup("0000:41:00.2") == 0);
    CHECK(lookup("0000:42:00.0") == 0);
    CHECK(lookup("0000:00:00.0") == 0);

    fs::remove_all(nodes / "1");
    CHECK(lookup("0000:41:00.1") == 202); // Must not require function zero to be present.
    fs::remove(nodes / "2" / "properties");
    CHECK(lookup("0000:41:00.1") == 0);
    fs::remove_all(nodes);
    CHECK(lookup("0000:41:00.1") == 0); // KFD may be unavailable to this process.
}

// A model card's default profile is matched on these GPUs before HIP starts,
// so they must follow HIP's device order and its visibility rules.
TEST_CASE(KfdTopologyFixture, test_kfd_visible_gpus_follow_hip_order) {
    namespace fs = std::filesystem;
    std::string path = (fs::temp_directory_path() / "luce-kfd-archs-XXXXXX").string();
    REQUIRE(mkdtemp(path.data()) != nullptr);
    const fs::path root = path, nodes = root / "nodes", dri = root / "dri";
    fs::create_directory(nodes);
    fs::create_directory(dri);
    struct Cleanup {
        fs::path root;
        ~Cleanup() { std::error_code ec; fs::remove_all(root, ec); }
    } cleanup{root};
    // A GPU node names its render node; this process can open the ones in `dri`.
    // Its memory banks hold its memory, in bytes.
    auto add_node = [&](const char * name, unsigned version, int render_minor,
                        std::vector<uint64_t> banks = {}) {
        fs::create_directory(nodes / name);
        std::ofstream props(nodes / name / "properties");
        props << "cpu_cores_count 16\ngfx_target_version " << version << "\n";
        if (render_minor >= 0) {
            props << "drm_render_minor " << render_minor << "\n";
            std::ofstream(dri / ("renderD" + std::to_string(render_minor)));
        }
        props << "simd_count 64\n";
        for (size_t i = 0; i < banks.size(); ++i) {
            const fs::path bank = nodes / name / "mem_banks" / std::to_string(i);
            fs::create_directories(bank);
            std::ofstream(bank / "properties") << "heap_type 1\nsize_in_bytes " << banks[i]
                                               << "\nflags 0\nwidth 256\n";
        }
    };
    using luce::common::detail::KfdGpu;
    using luce::common::detail::kfd_visible_gpus;
    std::vector<KfdGpu> gpus;
    std::vector<std::string> archs;
    std::vector<uint64_t> memory;
    const auto visible = [&](const char * rocr, const char * hip) {
        const bool known = kfd_visible_gpus(nodes, dri, rocr, hip, gpus);
        archs.clear();
        memory.clear();
        for (const KfdGpu & gpu : gpus) {
            archs.push_back(gpu.arch);
            memory.push_back(gpu.memory_bytes);
        }
        return known;
    };

    CHECK(!kfd_visible_gpus(root / "missing", dri, nullptr, nullptr, gpus));
    add_node("0", 0, 0);            // the CPU (render minor 0, as KFD reports it)
    CHECK(visible(nullptr, nullptr) && archs.empty());
    add_node("1", 120001, 128, {34208743424});    // R9700: its VRAM
    add_node("2", 110501, 129, {103079215104});   // Strix Halo: its GTT
    add_node("10", 90010, 130, {1ull << 30, 1ull << 30});  // node 10 sorts after node 2;
                                                            // minor/stepping in hex; banks add up
    CHECK(visible(nullptr, nullptr));
    CHECK(archs == std::vector<std::string>({"gfx1201", "gfx1151", "gfx90a"}));
    CHECK(memory == std::vector<uint64_t>({34208743424, 103079215104, 2ull << 30}));

    // ROCR_VISIBLE_DEVICES picks from the GPUs, HIP_VISIBLE_DEVICES from those.
    CHECK(visible("1", nullptr));
    CHECK(archs == std::vector<std::string>({"gfx1151"}));
    CHECK(memory == std::vector<uint64_t>({103079215104}));
    CHECK(visible("2,0", "1"));
    CHECK(archs == std::vector<std::string>({"gfx1201"}));
    CHECK(visible(nullptr, "1,0"));
    CHECK(archs == std::vector<std::string>({"gfx1151", "gfx1201"}));
    CHECK(memory == std::vector<uint64_t>({103079215104, 34208743424}));
    // HIP skips an index it already took and takes an empty list as unset;
    // an empty ROCR_VISIBLE_DEVICES hides every GPU.
    CHECK(visible(nullptr, "0,0,1"));
    CHECK(archs == std::vector<std::string>({"gfx1201", "gfx1151"}));
    CHECK(visible(nullptr, "") && archs.size() == 3);
    CHECK(visible("", nullptr) && archs.empty());
    // UUIDs, hidden devices, bad indices and repeats for ROCr leave the order unknown.
    for (const char * list : {"GPU-4a2b", "-1", "3", "0,", "0;1", "0, 1", "+1", "01", " 0"}) {
        CHECK(!visible(list, nullptr));
        CHECK(gpus.empty());
        CHECK(!visible(nullptr, list));
    }
    CHECK(!visible("0,0", nullptr));

    // A container given only the Strix Halo's render node: ROCm numbers it hip:0.
    fs::remove(dri / "renderD128");
    CHECK(visible(nullptr, nullptr));
    CHECK(archs == std::vector<std::string>({"gfx1151", "gfx90a"}));
    // A GPU without memory banks reports none.
    add_node("5", 120001, 132);
    CHECK(visible(nullptr, nullptr));
    CHECK(archs == std::vector<std::string>({"gfx1151", "gfx1201", "gfx90a"}));
    CHECK(memory == std::vector<uint64_t>({103079215104, 0, 2ull << 30}));
    fs::remove_all(nodes / "5");
    // A GPU node without a render node, or without a version: unknown.
    add_node("3", 120001, -1);
    CHECK(!visible(nullptr, nullptr));
    fs::remove_all(nodes / "3");
    add_node("4", 0, 131);
    CHECK(!visible(nullptr, nullptr));
}
#endif
