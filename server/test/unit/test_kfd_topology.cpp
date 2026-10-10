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
#endif
