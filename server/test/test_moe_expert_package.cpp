#include "CppUnitTestFramework.hpp"
#include "common/moe_expert_package.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace dflash::common;

#define PACKAGE_REQUIRE(cond) do { \
    if (!(cond)) throw std::runtime_error(std::string(__FILE__) + ":" + \
        std::to_string(__LINE__) + ": " + #cond); \
} while (0)

namespace {

struct MoeExpertPackageFixture {};

bool aligned_allocate(void ** ptr, size_t bytes, void *) {
#if defined(_WIN32)
    *ptr = _aligned_malloc(bytes, 4096);
    return *ptr != nullptr;
#else
    return ::posix_memalign(ptr, 4096, bytes) == 0;
#endif
}

void aligned_free(void * ptr, void *) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

uint8_t pattern(int layer, int component, int expert, size_t byte) {
    return (uint8_t) ((layer * 67 + component * 29 + expert * 13 + byte * 3) & 0xff);
}

void fill_component(std::vector<uint8_t> & source,
                    const ExpertFileRegion & region,
                    size_t expert_bytes, int layer, int component,
                    uint32_t experts) {
    for (uint32_t expert = 0; expert < experts; ++expert) {
        for (size_t byte = 0; byte < expert_bytes; ++byte) {
            source[region.offset + (size_t) expert * expert_bytes + byte] =
                pattern(layer, component, (int) expert, byte);
        }
    }
}

struct PackageModel {
    std::vector<uint8_t> shard0 = std::vector<uint8_t>(64 * 1024, 0xa5);
    std::vector<uint8_t> shard1 = std::vector<uint8_t>(64 * 1024, 0x5a);
    std::vector<LayerExpertRegions> layers{2};
    std::vector<uint32_t> experts{3, 2};

    PackageModel() {
        LayerExpertRegions & ordinary = layers[0];
        ordinary.expert_bytes_gate = 23;
        ordinary.expert_bytes_up = 19;
        ordinary.expert_bytes_down = 31;
        ordinary.gate_exps = {101, ordinary.expert_bytes_gate * experts[0], 0};
        ordinary.up_exps = {211, ordinary.expert_bytes_up * experts[0], 1};
        ordinary.down_exps = {401, ordinary.expert_bytes_down * experts[0], 1};
        fill_component(shard0, ordinary.gate_exps,
                       ordinary.expert_bytes_gate, 0, 0, experts[0]);
        fill_component(shard1, ordinary.up_exps,
                       ordinary.expert_bytes_up, 0, 1, experts[0]);
        fill_component(shard1, ordinary.down_exps,
                       ordinary.expert_bytes_down, 0, 2, experts[0]);

        LayerExpertRegions & fused = layers[1];
        fused.fused_gate_up = true;
        fused.expert_bytes_gate_up = 47;
        fused.expert_bytes_down = 37;
        fused.gate_up_exps = {1001, fused.expert_bytes_gate_up * experts[1], 0};
        fused.down_exps = {1401, fused.expert_bytes_down * experts[1], 1};
        fill_component(shard0, fused.gate_up_exps,
                       fused.expert_bytes_gate_up, 1, 0, experts[1]);
        fill_component(shard1, fused.down_exps,
                       fused.expert_bytes_down, 1, 1, experts[1]);
    }

    std::vector<MoeNvmeSource> sources() const {
        return {
            {shard0.data(), shard0.size(), -1},
            {shard1.data(), shard1.size(), -1},
        };
    }
};

int make_temp_file(char * path) {
#if defined(_WIN32)
    (void) path;
    char directory[MAX_PATH]{};
    char generated[MAX_PATH]{};
    if (::GetTempPathA(MAX_PATH, directory) == 0 ||
        ::GetTempFileNameA(directory, "lbm", 0, generated) == 0) {
        return -1;
    }
    return ::_open(generated,
                   _O_CREAT | _O_TRUNC | _O_RDWR | _O_BINARY | _O_TEMPORARY,
                   _S_IREAD | _S_IWRITE);
#else
    const int fd = ::mkstemp(path);
    if (fd >= 0) ::unlink(path);
    return fd;
#endif
}

void close_file(int fd) {
#if defined(_WIN32)
    ::_close(fd);
#else
    ::close(fd);
#endif
}

size_t file_size(int fd) {
#if defined(_WIN32)
    struct _stat64 st {};
    return ::_fstat64(fd, &st) == 0 ? (size_t) st.st_size : 0;
#else
    struct stat st {};
    return ::fstat(fd, &st) == 0 ? (size_t) st.st_size : 0;
#endif
}

void verify_component(const MoeNvmeLease & lease,
                      MoeExpertComponentKind kind,
                      int layer, int component, int expert, size_t bytes) {
    const MoeExpertComponentLayout * layout = lease.layout().component(kind);
    PACKAGE_REQUIRE(layout != nullptr);
    const MoeExpertIoSpan & span = lease.layout().spans[0];
    PACKAGE_REQUIRE(layout->device_offset >= span.device_offset);
    const size_t delta = layout->device_offset - span.device_offset;
    const uint8_t * data = lease.data() + span.buffer_offset + delta;
    PACKAGE_REQUIRE(data[0] == pattern(layer, component, expert, 0));
    PACKAGE_REQUIRE(data[bytes / 2] == pattern(layer, component, expert, bytes / 2));
    PACKAGE_REQUIRE(data[bytes - 1] == pattern(layer, component, expert, bytes - 1));
}

} // namespace

TEST_CASE(MoeExpertPackageFixture, plans_aligned_one_read_records) {
    PackageModel model;
    MoeExpertPackageManifest manifest;
    MoeExpertPackageOptions options;
    options.sync_on_finish = false;
    std::string err;
    PACKAGE_REQUIRE(plan_moe_expert_package(
        model.sources(), model.layers, model.experts,
        options, manifest, &err));
    PACKAGE_REQUIRE(manifest.version == 1);
    PACKAGE_REQUIRE(manifest.layer_regions.size() == 2);
    PACKAGE_REQUIRE(manifest.source_layout_hash != 0);
    for (const LayerExpertRegions & layer : manifest.layer_regions) {
        PACKAGE_REQUIRE(layer.expert_major.enabled);
        PACKAGE_REQUIRE(layer.expert_major.expert_stride % 4096 == 0);
        PACKAGE_REQUIRE(layer.expert_major.experts.offset % 4096 == 0);
        PACKAGE_REQUIRE(layer.expert_major.experts.source_index == 0);
    }
}

TEST_CASE(MoeExpertPackageFixture, roundtrip_preserves_split_source_bytes) {
    PackageModel model;
    char path[] = "/tmp/moe_expert_package_XXXXXX";
    const int fd = make_temp_file(path);
    PACKAGE_REQUIRE(fd >= 0);

    MoeExpertPackageOptions options;
    options.sync_on_finish = false;
    MoeExpertPackageManifest written;
    std::string err;
    PACKAGE_REQUIRE(write_moe_expert_package(
        fd, model.sources(), model.layers, model.experts,
        options, &written, &err));
    PACKAGE_REQUIRE(file_size(fd) == written.file_bytes);

    MoeNvmeSource package{nullptr, file_size(fd), fd};
    MoeExpertPackageManifest loaded;
    PACKAGE_REQUIRE(read_moe_expert_package(package, loaded, &err));
    PACKAGE_REQUIRE(loaded.source_layout_hash ==
                    moe_expert_source_layout_hash(
                        model.sources(), model.layers, model.experts));
    PACKAGE_REQUIRE(loaded.expert_counts == model.experts);

    MoeNvmeConfig config;
    config.backend = MoeNvmeBackend::ThreadPool;
    config.direct_io = MoeNvmeDirectMode::Disabled;
    config.host_slots = 2;
    config.io_threads = 1;
    MoeNvmeScheduler scheduler;
    PACKAGE_REQUIRE(scheduler.init(
        config, loaded.max_record_bytes,
        aligned_allocate, aligned_free, nullptr, &err));
    PACKAGE_REQUIRE(scheduler.bind_source(
        package, loaded.layer_regions, &err));

    MoeNvmeLease ordinary;
    PACKAGE_REQUIRE(scheduler.acquire(0, 2, ordinary, &err));
    PACKAGE_REQUIRE(ordinary.layout().span_count == 1);
    verify_component(ordinary, MoeExpertComponentKind::Gate,
                     0, 0, 2, model.layers[0].expert_bytes_gate);
    verify_component(ordinary, MoeExpertComponentKind::Up,
                     0, 1, 2, model.layers[0].expert_bytes_up);
    verify_component(ordinary, MoeExpertComponentKind::Down,
                     0, 2, 2, model.layers[0].expert_bytes_down);
    ordinary.reset();

    MoeNvmeLease fused;
    PACKAGE_REQUIRE(scheduler.acquire(1, 1, fused, &err));
    PACKAGE_REQUIRE(fused.layout().span_count == 1);
    verify_component(fused, MoeExpertComponentKind::FusedGateUp,
                     1, 0, 1, model.layers[1].expert_bytes_gate_up);
    verify_component(fused, MoeExpertComponentKind::Down,
                     1, 1, 1, model.layers[1].expert_bytes_down);
    fused.reset();
    PACKAGE_REQUIRE(scheduler.stats().read_ops == 2);
    PACKAGE_REQUIRE(scheduler.stats().errors == 0);
    scheduler.destroy();
    close_file(fd);
}

TEST_CASE(MoeExpertPackageFixture, rejects_incomplete_and_mismatched_sources) {
    PackageModel model;
    const uint64_t original_hash = moe_expert_source_layout_hash(
        model.sources(), model.layers, model.experts);
    PACKAGE_REQUIRE(original_hash != 0);
    model.shard0[model.layers[0].gate_exps.offset] ^= 0x1;
    PACKAGE_REQUIRE(moe_expert_source_layout_hash(
        model.sources(), model.layers, model.experts) != original_hash);

    std::vector<MoeNvmeSource> sources = model.sources();
    sources[1].mmap_size = 256;
    MoeExpertPackageManifest manifest;
    MoeExpertPackageOptions options;
    options.sync_on_finish = false;
    std::string err;
    PACKAGE_REQUIRE(!plan_moe_expert_package(
        sources, model.layers, model.experts,
        options, manifest, &err));
    PACKAGE_REQUIRE(!err.empty());

    std::vector<uint8_t> incomplete(4096, 0);
    PACKAGE_REQUIRE(!read_moe_expert_package(
        {incomplete.data(), incomplete.size(), -1}, manifest, &err));
    PACKAGE_REQUIRE(err.find("magic") != std::string::npos);
}
