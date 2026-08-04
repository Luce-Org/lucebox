#include "CppUnitTestFramework.hpp"
#include "common/moe_nvme_scheduler.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace dflash::common;

#define NVME_REQUIRE(cond) do { \
    if (!(cond)) throw std::runtime_error(std::string(__FILE__) + ":" + \
        std::to_string(__LINE__) + ": " + #cond); \
} while (0)

namespace {

struct MoeNvmeSchedulerFixture {};

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

uint8_t expected_byte(int layer, int tensor, int expert, size_t offset) {
    return (uint8_t) ((layer * 61 + tensor * 31 + expert * 17 + offset * 7) & 0xff);
}

void fill_tensor(std::vector<uint8_t> & file, const ExpertFileRegion & region,
                 size_t expert_bytes, int layer, int tensor, int experts) {
    for (int expert = 0; expert < experts; ++expert) {
        for (size_t i = 0; i < expert_bytes; ++i) {
            file[region.offset + (size_t) expert * expert_bytes + i] =
                expected_byte(layer, tensor, expert, i);
        }
    }
}

struct SyntheticModel {
    static constexpr int kExperts = 8;
    std::vector<uint8_t> file;
    std::vector<LayerExpertRegions> regions;

    SyntheticModel() : file(2 * 1024 * 1024, 0xa5), regions(2) {
        auto & a = regions[0];
        a.fused_gate_up = false;
        a.expert_bytes_gate = 4093;
        a.expert_bytes_up = 6141;
        a.expert_bytes_down = 8189;
        a.gate_exps = {257, a.expert_bytes_gate * kExperts};
        a.up_exps = {a.gate_exps.offset + a.gate_exps.size + 113,
                     a.expert_bytes_up * kExperts};
        a.down_exps = {a.up_exps.offset + a.up_exps.size + 197,
                       a.expert_bytes_down * kExperts};
        fill_tensor(file, a.gate_exps, a.expert_bytes_gate, 0, 0, kExperts);
        fill_tensor(file, a.up_exps, a.expert_bytes_up, 0, 1, kExperts);
        fill_tensor(file, a.down_exps, a.expert_bytes_down, 0, 2, kExperts);

        auto & b = regions[1];
        b.fused_gate_up = true;
        b.expert_bytes_gate_up = 10007;
        b.expert_bytes_down = 5003;
        b.gate_up_exps = {512 * 1024 + 73, b.expert_bytes_gate_up * kExperts};
        b.down_exps = {b.gate_up_exps.offset + b.gate_up_exps.size + 89,
                       b.expert_bytes_down * kExperts};
        fill_tensor(file, b.gate_up_exps, b.expert_bytes_gate_up, 1, 0, kExperts);
        fill_tensor(file, b.down_exps, b.expert_bytes_down, 1, 1, kExperts);
    }
};

void verify_lease(const MoeNvmeLease & lease, int layer, int expert) {
    NVME_REQUIRE(lease);
    const MoeExpertIoLayout & layout = lease.layout();
    NVME_REQUIRE(layout.key.layer == layer);
    NVME_REQUIRE(layout.key.expert == expert);
    const int expected_spans = layer == 0 ? 3 : 2;
    NVME_REQUIRE(layout.span_count == expected_spans);
    NVME_REQUIRE(layout.component_count == expected_spans);
    for (int tensor = 0; tensor < layout.span_count; ++tensor) {
        const MoeExpertIoSpan & span = layout.spans[tensor];
        const uint8_t * payload = lease.data() + span.buffer_offset;
        NVME_REQUIRE(payload[0] == expected_byte(layer, tensor, expert, 0));
        NVME_REQUIRE(payload[span.bytes / 2] ==
                     expected_byte(layer, tensor, expert, span.bytes / 2));
        NVME_REQUIRE(payload[span.bytes - 1] ==
                     expected_byte(layer, tensor, expert, span.bytes - 1));
        if (tensor > 0) {
            NVME_REQUIRE(span.device_offset ==
                         layout.spans[tensor - 1].device_offset +
                         layout.spans[tensor - 1].bytes);
        }
        NVME_REQUIRE((span.io_file_offset & 4095) == 0);
        NVME_REQUIRE((span.io_buffer_offset & 4095) == 0);
        NVME_REQUIRE((span.io_bytes & 4095) == 0);
    }
}

} // namespace

TEST_CASE(MoeNvmeSchedulerFixture, exact_layout_bounds_and_alignment) {
    SyntheticModel model;
    MoeExpertIoLayout layout;
    std::string err;
    NVME_REQUIRE(make_moe_expert_io_layout(
        0, 3, model.regions[0], model.file.size(), 4096, layout, &err));
    NVME_REQUIRE(layout.span_count == 3);
    NVME_REQUIRE(layout.component_count == 3);
    NVME_REQUIRE(layout.payload_bytes ==
                 model.regions[0].expert_bytes_gate +
                 model.regions[0].expert_bytes_up +
                 model.regions[0].expert_bytes_down);
    NVME_REQUIRE(layout.host_bytes >= layout.payload_bytes);
    NVME_REQUIRE(!make_moe_expert_io_layout(
        0, SyntheticModel::kExperts, model.regions[0], model.file.size(),
        4096, layout, &err));
    NVME_REQUIRE(!err.empty());
}

TEST_CASE(MoeNvmeSchedulerFixture, expert_major_layout_uses_one_read_without_changing_components) {
    constexpr int experts = SyntheticModel::kExperts;
    constexpr size_t gate_bytes = 4093;
    constexpr size_t up_bytes = 6141;
    constexpr size_t down_bytes = 8189;
    constexpr size_t gate_offset = 0;
    constexpr size_t up_offset = 4352;
    constexpr size_t down_offset = 10752;
    constexpr size_t stride = 19456;
    constexpr size_t file_offset = 4096;

    std::vector<uint8_t> file(file_offset + stride * experts + 4096, 0xa5);
    LayerExpertRegions layer;
    layer.expert_bytes_gate = gate_bytes;
    layer.expert_bytes_up = up_bytes;
    layer.expert_bytes_down = down_bytes;
    layer.expert_major.enabled = true;
    layer.expert_major.expert_stride = stride;
    layer.expert_major.gate_offset = gate_offset;
    layer.expert_major.up_offset = up_offset;
    layer.expert_major.down_offset = down_offset;
    layer.expert_major.experts = {file_offset, stride * experts, 0};
    for (int expert = 0; expert < experts; ++expert) {
        const size_t base = file_offset + (size_t) expert * stride;
        for (size_t i = 0; i < gate_bytes; ++i) {
            file[base + gate_offset + i] = expected_byte(0, 0, expert, i);
        }
        for (size_t i = 0; i < up_bytes; ++i) {
            file[base + up_offset + i] = expected_byte(0, 1, expert, i);
        }
        for (size_t i = 0; i < down_bytes; ++i) {
            file[base + down_offset + i] = expected_byte(0, 2, expert, i);
        }
    }

    MoeExpertIoLayout layout;
    std::string err;
    NVME_REQUIRE(make_moe_expert_io_layout(
        0, 5, layer, file.size(), 4096, layout, &err));
    NVME_REQUIRE(layout.span_count == 1);
    NVME_REQUIRE(layout.component_count == 3);
    NVME_REQUIRE(layout.payload_bytes == stride);
    NVME_REQUIRE(layout.component(MoeExpertComponentKind::Gate)->device_offset == gate_offset);
    NVME_REQUIRE(layout.component(MoeExpertComponentKind::Up)->device_offset == up_offset);
    NVME_REQUIRE(layout.component(MoeExpertComponentKind::Down)->device_offset == down_offset);

    MoeNvmeConfig config;
    config.backend = MoeNvmeBackend::Mmap;
    config.direct_io = MoeNvmeDirectMode::Disabled;
    config.host_slots = 4;
    MoeNvmeScheduler scheduler;
    NVME_REQUIRE(scheduler.init(config, stride, aligned_allocate,
                                aligned_free, nullptr, &err));
    NVME_REQUIRE(scheduler.bind_source(
        {file.data(), file.size(), -1}, {layer}, &err));
    MoeNvmeLease lease;
    NVME_REQUIRE(scheduler.acquire(0, 5, lease, &err));
    const MoeExpertIoSpan & record = lease.layout().spans[0];
    const uint8_t * payload = lease.data() + record.buffer_offset;
    NVME_REQUIRE(payload[gate_offset] == expected_byte(0, 0, 5, 0));
    NVME_REQUIRE(payload[up_offset + up_bytes / 2] ==
                 expected_byte(0, 1, 5, up_bytes / 2));
    NVME_REQUIRE(payload[down_offset + down_bytes - 1] ==
                 expected_byte(0, 2, 5, down_bytes - 1));
    lease.reset();
    NVME_REQUIRE(scheduler.stats().read_ops == 1);
    NVME_REQUIRE(scheduler.stats().errors == 0);
}

TEST_CASE(MoeNvmeSchedulerFixture, async_exact_reads_dedupe_and_cache) {
    SyntheticModel model;
    MoeNvmeConfig config;
    config.backend = MoeNvmeBackend::Mmap;
    config.direct_io = MoeNvmeDirectMode::Disabled;
    config.host_slots = 4;
    config.io_threads = 2;
    config.demand_reserve = 2;

    MoeNvmeScheduler scheduler;
    std::string err;
    const size_t max_payload = 32 * 1024;
    NVME_REQUIRE(scheduler.init(config, max_payload, aligned_allocate,
                                aligned_free, nullptr, &err));
    NVME_REQUIRE(scheduler.bind_source(
        {model.file.data(), model.file.size(), -1}, model.regions, &err));
    NVME_REQUIRE(std::string(scheduler.effective_backend_name()) == "mmap-workers");

    NVME_REQUIRE(scheduler.request(0, 3, MoeNvmePriority::Prefetch, &err));
    NVME_REQUIRE(scheduler.request(0, 3, MoeNvmePriority::Prefetch, &err));

    MoeNvmeLease first;
    NVME_REQUIRE(scheduler.acquire(0, 3, first, &err));
    verify_lease(first, 0, 3);
    first.reset();

    MoeNvmeLease cached;
    NVME_REQUIRE(scheduler.acquire(0, 3, cached, &err));
    verify_lease(cached, 0, 3);
    cached.reset();

    MoeNvmeLease fused;
    NVME_REQUIRE(scheduler.acquire(1, 5, fused, &err));
    verify_lease(fused, 1, 5);
    fused.reset();

    const MoeNvmeStats stats = scheduler.stats();
    NVME_REQUIRE(stats.requests >= 5);
    NVME_REQUIRE(stats.cache_hits >= 1);
    NVME_REQUIRE(stats.inflight_deduplications + stats.cache_hits >= 2);
    NVME_REQUIRE(stats.errors == 0);
    NVME_REQUIRE(stats.payload_bytes > 0);
    NVME_REQUIRE(stats.active_io_ns > 0);
}

TEST_CASE(MoeNvmeSchedulerFixture, speculation_cannot_consume_demand_reserve) {
    SyntheticModel model;
    MoeNvmeConfig config;
    config.backend = MoeNvmeBackend::Mmap;
    config.direct_io = MoeNvmeDirectMode::Disabled;
    config.host_slots = 4;
    config.io_threads = 1;
    config.demand_reserve = 2;

    MoeNvmeScheduler scheduler;
    std::string err;
    NVME_REQUIRE(scheduler.init(config, 32 * 1024, aligned_allocate,
                                aligned_free, nullptr, &err));
    NVME_REQUIRE(scheduler.bind_source(
        {model.file.data(), model.file.size(), -1}, model.regions, &err));

    NVME_REQUIRE(scheduler.request(0, 0, MoeNvmePriority::Prefetch, &err));
    NVME_REQUIRE(scheduler.request(0, 1, MoeNvmePriority::Prefetch, &err));
    NVME_REQUIRE(!scheduler.request(0, 2, MoeNvmePriority::Prefetch, nullptr));

    // Demand always has admission rights and upgrades a speculative resident.
    MoeNvmeLease demand;
    NVME_REQUIRE(scheduler.acquire(0, 2, demand, &err));
    verify_lease(demand, 0, 2);
    demand.reset();

    const MoeNvmeStats stats = scheduler.stats();
    NVME_REQUIRE(stats.prefetch_drops >= 1);
    NVME_REQUIRE(stats.errors == 0);
}

TEST_CASE(MoeNvmeSchedulerFixture, demand_timeout_prevents_busy_cache_deadlock) {
    SyntheticModel model;
    MoeNvmeConfig config;
    config.backend = MoeNvmeBackend::Mmap;
    config.direct_io = MoeNvmeDirectMode::Disabled;
    config.host_slots = 2;
    config.io_threads = 1;
    config.demand_timeout_ms = 25;

    MoeNvmeScheduler scheduler;
    std::string err;
    NVME_REQUIRE(scheduler.init(config, 32 * 1024, aligned_allocate,
                                aligned_free, nullptr, &err));
    NVME_REQUIRE(scheduler.bind_source(
        {model.file.data(), model.file.size(), -1}, model.regions, &err));

    MoeNvmeLease first;
    MoeNvmeLease second;
    MoeNvmeLease blocked;
    NVME_REQUIRE(scheduler.acquire(0, 0, first, &err));
    NVME_REQUIRE(scheduler.acquire(0, 1, second, &err));
    NVME_REQUIRE(!scheduler.acquire(0, 2, blocked, &err));
    NVME_REQUIRE(err.find("timed out") != std::string::npos);
    NVME_REQUIRE(scheduler.stats().demand_timeouts == 1);
    first.reset();
    second.reset();
}

TEST_CASE(MoeNvmeSchedulerFixture, split_model_reads_tensor_spans_from_multiple_shards) {
    constexpr int experts = SyntheticModel::kExperts;
    std::vector<uint8_t> shard_a(512 * 1024, 0xa5);
    std::vector<uint8_t> shard_b(512 * 1024, 0x5a);
    LayerExpertRegions layer;
    layer.fused_gate_up = false;
    layer.expert_bytes_gate = 4093;
    layer.expert_bytes_up = 6141;
    layer.expert_bytes_down = 8189;
    layer.gate_exps = {257, layer.expert_bytes_gate * experts, 0};
    layer.up_exps = {129, layer.expert_bytes_up * experts, 1};
    layer.down_exps = {
        layer.up_exps.offset + layer.up_exps.size + 197,
        layer.expert_bytes_down * experts, 1};
    fill_tensor(shard_a, layer.gate_exps, layer.expert_bytes_gate,
                0, 0, experts);
    fill_tensor(shard_b, layer.up_exps, layer.expert_bytes_up,
                0, 1, experts);
    fill_tensor(shard_b, layer.down_exps, layer.expert_bytes_down,
                0, 2, experts);

    MoeExpertIoLayout layout;
    std::string err;
    NVME_REQUIRE(make_moe_expert_io_layout(
        0, 6, layer, std::vector<size_t>{shard_a.size(), shard_b.size()},
        4096, layout, &err));
    NVME_REQUIRE(layout.spans[0].source_index == 0);
    NVME_REQUIRE(layout.spans[1].source_index == 1);
    NVME_REQUIRE(layout.spans[2].source_index == 1);

    MoeNvmeConfig config;
    config.backend = MoeNvmeBackend::Mmap;
    config.direct_io = MoeNvmeDirectMode::Disabled;
    config.host_slots = 4;
    config.io_threads = 2;
    MoeNvmeScheduler scheduler;
    NVME_REQUIRE(scheduler.init(config, 32 * 1024, aligned_allocate,
                                aligned_free, nullptr, &err));
    const std::vector<MoeNvmeSource> sources = {
        {shard_a.data(), shard_a.size(), -1},
        {shard_b.data(), shard_b.size(), -1},
    };
    NVME_REQUIRE(scheduler.bind_sources(sources, {layer}, &err));
    MoeNvmeLease lease;
    NVME_REQUIRE(scheduler.acquire(0, 6, lease, &err));
    verify_lease(lease, 0, 6);
    lease.reset();
    NVME_REQUIRE(scheduler.stats().read_ops == 3);
    NVME_REQUIRE(scheduler.stats().errors == 0);
}

#if !defined(_WIN32)
TEST_CASE(MoeNvmeSchedulerFixture, declared_shard_size_cannot_exceed_real_file) {
    SyntheticModel model;
    char path[] = "/tmp/moe_nvme_truncated_XXXXXX";
    const int fd = ::mkstemp(path);
    NVME_REQUIRE(fd >= 0);
    ::unlink(path);
    std::vector<uint8_t> bytes(4096, 0xa5);
    NVME_REQUIRE(::write(fd, bytes.data(), bytes.size()) == (ssize_t) bytes.size());

    MoeNvmeConfig config;
    config.backend = MoeNvmeBackend::ThreadPool;
    config.direct_io = MoeNvmeDirectMode::Disabled;
    config.host_slots = 2;
    MoeNvmeScheduler scheduler;
    std::string err;
    NVME_REQUIRE(scheduler.init(config, 32 * 1024, aligned_allocate,
                                aligned_free, nullptr, &err));
    NVME_REQUIRE(!scheduler.bind_source(
        {nullptr, bytes.size() * 2, fd}, model.regions, &err));
    NVME_REQUIRE(err.find("shorter") != std::string::npos);
    ::close(fd);
}

#if defined(__linux__) && defined(O_DIRECT)
TEST_CASE(MoeNvmeSchedulerFixture, direct_io_accepts_valid_unaligned_shard_tail) {
    constexpr size_t gate_bytes = 1000;
    constexpr size_t up_bytes = 1000;
    constexpr size_t down_bytes = 1300;
    LayerExpertRegions layer;
    layer.expert_bytes_gate = gate_bytes;
    layer.expert_bytes_up = up_bytes;
    layer.expert_bytes_down = down_bytes;
    layer.gate_exps = {123, gate_bytes};
    layer.up_exps = {2125, up_bytes};
    layer.down_exps = {8192 + 37, down_bytes};
    std::vector<uint8_t> file(layer.down_exps.offset + down_bytes, 0xa5);
    fill_tensor(file, layer.gate_exps, gate_bytes, 0, 0, 1);
    fill_tensor(file, layer.up_exps, up_bytes, 0, 1, 1);
    fill_tensor(file, layer.down_exps, down_bytes, 0, 2, 1);

    char path[] = "/tmp/moe_nvme_direct_tail_XXXXXX";
    const int fd = ::mkstemp(path);
    NVME_REQUIRE(fd >= 0);
    ::unlink(path);
    size_t written = 0;
    while (written < file.size()) {
        const ssize_t result = ::write(
            fd, file.data() + written, file.size() - written);
        NVME_REQUIRE(result > 0);
        written += (size_t) result;
    }

    MoeNvmeConfig config;
    config.backend = MoeNvmeBackend::ThreadPool;
    config.direct_io = MoeNvmeDirectMode::Enabled;
    config.host_slots = 2;
    config.io_threads = 1;
    MoeNvmeScheduler scheduler;
    std::string err;
    NVME_REQUIRE(scheduler.init(config, gate_bytes + up_bytes + down_bytes,
                                aligned_allocate, aligned_free, nullptr, &err));
    NVME_REQUIRE(scheduler.bind_source(
        {file.data(), file.size(), fd}, {layer}, &err));
    NVME_REQUIRE(scheduler.direct_io_active());
    MoeNvmeLease lease;
    NVME_REQUIRE(scheduler.acquire(0, 0, lease, &err));
    verify_lease(lease, 0, 0);
    lease.reset();
    NVME_REQUIRE(scheduler.stats().physical_bytes < 3 * 4096);
    NVME_REQUIRE(scheduler.stats().errors == 0);
    scheduler.destroy();
    ::close(fd);
}
#endif

TEST_CASE(MoeNvmeSchedulerFixture, split_real_files_use_the_fd_backend) {
    constexpr int experts = SyntheticModel::kExperts;
    std::vector<uint8_t> shard_a(512 * 1024, 0xa5);
    std::vector<uint8_t> shard_b(512 * 1024, 0x5a);
    LayerExpertRegions layer;
    layer.expert_bytes_gate = 4093;
    layer.expert_bytes_up = 6141;
    layer.expert_bytes_down = 8189;
    layer.gate_exps = {257, layer.expert_bytes_gate * experts, 0};
    layer.up_exps = {129, layer.expert_bytes_up * experts, 1};
    layer.down_exps = {
        layer.up_exps.offset + layer.up_exps.size + 197,
        layer.expert_bytes_down * experts, 1};
    fill_tensor(shard_a, layer.gate_exps, layer.expert_bytes_gate,
                0, 0, experts);
    fill_tensor(shard_b, layer.up_exps, layer.expert_bytes_up,
                0, 1, experts);
    fill_tensor(shard_b, layer.down_exps, layer.expert_bytes_down,
                0, 2, experts);

    char path_a[] = "/tmp/moe_nvme_shard_a_XXXXXX";
    char path_b[] = "/tmp/moe_nvme_shard_b_XXXXXX";
    const int fd_a = ::mkstemp(path_a);
    const int fd_b = ::mkstemp(path_b);
    NVME_REQUIRE(fd_a >= 0 && fd_b >= 0);
    ::unlink(path_a);
    ::unlink(path_b);
    auto write_all = [](int fd, const std::vector<uint8_t> & bytes) {
        size_t written = 0;
        while (written < bytes.size()) {
            const ssize_t result = ::write(
                fd, bytes.data() + written, bytes.size() - written);
            NVME_REQUIRE(result > 0);
            written += (size_t) result;
        }
    };
    write_all(fd_a, shard_a);
    write_all(fd_b, shard_b);

    MoeNvmeConfig config;
#if defined(__linux__)
    config.backend = MoeNvmeBackend::IoUring;
#else
    config.backend = MoeNvmeBackend::Auto;
#endif
    config.direct_io = MoeNvmeDirectMode::Disabled;
    config.host_slots = 4;
    config.io_threads = 2;
    MoeNvmeScheduler scheduler;
    std::string err;
    NVME_REQUIRE(scheduler.init(config, 32 * 1024, aligned_allocate,
                                aligned_free, nullptr, &err));
    NVME_REQUIRE(scheduler.bind_sources({
        {shard_a.data(), shard_a.size(), fd_a},
        {shard_b.data(), shard_b.size(), fd_b},
    }, {layer}, &err));
    NVME_REQUIRE(std::string(scheduler.effective_backend_name()) != "mmap-workers");
    MoeNvmeLease lease;
    NVME_REQUIRE(scheduler.acquire(0, 4, lease, &err));
    verify_lease(lease, 0, 4);
    lease.reset();
    NVME_REQUIRE(scheduler.stats().read_ops == 3);
    NVME_REQUIRE(scheduler.stats().errors == 0);
    scheduler.destroy();
    ::close(fd_a);
    ::close(fd_b);
}

TEST_CASE(MoeNvmeSchedulerFixture, real_file_backend_reads_exact_bytes) {
    SyntheticModel model;
    char path[] = "/tmp/moe_nvme_scheduler_XXXXXX";
    const int fd = ::mkstemp(path);
    NVME_REQUIRE(fd >= 0);
    ::unlink(path);
    size_t written = 0;
    while (written < model.file.size()) {
        const ssize_t result = ::write(fd, model.file.data() + written,
                                       model.file.size() - written);
        NVME_REQUIRE(result > 0);
        written += (size_t) result;
    }

    MoeNvmeConfig config;
    config.backend = MoeNvmeBackend::Auto;
    config.direct_io = MoeNvmeDirectMode::Disabled;
    config.host_slots = 4;
    config.io_threads = 2;
    MoeNvmeScheduler scheduler;
    std::string err;
    NVME_REQUIRE(scheduler.init(config, 32 * 1024, aligned_allocate,
                                aligned_free, nullptr, &err));
    NVME_REQUIRE(scheduler.bind_source(
        {model.file.data(), model.file.size(), fd}, model.regions, &err));
#if defined(__linux__)
    NVME_REQUIRE(std::string(scheduler.effective_backend_name()) == "io_uring");
#endif
    MoeNvmeLease lease;
    NVME_REQUIRE(scheduler.acquire(1, 7, lease, &err));
    verify_lease(lease, 1, 7);
    lease.reset();
    NVME_REQUIRE(scheduler.stats().read_ops == 2);
    NVME_REQUIRE(scheduler.stats().errors == 0);
    scheduler.destroy();
    ::close(fd);
}
#endif
