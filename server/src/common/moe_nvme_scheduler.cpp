// Model-neutral asynchronous SSD scheduler for routed MoE weights.

#include "moe_nvme_scheduler.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>
#include <utility>

#if defined(_WIN32)
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#endif

namespace dflash::common {
namespace {

using Clock = std::chrono::steady_clock;

uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

bool checked_add(size_t a, size_t b, size_t & out) {
    if (a > std::numeric_limits<size_t>::max() - b) return false;
    out = a + b;
    return true;
}

bool checked_mul(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    out = a * b;
    return true;
}

bool range_in_bounds(size_t offset, size_t length, size_t total) {
    return offset <= total && length <= total - offset;
}

bool is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool align_up_checked(size_t value, size_t alignment, size_t & out) {
    if (!is_power_of_two(alignment)) return false;
    const size_t mask = alignment - 1;
    if (value > std::numeric_limits<size_t>::max() - mask) return false;
    out = (value + mask) & ~mask;
    return true;
}

size_t align_down(size_t value, size_t alignment) {
    return value & ~(alignment - 1);
}

int parse_bounded_int(const char * name, int current, int lo, int hi) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) return current;
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < lo || parsed > hi) {
        std::fprintf(stderr, "[moe-nvme] ignoring invalid %s=%s (want %d..%d)\n",
                     name, value, lo, hi);
        return current;
    }
    return (int) parsed;
}

std::string lowercase(const char * value) {
    std::string out = value ? value : "";
    for (char & ch : out) {
        if (ch >= 'A' && ch <= 'Z') ch = (char) (ch - 'A' + 'a');
    }
    return out;
}

struct KeyHash {
    size_t operator()(const MoeExpertKey & key) const noexcept {
        const uint64_t a = (uint32_t) key.layer;
        const uint64_t b = (uint32_t) key.expert;
        uint64_t x = (a << 32) | b;
        x ^= x >> 30;
        x *= UINT64_C(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x *= UINT64_C(0x94d049bb133111eb);
        x ^= x >> 31;
        return (size_t) x;
    }
};

uint64_t physical_memory_bytes() {
#if defined(_WIN32)
    return 0;
#else
    const long pages = ::sysconf(_SC_PHYS_PAGES);
    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0) return 0;
    const uint64_t p = (uint64_t) pages;
    const uint64_t s = (uint64_t) page_size;
    if (p > std::numeric_limits<uint64_t>::max() / s) return 0;
    return p * s;
#endif
}

#if !defined(_WIN32)
bool pread_at_least(int fd, uint8_t * dst, size_t request_bytes,
                    size_t required_bytes, size_t offset,
                    size_t & bytes_read, std::string & err) {
    bytes_read = 0;
    if (required_bytes > request_bytes) {
        err = "invalid minimum read length";
        return false;
    }
    size_t done = 0;
    while (done < required_bytes) {
        const size_t remaining = request_bytes - done;
        const size_t chunk = std::min(remaining, (size_t) std::numeric_limits<ssize_t>::max());
        const ssize_t got = ::pread(fd, dst + done, chunk, (off_t) (offset + done));
        if (got < 0) {
            if (errno == EINTR) continue;
            err = std::string("pread failed: ") + std::strerror(errno);
            return false;
        }
        if (got == 0) {
            err = "short read at end of model file";
            return false;
        }
        done += (size_t) got;
    }
    bytes_read = done;
    return true;
}
#endif

#if defined(__linux__)

// Small dependency-free io_uring wrapper. The ABI is Linux UAPI, so the
// inference binary does not need liburing. The implementation follows the
// kernel io_uring interface and uses conservative flags for old enterprise
// kernels. Files and page-locked slots are registered when the kernel accepts
// them, reducing per-read pinning and fd-table overhead.
class RawIoUring {
public:
    ~RawIoUring() { close(); }

    bool open(unsigned entries, const std::vector<int> & active_fds,
              const std::vector<void *> & buffers, size_t buffer_bytes,
              std::string & err) {
        close();
        std::memset(&params_, 0, sizeof(params_));
        fd_ = (int) ::syscall(SYS_io_uring_setup, entries, &params_);
        if (fd_ < 0) {
            err = std::string("io_uring_setup failed: ") + std::strerror(errno);
            return false;
        }

        sq_ring_bytes_ = params_.sq_off.array + params_.sq_entries * sizeof(unsigned);
        cq_ring_bytes_ = params_.cq_off.cqes + params_.cq_entries * sizeof(io_uring_cqe);
        if (params_.features & IORING_FEAT_SINGLE_MMAP) {
            const size_t both = std::max(sq_ring_bytes_, cq_ring_bytes_);
            sq_ring_ = ::mmap(nullptr, both, PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQ_RING);
            if (sq_ring_ == MAP_FAILED) {
                sq_ring_ = nullptr;
                err = std::string("io_uring SQ/CQ mmap failed: ") + std::strerror(errno);
                close();
                return false;
            }
            cq_ring_ = sq_ring_;
            sq_ring_bytes_ = both;
            cq_ring_bytes_ = both;
            single_mmap_ = true;
        } else {
            sq_ring_ = ::mmap(nullptr, sq_ring_bytes_, PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQ_RING);
            cq_ring_ = ::mmap(nullptr, cq_ring_bytes_, PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_CQ_RING);
            if (sq_ring_ == MAP_FAILED || cq_ring_ == MAP_FAILED) {
                if (sq_ring_ == MAP_FAILED) sq_ring_ = nullptr;
                if (cq_ring_ == MAP_FAILED) cq_ring_ = nullptr;
                err = std::string("io_uring ring mmap failed: ") + std::strerror(errno);
                close();
                return false;
            }
        }

        sqes_bytes_ = params_.sq_entries * sizeof(io_uring_sqe);
        sqes_ = static_cast<io_uring_sqe *>(::mmap(
            nullptr, sqes_bytes_, PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQES));
        if (sqes_ == MAP_FAILED) {
            sqes_ = nullptr;
            err = std::string("io_uring SQE mmap failed: ") + std::strerror(errno);
            close();
            return false;
        }

        auto * sq = static_cast<uint8_t *>(sq_ring_);
        auto * cq = static_cast<uint8_t *>(cq_ring_);
        sq_head_ = reinterpret_cast<unsigned *>(sq + params_.sq_off.head);
        sq_tail_ = reinterpret_cast<unsigned *>(sq + params_.sq_off.tail);
        sq_mask_ = reinterpret_cast<unsigned *>(sq + params_.sq_off.ring_mask);
        sq_entries_ = reinterpret_cast<unsigned *>(sq + params_.sq_off.ring_entries);
        sq_array_ = reinterpret_cast<unsigned *>(sq + params_.sq_off.array);
        cq_head_ = reinterpret_cast<unsigned *>(cq + params_.cq_off.head);
        cq_tail_ = reinterpret_cast<unsigned *>(cq + params_.cq_off.tail);
        cq_mask_ = reinterpret_cast<unsigned *>(cq + params_.cq_off.ring_mask);
        cqes_ = reinterpret_cast<io_uring_cqe *>(cq + params_.cq_off.cqes);
        sqe_head_ = sqe_tail_ = 0;

        if (!active_fds.empty() &&
            ::syscall(SYS_io_uring_register, fd_, IORING_REGISTER_FILES,
                      active_fds.data(), (unsigned) active_fds.size()) == 0) {
            fixed_file_ = true;
        }

        iovecs_.resize(buffers.size());
        for (size_t i = 0; i < buffers.size(); ++i) {
            iovecs_[i].iov_base = buffers[i];
            iovecs_[i].iov_len = buffer_bytes;
        }
        if (!iovecs_.empty() &&
            ::syscall(SYS_io_uring_register, fd_, IORING_REGISTER_BUFFERS,
                      iovecs_.data(), (unsigned) iovecs_.size()) == 0) {
            fixed_buffers_ = true;
        }
        return true;
    }

    void close() {
        if (fd_ >= 0 && fixed_buffers_) {
            (void) ::syscall(SYS_io_uring_register, fd_, IORING_UNREGISTER_BUFFERS,
                             nullptr, 0U);
        }
        if (fd_ >= 0 && fixed_file_) {
            (void) ::syscall(SYS_io_uring_register, fd_, IORING_UNREGISTER_FILES,
                             nullptr, 0U);
        }
        fixed_buffers_ = false;
        fixed_file_ = false;
        iovecs_.clear();
        if (sqes_) ::munmap(sqes_, sqes_bytes_);
        sqes_ = nullptr;
        if (single_mmap_) {
            if (sq_ring_) ::munmap(sq_ring_, sq_ring_bytes_);
        } else {
            if (sq_ring_) ::munmap(sq_ring_, sq_ring_bytes_);
            if (cq_ring_) ::munmap(cq_ring_, cq_ring_bytes_);
        }
        sq_ring_ = cq_ring_ = nullptr;
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        single_mmap_ = false;
        sq_ring_bytes_ = cq_ring_bytes_ = sqes_bytes_ = 0;
        sq_head_ = sq_tail_ = sq_mask_ = sq_entries_ = sq_array_ = nullptr;
        cq_head_ = cq_tail_ = cq_mask_ = nullptr;
        cqes_ = nullptr;
        sqe_head_ = sqe_tail_ = 0;
    }

    io_uring_sqe * get_sqe() {
        const unsigned kernel_head = __atomic_load_n(sq_head_, __ATOMIC_ACQUIRE);
        if (sqe_tail_ - kernel_head >= *sq_entries_) return nullptr;
        io_uring_sqe * sqe = &sqes_[sqe_tail_ & *sq_mask_];
        std::memset(sqe, 0, sizeof(*sqe));
        ++sqe_tail_;
        return sqe;
    }

    void prepare_read(io_uring_sqe * sqe,
                      const std::vector<int> & active_fds,
                      uint32_t source_index, int slot,
                      void * dst, uint32_t bytes, uint64_t offset,
                      uint64_t user_data, bool force_async) const {
        sqe->opcode = fixed_buffers_ ? IORING_OP_READ_FIXED : IORING_OP_READ;
        sqe->fd = fixed_file_ ? (int) source_index : active_fds[source_index];
        sqe->off = offset;
        sqe->addr = (uint64_t) (uintptr_t) dst;
        sqe->len = bytes;
        sqe->user_data = user_data;
        if (fixed_buffers_) sqe->buf_index = (uint16_t) slot;
        if (fixed_file_) sqe->flags |= IOSQE_FIXED_FILE;
        if (force_async) sqe->flags |= IOSQE_ASYNC;
    }

    bool submit_all(std::string & err) {
        unsigned kernel_tail = __atomic_load_n(sq_tail_, __ATOMIC_RELAXED);
        const unsigned mask = *sq_mask_;
        const unsigned count = sqe_tail_ - sqe_head_;
        for (unsigned i = 0; i < count; ++i) {
            sq_array_[kernel_tail & mask] = sqe_head_ & mask;
            ++kernel_tail;
            ++sqe_head_;
        }
        __atomic_store_n(sq_tail_, kernel_tail, __ATOMIC_RELEASE);

        while (__atomic_load_n(sq_head_, __ATOMIC_ACQUIRE) != kernel_tail) {
            const unsigned pending = kernel_tail - __atomic_load_n(sq_head_, __ATOMIC_ACQUIRE);
            const int rc = (int) ::syscall(SYS_io_uring_enter, fd_, pending, 0U, 0U, nullptr, 0U);
            if (rc >= 0) continue;
            if (errno == EINTR) continue;
            err = std::string("io_uring_enter submit failed: ") + std::strerror(errno);
            return false;
        }
        return true;
    }

    bool wait_cqe(io_uring_cqe & out, std::string & err) {
        for (;;) {
            const unsigned head = __atomic_load_n(cq_head_, __ATOMIC_RELAXED);
            const unsigned tail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
            if (head != tail) {
                out = cqes_[head & *cq_mask_];
                __atomic_store_n(cq_head_, head + 1, __ATOMIC_RELEASE);
                return true;
            }
            const int rc = (int) ::syscall(SYS_io_uring_enter, fd_, 0U, 1U,
                                            IORING_ENTER_GETEVENTS, nullptr, 0U);
            if (rc >= 0) continue;
            if (errno == EINTR) continue;
            err = std::string("io_uring_enter wait failed: ") + std::strerror(errno);
            return false;
        }
    }

private:
    int fd_ = -1;
    io_uring_params params_{};
    void * sq_ring_ = nullptr;
    void * cq_ring_ = nullptr;
    io_uring_sqe * sqes_ = nullptr;
    size_t sq_ring_bytes_ = 0;
    size_t cq_ring_bytes_ = 0;
    size_t sqes_bytes_ = 0;
    bool single_mmap_ = false;
    bool fixed_file_ = false;
    bool fixed_buffers_ = false;
    std::vector<iovec> iovecs_;

    unsigned * sq_head_ = nullptr;
    unsigned * sq_tail_ = nullptr;
    unsigned * sq_mask_ = nullptr;
    unsigned * sq_entries_ = nullptr;
    unsigned * sq_array_ = nullptr;
    unsigned * cq_head_ = nullptr;
    unsigned * cq_tail_ = nullptr;
    unsigned * cq_mask_ = nullptr;
    io_uring_cqe * cqes_ = nullptr;
    unsigned sqe_head_ = 0;
    unsigned sqe_tail_ = 0;
};

#endif // __linux__

} // namespace

MoeNvmeConfig MoeNvmeConfig::from_env() {
    return from_env(MoeNvmeConfig{});
}

MoeNvmeConfig MoeNvmeConfig::from_env(MoeNvmeConfig base) {
    base.host_slots = parse_bounded_int("DFLASH_MOE_NVME_SLOTS", base.host_slots, 2, 64);
    base.io_threads = parse_bounded_int("DFLASH_MOE_NVME_IO_THREADS", base.io_threads, 1, 32);
    base.demand_reserve = parse_bounded_int(
        "DFLASH_MOE_NVME_DEMAND_RESERVE", base.demand_reserve, 1, base.host_slots - 1);
    base.max_prefetch_batch = parse_bounded_int(
        "DFLASH_MOE_NVME_PREFETCH_BATCH", base.max_prefetch_batch, 1, base.host_slots);
    base.demand_timeout_ms = parse_bounded_int(
        "DFLASH_MOE_NVME_DEMAND_TIMEOUT_MS", base.demand_timeout_ms, 0, 600000);

    if (const char * value = std::getenv("DFLASH_MOE_NVME_BACKEND")) {
        const std::string mode = lowercase(value);
        if (mode == "auto") base.backend = MoeNvmeBackend::Auto;
        else if (mode == "thread" || mode == "threads" || mode == "pread") {
            base.backend = MoeNvmeBackend::ThreadPool;
        } else if (mode == "uring" || mode == "io_uring") {
            base.backend = MoeNvmeBackend::IoUring;
        } else if (mode == "mmap") {
            base.backend = MoeNvmeBackend::Mmap;
        } else {
            std::fprintf(stderr, "[moe-nvme] ignoring invalid DFLASH_MOE_NVME_BACKEND=%s\n", value);
        }
    }
    if (const char * value = std::getenv("DFLASH_MOE_NVME_DIRECT")) {
        const std::string mode = lowercase(value);
        if (mode == "auto") base.direct_io = MoeNvmeDirectMode::Auto;
        else if (mode == "1" || mode == "on" || mode == "true") {
            base.direct_io = MoeNvmeDirectMode::Enabled;
        } else if (mode == "0" || mode == "off" || mode == "false") {
            base.direct_io = MoeNvmeDirectMode::Disabled;
        } else {
            std::fprintf(stderr, "[moe-nvme] ignoring invalid DFLASH_MOE_NVME_DIRECT=%s\n", value);
        }
    }
    return base;
}

bool make_moe_expert_io_layout(
    int layer,
    int expert,
    const LayerExpertRegions & regions,
    size_t source_size,
    size_t direct_alignment,
    MoeExpertIoLayout & out,
    std::string * err) {

    return make_moe_expert_io_layout(
        layer, expert, regions, std::vector<size_t>{source_size},
        direct_alignment, out, err);
}

bool make_moe_expert_io_layout(
    int layer,
    int expert,
    const LayerExpertRegions & regions,
    const std::vector<size_t> & source_sizes,
    size_t direct_alignment,
    MoeExpertIoLayout & out,
    std::string * err) {

    out = MoeExpertIoLayout{};
    out.key = { (int32_t) layer, (int32_t) expert };
    out.fused_gate_up = regions.fused_gate_up;
    if (layer < 0 || expert < 0) {
        if (err) *err = "negative layer or expert id";
        return false;
    }
    if (!is_power_of_two(direct_alignment) || direct_alignment < 512) {
        if (err) *err = "direct I/O alignment must be a power of two >= 512";
        return false;
    }

    size_t host_cursor = 0;
    size_t device_cursor = 0;
    auto add_span = [&](const ExpertFileRegion & region, size_t expert_bytes,
                        const char * label) -> bool {
        if (out.span_count >= 3 || expert_bytes == 0 || region.size == 0) {
            if (err) *err = std::string("missing or invalid ") + label + " expert region";
            return false;
        }
        if (region.source_index >= source_sizes.size()) {
            if (err) *err = std::string(label) + " tensor references a missing model shard";
            return false;
        }
        const size_t source_size = source_sizes[region.source_index];
        if (!range_in_bounds(region.offset, region.size, source_size)) {
            if (err) *err = std::string(label) + " tensor region is outside its model shard";
            return false;
        }
        const size_t expert_count = region.size / expert_bytes;
        if ((size_t) expert >= expert_count) {
            if (err) *err = std::string(label) + " expert id exceeds tensor extent";
            return false;
        }
        size_t expert_delta = 0;
        size_t file_offset = 0;
        if (!checked_mul((size_t) expert, expert_bytes, expert_delta) ||
            !checked_add(region.offset, expert_delta, file_offset) ||
            !range_in_bounds(expert_delta, expert_bytes, region.size) ||
            !range_in_bounds(file_offset, expert_bytes, source_size)) {
            if (err) *err = std::string(label) + " expert range is outside the model file";
            return false;
        }

        MoeExpertIoSpan & span = out.spans[out.span_count++];
        span.file_offset = file_offset;
        span.source_index = region.source_index;
        span.bytes = expert_bytes;
        span.device_offset = device_cursor;

        const size_t aligned_file = align_down(file_offset, direct_alignment);
        const size_t lead = file_offset - aligned_file;
        size_t aligned_buffer = 0;
        size_t raw_io_bytes = 0;
        size_t aligned_io_bytes = 0;
        if (!align_up_checked(host_cursor, direct_alignment, aligned_buffer) ||
            !checked_add(lead, expert_bytes, raw_io_bytes) ||
            !align_up_checked(raw_io_bytes, direct_alignment, aligned_io_bytes) ||
            !checked_add(aligned_buffer, lead, span.buffer_offset) ||
            !checked_add(aligned_buffer, aligned_io_bytes, host_cursor) ||
            !checked_add(device_cursor, expert_bytes, device_cursor)) {
            if (err) *err = "expert I/O layout size overflow";
            return false;
        }
        span.io_file_offset = aligned_file;
        span.io_buffer_offset = aligned_buffer;
        span.io_bytes = aligned_io_bytes;
        span.io_required_bytes = raw_io_bytes;
        return true;
    };

    auto add_component = [&](MoeExpertComponentKind kind, size_t offset,
                             size_t bytes, size_t record_bytes,
                             const char * label) -> bool {
        if (out.component_count >= 3 || bytes == 0 ||
            !range_in_bounds(offset, bytes, record_bytes)) {
            if (err) *err = std::string("invalid ") + label +
                            " component in expert record";
            return false;
        }
        for (int i = 0; i < out.component_count; ++i) {
            const MoeExpertComponentLayout & prior = out.components[i];
            const size_t prior_end = prior.device_offset + prior.bytes;
            const size_t end = offset + bytes;
            if (offset < prior_end && prior.device_offset < end) {
                if (err) *err = std::string(label) +
                                " overlaps another expert component";
                return false;
            }
        }
        out.components[out.component_count++] = {kind, offset, bytes};
        return true;
    };

    if (regions.expert_major.enabled) {
        const ExpertMajorFileLayout & packed = regions.expert_major;
        if (!add_span(packed.experts, packed.expert_stride, "expert-major")) {
            return false;
        }
        if (regions.fused_gate_up) {
            if (!add_component(MoeExpertComponentKind::FusedGateUp,
                               packed.gate_up_offset,
                               regions.expert_bytes_gate_up,
                               packed.expert_stride, "gate_up")) {
                return false;
            }
        } else {
            if (!add_component(MoeExpertComponentKind::Gate,
                               packed.gate_offset, regions.expert_bytes_gate,
                               packed.expert_stride, "gate") ||
                !add_component(MoeExpertComponentKind::Up,
                               packed.up_offset, regions.expert_bytes_up,
                               packed.expert_stride, "up")) {
                return false;
            }
        }
        if (!add_component(MoeExpertComponentKind::Down,
                           packed.down_offset, regions.expert_bytes_down,
                           packed.expert_stride, "down")) {
            return false;
        }
    } else if (regions.fused_gate_up) {
        if (!add_span(regions.gate_up_exps, regions.expert_bytes_gate_up, "gate_up") ||
            !add_component(MoeExpertComponentKind::FusedGateUp,
                           out.spans[out.span_count - 1].device_offset,
                           regions.expert_bytes_gate_up, device_cursor,
                           "gate_up")) {
            return false;
        }
    } else {
        if (!add_span(regions.gate_exps, regions.expert_bytes_gate, "gate") ||
            !add_component(MoeExpertComponentKind::Gate,
                           out.spans[out.span_count - 1].device_offset,
                           regions.expert_bytes_gate, device_cursor, "gate") ||
            !add_span(regions.up_exps, regions.expert_bytes_up, "up") ||
            !add_component(MoeExpertComponentKind::Up,
                           out.spans[out.span_count - 1].device_offset,
                           regions.expert_bytes_up, device_cursor, "up")) {
            return false;
        }
    }

    if (!regions.expert_major.enabled) {
        if (!add_span(regions.down_exps, regions.expert_bytes_down, "down") ||
            !add_component(MoeExpertComponentKind::Down,
                           out.spans[out.span_count - 1].device_offset,
                           regions.expert_bytes_down, device_cursor, "down")) {
            return false;
        }
    }

    out.payload_bytes = device_cursor;
    out.host_bytes = host_cursor;
    return true;
}

struct MoeNvmeScheduler::Impl {
    enum class SlotState : uint8_t { Free, Queued, Reading, Ready, Failed };

    struct Slot {
        void * data = nullptr;
        SlotState state = SlotState::Free;
        MoeExpertKey key{};
        MoeExpertIoLayout layout{};
        MoeNvmePriority priority = MoeNvmePriority::Prefetch;
        uint64_t generation = 0;
        uint64_t queue_epoch = 0;
        uint64_t last_touch = 0;
        uint64_t frequency = 0;
        int leases = 0;
        bool demand_resident = false;
        std::string error;
    };

    struct Job {
        int slot = -1;
        uint64_t generation = 0;
        uint64_t queue_epoch = 0;
        MoeNvmePriority priority = MoeNvmePriority::Prefetch;
    };

    struct SlotRef {
        int slot = -1;
        uint64_t generation = 0;
    };

    MoeNvmeConfig config{};
    size_t max_payload_bytes = 0;
    size_t bytes_per_slot = 0;
    AllocateFn allocate = nullptr;
    FreeFn free_fn = nullptr;
    void * allocator_opaque = nullptr;
    std::vector<Slot> slots;
    std::vector<LayerExpertRegions> regions;
    std::vector<MoeNvmeSource> sources;
    std::vector<size_t> source_sizes;

    MoeNvmeBackend effective_backend = MoeNvmeBackend::ThreadPool;
    bool direct_active = false;
    bool initialized = false;
    bool bound = false;
    bool stopping = false;
    std::vector<int> source_fds;
    std::vector<int> direct_fds;
    std::vector<int> active_fds;

    mutable std::mutex mutex;
    std::condition_variable work_cv;
    std::condition_variable state_cv;
    std::deque<Job> demand_queue;
    std::deque<Job> prefetch_queue;
    std::unordered_map<MoeExpertKey, SlotRef, KeyHash> index;
    std::vector<std::thread> workers;
    uint64_t clock = 0;
    int active_prefetch = 0;
    int max_active_prefetch = 1;

#if defined(__linux__)
    std::unique_ptr<RawIoUring> ring;
#endif

    std::atomic<uint64_t> requests{0};
    std::atomic<uint64_t> demand_requests{0};
    std::atomic<uint64_t> prefetch_requests{0};
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> inflight_deduplications{0};
    std::atomic<uint64_t> demand_upgrades{0};
    std::atomic<uint64_t> prefetch_drops{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> read_ops{0};
    std::atomic<uint64_t> payload_bytes{0};
    std::atomic<uint64_t> physical_bytes{0};
    mutable std::mutex io_time_mutex;
    int io_active = 0;
    Clock::time_point io_active_begin{};
    uint64_t active_io_ns = 0;
    std::atomic<uint64_t> read_ns{0};
    std::atomic<uint64_t> wait_ns{0};
    std::atomic<uint64_t> demand_timeouts{0};
    std::atomic<uint64_t> errors{0};

    void begin_io_activity() {
        std::lock_guard<std::mutex> lock(io_time_mutex);
        if (io_active++ == 0) io_active_begin = Clock::now();
    }

    void end_io_activity() {
        const auto now = Clock::now();
        std::lock_guard<std::mutex> lock(io_time_mutex);
        if (io_active <= 0) return;
        if (--io_active == 0) active_io_ns += elapsed_ns(io_active_begin, now);
    }

    uint64_t active_io_time() const {
        const auto now = Clock::now();
        std::lock_guard<std::mutex> lock(io_time_mutex);
        return active_io_ns + (io_active > 0 ? elapsed_ns(io_active_begin, now) : 0);
    }

    void reset_io_time() {
        std::lock_guard<std::mutex> lock(io_time_mutex);
        active_io_ns = 0;
        if (io_active > 0) io_active_begin = Clock::now();
    }

    bool valid_job_locked(const Job & job) const {
        if (job.slot < 0 || job.slot >= (int) slots.size()) return false;
        const Slot & slot = slots[(size_t) job.slot];
        return slot.state == SlotState::Queued &&
               slot.generation == job.generation &&
               slot.queue_epoch == job.queue_epoch;
    }

    bool queue_has_valid_locked(std::deque<Job> & queue) {
        while (!queue.empty() && !valid_job_locked(queue.front())) queue.pop_front();
        return !queue.empty();
    }

    bool take_one_locked(Job & out, bool allow_prefetch) {
        if (queue_has_valid_locked(demand_queue)) {
            out = demand_queue.front();
            demand_queue.pop_front();
        } else {
            if (!allow_prefetch || active_prefetch >= max_active_prefetch ||
                !queue_has_valid_locked(prefetch_queue)) {
                return false;
            }
            out = prefetch_queue.front();
            prefetch_queue.pop_front();
            ++active_prefetch;
        }
        Slot & slot = slots[(size_t) out.slot];
        slot.state = SlotState::Reading;
        return true;
    }

    int speculative_occupancy_locked() const {
        int count = 0;
        for (const Slot & slot : slots) {
            if (slot.state != SlotState::Free && !slot.demand_resident &&
                slot.priority == MoeNvmePriority::Prefetch) {
                ++count;
            }
        }
        return count;
    }

    uint64_t eviction_score_locked(const Slot & slot) const {
        // Frequency dominates; recency breaks ties. This is an LFRU score with
        // enough hysteresis that one recent speculative touch cannot displace
        // a repeatedly demanded expert.
        const uint64_t age = clock >= slot.last_touch ? clock - slot.last_touch : 0;
        const uint64_t recency = age < 255 ? 255 - age : 0;
        return (slot.frequency << 8) | recency;
    }

    int choose_slot_locked(MoeNvmePriority priority) {
        if (priority == MoeNvmePriority::Prefetch) {
            const int limit = std::max(1, (int) slots.size() - config.demand_reserve);
            if (speculative_occupancy_locked() >= limit) return -1;
        }
        for (size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].state == SlotState::Free) return (int) i;
        }

        if (priority == MoeNvmePriority::Demand) {
            // Demand may cancel a queued speculative read before evicting data.
            for (size_t i = 0; i < slots.size(); ++i) {
                Slot & slot = slots[i];
                if (slot.state == SlotState::Queued &&
                    slot.priority == MoeNvmePriority::Prefetch && slot.leases == 0) {
                    index.erase(slot.key);
                    ++slot.queue_epoch; // invalidates the old queue entry
                    return (int) i;
                }
            }
        }

        int victim = -1;
        uint64_t best = std::numeric_limits<uint64_t>::max();
        for (size_t i = 0; i < slots.size(); ++i) {
            const Slot & slot = slots[i];
            if ((slot.state != SlotState::Ready && slot.state != SlotState::Failed) ||
                slot.leases != 0) {
                continue;
            }
            if (priority == MoeNvmePriority::Prefetch && slot.demand_resident) continue;
            uint64_t score = slot.state == SlotState::Failed ? 0 : eviction_score_locked(slot);
            if (!slot.demand_resident) score >>= 2;
            if (score < best) {
                best = score;
                victim = (int) i;
            }
        }
        return victim;
    }

    enum class Admission { New, ReadyHit, Inflight, NoSlot, Invalid };

    Admission admit_locked(int layer, int expert, MoeNvmePriority priority,
                           int & slot_out, std::string * err) {
        slot_out = -1;
        if (stopping) {
            if (err) *err = "SSD scheduler is stopping";
            return Admission::Invalid;
        }
        if (!bound) {
            if (err) *err = "SSD scheduler has no bound model source";
            return Admission::Invalid;
        }
        if (layer < 0 || layer >= (int) regions.size()) {
            if (err) *err = "SSD request layer is out of range";
            return Admission::Invalid;
        }
        const MoeExpertKey key{ (int32_t) layer, (int32_t) expert };
        auto found = index.find(key);
        if (found != index.end()) {
            Slot & slot = slots[(size_t) found->second.slot];
            if (slot.generation == found->second.generation && slot.state != SlotState::Free) {
                slot_out = found->second.slot;
                slot.last_touch = ++clock;
                if (slot.state == SlotState::Ready) {
                    ++slot.frequency;
                    if (priority == MoeNvmePriority::Demand) slot.demand_resident = true;
                    return Admission::ReadyHit;
                }
                if (slot.state == SlotState::Failed) {
                    if (err) *err = slot.error;
                    return Admission::Invalid;
                }
                if (priority == MoeNvmePriority::Demand &&
                    slot.priority == MoeNvmePriority::Prefetch) {
                    slot.priority = MoeNvmePriority::Demand;
                    slot.demand_resident = true;
                    demand_upgrades.fetch_add(1, std::memory_order_relaxed);
                    if (slot.state == SlotState::Queued) {
                        ++slot.queue_epoch;
                        demand_queue.push_back({slot_out, slot.generation,
                                                slot.queue_epoch, MoeNvmePriority::Demand});
                        work_cv.notify_one();
                    }
                }
                return Admission::Inflight;
            }
            index.erase(found);
        }

        const int chosen = choose_slot_locked(priority);
        if (chosen < 0) return Admission::NoSlot;
        Slot & slot = slots[(size_t) chosen];
        if (slot.state != SlotState::Free) {
            index.erase(slot.key);
            evictions.fetch_add(1, std::memory_order_relaxed);
        }

        MoeExpertIoLayout layout;
        if (!make_moe_expert_io_layout(layer, expert, regions[(size_t) layer],
                                       source_sizes, config.direct_alignment,
                                       layout, err)) {
            slot.state = SlotState::Free;
            return Admission::Invalid;
        }
        if (layout.payload_bytes > max_payload_bytes || layout.host_bytes > bytes_per_slot) {
            if (err) *err = "expert read plan exceeds the configured SSD slot size";
            slot.state = SlotState::Free;
            return Admission::Invalid;
        }

        slot.state = SlotState::Queued;
        slot.key = key;
        slot.layout = layout;
        slot.priority = priority;
        ++slot.generation;
        ++slot.queue_epoch;
        slot.last_touch = ++clock;
        slot.frequency = priority == MoeNvmePriority::Demand ? 1 : 0;
        slot.leases = 0;
        slot.demand_resident = priority == MoeNvmePriority::Demand;
        slot.error.clear();
        slot_out = chosen;
        index[key] = {chosen, slot.generation};
        Job job{chosen, slot.generation, slot.queue_epoch, priority};
        if (priority == MoeNvmePriority::Demand) demand_queue.push_back(job);
        else prefetch_queue.push_back(job);
        work_cv.notify_one();
        return Admission::New;
    }

    bool read_job_threaded(const Job & job, std::string & err,
                           uint64_t & ops, uint64_t & logical, uint64_t & physical) {
        const Slot & slot = slots[(size_t) job.slot];
        uint8_t * base = static_cast<uint8_t *>(slot.data);
        for (int i = 0; i < slot.layout.span_count; ++i) {
            const MoeExpertIoSpan & span = slot.layout.spans[i];
            if (span.source_index >= sources.size()) {
                err = "expert read references a missing model shard";
                return false;
            }
            const MoeNvmeSource & source = sources[span.source_index];
            const int active_fd = active_fds.empty()
                ? -1 : active_fds[span.source_index];
            ++ops;
            logical += span.bytes;
            if (effective_backend == MoeNvmeBackend::Mmap || active_fd < 0) {
                if (!source.mmap_data ||
                    !range_in_bounds(span.file_offset, span.bytes, source.mmap_size)) {
                    err = "mmap expert read is outside the model file";
                    return false;
                }
                const auto * src = static_cast<const uint8_t *>(source.mmap_data);
                std::memcpy(base + span.buffer_offset, src + span.file_offset, span.bytes);
                physical += span.bytes;
            } else {
#if defined(_WIN32)
                (void) base;
                err = "pread backend is unavailable on Windows";
                return false;
#else
                const size_t read_offset = direct_active ? span.io_file_offset : span.file_offset;
                const size_t read_bytes = direct_active ? span.io_bytes : span.bytes;
                const size_t required_bytes = direct_active
                    ? span.io_required_bytes : span.bytes;
                const size_t buffer_offset = direct_active ? span.io_buffer_offset : span.buffer_offset;
                size_t actual_bytes = 0;
                if (!pread_at_least(active_fd, base + buffer_offset,
                                    read_bytes, required_bytes, read_offset,
                                    actual_bytes, err)) {
                    return false;
                }
                physical += actual_bytes;
#endif
            }
        }
        return true;
    }

    void complete_job(const Job & job, bool ok, const std::string & err,
                      uint64_t ops, uint64_t logical, uint64_t physical,
                      uint64_t duration_ns) {
        read_ops.fetch_add(ops, std::memory_order_relaxed);
        payload_bytes.fetch_add(logical, std::memory_order_relaxed);
        physical_bytes.fetch_add(physical, std::memory_order_relaxed);
        read_ns.fetch_add(duration_ns, std::memory_order_relaxed);
        if (!ok) errors.fetch_add(1, std::memory_order_relaxed);

        std::lock_guard<std::mutex> lock(mutex);
        if (job.priority == MoeNvmePriority::Prefetch && active_prefetch > 0) {
            --active_prefetch;
        }
        if (job.slot >= 0 && job.slot < (int) slots.size()) {
            Slot & slot = slots[(size_t) job.slot];
            if (slot.generation == job.generation && slot.state == SlotState::Reading) {
                slot.state = ok ? SlotState::Ready : SlotState::Failed;
                slot.error = ok ? std::string() : err;
                slot.last_touch = ++clock;
            }
        }
        state_cv.notify_all();
        work_cv.notify_all();
    }

    void thread_worker() {
        for (;;) {
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                work_cv.wait(lock, [&] {
                    return stopping || queue_has_valid_locked(demand_queue) ||
                           (active_prefetch < max_active_prefetch &&
                            queue_has_valid_locked(prefetch_queue));
                });
                if (stopping) return;
                if (!take_one_locked(job, true)) continue;
            }

            const auto begin = Clock::now();
            std::string err;
            uint64_t ops = 0, logical = 0, physical = 0;
            begin_io_activity();
            const bool ok = read_job_threaded(job, err, ops, logical, physical);
            end_io_activity();
            complete_job(job, ok, err, ops, logical, physical,
                         elapsed_ns(begin, Clock::now()));
        }
    }

#if defined(__linux__)
    void uring_worker() {
        struct Op {
            size_t job = 0;
            uint32_t expected = 0;
            uint32_t required = 0;
        };
        struct Progress {
            int pending = 0;
            uint64_t ops = 0;
            uint64_t logical = 0;
            uint64_t physical = 0;
            bool ok = true;
            bool completed = false;
            std::string error;
        };
        for (;;) {
            std::vector<Job> jobs;
            {
                std::unique_lock<std::mutex> lock(mutex);
                work_cv.wait(lock, [&] {
                    return stopping || queue_has_valid_locked(demand_queue) ||
                           queue_has_valid_locked(prefetch_queue);
                });
                if (stopping) return;

                Job job;
                while ((int) jobs.size() < config.host_slots &&
                       take_one_locked(job, false)) {
                    jobs.push_back(job);
                }
                if (jobs.empty()) {
                    int speculative = 0;
                    while ((int) jobs.size() < config.max_prefetch_batch &&
                           speculative < config.max_prefetch_batch &&
                           take_one_locked(job, true)) {
                        jobs.push_back(job);
                        ++speculative;
                    }
                }
            }
            if (jobs.empty()) continue;

            const auto begin = Clock::now();
            std::vector<Op> operations;
            std::vector<Progress> progress(jobs.size());

            for (size_t j = 0; j < jobs.size(); ++j) {
                const Slot & slot = slots[(size_t) jobs[j].slot];
                auto * base = static_cast<uint8_t *>(slot.data);
                progress[j].logical = slot.layout.payload_bytes;
                for (int s = 0; s < slot.layout.span_count; ++s) {
                    const MoeExpertIoSpan & span = slot.layout.spans[s];
                    const size_t read_offset = direct_active ? span.io_file_offset : span.file_offset;
                    const size_t read_bytes = direct_active ? span.io_bytes : span.bytes;
                    const size_t required_bytes = direct_active
                        ? span.io_required_bytes : span.bytes;
                    const size_t buffer_offset = direct_active ? span.io_buffer_offset : span.buffer_offset;
                    if (read_bytes > (size_t) std::numeric_limits<int32_t>::max() ||
                        required_bytes > read_bytes) {
                        progress[j].ok = false;
                        progress[j].error =
                            "one expert tensor read exceeds io_uring's supported length";
                        continue;
                    }
                    io_uring_sqe * sqe = ring->get_sqe();
                    if (!sqe) {
                        progress[j].ok = false;
                        progress[j].error = "io_uring submission queue is full";
                        continue;
                    }
                    const uint64_t op_index = operations.size();
                    operations.push_back({j, (uint32_t) read_bytes,
                                          (uint32_t) required_bytes});
                    ring->prepare_read(sqe, active_fds, span.source_index,
                                       jobs[j].slot,
                                       base + buffer_offset, (uint32_t) read_bytes,
                                       read_offset, op_index, !direct_active);
                    ++progress[j].pending;
                    ++progress[j].ops;
                }
            }

            // Publish a slot as soon as that expert's last tensor slice
            // completes. Waiting for the slowest expert in the whole ring
            // batch creates a barrier that prevents SSD N+1 from overlapping
            // H2D/compute N.
            auto finish_job = [&](size_t j) {
                Progress & item = progress[j];
                if (item.completed) return;
                item.completed = true;
                complete_job(jobs[j], item.ok, item.error, item.ops,
                             item.logical, item.physical,
                             elapsed_ns(begin, Clock::now()));
            };

            std::string ring_error;
            bool fatal_ring_error = false;
            if (!operations.empty()) begin_io_activity();
            if (operations.empty()) {
                for (Progress & item : progress) {
                    item.ok = false;
                    if (item.error.empty()) item.error = "io_uring batch contained no readable spans";
                }
            } else if (!ring->submit_all(ring_error)) {
                fatal_ring_error = true;
                for (Progress & item : progress) {
                    item.ok = false;
                    item.error = ring_error;
                }
            } else {
                for (size_t j = 0; j < progress.size(); ++j) {
                    if (progress[j].pending == 0) finish_job(j);
                }
                for (size_t completed = 0; completed < operations.size(); ++completed) {
                    io_uring_cqe cqe{};
                    if (!ring->wait_cqe(cqe, ring_error)) {
                        fatal_ring_error = true;
                        for (Progress & item : progress) {
                            if (!item.completed) {
                                item.ok = false;
                                item.error = ring_error;
                            }
                        }
                        break;
                    }
                    if (cqe.user_data >= operations.size()) {
                        for (Progress & item : progress) {
                            if (!item.completed) {
                                item.ok = false;
                                item.error = "io_uring returned an invalid completion tag";
                            }
                        }
                        continue;
                    }
                    const Op & op = operations[(size_t) cqe.user_data];
                    Progress & item = progress[op.job];
                    if (cqe.res < 0) {
                        item.ok = false;
                        item.error = std::string("io_uring read failed: ") +
                                     std::strerror(-cqe.res);
                    } else {
                        item.physical += (uint32_t) cqe.res;
                        if ((uint32_t) cqe.res < op.required ||
                            (uint32_t) cqe.res > op.expected) {
                            item.ok = false;
                            item.error = "io_uring returned an incomplete model read";
                        }
                    }
                    if (item.pending > 0 && --item.pending == 0) finish_job(op.job);
                }
            }
            if (!operations.empty()) end_io_activity();

            for (size_t j = 0; j < jobs.size(); ++j) {
                if (!progress[j].completed) {
                    if (progress[j].pending != 0 && progress[j].error.empty()) {
                        progress[j].ok = false;
                        progress[j].error = "io_uring did not complete every expert span";
                    }
                    finish_job(j);
                }
            }
            if (fatal_ring_error) {
                std::lock_guard<std::mutex> lock(mutex);
                stopping = true;
                state_cv.notify_all();
                work_cv.notify_all();
                return;
            }
        }
    }
#endif
};

MoeNvmeLease::~MoeNvmeLease() { reset(); }

MoeNvmeLease::MoeNvmeLease(MoeNvmeLease && other) noexcept
    : scheduler_(other.scheduler_), data_(other.data_), layout_(other.layout_),
      slot_(other.slot_), generation_(other.generation_) {
    other.scheduler_ = nullptr;
    other.data_ = nullptr;
    other.slot_ = -1;
    other.generation_ = 0;
}

MoeNvmeLease & MoeNvmeLease::operator=(MoeNvmeLease && other) noexcept {
    if (this != &other) {
        reset();
        scheduler_ = other.scheduler_;
        data_ = other.data_;
        layout_ = other.layout_;
        slot_ = other.slot_;
        generation_ = other.generation_;
        other.scheduler_ = nullptr;
        other.data_ = nullptr;
        other.slot_ = -1;
        other.generation_ = 0;
    }
    return *this;
}

void MoeNvmeLease::reset() {
    if (scheduler_) scheduler_->release_lease(slot_, generation_);
    scheduler_ = nullptr;
    data_ = nullptr;
    slot_ = -1;
    generation_ = 0;
    layout_ = MoeExpertIoLayout{};
}

MoeNvmeScheduler::MoeNvmeScheduler() : impl_(new Impl) {}
MoeNvmeScheduler::~MoeNvmeScheduler() { destroy(); }

bool MoeNvmeScheduler::init(const MoeNvmeConfig & requested,
                            size_t max_expert_payload_bytes,
                            AllocateFn allocate,
                            FreeFn free_fn,
                            void * allocator_opaque,
                            std::string * err) {
    destroy();
    impl_.reset(new (std::nothrow) Impl);
    if (!impl_) {
        if (err) *err = "failed to allocate SSD scheduler state";
        return false;
    }
    Impl & p = *impl_;
    p.config = requested;
    p.config.host_slots = std::max(2, p.config.host_slots);
    p.config.io_threads = std::max(1, p.config.io_threads);
    p.config.demand_reserve = std::max(1, std::min(p.config.demand_reserve,
                                                   p.config.host_slots - 1));
    p.config.max_prefetch_batch = std::max(1, std::min(p.config.max_prefetch_batch,
                                                       p.config.host_slots));
    p.config.demand_timeout_ms = std::max(0, p.config.demand_timeout_ms);
    if (!is_power_of_two(p.config.direct_alignment) || p.config.direct_alignment < 512 ||
        max_expert_payload_bytes == 0 || !allocate || !free_fn) {
        if (err) *err = "invalid SSD scheduler initialization arguments";
        return false;
    }
    p.max_payload_bytes = max_expert_payload_bytes;
    p.allocate = allocate;
    p.free_fn = free_fn;
    p.allocator_opaque = allocator_opaque;

    size_t overhead = 0;
    if (!checked_mul((size_t) 16, p.config.direct_alignment, overhead) ||
        !checked_add(max_expert_payload_bytes, overhead, p.bytes_per_slot) ||
        !align_up_checked(p.bytes_per_slot, p.config.direct_alignment, p.bytes_per_slot)) {
        if (err) *err = "SSD slot size overflow";
        return false;
    }

    p.slots.resize((size_t) p.config.host_slots);
    for (size_t i = 0; i < p.slots.size(); ++i) {
        if (!p.allocate(&p.slots[i].data, p.bytes_per_slot, p.allocator_opaque) ||
            !p.slots[i].data) {
            if (err) *err = "failed to allocate page-locked SSD host slot";
            for (size_t j = 0; j < i; ++j) p.free_fn(p.slots[j].data, p.allocator_opaque);
            p.slots.clear();
            return false;
        }
    }
    p.initialized = true;
    return true;
}

bool MoeNvmeScheduler::bind_source(const MoeNvmeSource & source,
                                   const std::vector<LayerExpertRegions> & layer_regions,
                                   std::string * err) {
    return bind_sources({source}, layer_regions, err);
}

bool MoeNvmeScheduler::bind_sources(
        const std::vector<MoeNvmeSource> & sources,
        const std::vector<LayerExpertRegions> & layer_regions,
        std::string * err) {
    if (!impl_ || !impl_->initialized) {
        if (err) *err = "SSD scheduler is not initialized";
        return false;
    }
    Impl & p = *impl_;
    std::lock_guard<std::mutex> lock(p.mutex);
    if (p.bound) {
        bool same = p.sources.size() == sources.size();
        for (size_t i = 0; same && i < sources.size(); ++i) {
            same = p.sources[i].mmap_data == sources[i].mmap_data &&
                   p.sources[i].mmap_size == sources[i].mmap_size &&
                   (sources[i].mmap_data || p.sources[i].fd == sources[i].fd);
        }
        if (!same && err) *err = "SSD scheduler cannot rebind an active model source";
        return same;
    }
    if (sources.empty() || layer_regions.empty()) {
        if (err) *err = "empty SSD model source or expert-region table";
        return false;
    }
    bool all_mapped = true;
    bool all_have_fds = true;
    uint64_t total_source_bytes = 0;
    for (const MoeNvmeSource & source : sources) {
        if (source.mmap_size == 0 || (!source.mmap_data && source.fd < 0)) {
            if (err) *err = "invalid SSD model shard";
            return false;
        }
        all_mapped = all_mapped && source.mmap_data != nullptr;
        all_have_fds = all_have_fds && source.fd >= 0;
#if defined(_WIN32)
        if (source.fd >= 0) {
            struct _stat64 file_stat{};
            if (::_fstat64(source.fd, &file_stat) != 0 || file_stat.st_size < 0 ||
                (uint64_t) file_stat.st_size < (uint64_t) source.mmap_size) {
                if (err) *err = "model shard fd is unreadable or shorter than its declared size";
                return false;
            }
        }
#else
        if (source.fd >= 0) {
            struct stat file_stat{};
            if (::fstat(source.fd, &file_stat) != 0 || file_stat.st_size < 0 ||
                (uint64_t) file_stat.st_size < (uint64_t) source.mmap_size) {
                if (err) *err = "model shard fd is unreadable or shorter than its declared size";
                return false;
            }
        }
#endif
        if ((uint64_t) source.mmap_size >
            std::numeric_limits<uint64_t>::max() - total_source_bytes) {
            if (err) *err = "SSD model shard sizes overflow";
            return false;
        }
        total_source_bytes += (uint64_t) source.mmap_size;
    }
    p.sources = sources;
    p.source_sizes.clear();
    p.source_sizes.reserve(sources.size());
    for (const MoeNvmeSource & source : sources) {
        p.source_sizes.push_back(source.mmap_size);
    }
    p.regions = layer_regions;
    p.source_fds.assign(sources.size(), -1);
    p.direct_fds.assign(sources.size(), -1);

    auto close_model_fds = [&]() {
#if !defined(_WIN32)
        for (int fd : p.direct_fds) if (fd >= 0) ::close(fd);
        for (int fd : p.source_fds) if (fd >= 0) ::close(fd);
#endif
        p.direct_fds.clear();
        p.source_fds.clear();
        p.active_fds.clear();
        p.direct_active = false;
    };

#if !defined(_WIN32)
    if (all_have_fds) {
        for (size_t i = 0; i < sources.size(); ++i) {
            p.source_fds[i] = ::dup(sources[i].fd);
            if (p.source_fds[i] < 0) {
                if (err) {
                    *err = std::string("failed to duplicate model shard fd: ") +
                           std::strerror(errno);
                }
                close_model_fds();
                return false;
            }
        }
    }
#endif
    const bool duplicated_all_fds = all_have_fds &&
        std::all_of(p.source_fds.begin(), p.source_fds.end(),
                    [](int fd) { return fd >= 0; });

    bool want_direct = p.config.direct_io == MoeNvmeDirectMode::Enabled;
    if (p.config.direct_io == MoeNvmeDirectMode::Auto) {
        const uint64_t ram = physical_memory_bytes();
        // Direct I/O avoids keeping both an explicit expert cache and a second
        // model-sized page cache when the model itself nearly fills RAM.
        want_direct = ram != 0 && total_source_bytes > ram - ram / 4;
    }
    if (want_direct && !duplicated_all_fds &&
        p.config.direct_io == MoeNvmeDirectMode::Enabled) {
        if (err) *err = "O_DIRECT requires a readable fd for every model shard";
        close_model_fds();
        return false;
    }

#if defined(__linux__) && defined(O_DIRECT)
    if (want_direct && duplicated_all_fds) {
        bool aligned = true;
        for (const Impl::Slot & slot : p.slots) {
            if (((uintptr_t) slot.data & (p.config.direct_alignment - 1)) != 0) {
                aligned = false;
                break;
            }
        }
        if (aligned) {
            p.direct_active = true;
            for (size_t i = 0; i < p.source_fds.size(); ++i) {
                char proc_path[64];
                std::snprintf(proc_path, sizeof(proc_path),
                              "/proc/self/fd/%d", p.source_fds[i]);
                p.direct_fds[i] = ::open(
                    proc_path, O_RDONLY | O_CLOEXEC | O_DIRECT);
                if (p.direct_fds[i] < 0) {
                    p.direct_active = false;
                    break;
                }
            }
            if (!p.direct_active) {
                for (int & fd : p.direct_fds) {
                    if (fd >= 0) ::close(fd);
                    fd = -1;
                }
            }
        }
        if (!p.direct_active && p.config.direct_io == MoeNvmeDirectMode::Enabled) {
            if (err) {
                *err = "O_DIRECT was requested but one model shard or the pinned buffers do not support it";
            }
            close_model_fds();
            return false;
        }
    }
#else
    if (want_direct && p.config.direct_io == MoeNvmeDirectMode::Enabled) {
        if (err) *err = "O_DIRECT was requested on an unsupported platform";
        close_model_fds();
        return false;
    }
#endif
    if (duplicated_all_fds) {
        p.active_fds = p.direct_active ? p.direct_fds : p.source_fds;
    }

    if (p.config.backend == MoeNvmeBackend::Mmap) {
        if (!all_mapped) {
            if (err) *err = "mmap SSD backend requested without every shard mapped";
            close_model_fds();
            return false;
        }
        p.effective_backend = MoeNvmeBackend::Mmap;
        close_model_fds();
    } else if (p.config.backend == MoeNvmeBackend::ThreadPool) {
        if (!p.active_fds.empty()) p.effective_backend = MoeNvmeBackend::ThreadPool;
        else if (all_mapped) p.effective_backend = MoeNvmeBackend::Mmap;
        else {
            if (err) *err = "threaded SSD backend needs every shard readable";
            close_model_fds();
            return false;
        }
    } else {
#if defined(__linux__)
        if (!p.active_fds.empty()) {
            p.ring.reset(new (std::nothrow) RawIoUring);
            std::string ring_error;
            std::vector<void *> buffers;
            buffers.reserve(p.slots.size());
            for (const Impl::Slot & slot : p.slots) buffers.push_back(slot.data);
            const unsigned entries = (unsigned) std::max<size_t>(
                32, p.slots.size() * 3 + 4);
            if (p.ring && p.ring->open(entries, p.active_fds, buffers,
                                       p.bytes_per_slot, ring_error)) {
                p.effective_backend = MoeNvmeBackend::IoUring;
            } else {
                p.ring.reset();
                if (p.config.backend == MoeNvmeBackend::IoUring) {
                    if (err) *err = ring_error.empty() ? "io_uring initialization failed" : ring_error;
                    close_model_fds();
                    return false;
                }
                p.effective_backend = MoeNvmeBackend::ThreadPool;
            }
        } else {
            if (p.config.backend == MoeNvmeBackend::IoUring) {
                if (err) *err = "io_uring backend requested without every model shard readable";
                close_model_fds();
                return false;
            }
            if (all_mapped) p.effective_backend = MoeNvmeBackend::Mmap;
            else {
                if (err) *err = "SSD backend cannot read every model shard";
                close_model_fds();
                return false;
            }
        }
#else
        if (p.config.backend == MoeNvmeBackend::IoUring) {
            if (err) *err = "io_uring backend is Linux-only";
            close_model_fds();
            return false;
        }
        if (!p.active_fds.empty()) p.effective_backend = MoeNvmeBackend::ThreadPool;
        else if (all_mapped) p.effective_backend = MoeNvmeBackend::Mmap;
        else {
            if (err) *err = "SSD backend cannot read every model shard";
            close_model_fds();
            return false;
        }
#endif
    }

    p.stopping = false;
    if (p.effective_backend == MoeNvmeBackend::IoUring) {
#if defined(__linux__)
        p.max_active_prefetch = std::max(1, p.config.max_prefetch_batch);
        p.workers.emplace_back([&p] { p.uring_worker(); });
#endif
    } else {
        const int workers = p.effective_backend == MoeNvmeBackend::Mmap
            ? std::min(2, p.config.io_threads) : p.config.io_threads;
        p.max_active_prefetch = std::max(1, workers - 1);
        for (int i = 0; i < workers; ++i) {
            p.workers.emplace_back([&p] { p.thread_worker(); });
        }
    }
    p.bound = true;
    return true;
}

bool MoeNvmeScheduler::is_initialized() const {
    return impl_ && impl_->initialized;
}

bool MoeNvmeScheduler::is_bound() const {
    return impl_ && impl_->bound;
}

void MoeNvmeScheduler::destroy() {
    if (!impl_) return;
    Impl & p = *impl_;
    {
        std::lock_guard<std::mutex> lock(p.mutex);
        p.stopping = true;
        p.work_cv.notify_all();
        p.state_cv.notify_all();
    }
    for (std::thread & worker : p.workers) {
        if (worker.joinable()) worker.join();
    }
    p.workers.clear();
#if defined(__linux__)
    p.ring.reset();
#endif
#if !defined(_WIN32)
    for (int fd : p.direct_fds) if (fd >= 0) ::close(fd);
    for (int fd : p.source_fds) if (fd >= 0) ::close(fd);
#endif
    p.direct_fds.clear();
    p.source_fds.clear();
    p.active_fds.clear();
    for (Impl::Slot & slot : p.slots) {
        if (slot.data && p.free_fn) p.free_fn(slot.data, p.allocator_opaque);
        slot.data = nullptr;
    }
    p.slots.clear();
    p.index.clear();
    p.demand_queue.clear();
    p.prefetch_queue.clear();
    p.regions.clear();
    p.sources.clear();
    p.source_sizes.clear();
    p.initialized = false;
    p.bound = false;
}

bool MoeNvmeScheduler::request(int layer, int expert, MoeNvmePriority priority,
                               std::string * err) {
    if (!impl_ || !impl_->initialized) {
        if (err) *err = "SSD scheduler is not initialized";
        return false;
    }
    Impl & p = *impl_;
    p.requests.fetch_add(1, std::memory_order_relaxed);
    if (priority == MoeNvmePriority::Demand) {
        p.demand_requests.fetch_add(1, std::memory_order_relaxed);
    } else {
        p.prefetch_requests.fetch_add(1, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> lock(p.mutex);
    int slot = -1;
    const Impl::Admission result = p.admit_locked(layer, expert, priority, slot, err);
    if (result == Impl::Admission::ReadyHit) {
        p.cache_hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    if (result == Impl::Admission::Inflight) {
        p.inflight_deduplications.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    if (result == Impl::Admission::New) return true;
    if (result == Impl::Admission::NoSlot && priority == MoeNvmePriority::Prefetch) {
        p.prefetch_drops.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (result == Impl::Admission::NoSlot && err) *err = "all SSD expert slots are busy";
    return false;
}

bool MoeNvmeScheduler::acquire(int layer, int expert, MoeNvmeLease & out,
                               std::string * err) {
    out.reset();
    if (!impl_ || !impl_->initialized) {
        if (err) *err = "SSD scheduler is not initialized";
        return false;
    }
    Impl & p = *impl_;
    const auto begin = Clock::now();
    const bool timeout_enabled = p.config.demand_timeout_ms > 0;
    const auto deadline = timeout_enabled
        ? begin + std::chrono::milliseconds(p.config.demand_timeout_ms)
        : Clock::time_point::max();
    p.requests.fetch_add(1, std::memory_order_relaxed);
    p.demand_requests.fetch_add(1, std::memory_order_relaxed);
    const MoeExpertKey key{(int32_t) layer, (int32_t) expert};

    std::unique_lock<std::mutex> lock(p.mutex);
    auto wait_for_state = [&]() -> bool {
        if (!timeout_enabled) {
            p.state_cv.wait(lock);
            return true;
        }
        if (p.state_cv.wait_until(lock, deadline) != std::cv_status::timeout) {
            return true;
        }
        p.demand_timeouts.fetch_add(1, std::memory_order_relaxed);
        p.wait_ns.fetch_add(elapsed_ns(begin, Clock::now()),
                            std::memory_order_relaxed);
        if (err) {
            *err = "timed out waiting for an SSD expert after " +
                   std::to_string(p.config.demand_timeout_ms) + " ms";
        }
        return false;
    };
    bool admitted = false;
    for (;;) {
        if (p.stopping) {
            if (err) *err = "SSD scheduler is stopping";
            return false;
        }
        auto found = p.index.find(key);
        if (found == p.index.end()) {
            int slot = -1;
            const Impl::Admission result = p.admit_locked(
                layer, expert, MoeNvmePriority::Demand, slot, err);
            if (result == Impl::Admission::Invalid) return false;
            if (result == Impl::Admission::NoSlot) {
                if (!wait_for_state()) return false;
                continue;
            }
            admitted = true;
            if (result == Impl::Admission::ReadyHit) {
                p.cache_hits.fetch_add(1, std::memory_order_relaxed);
            } else if (result == Impl::Admission::Inflight) {
                p.inflight_deduplications.fetch_add(1, std::memory_order_relaxed);
            }
            found = p.index.find(key);
            if (found == p.index.end()) continue;
        }

        Impl::Slot & slot = p.slots[(size_t) found->second.slot];
        if (slot.generation != found->second.generation) {
            p.index.erase(found);
            continue;
        }
        if (!admitted) {
            if (slot.state == Impl::SlotState::Ready) {
                p.cache_hits.fetch_add(1, std::memory_order_relaxed);
            } else if (slot.state == Impl::SlotState::Queued ||
                       slot.state == Impl::SlotState::Reading) {
                p.inflight_deduplications.fetch_add(1, std::memory_order_relaxed);
            }
            admitted = true;
        }
        if (slot.priority == MoeNvmePriority::Prefetch &&
            (slot.state == Impl::SlotState::Queued || slot.state == Impl::SlotState::Reading)) {
            slot.priority = MoeNvmePriority::Demand;
            slot.demand_resident = true;
            p.demand_upgrades.fetch_add(1, std::memory_order_relaxed);
            if (slot.state == Impl::SlotState::Queued) {
                ++slot.queue_epoch;
                p.demand_queue.push_back({found->second.slot, slot.generation,
                                          slot.queue_epoch, MoeNvmePriority::Demand});
                p.work_cv.notify_one();
            }
        }
        if (slot.state == Impl::SlotState::Failed) {
            if (err) *err = slot.error;
            return false;
        }
        if (slot.state != Impl::SlotState::Ready) {
            if (!wait_for_state()) return false;
            continue;
        }

        ++slot.leases;
        ++slot.frequency;
        slot.last_touch = ++p.clock;
        slot.demand_resident = true;
        out.scheduler_ = this;
        out.data_ = static_cast<const uint8_t *>(slot.data);
        out.layout_ = slot.layout;
        out.slot_ = found->second.slot;
        out.generation_ = slot.generation;
        p.wait_ns.fetch_add(elapsed_ns(begin, Clock::now()), std::memory_order_relaxed);
        return true;
    }
}

void MoeNvmeScheduler::release_lease(int slot_index, uint64_t generation) {
    if (!impl_) return;
    Impl & p = *impl_;
    std::lock_guard<std::mutex> lock(p.mutex);
    if (slot_index >= 0 && slot_index < (int) p.slots.size()) {
        Impl::Slot & slot = p.slots[(size_t) slot_index];
        if (slot.generation == generation && slot.leases > 0) --slot.leases;
    }
    p.state_cv.notify_all();
}

MoeNvmeStats MoeNvmeScheduler::stats() const {
    MoeNvmeStats out;
    if (!impl_) return out;
    const Impl & p = *impl_;
    out.requests = p.requests.load(std::memory_order_relaxed);
    out.demand_requests = p.demand_requests.load(std::memory_order_relaxed);
    out.prefetch_requests = p.prefetch_requests.load(std::memory_order_relaxed);
    out.cache_hits = p.cache_hits.load(std::memory_order_relaxed);
    out.inflight_deduplications = p.inflight_deduplications.load(std::memory_order_relaxed);
    out.demand_upgrades = p.demand_upgrades.load(std::memory_order_relaxed);
    out.prefetch_drops = p.prefetch_drops.load(std::memory_order_relaxed);
    out.evictions = p.evictions.load(std::memory_order_relaxed);
    out.read_ops = p.read_ops.load(std::memory_order_relaxed);
    out.payload_bytes = p.payload_bytes.load(std::memory_order_relaxed);
    out.physical_bytes = p.physical_bytes.load(std::memory_order_relaxed);
    out.active_io_ns = p.active_io_time();
    out.read_ns = p.read_ns.load(std::memory_order_relaxed);
    out.wait_ns = p.wait_ns.load(std::memory_order_relaxed);
    out.demand_timeouts = p.demand_timeouts.load(std::memory_order_relaxed);
    out.errors = p.errors.load(std::memory_order_relaxed);
    return out;
}

void MoeNvmeScheduler::reset_stats() {
    if (!impl_) return;
    Impl & p = *impl_;
    p.requests.store(0, std::memory_order_relaxed);
    p.demand_requests.store(0, std::memory_order_relaxed);
    p.prefetch_requests.store(0, std::memory_order_relaxed);
    p.cache_hits.store(0, std::memory_order_relaxed);
    p.inflight_deduplications.store(0, std::memory_order_relaxed);
    p.demand_upgrades.store(0, std::memory_order_relaxed);
    p.prefetch_drops.store(0, std::memory_order_relaxed);
    p.evictions.store(0, std::memory_order_relaxed);
    p.read_ops.store(0, std::memory_order_relaxed);
    p.payload_bytes.store(0, std::memory_order_relaxed);
    p.physical_bytes.store(0, std::memory_order_relaxed);
    p.reset_io_time();
    p.read_ns.store(0, std::memory_order_relaxed);
    p.wait_ns.store(0, std::memory_order_relaxed);
    p.demand_timeouts.store(0, std::memory_order_relaxed);
    p.errors.store(0, std::memory_order_relaxed);
}

size_t MoeNvmeScheduler::slot_bytes() const {
    return impl_ ? impl_->bytes_per_slot : 0;
}

size_t MoeNvmeScheduler::total_host_bytes() const {
    return impl_ ? impl_->bytes_per_slot * impl_->slots.size() : 0;
}

int MoeNvmeScheduler::slot_count() const {
    return impl_ ? (int) impl_->slots.size() : 0;
}

const char * MoeNvmeScheduler::effective_backend_name() const {
    if (!impl_ || !impl_->bound) return "unbound";
    switch (impl_->effective_backend) {
        case MoeNvmeBackend::IoUring: return impl_->direct_active ? "io_uring+direct" : "io_uring";
        case MoeNvmeBackend::ThreadPool: return impl_->direct_active ? "pread-pool+direct" : "pread-pool";
        case MoeNvmeBackend::Mmap: return "mmap-workers";
        case MoeNvmeBackend::Auto: break;
    }
    return "unknown";
}

bool MoeNvmeScheduler::direct_io_active() const {
    return impl_ && impl_->direct_active;
}

} // namespace dflash::common
