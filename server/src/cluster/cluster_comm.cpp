// cluster_comm.cpp - collective communication between cluster ranks
// (cluster_comm.h).
//
// Two implementations:
//   * LoopbackComm: always compiled. size-1 communicator whose collectives
//     are no-ops that still count into the stats, so cluster-enabled code
//     paths run unchanged on a single node and in GPU-free unit tests.
//   * RcclComm: compiled only under LUCE_CLUSTER_RCCL (set by
//     -DLUCE_CLUSTER=ON on a HIP build). Wraps one ncclComm_t created
//     with the non-blocking config so a watchdog can ncclCommAbort a hung
//     collective; every device collective is enqueued on the caller's HIP
//     stream and never synchronizes the host.
//
// Without the define the RCCL entry points return false / nullptr with
// err = "RCCL support not compiled in (build with -DLUCE_CLUSTER=ON on HIP)".

#include "cluster/cluster_comm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#ifdef LUCE_CLUSTER_RCCL
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#endif

namespace luce::cluster {

uint64_t cluster_device_bytes_in_use(int device) {
#ifdef LUCE_CLUSTER_RCCL
    if (device < 0) return 0;
    int previous = 0;
    if (hipGetDevice(&previous) != hipSuccess) return 0;
    if (device != previous && hipSetDevice(device) != hipSuccess) return 0;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const hipError_t rc = hipMemGetInfo(&free_bytes, &total_bytes);
    if (device != previous) hipSetDevice(previous);
    if (rc != hipSuccess || total_bytes == 0 || free_bytes > total_bytes) return 0;
    return (uint64_t) (total_bytes - free_bytes);
#else
    (void) device;
    return 0;
#endif
}

namespace {

const char * const kNoRcclMsg =
    "RCCL support not compiled in (build with -DLUCE_CLUSTER=ON on HIP)";

void set_err(std::string * err, const std::string & msg) {
    if (err) *err = msg;
}

uint64_t now_us() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

// ─── Loopback ───────────────────────────────────────────────────────────

class LoopbackComm final : public IClusterComm {
public:
    LoopbackComm(int rank, int size) : rank_(rank), size_(size) {}

    int rank() const override { return rank_; }
    int size() const override { return size_; }

    bool allreduce_sum_f32(void * dev_ptr, size_t n, DeviceStream stream, std::string * err) override {
        (void)dev_ptr; (void)stream; (void)err;
        std::lock_guard<std::mutex> lk(mu_);
        stats_.allreduce_calls++;
        stats_.allreduce_bytes += n * sizeof(float);
        return true;
    }

    bool allreduce_sum_bf16_compressed(void * dev_f32, size_t n, void * scratch_bf16,
                                       DeviceStream stream, std::string * err) override {
        (void)dev_f32; (void)scratch_bf16; (void)stream; (void)err;
        std::lock_guard<std::mutex> lk(mu_);
        stats_.allreduce_calls++;
        stats_.allreduce_bytes += n * 2;
        return true;
    }

    bool broadcast_bytes(void * dev_buf, size_t bytes, int root, DeviceStream stream,
                         std::string * err) override {
        (void)dev_buf; (void)stream;
        if (root != rank_) {
            set_err(err, "loopback broadcast root " + std::to_string(root) + " is not this rank");
            return false;
        }
        std::lock_guard<std::mutex> lk(mu_);
        stats_.broadcast_calls++;
        stats_.broadcast_bytes += bytes;
        return true;
    }

    bool broadcast_host_i32(int32_t * host_buf, size_t count, int root, std::string * err) override {
        (void)host_buf;
        if (root != rank_) {
            set_err(err, "loopback broadcast root " + std::to_string(root) + " is not this rank");
            return false;
        }
        std::lock_guard<std::mutex> lk(mu_);
        stats_.broadcast_calls++;
        stats_.broadcast_bytes += count * sizeof(int32_t);
        return true;
    }

    bool barrier(std::string * err) override {
        (void)err;
        std::lock_guard<std::mutex> lk(mu_);
        stats_.barrier_calls++;
        return true;
    }

    bool wait_stream(DeviceStream stream, uint32_t deadline_ms, std::string * err) override {
        (void)stream; (void)deadline_ms; (void)err;
        return true;
    }

    void abort() override {
        std::lock_guard<std::mutex> lk(mu_);
        if (!aborted_) {
            aborted_ = true;
            stats_.aborts++;
        }
    }

    bool async_error(std::string * err) const override {
        (void)err;
        return false;
    }

    ClusterCommStats stats() const override {
        std::lock_guard<std::mutex> lk(mu_);
        return stats_;
    }

    void reset_stats() override {
        std::lock_guard<std::mutex> lk(mu_);
        stats_ = ClusterCommStats{};
    }

    const char * backend_name() const override { return "loopback"; }

private:
    const int rank_;
    const int size_;
    mutable std::mutex mu_;
    ClusterCommStats stats_;
    bool aborted_ = false;
};

}  // namespace

std::unique_ptr<IClusterComm> create_loopback_cluster_comm(int rank, int size) {
    if (size < 1) size = 1;
    if (rank < 0 || rank >= size) rank = 0;
    return std::unique_ptr<IClusterComm>(new LoopbackComm(rank, size));
}

// ═══════════════════════════════════════════════════════════════════════
#ifdef LUCE_CLUSTER_RCCL

static_assert(sizeof(ncclUniqueId) == kRcclUniqueIdSize,
              "ncclUniqueId size differs from kRcclUniqueIdSize (NCCL_UNIQUE_ID_BYTES)");

// Defined in cluster_comm_kernels.cu (LANGUAGE HIP).
bool cluster_kernel_f32_to_bf16(const float * in, uint16_t * out, size_t n, void * stream);
bool cluster_kernel_bf16_to_f32(const uint16_t * in, float * out, size_t n, void * stream);

namespace {

std::string hip_err(const char * what, hipError_t e) {
    return std::string(what) + ": " + hipGetErrorString(e);
}

std::string nccl_err(const char * what, ncclResult_t r) {
    return std::string(what) + ": " + ncclGetErrorString(r);
}

// Non-blocking communicators report enqueue success as ncclInProgress.
bool nccl_ok(ncclResult_t r) {
    return r == ncclSuccess || r == ncclInProgress;
}

class RcclComm final : public IClusterComm {
public:
    RcclComm(int rank, int size, int device, uint32_t timeout_ms)
        : rank_(rank), size_(size), device_(device), timeout_ms_(timeout_ms) {}

    ~RcclComm() override {
        (void)hipSetDevice(device_);
        if (comm_ != nullptr) {
            if (!aborted_) {
                // Non-blocking comms may report InProgress here; nothing to do
                // about it at teardown.
                (void)ncclCommDestroy(comm_);
            }
            comm_ = nullptr;
        }
        if (wait_event_ != nullptr) (void)hipEventDestroy(wait_event_);
        if (internal_stream_ != nullptr) (void)hipStreamDestroy(internal_stream_);
        if (staging_dev_ != nullptr) (void)hipFree(staging_dev_);
        if (staging_host_ != nullptr) (void)hipHostFree(staging_host_);
    }

    bool init(const RcclUniqueId & id, bool blocking, std::string * err) {
        hipError_t he = hipSetDevice(device_);
        if (he != hipSuccess) {
            set_err(err, hip_err("hipSetDevice", he));
            return false;
        }
        he = hipStreamCreateWithFlags(&internal_stream_, hipStreamNonBlocking);
        if (he != hipSuccess) {
            set_err(err, hip_err("hipStreamCreateWithFlags", he));
            return false;
        }
        he = hipEventCreateWithFlags(&wait_event_, hipEventDisableTiming);
        if (he != hipSuccess) {
            set_err(err, hip_err("hipEventCreateWithFlags", he));
            return false;
        }
        he = hipMalloc(&staging_dev_, kStagingBytes);
        if (he != hipSuccess) {
            set_err(err, hip_err("hipMalloc(staging)", he));
            return false;
        }
        he = hipHostMalloc(&staging_host_, kStagingBytes, hipHostMallocDefault);
        if (he != hipSuccess) {
            set_err(err, hip_err("hipHostMalloc(staging)", he));
            return false;
        }

        ncclUniqueId nid;
        std::memcpy(&nid, id.data(), sizeof(nid));
        ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
        cfg.blocking = blocking ? 1 : 0;

        ncclResult_t r = ncclCommInitRankConfig(&comm_, size_, nid, rank_, &cfg);
        if (!nccl_ok(r)) {
            set_err(err, nccl_err("ncclCommInitRankConfig", r));
            comm_ = nullptr;
            return false;
        }
        if (!blocking) {
            // Poll until the communicator leaves InProgress or the deadline
            // hits; on timeout abort so peers unblock too.
            const uint64_t deadline = now_us() + (uint64_t)timeout_ms_ * 1000ull;
            for (;;) {
                ncclResult_t state = ncclSuccess;
                r = ncclCommGetAsyncError(comm_, &state);
                if (r != ncclSuccess) {
                    set_err(err, nccl_err("ncclCommGetAsyncError during init", r));
                    abort();
                    return false;
                }
                if (state == ncclSuccess) break;
                if (state != ncclInProgress) {
                    set_err(err, nccl_err("RCCL communicator init failed", state));
                    abort();
                    return false;
                }
                if (now_us() > deadline) {
                    set_err(err, "RCCL communicator init timed out after " +
                                 std::to_string(timeout_ms_) + " ms (rank " +
                                 std::to_string(rank_) + "/" + std::to_string(size_) + ")");
                    abort();
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        ready_ = true;
        return true;
    }

    int rank() const override { return rank_; }
    int size() const override { return size_; }

    bool allreduce_sum_f32(void * dev_ptr, size_t n, DeviceStream stream, std::string * err) override {
        if (!check_ready(err)) return false;
        if (n == 0) return true;
        (void)hipSetDevice(device_);
        const ncclResult_t r = ncclAllReduce(dev_ptr, dev_ptr, n, ncclFloat32, ncclSum, comm_,
                                             (hipStream_t)stream);
        if (!nccl_ok(r)) {
            set_err(err, nccl_err("ncclAllReduce(f32)", r));
            return false;
        }
        std::lock_guard<std::mutex> lk(mu_);
        stats_.allreduce_calls++;
        stats_.allreduce_bytes += n * sizeof(float);
        return true;
    }

    bool allreduce_sum_bf16_compressed(void * dev_f32, size_t n, void * scratch_bf16,
                                       DeviceStream stream, std::string * err) override {
        if (!check_ready(err)) return false;
        if (n == 0) return true;
        if (scratch_bf16 == nullptr) {
            set_err(err, "allreduce_sum_bf16_compressed needs a device scratch buffer");
            return false;
        }
        (void)hipSetDevice(device_);
        if (!cluster_kernel_f32_to_bf16((const float *)dev_f32, (uint16_t *)scratch_bf16, n, stream)) {
            set_err(err, hip_err("f32->bf16 conversion launch", hipGetLastError()));
            return false;
        }
        const ncclResult_t r = ncclAllReduce(scratch_bf16, scratch_bf16, n, ncclBfloat16, ncclSum,
                                             comm_, (hipStream_t)stream);
        if (!nccl_ok(r)) {
            set_err(err, nccl_err("ncclAllReduce(bf16)", r));
            return false;
        }
        if (!cluster_kernel_bf16_to_f32((const uint16_t *)scratch_bf16, (float *)dev_f32, n, stream)) {
            set_err(err, hip_err("bf16->f32 conversion launch", hipGetLastError()));
            return false;
        }
        std::lock_guard<std::mutex> lk(mu_);
        stats_.allreduce_calls++;
        stats_.allreduce_bytes += n * 2;
        return true;
    }

    bool broadcast_bytes(void * dev_buf, size_t bytes, int root, DeviceStream stream,
                         std::string * err) override {
        if (!check_ready(err)) return false;
        if (root < 0 || root >= size_) {
            set_err(err, "broadcast root " + std::to_string(root) + " out of range");
            return false;
        }
        if (bytes == 0) return true;
        (void)hipSetDevice(device_);
        const ncclResult_t r = ncclBroadcast(dev_buf, dev_buf, bytes, ncclUint8, root, comm_,
                                             (hipStream_t)stream);
        if (!nccl_ok(r)) {
            set_err(err, nccl_err("ncclBroadcast", r));
            return false;
        }
        std::lock_guard<std::mutex> lk(mu_);
        stats_.broadcast_calls++;
        stats_.broadcast_bytes += bytes;
        return true;
    }

    bool broadcast_host_i32(int32_t * host_buf, size_t count, int root, std::string * err) override {
        if (!check_ready(err)) return false;
        if (root < 0 || root >= size_) {
            set_err(err, "broadcast root " + std::to_string(root) + " out of range");
            return false;
        }
        (void)hipSetDevice(device_);
        // Serialize users of the internal staging buffers.
        std::lock_guard<std::mutex> slk(staging_mu_);
        const size_t max_per_chunk = kStagingBytes / sizeof(int32_t);
        size_t done = 0;
        while (done < count) {
            const size_t n = std::min(max_per_chunk, count - done);
            const size_t bytes = n * sizeof(int32_t);
            if (rank_ == root) {
                std::memcpy(staging_host_, host_buf + done, bytes);
                hipError_t he = hipMemcpyAsync(staging_dev_, staging_host_, bytes,
                                               hipMemcpyHostToDevice, internal_stream_);
                if (he != hipSuccess) {
                    set_err(err, hip_err("hipMemcpyAsync(H2D staging)", he));
                    return false;
                }
            }
            const ncclResult_t r = ncclBroadcast(staging_dev_, staging_dev_, bytes, ncclUint8, root,
                                                 comm_, internal_stream_);
            if (!nccl_ok(r)) {
                set_err(err, nccl_err("ncclBroadcast(host_i32)", r));
                return false;
            }
            hipError_t he = hipMemcpyAsync(staging_host_, staging_dev_, bytes,
                                           hipMemcpyDeviceToHost, internal_stream_);
            if (he != hipSuccess) {
                set_err(err, hip_err("hipMemcpyAsync(D2H staging)", he));
                return false;
            }
            if (!wait_stream_impl(internal_stream_, timeout_ms_, /*count_allreduce=*/false, err)) {
                return false;
            }
            std::memcpy(host_buf + done, staging_host_, bytes);
            done += n;
        }
        std::lock_guard<std::mutex> lk(mu_);
        stats_.broadcast_calls++;
        stats_.broadcast_bytes += count * sizeof(int32_t);
        return true;
    }

    bool barrier(std::string * err) override {
        if (!check_ready(err)) return false;
        const uint64_t t0 = now_us();
        (void)hipSetDevice(device_);
        std::lock_guard<std::mutex> slk(staging_mu_);
        // Tiny all-reduce on the internal stream + host sync = barrier.
        const ncclResult_t r = ncclAllReduce(staging_dev_, staging_dev_, 1, ncclFloat32, ncclSum,
                                             comm_, internal_stream_);
        if (!nccl_ok(r)) {
            set_err(err, nccl_err("ncclAllReduce(barrier)", r));
            return false;
        }
        if (!wait_stream_impl(internal_stream_, timeout_ms_, /*count_allreduce=*/false, err)) {
            return false;
        }
        std::lock_guard<std::mutex> lk(mu_);
        stats_.barrier_calls++;
        stats_.barrier_us += now_us() - t0;
        return true;
    }

    bool wait_stream(DeviceStream stream, uint32_t deadline_ms, std::string * err) override {
        if (!check_ready(err)) return false;
        return wait_stream_impl((hipStream_t)stream, deadline_ms, /*count_allreduce=*/true, err);
    }

    void abort() override {
        std::lock_guard<std::mutex> lk(mu_);
        if (aborted_) return;
        aborted_ = true;
        stats_.aborts++;
        if (comm_ != nullptr) {
            (void)ncclCommAbort(comm_);
            comm_ = nullptr;
        }
        ready_ = false;
    }

    bool async_error(std::string * err) const override {
        std::lock_guard<std::mutex> lk(mu_);
        if (comm_ == nullptr) {
            if (aborted_) set_err(err, "RCCL communicator aborted");
            return aborted_;
        }
        ncclResult_t state = ncclSuccess;
        const ncclResult_t r = ncclCommGetAsyncError(comm_, &state);
        if (r != ncclSuccess) {
            set_err(err, nccl_err("ncclCommGetAsyncError", r));
            return true;
        }
        if (state != ncclSuccess && state != ncclInProgress) {
            set_err(err, nccl_err("RCCL asynchronous error", state));
            return true;
        }
        return false;
    }

    ClusterCommStats stats() const override {
        std::lock_guard<std::mutex> lk(mu_);
        return stats_;
    }

    void reset_stats() override {
        std::lock_guard<std::mutex> lk(mu_);
        stats_ = ClusterCommStats{};
    }

    const char * backend_name() const override { return "rccl"; }

private:
    static constexpr size_t kStagingBytes = 64 * 1024;

    bool check_ready(std::string * err) const {
        std::lock_guard<std::mutex> lk(mu_);
        if (aborted_) {
            set_err(err, "RCCL communicator aborted");
            return false;
        }
        if (!ready_ || comm_ == nullptr) {
            set_err(err, "RCCL communicator not initialized");
            return false;
        }
        return true;
    }

    // Record an event on `stream` and poll it until completion or deadline.
    // On timeout: abort() so peers blocked in the same collective fail too.
    bool wait_stream_impl(hipStream_t stream, uint32_t deadline_ms, bool count_allreduce,
                          std::string * err) {
        const uint64_t t0 = now_us();
        const uint64_t deadline = t0 + (uint64_t)deadline_ms * 1000ull;
        hipEvent_t ev = nullptr;
        {
            std::lock_guard<std::mutex> lk(event_mu_);
            ev = wait_event_;
            const hipError_t he = hipEventRecord(ev, stream);
            if (he != hipSuccess) {
                set_err(err, hip_err("hipEventRecord", he));
                return false;
            }
            // Poll under event_mu_ so a concurrent wait cannot re-record the
            // single event underneath us.
            uint32_t spins = 0;
            for (;;) {
                const hipError_t q = hipEventQuery(ev);
                if (q == hipSuccess) break;
                if (q != hipErrorNotReady) {
                    set_err(err, hip_err("hipEventQuery", q));
                    return false;
                }
                std::string aerr;
                if (async_error(&aerr)) {
                    set_err(err, "collective failed while waiting: " + aerr);
                    abort();
                    return false;
                }
                if (now_us() > deadline) {
                    set_err(err, "wait_stream timed out after " + std::to_string(deadline_ms) +
                                 " ms on rank " + std::to_string(rank_) + "; aborting communicator");
                    abort();
                    return false;
                }
                // Busy-poll briefly (sub-100 us collectives), then back off.
                if (++spins < 2000) {
                    std::this_thread::yield();
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
            }
        }
        if (count_allreduce) {
            std::lock_guard<std::mutex> lk(mu_);
            stats_.allreduce_wait_us += now_us() - t0;
        }
        return true;
    }

    const int rank_;
    const int size_;
    const int device_;
    const uint32_t timeout_ms_;

    ncclComm_t comm_ = nullptr;
    hipStream_t internal_stream_ = nullptr;
    hipEvent_t wait_event_ = nullptr;
    void * staging_dev_ = nullptr;
    void * staging_host_ = nullptr;

    mutable std::mutex mu_;        // stats_, aborted_, ready_, comm_ lifetime
    std::mutex staging_mu_;        // staging buffers + internal stream users
    std::mutex event_mu_;          // wait_event_
    ClusterCommStats stats_;
    bool aborted_ = false;
    bool ready_ = false;
};

// ─── Self-test (device path) ────────────────────────────────────────────

struct SelftestBuffers {
    int device = 0;
    hipStream_t stream = nullptr;
    hipEvent_t e0 = nullptr;
    hipEvent_t e1 = nullptr;
    float * dev = nullptr;
    uint16_t * scratch = nullptr;
    float * host_src = nullptr;
    std::vector<float> host_out;

    ~SelftestBuffers() {
        (void)hipSetDevice(device);
        if (e0) (void)hipEventDestroy(e0);
        if (e1) (void)hipEventDestroy(e1);
        if (dev) (void)hipFree(dev);
        if (scratch) (void)hipFree(scratch);
        if (host_src) (void)hipHostFree(host_src);
        if (stream) (void)hipStreamDestroy(stream);
    }

    bool init(int dev_id, size_t max_n, float fill, std::string * err) {
        device = dev_id;
        hipError_t he = hipSetDevice(device);
        if (he != hipSuccess) { set_err(err, hip_err("hipSetDevice", he)); return false; }
        he = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
        if (he != hipSuccess) { set_err(err, hip_err("hipStreamCreate", he)); return false; }
        he = hipEventCreate(&e0);
        if (he != hipSuccess) { set_err(err, hip_err("hipEventCreate", he)); return false; }
        he = hipEventCreate(&e1);
        if (he != hipSuccess) { set_err(err, hip_err("hipEventCreate", he)); return false; }
        he = hipMalloc((void **)&dev, max_n * sizeof(float));
        if (he != hipSuccess) { set_err(err, hip_err("hipMalloc(selftest)", he)); return false; }
        he = hipMalloc((void **)&scratch, max_n * sizeof(uint16_t));
        if (he != hipSuccess) { set_err(err, hip_err("hipMalloc(selftest scratch)", he)); return false; }
        he = hipHostMalloc((void **)&host_src, max_n * sizeof(float), hipHostMallocDefault);
        if (he != hipSuccess) { set_err(err, hip_err("hipHostMalloc(selftest)", he)); return false; }
        for (size_t i = 0; i < max_n; ++i) host_src[i] = fill;
        host_out.resize(max_n);
        return true;
    }

    // Upload fresh inputs, run `op`, and measure its device time in µs.
    template <typename Op>
    bool timed(size_t n, Op && op, double & us, IClusterComm & comm, uint32_t timeout_ms,
               std::string * err) {
        hipError_t he = hipMemcpyAsync(dev, host_src, n * sizeof(float), hipMemcpyHostToDevice, stream);
        if (he != hipSuccess) { set_err(err, hip_err("hipMemcpyAsync(H2D)", he)); return false; }
        he = hipEventRecord(e0, stream);
        if (he != hipSuccess) { set_err(err, hip_err("hipEventRecord", he)); return false; }
        if (!op()) return false;
        he = hipEventRecord(e1, stream);
        if (he != hipSuccess) { set_err(err, hip_err("hipEventRecord", he)); return false; }
        if (!comm.wait_stream(stream, timeout_ms, err)) return false;
        float ms = 0.0f;
        he = hipEventElapsedTime(&ms, e0, e1);
        if (he != hipSuccess) { set_err(err, hip_err("hipEventElapsedTime", he)); return false; }
        us = (double)ms * 1000.0;
        return true;
    }

    bool verify(size_t n, float expected, const char * what, std::string * err) {
        const hipError_t he = hipMemcpy(host_out.data(), dev, n * sizeof(float), hipMemcpyDeviceToHost);
        if (he != hipSuccess) { set_err(err, hip_err("hipMemcpy(D2H verify)", he)); return false; }
        for (size_t i = 0; i < n; ++i) {
            if (host_out[i] != expected) {
                set_err(err, std::string(what) + ": element " + std::to_string(i) + " = " +
                             std::to_string(host_out[i]) + ", expected " + std::to_string(expected));
                return false;
            }
        }
        return true;
    }
};

}  // namespace

bool rccl_get_unique_id(RcclUniqueId & out, std::string * err) {
    ncclUniqueId id;
    const ncclResult_t r = ncclGetUniqueId(&id);
    if (r != ncclSuccess) {
        set_err(err, nccl_err("ncclGetUniqueId", r));
        return false;
    }
    std::memcpy(out.data(), &id, sizeof(id));
    return true;
}

std::unique_ptr<IClusterComm> create_rccl_cluster_comm(const RcclCommInit & init, std::string * err) {
    if (init.size < 1 || init.rank < 0 || init.rank >= init.size) {
        set_err(err, "bad RCCL rank/size (" + std::to_string(init.rank) + "/" +
                     std::to_string(init.size) + ")");
        return nullptr;
    }
    std::unique_ptr<RcclComm> comm(new RcclComm(init.rank, init.size, init.device,
                                                init.timeout_ms == 0 ? 30000u : init.timeout_ms));
    if (!comm->init(init.unique_id, init.blocking, err)) {
        return nullptr;
    }
    return std::unique_ptr<IClusterComm>(comm.release());
}

std::string rccl_version_string() {
    int v = 0;
    if (ncclGetVersion(&v) != ncclSuccess || v <= 0) return "n/a";
    // NCCL >= 2.9 encodes MAJOR*10000 + MINOR*100 + PATCH.
    const int major = v / 10000;
    const int minor = (v % 10000) / 100;
    const int patch = v % 100;
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

#else  // !LUCE_CLUSTER_RCCL ──────────────────────────────────────────

bool rccl_get_unique_id(RcclUniqueId & out, std::string * err) {
    out.fill(0);
    set_err(err, kNoRcclMsg);
    return false;
}

std::unique_ptr<IClusterComm> create_rccl_cluster_comm(const RcclCommInit & init, std::string * err) {
    (void)init;
    set_err(err, kNoRcclMsg);
    return nullptr;
}

std::string rccl_version_string() {
    return "n/a";
}

#endif  // LUCE_CLUSTER_RCCL

// ─── Self-test ──────────────────────────────────────────────────────────

namespace {

struct LatencyStats {
    std::vector<double> samples;

    double percentile(double p) const {
        if (samples.empty()) return 0.0;
        std::vector<double> s = samples;
        std::sort(s.begin(), s.end());
        const double idx = p / 100.0 * (double)(s.size() - 1);
        const size_t lo = (size_t)std::floor(idx);
        const size_t hi = std::min(lo + 1, s.size() - 1);
        const double frac = idx - (double)lo;
        return s[lo] + (s[hi] - s[lo]) * frac;
    }

    void print(const char * label, size_t n_floats, size_t bytes) const {
        std::printf("[cluster-selftest] %-14s n=%zu bytes=%zu iters=%zu  p50=%.1f us  p90=%.1f us  "
                    "p99=%.1f us  max=%.1f us\n",
                    label, n_floats, bytes, samples.size(), percentile(50.0), percentile(90.0),
                    percentile(99.0), percentile(100.0));
    }
};

}  // namespace

bool run_cluster_selftest(IClusterComm & comm, int device, int iters, size_t small_n, size_t large_n,
                          std::string * err) {
    if (iters < 1) iters = 1;
    if (small_n == 0) small_n = 16384;
    if (large_n == 0) large_n = small_n;
    const int size = comm.size();
    const float fill = (float)(comm.rank() + 1);
    const float expected = (float)((size * (size + 1)) / 2);
    constexpr int kLargeIters = 43;  // one per DeepSeek4 MoE layer

    std::printf("[cluster-selftest] rank %d/%d backend=%s rccl=%s device=%d small_n=%zu large_n=%zu "
                "iters=%d\n",
                comm.rank(), size, comm.backend_name(), rccl_version_string().c_str(), device,
                small_n, large_n, iters);

#ifdef LUCE_CLUSTER_RCCL
    SelftestBuffers buf;
    const size_t max_n = std::max(small_n, large_n);
    if (!buf.init(device, max_n, fill, err)) return false;
    const uint32_t timeout_ms = 30000;

    LatencyStats small_lat;
    const int check_every = std::max(1, iters / 10);
    for (int i = 0; i < iters; ++i) {
        double us = 0.0;
        auto op = [&]() {
            return comm.allreduce_sum_f32(buf.dev, small_n, buf.stream, err);
        };
        if (!buf.timed(small_n, op, us, comm, timeout_ms, err)) return false;
        small_lat.samples.push_back(us);
        if (i % check_every == 0 || i + 1 == iters) {
            if (!buf.verify(small_n, expected, "small f32 all-reduce", err)) return false;
        }
    }
    small_lat.print("f32 small", small_n, small_n * sizeof(float));

    LatencyStats large_f32;
    LatencyStats large_bf16;
    for (int i = 0; i < kLargeIters; ++i) {
        const bool use_bf16 = (i % 2) == 1;
        double us = 0.0;
        auto op = [&]() {
            return use_bf16
                ? comm.allreduce_sum_bf16_compressed(buf.dev, large_n, buf.scratch, buf.stream, err)
                : comm.allreduce_sum_f32(buf.dev, large_n, buf.stream, err);
        };
        if (!buf.timed(large_n, op, us, comm, timeout_ms, err)) return false;
        (use_bf16 ? large_bf16 : large_f32).samples.push_back(us);
        // rank+1 sums are small integers: exactly representable in bf16.
        if (!buf.verify(large_n, expected, use_bf16 ? "large bf16 all-reduce" : "large f32 all-reduce",
                        err)) {
            return false;
        }
    }
    large_f32.print("f32 large", large_n, large_n * sizeof(float));
    large_bf16.print("bf16 large", large_n, large_n * 2);

    // Host broadcast of a few decision-sized words from rank 0.
    int32_t words[8];
    for (int i = 0; i < 8; ++i) words[i] = comm.rank() == 0 ? 1000 + i : -1;
    if (!comm.broadcast_host_i32(words, 8, 0, err)) return false;
    for (int i = 0; i < 8; ++i) {
        if (words[i] != 1000 + i) {
            set_err(err, "broadcast_host_i32 mismatch at " + std::to_string(i) + ": " +
                         std::to_string(words[i]));
            return false;
        }
    }
    if (!comm.barrier(err)) return false;

    std::string aerr;
    if (comm.async_error(&aerr)) {
        set_err(err, "async error after self-test: " + aerr);
        return false;
    }
    const ClusterCommStats st = comm.stats();
    std::printf("[cluster-selftest] rank %d OK: allreduce_calls=%llu allreduce_bytes=%llu "
                "allreduce_wait_us=%llu broadcast_calls=%llu barrier_us=%llu\n",
                comm.rank(), (unsigned long long)st.allreduce_calls,
                (unsigned long long)st.allreduce_bytes, (unsigned long long)st.allreduce_wait_us,
                (unsigned long long)st.broadcast_calls, (unsigned long long)st.barrier_us);
    return true;
#else
    // Without RCCL only the loopback communicator exists; it never touches
    // the buffers, so host memory stands in for device memory.
    (void)device;
    if (std::string(comm.backend_name()) != "loopback") {
        set_err(err, kNoRcclMsg);
        return false;
    }
    std::vector<float> host(std::max(small_n, large_n), fill);
    std::vector<uint16_t> scratch(host.size());
    LatencyStats small_lat;
    for (int i = 0; i < iters; ++i) {
        const uint64_t t0 = now_us();
        if (!comm.allreduce_sum_f32(host.data(), small_n, nullptr, err)) return false;
        if (!comm.wait_stream(nullptr, 30000, err)) return false;
        small_lat.samples.push_back((double)(now_us() - t0));
    }
    small_lat.print("f32 small", small_n, small_n * sizeof(float));
    for (int i = 0; i < kLargeIters; ++i) {
        if (!comm.allreduce_sum_bf16_compressed(host.data(), large_n, scratch.data(), nullptr, err)) {
            return false;
        }
    }
    for (size_t i = 0; i < small_n; ++i) {
        if (host[i] != expected) {
            set_err(err, "loopback self-test value mismatch at " + std::to_string(i));
            return false;
        }
    }
    if (!comm.barrier(err)) return false;
    std::printf("[cluster-selftest] rank %d OK (loopback)\n", comm.rank());
    return true;
#endif
}

}  // namespace luce::cluster
