// cluster_comm.h - collective communication between cluster ranks.
//
// Thin interface over RCCL (ROCm's NCCL) so the model code never includes
// rccl.h directly and unit tests can substitute a loopback or mock. All
// device-side collectives are enqueued on the caller-supplied HIP stream (the
// ggml backend's own stream, see ggml_backend_cuda_get_stream) so they are
// stream-ordered with the kernels that produce and consume the tensors; no
// host synchronization is needed around a call.
//
// Bootstrap: rank 0 obtains a unique id (rccl_get_unique_id), ships it to the
// workers over the control channel (WelcomeMsg), and every rank then calls
// create_rccl_cluster_comm. The communicator is created with the
// non-blocking config so a hung collective can be aborted from a watchdog
// (wait_stream deadline -> abort()).
//
// gfx1151 has no GPUDirect RDMA; RCCL bounces through host memory
// (NCCL_NET_GDR_LEVEL=0). On Strix Halo that host memory is the same DRAM
// as GPU memory, so the bounce is a memcpy, not a PCIe hop.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace luce::cluster {

inline constexpr size_t kRcclUniqueIdSize = 128;  // == NCCL_UNIQUE_ID_BYTES
using RcclUniqueId = std::array<uint8_t, kRcclUniqueIdSize>;

// Device memory in use on `device` right now (total - free), in bytes, or 0
// when the backend cannot report it. Sampled at request boundaries to fill
// RequestReportMsg::peak_device_bytes, which is therefore a high-water mark
// over sampled request ends, not a continuous peak.
uint64_t cluster_device_bytes_in_use(int device);

struct ClusterCommStats {
    uint64_t allreduce_calls    = 0;
    uint64_t allreduce_bytes    = 0;   // payload bytes handed to RCCL (f32 or bf16)
    uint64_t allreduce_wait_us  = 0;   // time spent in wait_stream() attributable to collectives
    uint64_t broadcast_calls    = 0;
    uint64_t broadcast_bytes    = 0;
    uint64_t barrier_calls      = 0;
    uint64_t barrier_us         = 0;
    uint64_t aborts             = 0;
};

// Opaque stream handle: hipStream_t on HIP builds, cudaStream_t on CUDA
// builds, ignored by the loopback implementation.
using DeviceStream = void *;

class IClusterComm {
public:
    virtual ~IClusterComm() = default;

    virtual int rank() const = 0;
    virtual int size() const = 0;

    // In-place sum of n float32 values at dev_ptr across all ranks. Enqueued
    // on `stream`; returns as soon as the collective is enqueued (or false on
    // enqueue failure). Result is bit-identical on every rank.
    virtual bool allreduce_sum_f32(void * dev_ptr, size_t n, DeviceStream stream,
                                   std::string * err) = 0;

    // Bandwidth-saving variant for large prefill partials: converts dev_f32
    // to bf16 in `scratch_bf16` (device, >= n * 2 bytes), all-reduces the
    // bf16 buffer with f32 accumulation semantics as in
    // ggml_backend_cuda_comm_allreduce_tensor, and writes the f32 sum back to
    // dev_f32. Same stream-ordering contract as allreduce_sum_f32.
    virtual bool allreduce_sum_bf16_compressed(void * dev_f32, size_t n, void * scratch_bf16,
                                               DeviceStream stream, std::string * err) = 0;

    // Broadcast `bytes` bytes at `buf` (device memory) from `root`.
    virtual bool broadcast_bytes(void * dev_buf, size_t bytes, int root, DeviceStream stream,
                                 std::string * err) = 0;

    // Host-side convenience: broadcast small host buffers (decisions, draft
    // tokens) via a pinned staging buffer and a dedicated internal stream;
    // blocks until the data is available on the host on every rank. Used
    // only when the control channel is not preferred for a message.
    virtual bool broadcast_host_i32(int32_t * host_buf, size_t count, int root,
                                    std::string * err) = 0;

    // Cluster-wide barrier (host blocking).
    virtual bool barrier(std::string * err) = 0;

    // Block the host until all work enqueued on `stream` so far has
    // completed, or until deadline_ms elapses. On timeout returns false,
    // records an abort and calls abort() so peers unblock with an error
    // instead of hanging.
    virtual bool wait_stream(DeviceStream stream, uint32_t deadline_ms, std::string * err) = 0;

    // Tear down the communicator abruptly (ncclCommAbort). Idempotent.
    virtual void abort() = 0;

    // True if a collective failed asynchronously (ncclCommGetAsyncError).
    virtual bool async_error(std::string * err) const = 0;

    virtual ClusterCommStats stats() const = 0;
    virtual void reset_stats() = 0;

    virtual const char * backend_name() const = 0;  // "rccl" | "loopback" | "mock"
};

struct RcclCommInit {
    int          rank = -1;
    int          size = 0;
    RcclUniqueId unique_id{};
    int          device = 0;            // local HIP device ordinal
    uint32_t     timeout_ms = 30000;    // default deadline for wait_stream
    bool         blocking = false;      // false -> ncclCommInitRankConfig non-blocking
};

// Rank 0 only: obtain the unique id to distribute in WelcomeMsg.
// Compiled only when LUCE_CLUSTER_RCCL is defined; otherwise returns
// false with err = "RCCL support not compiled in".
bool rccl_get_unique_id(RcclUniqueId & out, std::string * err);

// Create the RCCL-backed communicator for this rank (all ranks must call it
// with the same unique id). nullptr + err on failure.
std::unique_ptr<IClusterComm> create_rccl_cluster_comm(const RcclCommInit & init, std::string * err);

// size-1 communicator: every collective is a no-op that succeeds. Used for
// single-node runs of cluster-enabled code paths and for tests.
std::unique_ptr<IClusterComm> create_loopback_cluster_comm(int rank = 0, int size = 1);

// Human-readable RCCL version string ("2.27.7" etc.) or "n/a".
std::string rccl_version_string();

// Runs the startup self-test used by --cluster-selftest: `iters` all-reduces
// of `small_n` floats plus 43 all-reduces of `large_n` floats on device
// buffers filled with rank-dependent values, checks the sums against the
// host-computed expectation, and reports latency percentiles to stdout.
// Returns false on any mismatch or timeout.
bool run_cluster_selftest(IClusterComm & comm, int device, int iters, size_t small_n, size_t large_n,
                          std::string * err);

}  // namespace luce::cluster
