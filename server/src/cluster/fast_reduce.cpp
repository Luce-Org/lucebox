#include "cluster/fast_reduce.h"

// The RDMA fabric is built only with -DLUCE_CLUSTER=ON (HIP + libibverbs).
// Without it, FastReduce never comes up and callers keep RCCL, the same way
// cluster_comm.cpp compiles its RCCL half under LUCE_CLUSTER_RCCL.
#ifdef LUCE_CLUSTER

#include <hip/hip_runtime.h>
#include <infiniband/verbs.h>
#include <poll.h>

#include "cluster/cluster_control.h"
#include "common/platform_env.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace luce::cluster {

// The kernels live in fast_reduce_kernels.cu (LANGUAGE HIP). A kernel that
// writes an exchange row also raises the row's flag from its last block.
void fast_reduce_launch_pack(const float * src, void * dst, int n, int bf16,
                             uint32_t * flag, uint32_t value, uint32_t * counter, hipStream_t stream);
void fast_reduce_launch_add(float * dst, const void * peer, int n, int bf16, hipStream_t stream);
void fast_reduce_launch_hyb_combine(const float * cold, const float * hot, void * own_row, int n,
                                    uint32_t * flag, uint32_t value, uint32_t * counter, hipStream_t stream);
void fast_reduce_launch_hyb_final(const void * own_row, const void * peer_row, float * dst, int n,
                                  uint32_t * flag, uint32_t value, uint32_t * counter, hipStream_t stream);

namespace {

// What one rank has to tell every other rank to be written into.
constexpr int kMaxRanks = 8;

struct Endpoint {
    // qpn[r] is the queue pair this rank created to talk to rank r. Every pair
    // needs its own number, so one per rank is not enough: rank 2 must connect
    // to the queue pair rank 1 made *for rank 2*, not to whichever one rank 1
    // happened to make first.
    uint32_t qpn[kMaxRanks] = {0};
    uint32_t psn = 0;
    // One key per region. The payload and the flags are separate memory
    // regions -- deliberately, so a flag write cannot be reordered behind a
    // payload write -- and a remote write carries the key of the region it
    // lands in. Using one key for both is not a mismatch the sender notices:
    // the receiver drops it as an access violation and the peer simply waits.
    uint32_t rkey_data = 0;
    uint32_t rkey_flag = 0;
    uint64_t data_addr = 0;      // base of the receive region
    uint64_t flag_addr = 0;      // base of the flag region
    uint8_t  gid[16] = {0};
};

bool set_nodelay(int fd) {
    int one = 1;
    return setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) == 0;
}

// A star: rank 0 collects every endpoint and hands the full table back. Small
// and synchronous, which is what a one-off exchange should be.
bool exchange(const FastReduce::Config & cfg, const Endpoint & mine,
              std::vector<Endpoint> & all, std::string * err) {
    all.assign((size_t) cfg.size, Endpoint{});
    all[(size_t) cfg.rank] = mine;
    if (cfg.size <= 1) return true;

    auto fail = [&](const char * what) {
        if (err) *err = std::string("fast-reduce bootstrap: ") + what + ": " + std::strerror(errno);
        return false;
    };
    auto fail_io = [&](const char * what, const std::string & why) {
        if (err) *err = std::string("fast-reduce bootstrap: ") + what + ": " + why;
        return false;
    };

    if (cfg.rank == 0) {
        const int lst = ::socket(AF_INET, SOCK_STREAM, 0);
        if (lst < 0) return fail("socket");
        int one = 1;
        setsockopt(lst, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = INADDR_ANY;
        a.sin_port = htons((uint16_t) cfg.bootstrap_port);
        if (::bind(lst, (sockaddr *) &a, sizeof(a)) != 0) { ::close(lst); return fail("bind"); }
        if (::listen(lst, cfg.size) != 0) { ::close(lst); return fail("listen"); }

        // Every wait is bounded by timeout_ms: a worker that died during its
        // model load must fail rank 0's load, not park it forever.
        std::vector<int> peers((size_t) cfg.size, -1);
        auto close_peers = [&]() {
            for (int & fd : peers) {
                if (fd >= 0) ::close(fd);
                fd = -1;
            }
        };
        std::string ioerr;
        for (int i = 1; i < cfg.size; ++i) {
            pollfd pfd{lst, POLLIN, 0};
            const int ready = ::poll(&pfd, 1, (int) cfg.timeout_ms);
            if (ready <= 0) {
                if (ready == 0) errno = ETIMEDOUT;
                ::close(lst); close_peers();
                return fail("accept");
            }
            const int fd = ::accept(lst, nullptr, nullptr);
            if (fd < 0) { ::close(lst); close_peers(); return fail("accept"); }
            set_nodelay(fd);
            int32_t r = -1;
            Endpoint e{};
            if (!read_exact_deadline(fd, &r, sizeof(r), cfg.timeout_ms, &ioerr) ||
                !read_exact_deadline(fd, &e, sizeof(e), cfg.timeout_ms, &ioerr) ||
                r <= 0 || r >= cfg.size || peers[(size_t) r] >= 0) {
                ::close(fd); ::close(lst); close_peers();
                return fail_io("read endpoint", ioerr.empty() ? "bad or duplicate rank" : ioerr);
            }
            all[(size_t) r] = e;
            peers[(size_t) r] = fd;
        }
        ::close(lst);
        for (int i = 1; i < cfg.size; ++i) {
            if (!write_exact_deadline(peers[(size_t) i], all.data(), sizeof(Endpoint) * all.size(),
                                      cfg.timeout_ms, &ioerr)) {
                close_peers();
                return fail_io("write table", ioerr);
            }
        }
        close_peers();
        return true;
    }

    // Workers retry: rank 0 may still be loading when they get here.
    int fd = -1;
    for (int attempt = 0; attempt < 600; ++attempt) {
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return fail("socket");
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t) cfg.bootstrap_port);
        if (::inet_pton(AF_INET, cfg.bootstrap_host.c_str(), &a.sin_addr) != 1) {
            ::close(fd);
            if (err) *err = "fast-reduce bootstrap: bad host " + cfg.bootstrap_host;
            return false;
        }
        if (::connect(fd, (sockaddr *) &a, sizeof(a)) == 0) break;
        ::close(fd);
        fd = -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (fd < 0) return fail("connect");
    set_nodelay(fd);
    const int32_t r = cfg.rank;
    std::string ioerr;
    if (!write_exact_deadline(fd, &r, sizeof(r), cfg.timeout_ms, &ioerr) ||
        !write_exact_deadline(fd, &mine, sizeof(mine), cfg.timeout_ms, &ioerr) ||
        !read_exact_deadline(fd, all.data(), sizeof(Endpoint) * all.size(), cfg.timeout_ms, &ioerr)) {
        ::close(fd);
        return fail_io("exchange", ioerr);
    }
    ::close(fd);
    return true;
}

}  // namespace

// A switch is on when set to anything but empty or "0".
static bool fr_env_on(const char * name) {
    return luce::common::env_flag_enabled(name);
}

struct FastReduceImpl {
    // Last-block counters for in-kernel publishes, per device (a kernel counts
    // in its own device memory): [slot][kind], kinds 0 attention row, 1 hybrid
    // export, 2 hybrid combine, 3 hybrid final. Zeroed on the first stream that
    // needs them, ahead of its first publish.
    static constexpr int kPubKinds = 4;
    std::mutex pub_counter_mu;
    std::map<int, uint32_t *> pub_counters;
    uint32_t * pub_counter(hipStream_t stm, int slot, int kind);
    FastReduce::Config cfg;
    bool up = false;

    ibv_context * ctx = nullptr;
    ibv_pd *      pd = nullptr;
    ibv_cq *      cq = nullptr;
    ibv_mr *      mr_data = nullptr;
    ibv_mr *      mr_flag = nullptr;
    std::vector<ibv_qp *> qps;              // one per peer, null at own rank
    std::vector<Endpoint> peers;

    // Pinned, NIC-registered, GPU-mapped. One region for payloads and one for
    // flags, so a flag write cannot be reordered behind a payload write on a
    // different memory region.
    float *    data_host = nullptr;         // [slots][size][max_elems]
    float *    data_dev = nullptr;
    // Two flag arrays, not one. The NIC writes `arrival` by DMA and only the
    // host reads it: a DMA write into system memory is coherent with a CPU
    // read, but nothing invalidates the GPU's caches for it, so a kernel
    // spinning on a NIC-written word can spin on a stale line forever. The
    // host progress thread sees the arrival and then writes `visible` with an
    // ordinary store, which is the path the probe measured at 0.79 us. This is
    // the same reason a general collective has a proxy thread at all.
    uint32_t * flag_host = nullptr;         // [slots][size] arrival | [slots][size] visible
                                            // | [slots] pub | [slots] hybrid export | [slots] hybrid final
    uint32_t * flag_dev = nullptr;


    // Payload length per reduction, on its own much larger ring. It cannot
    // share the slot ring: it is written when the host *enqueues* a reduction
    // and the host enqueues a whole step ahead of the GPU, so with one ring the
    // entry the progress thread still needs is overwritten a hundred
    // reductions before it reads it.
    static constexpr int kPendingRing = 8192;
    std::vector<int> pending_n;
    std::atomic<uint64_t> seq{0};
    std::atomic<bool> stop{false};
    // Set by the stall watchdog: every later submit becomes a no-op so the GPU
    // never parks on a peer that is gone, and the process exits once idle.
    std::atomic<bool> failed{false};
    // Hybrid exchange buffers, pinned: [part 0 = the dGPU's export | part 1 = the final sum][slots][hyb_elems].
    float *  hyb_host = nullptr;
    float *  hyb_dev = nullptr;
    int      hyb_elems = 0;
    // Exchange index -> sequence number: whichever of an exchange's calls the
    // host issues first takes the next sequence number for it.
    static constexpr int kHybRing = 4096;
    std::vector<std::pair<uint64_t, uint64_t>> hyb_ring;
    float * hyb_part(int part, int slot) const {
        return hyb_dev + ((size_t) part * (size_t) cfg.slots + (size_t) slot) * (size_t) hyb_elems;
    }
    std::thread progress;

    size_t slot_stride() const { return (size_t) cfg.max_elems * (size_t) cfg.size; }
    // My own row inside a slot is never written by a peer; peers write theirs.
    float * recv_row(int slot, int peer) {
        return data_host + (size_t) slot * slot_stride() + (size_t) peer * cfg.max_elems;
    }
    size_t arrival_off(int slot, int peer) const {
        return (size_t) slot * (size_t) cfg.size + (size_t) peer;
    }
    size_t visible_off(int slot, int peer) const {
        return (size_t) cfg.slots * (size_t) cfg.size + arrival_off(slot, peer);
    }
    size_t pub_off(int slot) const {
        return 2u * (size_t) cfg.slots * (size_t) cfg.size + (size_t) slot;
    }
    uint32_t * arrival_flag(int slot, int peer) { return flag_host + arrival_off(slot, peer); }
    uint32_t * visible_flag(int slot, int peer) { return flag_host + visible_off(slot, peer); }
    uint32_t * pub_flag(int slot) { return flag_host + pub_off(slot); }
    size_t hybh_off(int slot) const { return pub_off(0) + (size_t) cfg.slots + (size_t) slot; }
    size_t hybf_off(int slot) const { return hybh_off(0) + (size_t) cfg.slots + (size_t) slot; }
    size_t flag_count() const {
        return 2u * (size_t) cfg.size * (size_t) cfg.slots + 3u * (size_t) cfg.slots;
    }
};

FastReduce::FastReduce() : p_(new FastReduceImpl()) {}
uint32_t * FastReduceImpl::pub_counter(hipStream_t stm, int slot, int kind) {
    int dev = 0;
    if (hipStreamGetDevice(stm, &dev) != hipSuccess) return nullptr;
    std::lock_guard<std::mutex> lock(pub_counter_mu);
    auto it = pub_counters.find(dev);
    if (it == pub_counters.end()) {
        int prev = 0;
        if (hipGetDevice(&prev) != hipSuccess || hipSetDevice(dev) != hipSuccess) return nullptr;
        uint32_t * ctr = nullptr;
        const size_t bytes = (size_t) cfg.slots * kPubKinds * sizeof(uint32_t);
        const bool ok = hipMalloc((void **) &ctr, bytes) == hipSuccess &&
                        hipMemsetAsync(ctr, 0, bytes, stm) == hipSuccess;   // ahead of this stream's first publish
        (void) hipSetDevice(prev);
        if (!ok) return nullptr;
        it = pub_counters.emplace(dev, ctr).first;
    }
    return it->second + (size_t) slot * kPubKinds + (size_t) kind;
}

FastReduce::~FastReduce() { shutdown(); }

bool FastReduce::ok() const { return p_ && p_->up; }

bool FastReduce::init(const Config & cfg, std::string * err) {
    auto & s = *p_;
    s.cfg = cfg;
    if (cfg.size <= 1) { s.up = false; return true; }
    if (cfg.size != 2) {
        if (err) *err = "fast-reduce: a two-rank exchange, " + std::to_string(cfg.size) + " ranks keep RCCL";
        return false;
    }

    auto bail = [&](const std::string & what) {
        if (err) *err = "fast-reduce: " + what;
        shutdown();
        return false;
    };

    // ── device ────────────────────────────────────────────────────────────
    int n_dev = 0;
    ibv_device ** devs = ibv_get_device_list(&n_dev);
    if (!devs) return bail("ibv_get_device_list failed");
    ibv_device * dev = nullptr;
    for (int i = 0; i < n_dev; ++i) {
        if (cfg.hca.empty() || cfg.hca == ibv_get_device_name(devs[i])) {
            dev = devs[i];
            break;
        }
    }
    if (!dev) { ibv_free_device_list(devs); return bail("no HCA named " + cfg.hca); }
    s.ctx = ibv_open_device(dev);
    ibv_free_device_list(devs);
    if (!s.ctx) return bail("ibv_open_device failed");

    s.pd = ibv_alloc_pd(s.ctx);
    if (!s.pd) return bail("ibv_alloc_pd failed");
    // Two work requests per peer per reduction, a step's worth in flight.
    s.cq = ibv_create_cq(s.ctx, 4096, nullptr, nullptr, 0);
    if (!s.cq) return bail("ibv_create_cq failed");

    // ── buffers ───────────────────────────────────────────────────────────
    const size_t data_elems = (size_t) cfg.slots * s.slot_stride();
    if (hipHostMalloc((void **) &s.data_host, data_elems * sizeof(float),
                      hipHostMallocDefault) != hipSuccess) {
        return bail("hipHostMalloc for the payload region failed");
    }
    std::memset(s.data_host, 0, data_elems * sizeof(float));
    if (hipHostMalloc((void **) &s.flag_host, s.flag_count() * sizeof(uint32_t),
                      hipHostMallocDefault) != hipSuccess) {
        return bail("hipHostMalloc for the flag region failed");
    }
    std::memset(s.flag_host, 0, s.flag_count() * sizeof(uint32_t));
    if (fr_env_on("LUCE_CLUSTER_HYBRID_EXCHANGE")) {
        // Decode widths only: 5120 x 6 lanes fits; prefill keeps its own path.
        s.hyb_elems = std::min(cfg.max_elems, 32768);
        const size_t bytes = (size_t) 2 * (size_t) cfg.slots * (size_t) s.hyb_elems * sizeof(float);
        float * h = nullptr;
        if (hipHostMalloc((void **) &h, bytes, hipHostMallocDefault) != hipSuccess) {
            return bail("hipHostMalloc for the hybrid exchange failed");
        }
        s.hyb_host = h;
        if (hipHostGetDevicePointer((void **) &s.hyb_dev, s.hyb_host, 0) != hipSuccess) {
            return bail("hipHostGetDevicePointer for the hybrid exchange failed");
        }
        s.hyb_ring.assign(FastReduceImpl::kHybRing, {UINT64_MAX, 0});
        std::fprintf(stderr, "[fast-reduce] hybrid exchange buffers: %.1f MiB pinned\n", bytes / 1048576.0);
    }

    if (hipHostGetDevicePointer((void **) &s.data_dev, s.data_host, 0) != hipSuccess ||
        hipHostGetDevicePointer((void **) &s.flag_dev, s.flag_host, 0) != hipSuccess) {
        return bail("hipHostGetDevicePointer failed");
    }

    s.mr_data = ibv_reg_mr(s.pd, s.data_host, data_elems * sizeof(float),
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    s.mr_flag = ibv_reg_mr(s.pd, s.flag_host, s.flag_count() * sizeof(uint32_t),
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!s.mr_data || !s.mr_flag) return bail("ibv_reg_mr failed (memlock limit?)");

    // ── queue pairs ───────────────────────────────────────────────────────
    s.qps.assign((size_t) cfg.size, nullptr);
    for (int r = 0; r < cfg.size; ++r) {
        if (r == cfg.rank) continue;
        ibv_qp_init_attr ia{};
        ia.send_cq = s.cq;
        ia.recv_cq = s.cq;
        ia.qp_type = IBV_QPT_RC;
        ia.cap.max_send_wr = 4096;
        ia.cap.max_recv_wr = 1;
        ia.cap.max_send_sge = 1;
        ia.cap.max_recv_sge = 1;
        ia.cap.max_inline_data = 64;      // the flag write goes inline
        s.qps[(size_t) r] = ibv_create_qp(s.pd, &ia);
        if (!s.qps[(size_t) r]) return bail("ibv_create_qp failed");

        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_INIT;
        a.pkey_index = 0;
        a.port_num = (uint8_t) cfg.ib_port;
        a.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
        if (ibv_modify_qp(s.qps[(size_t) r], &a,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0) {
            return bail("ibv_modify_qp to INIT failed");
        }
    }

    // The memory keys and addresses are the same for every peer; only the
    // queue pair numbers differ.
    Endpoint mine{};
    mine.rkey_data = s.mr_data->rkey;
    mine.rkey_flag = s.mr_flag->rkey;
    mine.data_addr = (uint64_t) (uintptr_t) s.data_host;
    mine.flag_addr = (uint64_t) (uintptr_t) s.flag_host;
    mine.psn = 0;
    ibv_gid gid{};
    if (ibv_query_gid(s.ctx, (uint8_t) cfg.ib_port, cfg.gid_index, &gid) != 0) {
        return bail("ibv_query_gid failed");
    }
    std::memcpy(mine.gid, gid.raw, 16);

    if (cfg.size > kMaxRanks) return bail("more ranks than the endpoint carries");
    for (int r = 0; r < cfg.size; ++r) {
        if (r == cfg.rank) continue;
        mine.qpn[r] = s.qps[(size_t) r]->qp_num;
    }

    std::vector<Endpoint> table;
    if (!exchange(cfg, mine, table, err)) { shutdown(); return false; }
    s.peers = table;

    // LUCE_CLUSTER_FAST_REDUCE_MTU: the path MTU (default 4096, what a direct
    // ConnectX link runs).
    const char * mtu_env = std::getenv("LUCE_CLUSTER_FAST_REDUCE_MTU");
    const int mtu = mtu_env ? std::atoi(mtu_env) : 4096;

    for (int r = 0; r < cfg.size; ++r) {
        if (r == cfg.rank) continue;
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_RTR;
        a.path_mtu = mtu >= 4096 ? IBV_MTU_4096 : (mtu >= 2048 ? IBV_MTU_2048 : IBV_MTU_1024);
        a.dest_qp_num = s.peers[(size_t) r].qpn[cfg.rank];
        a.rq_psn = 0;
        a.max_dest_rd_atomic = 1;
        a.min_rnr_timer = 12;
        a.ah_attr.is_global = 1;
        a.ah_attr.port_num = (uint8_t) cfg.ib_port;
        a.ah_attr.grh.hop_limit = 64;
        a.ah_attr.grh.sgid_index = (uint8_t) cfg.gid_index;
        std::memcpy(a.ah_attr.grh.dgid.raw, s.peers[(size_t) r].gid, 16);
        if (ibv_modify_qp(s.qps[(size_t) r], &a,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            return bail("ibv_modify_qp to RTR failed");
        }
        ibv_qp_attr b{};
        b.qp_state = IBV_QPS_RTS;
        b.timeout = 14;
        b.retry_cnt = 7;
        b.rnr_retry = 7;
        b.sq_psn = 0;
        b.max_rd_atomic = 1;
        if (ibv_modify_qp(s.qps[(size_t) r], &b,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) != 0) {
            return bail("ibv_modify_qp to RTS failed");
        }
    }

    s.pending_n.assign((size_t) FastReduceImpl::kPendingRing, 0);

    // ── progress ──────────────────────────────────────────────────────────
    // LUCE_CLUSTER_FAST_REDUCE_STALL_S: how long a reduction may wait before
    // the rank releases its GPU and exits (0 turns the watchdog off).
    const char * stall_env = std::getenv("LUCE_CLUSTER_FAST_REDUCE_STALL_S");
    const double stall_s = stall_env ? std::atof(stall_env) : 20.0;

    s.up = true;
    s.progress = std::thread([&s, stall_s]() {
        uint64_t next = 1;
        std::vector<ibv_wc> wc(64);
        // Stall release: every flag a stream can be parked on for reduction q,
        // including the hybrid exchange's export and final words, so the GPUs
        // drain instead of needing a reset.
        auto release_one = [&s](uint64_t q) {
            const int qs = (int) (q % (uint64_t) s.cfg.slots);
            for (int r = 0; r < s.cfg.size; ++r) {
                if (r != s.cfg.rank) *(volatile uint32_t *) s.visible_flag(qs, r) = (uint32_t) q;
            }
            if (s.hyb_host) {
                *(volatile uint32_t *) (s.flag_host + s.hybh_off(qs)) = (uint32_t) q;
                *(volatile uint32_t *) (s.flag_host + s.hybf_off(qs)) = (uint32_t) q;
            }
        };
        // A rank that cannot finish reduction `first` (stalled, or its RDMA
        // write failed) releases every reduction the host has handed its GPU
        // so the queue drains, makes later submits no-ops, and exits once the
        // GPU is idle: exiting with a parked queue costs a GPU reset, and the
        // peer's own watchdog ends the other rank.
        auto release_and_exit = [&s, &release_one](uint64_t first) {
            s.failed.store(true);
            // Nothing reports a finished reduction, so keep releasing for the
            // whole grace period; late submits are released too.
            const auto drain_t0 = std::chrono::steady_clock::now();
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - drain_t0).count() < 15.0) {
                const uint64_t last = std::max(first, s.seq.load());
                for (uint64_t q = first; q <= last; ++q) release_one(q);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            std::fprintf(stderr, "[fast-reduce] rank %d: GPU released, exiting\n", s.cfg.rank);
            std::fflush(nullptr);
            std::_Exit(4);
        };
        // A broken connection also starves the reductions already published:
        // any of the last ring's worth may still wait on a peer's flag.
        auto oldest_in_flight = [&s](uint64_t q) {
            return q > (uint64_t) s.cfg.slots ? q - (uint64_t) s.cfg.slots + 1 : (uint64_t) 1;
        };
        while (!s.stop.load(std::memory_order_relaxed)) {
            const int slot = (int) (next % (uint64_t) s.cfg.slots);
            volatile uint32_t * pub = s.pub_flag(slot);
            uint64_t spins = 0;
            bool moaned = false;
            const auto wait_t0 = std::chrono::steady_clock::now();
            // The unpublished-reduction clock starts when the host hands the
            // reduction out, not when this thread starts waiting: an idle
            // server waits here between requests and that time is no stall.
            bool handed_out = false;
            auto handed_t0 = wait_t0;
            while (*pub != (uint32_t) next) {
                if (s.stop.load(std::memory_order_relaxed)) return;
                if (++spins < 200000ull) continue;
                spins = 0;
                std::this_thread::yield();
                if (!handed_out && s.seq.load() >= next) {
                    handed_out = true;
                    handed_t0 = std::chrono::steady_clock::now();
                }
                if (handed_out && stall_s > 0.0) {
                    const double w = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - handed_t0).count();
                    if (w > stall_s) {   // handed out, never published
                        std::fprintf(stderr,
                                     "[fast-reduce] rank %d: STALL, reduction %llu unpublished for %.0f s "
                                     "with %llu queued after it; releasing and exiting\n",
                                     s.cfg.rank, (unsigned long long) next, w,
                                     (unsigned long long) (s.seq.load() - next));
                        release_and_exit(next);
                    }
                }
                if (next <= 1) continue;
                // The GPU published reduction p = next-1 and is waiting on the
                // peer's flag for it. Arrived long ago is the normal idle case;
                // published but not arrived means a peer never delivered.
                const uint64_t p = next - 1;
                const int ps = (int) (p % (uint64_t) s.cfg.slots);
                bool arrived = true;
                for (int r = 0; r < s.cfg.size; ++r) {
                    if (r == s.cfg.rank) continue;
                    if (*(volatile uint32_t *) s.visible_flag(ps, r) != (uint32_t) p) arrived = false;
                }
                if (arrived) continue;
                const double waited = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - wait_t0).count();
                if (waited > 5.0 && !moaned) {
                    moaned = true;
                    std::string vis;
                    for (int r = 0; r < s.cfg.size; ++r) {
                        if (r == s.cfg.rank) continue;
                        vis += " r" + std::to_string(r) + "=" +
                               std::to_string(*(volatile uint32_t *) s.visible_flag(ps, r));
                    }
                    std::fprintf(stderr,
                                 "[fast-reduce] rank %d: reduction %llu published %.1f s ago, "
                                 "not arrived; peer flags%s (want %llu); host submitted %llu\n",
                                 s.cfg.rank, (unsigned long long) p, waited, vis.c_str(),
                                 (unsigned long long) p,
                                 (unsigned long long) s.seq.load());
                }
                if (stall_s <= 0.0 || waited < stall_s) continue;
                const uint64_t last = s.seq.load();
                std::fprintf(stderr,
                             "[fast-reduce] rank %d: STALL at reduction %llu for %.0f s "
                             "(peer gone?); releasing %llu pending reduction(s) and exiting\n",
                             s.cfg.rank, (unsigned long long) p, waited,
                             (unsigned long long) (last >= p ? last - p + 1 : 1));
                release_and_exit(p);
            }
            int n = s.pending_n[(size_t) (next % (uint64_t) FastReduceImpl::kPendingRing)];
            const bool exact_row = n < 0;   // submit(..., exact): f32 even on a bf16 wire
            if (exact_row) n = -n;
            for (int r = 0; r < s.cfg.size; ++r) {
                if (r == s.cfg.rank) continue;
                const Endpoint & pe = s.peers[(size_t) r];

                ibv_sge sge{};
                sge.addr = (uint64_t) (uintptr_t) (s.data_host +
                    (size_t) slot * s.slot_stride() + (size_t) s.cfg.rank * s.cfg.max_elems);
                const bool bf16_row = !exact_row;
                sge.length = (uint32_t) ((size_t) n * (bf16_row ? 2u : sizeof(float)));
                sge.lkey = s.mr_data->lkey;

                ibv_send_wr w{};
                w.wr_id = next;
                w.sg_list = &sge;
                w.num_sge = 1;
                w.opcode = IBV_WR_RDMA_WRITE;
                w.send_flags = 0;
                w.wr.rdma.remote_addr = pe.data_addr +
                    ((size_t) slot * s.slot_stride() + (size_t) s.cfg.rank * s.cfg.max_elems) * sizeof(float);
                w.wr.rdma.rkey = pe.rkey_data;

                // The flag rides the same queue pair, so reliable-connection
                // ordering puts it after the payload. That ordering is the
                // whole protocol: a flag that could overtake its data would
                // hand the peer a half-written buffer.
                const uint32_t flagv = (uint32_t) next;
                ibv_sge fsge{};
                fsge.addr = (uint64_t) (uintptr_t) &flagv;
                fsge.length = sizeof(uint32_t);
                fsge.lkey = 0;

                ibv_send_wr fw{};
                fw.wr_id = next;
                fw.sg_list = &fsge;
                fw.num_sge = 1;
                fw.opcode = IBV_WR_RDMA_WRITE;
                fw.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
                // Straight at the word the peer's stream is waiting on. When
                // a shader was the waiter this had to go through the host: a
                // DMA write into system memory is coherent with a CPU read but
                // nothing invalidates the GPU's caches for it. The command
                // processor is not behind those caches, so the hop is gone.
                fw.wr.rdma.remote_addr = pe.flag_addr +
                    s.visible_off(slot, s.cfg.rank) * sizeof(uint32_t);
                fw.wr.rdma.rkey = pe.rkey_flag;

                w.next = &fw;
                ibv_send_wr * bad = nullptr;
                if (ibv_post_send(s.qps[(size_t) r], &w, &bad) != 0) {
                    std::fprintf(stderr, "[fast-reduce] rank %d: ibv_post_send failed (seq %llu); "
                                 "releasing and exiting\n", s.cfg.rank, (unsigned long long) next);
                    release_and_exit(oldest_in_flight(next));
                }
            }
            // Drain completions so the send queue does not fill. A failed
            // write is not recoverable here: the peer is already waiting on a
            // flag that will never arrive, so this rank releases its own GPU
            // and exits, and the peer's watchdog ends the other rank.
            int done = 0;
            do {
                done = ibv_poll_cq(s.cq, (int) wc.size(), wc.data());
                for (int i = 0; i < done; ++i) {
                    if (wc[(size_t) i].status != IBV_WC_SUCCESS) {
                        std::fprintf(stderr,
                                     "[fast-reduce] write failed: %s (seq %llu); releasing and exiting\n",
                                     ibv_wc_status_str(wc[(size_t) i].status),
                                     (unsigned long long) wc[(size_t) i].wr_id);
                        release_and_exit(oldest_in_flight(next));
                    }
                }
            } while (done > 0);

            // Nothing to carry across any more: the peer's own stream is
            // waiting on the word this rank's NIC wrote.
            if (next <= 3 || next % 4096 == 0) {
                std::fprintf(stderr, "[fast-reduce] seq %llu delivered (%d floats)\n",
                             (unsigned long long) next, n);
            }
            ++next;
        }
    });

    return true;
}

bool FastReduce::submit(float * data, size_t n, void * stream, bool exact) {
    auto & s = *p_;
    if (s.failed.load(std::memory_order_relaxed)) return true;   // stalled: drain, then exit
    if (!s.up || n == 0 || n > (size_t) s.cfg.max_elems) return false;

    const uint64_t seq = s.seq.fetch_add(1, std::memory_order_relaxed) + 1;
    const int slot = (int) (seq % (uint64_t) s.cfg.slots);
    // The progress thread reads this once it sees the flag, and the flag is
    // raised further down the stream -- so writing it here is ordered ahead of
    // any possible read. An exact row (f32 on a bf16 wire) is stored negated.
    s.pending_n[(size_t) (seq % (uint64_t) FastReduceImpl::kPendingRing)] = exact ? -(int) n : (int) n;

    // The wait is a stream operation, not a kernel spin. On gfx1151 a kernel
    // spinning on a flag the host raised needs a system-scope acquire, which
    // invalidates the L2 the next layer is about to read; hipStreamWaitValue32
    // is handled by the command processor, outside the shader caches.
    //
    // No slot-reuse wait is needed: the ranks advance in lockstep (a rank
    // publishes reduction q only after it added the peer's row of q - 1), so a
    // slot comes round again only after both ranks consumed it `slots`
    // reductions earlier, and the stream orders this GPU's writes behind its
    // reads.
    //
    // Pack straight into the pinned row the NIC sends from (the kernel raises
    // the flag), wait for the peer's row, add it straight out of pinned memory.
    auto stm = (hipStream_t) stream;
    const int wire_bf16 = exact ? 0 : 1;
    const int peer = 1 - s.cfg.rank;
    float * own_row = s.data_dev + (size_t) slot * s.slot_stride() + (size_t) s.cfg.rank * s.cfg.max_elems;
    const float * peer_row = s.data_dev + (size_t) slot * s.slot_stride() + (size_t) peer * s.cfg.max_elems;
    uint32_t * ctr = s.pub_counter(stm, slot, 0);
    if (!ctr) return false;
    fast_reduce_launch_pack(data, own_row, (int) n, wire_bf16,
                            s.flag_dev + s.pub_off(slot), (uint32_t) seq, ctr, stm);
    if (hipStreamWaitValue32(stm, s.flag_dev + s.visible_off(slot, peer), (uint32_t) seq,
                             hipStreamWaitValueEq, 0xffffffffu) != hipSuccess) return false;
    fast_reduce_launch_add(data, peer_row, (int) n, wire_bf16, stm);
    return true;
}

bool FastReduce::hyb_available() const {
    const auto & s = *p_;
    return s.up && s.hyb_host != nullptr && !s.failed.load(std::memory_order_relaxed);
}

// The sequence number of hybrid exchange k, taken by its first call.
static uint64_t hyb_seq_for(FastReduceImpl & s, uint64_t k, size_t n) {
    auto & e = s.hyb_ring[(size_t) (k % (uint64_t) FastReduceImpl::kHybRing)];
    if (e.first == k) return e.second;
    const uint64_t seq = s.seq.fetch_add(1, std::memory_order_relaxed) + 1;
    s.pending_n[(size_t) (seq % (uint64_t) FastReduceImpl::kPendingRing)] = (int) n;
    e = {k, seq};
    return seq;
}

bool FastReduce::hyb_export(const float * data, size_t n, void * stream, uint64_t k) {
    auto & s = *p_;
    if (s.failed.load(std::memory_order_relaxed)) return true;
    if (!s.hyb_host || n == 0 || n > (size_t) s.hyb_elems) return false;
    const uint64_t seq = hyb_seq_for(s, k, n);
    const int slot = (int) (seq % (uint64_t) s.cfg.slots);
    auto stm = (hipStream_t) stream;
    uint32_t * ctr = s.pub_counter(stm, slot, 1);
    if (!ctr) return false;
    fast_reduce_launch_pack(data, s.hyb_part(0, slot), (int) n, 0,
                            s.flag_dev + s.hybh_off(slot), (uint32_t) seq, ctr, stm);
    return true;
}

bool FastReduce::hyb_combine(const float * data, size_t n, void * stream, uint64_t k) {
    auto & s = *p_;
    if (s.failed.load(std::memory_order_relaxed)) return true;
    if (!s.hyb_host || n == 0 || n > (size_t) s.hyb_elems) return false;
    const uint64_t seq = hyb_seq_for(s, k, n);
    const int slot = (int) (seq % (uint64_t) s.cfg.slots);
    const int peer = 1 - s.cfg.rank;
    float * own_row = s.data_dev + (size_t) slot * s.slot_stride() + (size_t) s.cfg.rank * s.cfg.max_elems;
    const float * peer_row = s.data_dev + (size_t) slot * s.slot_stride() + (size_t) peer * s.cfg.max_elems;
    auto stm = (hipStream_t) stream;
    if (hipStreamWaitValue32(stm, s.flag_dev + s.hybh_off(slot), (uint32_t) seq,
                             hipStreamWaitValueEq, 0xffffffffu) != hipSuccess) return false;
    uint32_t * ctr_pub = s.pub_counter(stm, slot, 2);
    uint32_t * ctr_fin = s.pub_counter(stm, slot, 3);
    if (!ctr_pub || !ctr_fin) return false;
    fast_reduce_launch_hyb_combine(data, s.hyb_part(0, slot), own_row, (int) n,
                                   s.flag_dev + s.pub_off(slot), (uint32_t) seq, ctr_pub, stm);
    if (hipStreamWaitValue32(stm, s.flag_dev + s.visible_off(slot, peer), (uint32_t) seq,
                             hipStreamWaitValueEq, 0xffffffffu) != hipSuccess) return false;
    fast_reduce_launch_hyb_final(own_row, peer_row, s.hyb_part(1, slot), (int) n,
                                 s.flag_dev + s.hybf_off(slot), (uint32_t) seq, ctr_fin, stm);
    return true;
}

bool FastReduce::hyb_import(float * data, size_t n, void * stream, uint64_t k) {
    auto & s = *p_;
    if (s.failed.load(std::memory_order_relaxed)) return true;
    if (!s.hyb_host || n == 0 || n > (size_t) s.hyb_elems) return false;
    const uint64_t seq = hyb_seq_for(s, k, n);
    const int slot = (int) (seq % (uint64_t) s.cfg.slots);
    auto stm = (hipStream_t) stream;
    if (hipStreamWaitValue32(stm, s.flag_dev + s.hybf_off(slot), (uint32_t) seq,
                             hipStreamWaitValueEq, 0xffffffffu) != hipSuccess) return false;
    return hipMemcpyAsync(data, s.hyb_part(1, slot), n * sizeof(float), hipMemcpyHostToDevice, stm) == hipSuccess;
}

bool FastReduce::submit_from_root(float * data, size_t n, void * stream, int root, bool exact) {
    auto & s = *p_;
    if (!s.up || n == 0 || n > (size_t) s.cfg.max_elems) return false;
    if (s.cfg.rank != root &&
        hipMemsetAsync(data, 0, n * sizeof(float), (hipStream_t) stream) != hipSuccess) {
        return false;
    }
    return submit(data, n, stream, exact);
}

void FastReduce::shutdown() {
    if (!p_) return;
    auto & s = *p_;
    s.stop.store(true);
    if (s.progress.joinable()) s.progress.join();
    s.up = false;

    for (ibv_qp * q : s.qps) if (q) ibv_destroy_qp(q);
    s.qps.clear();
    if (s.mr_data) { ibv_dereg_mr(s.mr_data); s.mr_data = nullptr; }
    if (s.mr_flag) { ibv_dereg_mr(s.mr_flag); s.mr_flag = nullptr; }
    if (s.cq) { ibv_destroy_cq(s.cq); s.cq = nullptr; }
    if (s.pd) { ibv_dealloc_pd(s.pd); s.pd = nullptr; }
    if (s.ctx) { ibv_close_device(s.ctx); s.ctx = nullptr; }

    for (auto & [dev, ctr] : s.pub_counters) {
        int prev = 0;
        if (hipGetDevice(&prev) == hipSuccess && hipSetDevice(dev) == hipSuccess) {
            (void) hipFree(ctr);
            (void) hipSetDevice(prev);
        }
    }
    s.pub_counters.clear();
    if (s.data_host) { (void) hipHostFree(s.data_host); s.data_host = nullptr; }
    if (s.flag_host) { (void) hipHostFree(s.flag_host); s.flag_host = nullptr; }
    if (s.hyb_host) { (void) hipHostFree(s.hyb_host); s.hyb_host = nullptr; }
}

}  // namespace luce::cluster

#else  // !LUCE_CLUSTER

namespace luce::cluster {

struct FastReduceImpl {};

FastReduce::FastReduce() = default;
FastReduce::~FastReduce() = default;

bool FastReduce::init(const Config &, std::string * err) {
    if (err) *err = "fast reduce not compiled in (build with -DLUCE_CLUSTER=ON on HIP)";
    return false;
}

void FastReduce::shutdown() {}
bool FastReduce::ok() const { return false; }
bool FastReduce::submit(float *, size_t, void *, bool) { return false; }
bool FastReduce::submit_from_root(float *, size_t, void *, int, bool) { return false; }
bool FastReduce::hyb_available() const { return false; }
bool FastReduce::hyb_export(const float *, size_t, void *, uint64_t) { return false; }
bool FastReduce::hyb_combine(const float *, size_t, void *, uint64_t) { return false; }
bool FastReduce::hyb_import(float *, size_t, void *, uint64_t) { return false; }

}  // namespace luce::cluster

#endif  // LUCE_CLUSTER
