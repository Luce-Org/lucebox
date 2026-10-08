#include "cluster/fast_reduce.h"

// The RDMA fabric is built only with -DLUCE_CLUSTER=ON (HIP + libibverbs).
// Without it, FastReduce never comes up and callers keep RCCL, the same way
// cluster_comm.cpp compiles its RCCL half under LUCE_CLUSTER_RCCL.
#ifdef LUCE_CLUSTER

#include <hip/hip_runtime.h>
#include <infiniband/verbs.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
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

// The kernels live in fast_reduce_kernels.cu (LANGUAGE HIP).
void fast_reduce_launch_pack(const float * src, void * dst, int n, int bf16, hipStream_t stream);
void fast_reduce_launch_pack_pub(const float * src, void * dst, int n, int bf16,
                                 uint32_t * flag, uint32_t value, uint32_t * counter, hipStream_t stream);
void fast_reduce_launch_hyb_combine_pub(const float * cold, const float * hot, const float * shared, void * own_row,
                                        int n, int bf16, uint32_t * flag, uint32_t value, uint32_t * counter,
                                        hipStream_t stream);
void fast_reduce_launch_hyb_final_pub(const void * own_row, const void * peer_row, float * dst, int n, int bf16,
                                      uint32_t * flag, uint32_t value, uint32_t * counter, hipStream_t stream);
void fast_reduce_launch_add_zc(float * dst, const void * peer, int n, int bf16, hipStream_t stream);
void fast_reduce_launch_flag_sys(uint32_t * flag, uint32_t value, hipStream_t stream);
void fast_reduce_launch_hyb_combine(const float * cold, const float * hot, const float * shared, void * own_row,
                                    int n, int bf16, hipStream_t stream);
void fast_reduce_launch_hyb_final(const void * own_row, const void * peer_row, float * dst, int n, int bf16,
                                  hipStream_t stream);
void fast_reduce_launch_set_flag(uint32_t * pub_flag, uint32_t * slot_done,
                                 uint32_t * progress, uint32_t seq,
                                 uint32_t slot_span, uint64_t spin_limit,
                                 hipStream_t stream);
void fast_reduce_launch_wait_flags(uint32_t * const * peer_flags,
                                   uint32_t * timed_out, uint32_t * progress,
                                   uint32_t seq, int n_peers,
                                   uint64_t spin_limit, hipStream_t stream);
void fast_reduce_launch_add(float * dst, const float * scratch, int n_peers,
                            int n, int stride, uint32_t * slot_done,
                            uint32_t * progress, uint32_t seq,
                            hipStream_t stream);

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

bool write_all(int fd, const void * p, size_t n) {
    const char * c = (const char *) p;
    while (n) {
        const ssize_t w = ::write(fd, c, n);
        if (w <= 0) return false;
        c += w;
        n -= (size_t) w;
    }
    return true;
}

bool read_all(int fd, void * p, size_t n) {
    char * c = (char *) p;
    while (n) {
        const ssize_t r = ::read(fd, c, n);
        if (r <= 0) return false;
        c += r;
        n -= (size_t) r;
    }
    return true;
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

        std::vector<int> peers((size_t) cfg.size, -1);
        for (int i = 1; i < cfg.size; ++i) {
            const int fd = ::accept(lst, nullptr, nullptr);
            if (fd < 0) { ::close(lst); return fail("accept"); }
            set_nodelay(fd);
            int32_t r = -1;
            Endpoint e{};
            if (!read_all(fd, &r, sizeof(r)) || !read_all(fd, &e, sizeof(e)) ||
                r <= 0 || r >= cfg.size) {
                ::close(fd); ::close(lst);
                return fail("read endpoint");
            }
            all[(size_t) r] = e;
            peers[(size_t) r] = fd;
        }
        ::close(lst);
        for (int i = 1; i < cfg.size; ++i) {
            if (!write_all(peers[(size_t) i], all.data(), sizeof(Endpoint) * all.size())) {
                for (int fd : peers) if (fd >= 0) ::close(fd);
                return fail("write table");
            }
        }
        for (int fd : peers) if (fd >= 0) ::close(fd);
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
    if (!write_all(fd, &r, sizeof(r)) || !write_all(fd, &mine, sizeof(mine)) ||
        !read_all(fd, all.data(), sizeof(Endpoint) * all.size())) {
        ::close(fd);
        return fail("exchange");
    }
    ::close(fd);
    return true;
}

}  // namespace

// A switch is on when set to anything but empty or "0". Cluster mode sets the
// measured configuration's switches by default (launch_profiles.h), so "0" is
// how an operator turns one off.
static bool fr_env_on(const char * name) {
    const char * v = std::getenv(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

// LUCE_CLUSTER_FAST_REDUCE_LEAN=1 drops the per-reduction slot-reuse waits and
// done writes on the zero-copy and hybrid paths. Neither carries data: the
// ranks advance in lockstep (a rank publishes reduction q only after it added
// the peer's row of q - 1), so a slot comes round again only after both ranks
// consumed it 256 reductions earlier, and the GPU's own stream orders its
// writes behind its reads. On a discrete GPU each of these stream operations
// is a blit kernel with its own dispatch gap, four per layer.
static bool fast_reduce_lean() {
    static const bool lean = fr_env_on("LUCE_CLUSTER_FAST_REDUCE_LEAN");
    return lean;
}

// The kernel that writes an exchange row also publishes its flag (one
// system-scope release by its grid's last block) instead of a
// hipStreamWriteValue32 after it: about 0.6 ms less per two-box verify step,
// same bits. LUCE_CLUSTER_FAST_REDUCE_KPUB=0 goes back to the stream writes.
static bool fast_reduce_kpub() {
    static const bool on = [] {
        const char * v = std::getenv("LUCE_CLUSTER_FAST_REDUCE_KPUB");
        return !(v && v[0] == '0' && v[1] == '\0');
    }();
    return on;
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
                                            // | [slots] pub | [slots] done
    uint32_t * flag_dev = nullptr;

    // Device-visible pointer table for the flags, one row per slot. The
    // payloads need no such table: they are moved by the copy engine, which
    // takes its addresses from the host.
    uint32_t **    peer_flags = nullptr;    // [slots][size-1]
    uint32_t **    peer_flags_dev = nullptr;

    // Device memory the peers' partials are copied into before they are added.
    // Adding straight out of the pinned staging buffer is what made the first
    // version 250 times slower than the collective it replaces: that memory is
    // uncached for the GPU.
    float *        scratch = nullptr;       // [slots][size-1][max_elems]

    uint32_t * timed_out_host = nullptr;
    uint32_t * timed_out_dev = nullptr;
    // seq*4 + {1 published, 2 waiting, 3 done}: which reduction the GPU is in
    // and what it is doing there.
    uint32_t * progress_host = nullptr;
    uint32_t * progress_dev = nullptr;

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
    // Hybrid exchange buffers, pinned: [part 0 = hot | part 1 = shared | 2 = final][slots][hyb_elems].
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
    size_t done_off(int slot) const {
        return pub_off(0) + (size_t) cfg.slots + (size_t) slot;
    }
    uint32_t * arrival_flag(int slot, int peer) { return flag_host + arrival_off(slot, peer); }
    uint32_t * visible_flag(int slot, int peer) { return flag_host + visible_off(slot, peer); }
    uint32_t * pub_flag(int slot) { return flag_host + pub_off(slot); }
    uint32_t * done_flag(int slot) { return flag_host + done_off(slot); }
    size_t hybh_off(int slot) const { return done_off(0) + (size_t) cfg.slots + (size_t) slot; }
    size_t hybf_off(int slot) const { return hybh_off(0) + (size_t) cfg.slots + (size_t) slot; }
    size_t flag_count() const {
        return 2u * (size_t) cfg.size * (size_t) cfg.slots + 4u * (size_t) cfg.slots;
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
uint64_t FastReduce::submitted() const { return p_ ? p_->seq.load() : 0; }
uint64_t FastReduce::timed_out() const {
    return (p_ && p_->timed_out_host) ? *p_->timed_out_host : 0;
}

bool FastReduce::init(const Config & cfg, std::string * err) {
    auto & s = *p_;
    s.cfg = cfg;
    if (cfg.size <= 1) { s.up = false; return true; }

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
    if (hipHostMalloc((void **) &s.timed_out_host, sizeof(uint32_t) * 16,
                      hipHostMallocDefault) != hipSuccess) {
        return bail("hipHostMalloc for the timeout word failed");
    }
    *s.timed_out_host = 0;
    if (hipHostMalloc((void **) &s.progress_host, sizeof(uint32_t) * 16,
                      hipHostMallocDefault) != hipSuccess) {
        return bail("hipHostMalloc for the progress word failed");
    }
    *s.progress_host = 0;
    if (fr_env_on("LUCE_CLUSTER_HYBRID_EXCHANGE") && cfg.size == 2) {
        // Decode widths only: 5120 x 6 lanes fits; prefill keeps its own path.
        s.hyb_elems = std::min(cfg.max_elems, 32768);
        const size_t bytes = (size_t) 3 * (size_t) cfg.slots * (size_t) s.hyb_elems * sizeof(float);
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
        hipHostGetDevicePointer((void **) &s.flag_dev, s.flag_host, 0) != hipSuccess ||
        hipHostGetDevicePointer((void **) &s.timed_out_dev, s.timed_out_host, 0) != hipSuccess ||
        hipHostGetDevicePointer((void **) &s.progress_dev, s.progress_host, 0) != hipSuccess) {
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

    for (int r = 0; r < cfg.size; ++r) {
        if (r == cfg.rank) continue;
        ibv_qp_attr a{};
        a.qp_state = IBV_QPS_RTR;
        // LUCE_CLUSTER_FAST_REDUCE_MTU=1024|2048|4096 (default 4096: the direct ConnectX link runs 4096).
        {
            static const int mtu_env = std::getenv("LUCE_CLUSTER_FAST_REDUCE_MTU") ? std::atoi(std::getenv("LUCE_CLUSTER_FAST_REDUCE_MTU")) : 4096;
            a.path_mtu = mtu_env >= 4096 ? IBV_MTU_4096 : (mtu_env >= 2048 ? IBV_MTU_2048 : IBV_MTU_1024);
        }
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

    // ── pointer tables, one row per slot ──────────────────────────────────
    const int n_peers = cfg.size - 1;
    const size_t rows = (size_t) cfg.slots * (size_t) n_peers;
    if (hipHostMalloc((void **) &s.peer_flags, rows * sizeof(uint32_t *),
                      hipHostMallocDefault) != hipSuccess) {
        return bail("hipHostMalloc for the flag table failed");
    }
    for (int slot = 0; slot < cfg.slots; ++slot) {
        int k = 0;
        for (int r = 0; r < cfg.size; ++r) {
            if (r == cfg.rank) continue;
            s.peer_flags[(size_t) slot * (size_t) n_peers + (size_t) k] =
                s.flag_dev + s.visible_off(slot, r);
            ++k;
        }
    }
    if (hipHostGetDevicePointer((void **) &s.peer_flags_dev, s.peer_flags, 0) != hipSuccess) {
        return bail("hipHostGetDevicePointer for the flag table failed");
    }
    if (hipMalloc((void **) &s.scratch,
                  rows * (size_t) cfg.max_elems * sizeof(float)) != hipSuccess) {
        return bail("hipMalloc for the peer scratch failed");
    }

    s.pending_n.assign((size_t) FastReduceImpl::kPendingRing, 0);

    // ── progress ──────────────────────────────────────────────────────────
    s.up = true;
    s.progress = std::thread([&s]() {
        const int n_peers_l = s.cfg.size - 1;
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
        while (!s.stop.load(std::memory_order_relaxed)) {
            const int slot = (int) (next % (uint64_t) s.cfg.slots);
            volatile uint32_t * pub = s.pub_flag(slot);
            uint64_t spins = 0;
            bool moaned = false;
            const auto wait_t0 = std::chrono::steady_clock::now();
            static const double stall_s = std::getenv("LUCE_CLUSTER_FAST_REDUCE_STALL_S")
                ? std::atof(std::getenv("LUCE_CLUSTER_FAST_REDUCE_STALL_S")) : 20.0;
            while (*pub != (uint32_t) next) {
                if (s.stop.load(std::memory_order_relaxed)) return;
                if (++spins < 200000ull) continue;
                spins = 0;
                std::this_thread::yield();
                {
                    const double w = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - wait_t0).count();
                    if (stall_s > 0.0 && w > stall_s && s.seq.load() >= next) {   // allocated, never published
                        s.failed.store(true);
                        const uint64_t last = s.seq.load();
                        std::fprintf(stderr,
                                     "[fast-reduce] rank %d: STALL, reduction %llu unpublished for %.0f s "
                                     "with %llu queued after it; releasing and exiting\n",
                                     s.cfg.rank, (unsigned long long) next, w,
                                     (unsigned long long) (last - next));
                        const auto t0 = std::chrono::steady_clock::now();
                        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 10.0) {
                            for (uint64_t q = next; q <= s.seq.load(); ++q) release_one(q);
                            std::this_thread::sleep_for(std::chrono::milliseconds(20));
                        }
                        std::fflush(nullptr);
                        std::_Exit(4);
                    }
                }
                if (next <= 1) continue;
                // The GPU published reduction p = next-1 and is waiting on the
                // peers' flags for it. Done long ago is the normal idle case;
                // published but not done means a peer never delivered.
                const uint64_t p = next - 1;
                const int ps = (int) (p % (uint64_t) s.cfg.slots);
                if (*(volatile uint32_t *) s.done_flag(ps) == (uint32_t) p) continue;
                if (fast_reduce_lean()) {
                    bool arrived = true;
                    for (int r = 0; r < s.cfg.size; ++r) {
                        if (r == s.cfg.rank) continue;
                        const uint32_t v = *(volatile uint32_t *) s.visible_flag(ps, r);
                        if (v != (uint32_t) p) arrived = false;
                    }
                    if (arrived) continue;
                }
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
                                 "not done; peer flags%s (want %llu); host submitted %llu\n",
                                 s.cfg.rank, (unsigned long long) p, waited, vis.c_str(),
                                 (unsigned long long) p,
                                 (unsigned long long) s.seq.load());
                }
                if (waited < stall_s) continue;
                // Stall: release every reduction the host has handed the GPU so
                // its queue drains, make later submits no-ops, and exit once the
                // GPU is idle. Exiting with a parked queue costs a GPU reset.
                s.failed.store(true);
                const uint64_t last = s.seq.load();
                std::fprintf(stderr,
                             "[fast-reduce] rank %d: STALL at reduction %llu for %.0f s "
                             "(peer gone?); releasing %llu pending reduction(s) and exiting\n",
                             s.cfg.rank, (unsigned long long) p, waited,
                             (unsigned long long) (last >= p ? last - p + 1 : 1));
                for (uint64_t q = p; q <= std::max(p, last); ++q) release_one(q);
                const auto drain_t0 = std::chrono::steady_clock::now();
                while (std::chrono::duration<double>(std::chrono::steady_clock::now() - drain_t0).count() < 15.0) {
                    const uint64_t l2 = std::max(p, s.seq.load());
                    for (uint64_t q = p; q <= l2; ++q) release_one(q);   // late submits too
                    if (*(volatile uint32_t *) s.done_flag((int) (l2 % (uint64_t) s.cfg.slots)) == (uint32_t) l2) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                }
                std::fprintf(stderr, "[fast-reduce] rank %d: GPU drained, exiting\n", s.cfg.rank);
                std::fflush(nullptr);
                std::_Exit(4);
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
                static const bool bf16_wire = fr_env_on("LUCE_CLUSTER_FAST_REDUCE_BF16") &&
                                              fr_env_on("LUCE_CLUSTER_FAST_REDUCE_ZC");
                sge.length = (uint32_t) ((size_t) n * (bf16_wire && !exact_row ? 2u : sizeof(float)));
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
                    s.stop.store(true);
                    return;
                }
            }
            // Drain completions so the send queue does not fill. A failed
            // write is not recoverable here -- the peer is already waiting on
            // a flag that will never arrive -- so it stops the thread and the
            // waiting kernels time out into a counted, visible failure.
            int done = 0;
            do {
                done = ibv_poll_cq(s.cq, (int) wc.size(), wc.data());
                for (int i = 0; i < done; ++i) {
                    if (wc[(size_t) i].status != IBV_WC_SUCCESS) {
                        std::fprintf(stderr,
                                     "[fast-reduce] write failed: %s (seq %llu)\n",
                                     ibv_wc_status_str(wc[(size_t) i].status),
                                     (unsigned long long) wc[(size_t) i].wr_id);
                        s.stop.store(true);
                        return;
                    }
                }
            } while (done > 0);

            // Nothing to carry across any more: the peer's own stream is
            // waiting on the word this rank's NIC wrote.
            if (next <= 3 || next % 4096 == 0) {
                std::fprintf(stderr,
                             "[fast-reduce] seq %llu delivered (%d floats), "
                             "gpu timeouts up to seq %u\n",
                             (unsigned long long) next, n,
                             *(volatile uint32_t *) s.timed_out_host);
            }
            ++next;
        }
    });

    return true;
}

bool FastReduce::submit(float * data, size_t n, void * stream, bool exact) {
    auto & s = *p_;
    if (s.failed.load(std::memory_order_relaxed)) return true;   // stalled: drain, then exit
    // The first handful of decisions, taken or declined, so a stall says which
    // reduction it stopped at rather than only that it stopped.
    static std::atomic<int> traced{0};
    const bool trace = traced.fetch_add(1, std::memory_order_relaxed) < 24;
    if (!s.up || n == 0 || n > (size_t) s.cfg.max_elems) {
        if (trace) {
            std::fprintf(stderr, "[fast-reduce] declined %zu floats (cap %d)\n",
                         n, s.cfg.max_elems);
        }
        return false;
    }
    if (trace) std::fprintf(stderr, "[fast-reduce] took %zu floats\n", n);

    const uint64_t seq = s.seq.fetch_add(1, std::memory_order_relaxed) + 1;
    const int slot = (int) (seq % (uint64_t) s.cfg.slots);
    // The progress thread reads this once it sees the flag, and the flag is
    // raised further down the stream -- so writing it here is ordered ahead of
    // any possible read. An exact row (f32 on a bf16 wire) is stored negated.
    s.pending_n[(size_t) (seq % (uint64_t) FastReduceImpl::kPendingRing)] = exact ? -(int) n : (int) n;

    const int n_peers = s.cfg.size - 1;
    auto stm = (hipStream_t) stream;

    // Loads, not cycles: roughly 100 ms of system-scope reads. What it waits
    // on takes about fourteen microseconds, so this only fires when something
    // is wrong -- and when it does it has to fire, not spin forever.
    constexpr uint64_t kSpin = 200000ull;

    // Publish and wait are stream operations, not kernels.
    //
    // A kernel cannot do this cheaply on gfx1151. Raising a flag the host will
    // see needs a system-scope release, which writes the L2 back; spinning on
    // one the host raised needs a system-scope acquire, which invalidates it.
    // Either way the weights the next layer is about to read are gone, and
    // measured that cost 34 ms per reduction -- while the same path with the
    // flags removed ran the step in 32.5 ms. Weakening the ordering does not
    // help: relaxed is cheap and the host then does not see the flag at all.
    //
    // hipStreamWriteValue32 and hipStreamWaitValue32 are handled by the
    // command processor instead, so the shader cores and their caches are not
    // involved at all -- which is what this needs and what a kernel cannot be.
    // LUCE_CLUSTER_FAST_REDUCE_KERNEL_FLAGS=1 restores the kernel form for
    // comparison.
    static const bool kernel_flags =
        std::getenv("LUCE_CLUSTER_FAST_REDUCE_KERNEL_FLAGS") != nullptr;

    // Do not write into a slot the progress thread has not finished with. The
    // kernel form waited on this; the stream form has to as well, or the GPU --
    // which runs a whole step's reductions ahead of the host -- laps the ring
    // and overwrites a payload that has not been sent. That is what made the
    // third request of a run stop rather than the first.
    // The previous user of this slot was seq - slots, and it leaves exactly that
    // value behind. Asking for anything higher is a wait nothing can satisfy.
    static const bool zc_lean = fast_reduce_lean() && fr_env_on("LUCE_CLUSTER_FAST_REDUCE_ZC") &&
                                s.cfg.size == 2;
    if (!kernel_flags && !zc_lean && seq > (uint64_t) s.cfg.slots) {
        const uint32_t need = (uint32_t) (seq - (uint64_t) s.cfg.slots);
        if (hipStreamWaitValue32(stm, s.flag_dev + s.done_off(slot), need,
                                 hipStreamWaitValueGte, 0xffffffffu) != hipSuccess) {
            return false;
        }
    }

    // Zero-copy (discrete GPU): pack straight into the pinned row, publish, wait
    // for the peer, add its row straight out of pinned memory, mark the slot done.
    static const bool zc = fr_env_on("LUCE_CLUSTER_FAST_REDUCE_ZC");
    static const int zc_bf16 = fr_env_on("LUCE_CLUSTER_FAST_REDUCE_BF16") ? 1 : 0;
    const int wire_bf16 = exact ? 0 : zc_bf16;
    if (zc && !kernel_flags && s.cfg.size == 2) {
        float * own_row = s.data_dev + (size_t) slot * s.slot_stride() + (size_t) s.cfg.rank * s.cfg.max_elems;
        const int peer = 1 - s.cfg.rank;
        const float * peer_row = s.data_dev + (size_t) slot * s.slot_stride() + (size_t) peer * s.cfg.max_elems;
        if (fast_reduce_kpub()) {
            uint32_t * ctr = s.pub_counter(stm, slot, 0);
            if (!ctr) return false;
            fast_reduce_launch_pack_pub(data, own_row, (int) n, wire_bf16,
                                        s.flag_dev + s.pub_off(slot), (uint32_t) seq, ctr, stm);
        } else {
            fast_reduce_launch_pack(data, own_row, (int) n, wire_bf16, stm);
            if (hipStreamWriteValue32(stm, s.flag_dev + s.pub_off(slot), (uint32_t) seq, 0) != hipSuccess) return false;
        }
        if (hipStreamWaitValue32(stm, s.flag_dev + s.visible_off(slot, peer), (uint32_t) seq,
                                 hipStreamWaitValueEq, 0xffffffffu) != hipSuccess) return false;
        fast_reduce_launch_add_zc(data, peer_row, (int) n, wire_bf16, stm);
        if (zc_lean) return true;
        return hipStreamWriteValue32(stm, s.flag_dev + s.done_off(slot), (uint32_t) seq, 0) == hipSuccess;
    }

    // Publish: the copy engine moves the partial into the pinned staging
    // buffer, then the flag is written. Both are on the stream, so the flag
    // cannot precede the data it announces.
    if (hipMemcpyAsync(s.data_host + (size_t) slot * s.slot_stride() +
                           (size_t) s.cfg.rank * s.cfg.max_elems,
                       data, n * sizeof(float), hipMemcpyDefault, stm) != hipSuccess) {
        return false;
    }
    if (kernel_flags) {
        fast_reduce_launch_set_flag(s.flag_dev + s.pub_off(slot),
                                    s.flag_dev + s.done_off(slot),
                                    s.progress_dev, (uint32_t) seq,
                                    (uint32_t) (s.cfg.slots - 2), kSpin, stm);
    } else if (hipStreamWriteValue32(stm, s.flag_dev + s.pub_off(slot),
                                     (uint32_t) seq, 0) != hipSuccess) {
        return false;
    }

    // Wait for the peers, then bring their partials into device memory before
    // touching them: the staging buffers are uncached for the GPU, and adding
    // straight out of them is what made the first version 250 times slower
    // than the collective it replaces.
    if (kernel_flags) {
        fast_reduce_launch_wait_flags(s.peer_flags_dev + (size_t) slot * (size_t) n_peers,
                                      s.timed_out_dev, s.progress_dev,
                                      (uint32_t) seq, n_peers, kSpin, stm);
    } else {
        for (int r = 0; r < s.cfg.size; ++r) {
            if (r == s.cfg.rank) continue;
            if (hipStreamWaitValue32(stm, s.flag_dev + s.visible_off(slot, r),
                                     (uint32_t) seq, hipStreamWaitValueEq,
                                     0xffffffffu) != hipSuccess) {
                return false;
            }
        }
    }
    float * scratch_row = s.scratch +
        ((size_t) slot * (size_t) n_peers) * (size_t) s.cfg.max_elems;
    int k = 0;
    for (int r = 0; r < s.cfg.size; ++r) {
        if (r == s.cfg.rank) continue;
        if (hipMemcpyAsync(scratch_row + (size_t) k * (size_t) s.cfg.max_elems,
                           s.data_host + (size_t) slot * s.slot_stride() +
                               (size_t) r * s.cfg.max_elems,
                           n * sizeof(float), hipMemcpyDefault, stm) != hipSuccess) {
            return false;
        }
        ++k;
    }
    fast_reduce_launch_add(data, scratch_row, n_peers, (int) n, s.cfg.max_elems,
                           !kernel_flags ? nullptr : s.flag_dev + s.done_off(slot),
                           s.progress_dev, (uint32_t) seq, stm);
    if (!kernel_flags &&
        hipStreamWriteValue32(stm, s.flag_dev + s.done_off(slot),
                              (uint32_t) seq, 0) != hipSuccess) {
        return false;
    }
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

bool FastReduce::hyb_export(const float * data, size_t n, void * stream, int part, uint64_t k) {
    auto & s = *p_;
    if (s.failed.load(std::memory_order_relaxed)) return true;
    if (!s.hyb_host || n == 0 || n > (size_t) s.hyb_elems || part < 0 || part > 1) return false;
    const uint64_t seq = hyb_seq_for(s, k, n);
    const int slot = (int) (seq % (uint64_t) s.cfg.slots);
    auto stm = (hipStream_t) stream;
    if (part == 0 && !fast_reduce_lean() && seq > (uint64_t) s.cfg.slots) {
        const uint32_t need = (uint32_t) (seq - (uint64_t) s.cfg.slots);
        if (hipStreamWaitValue32(stm, s.flag_dev + s.done_off(slot), need,
                                 hipStreamWaitValueGte, 0xffffffffu) != hipSuccess) return false;
    }
    static const bool one_export = fr_env_on("LUCE_CLUSTER_HYBRID_ONE_EXPORT");
    const bool last_part = part == 1 || one_export;
    if (last_part && fast_reduce_kpub()) {
        uint32_t * ctr = s.pub_counter(stm, slot, 1);
        if (!ctr) return false;
        fast_reduce_launch_pack_pub(data, s.hyb_part(part, slot), (int) n, 0,
                                    s.flag_dev + s.hybh_off(slot), (uint32_t) seq, ctr, stm);
        return true;
    }
    fast_reduce_launch_pack(data, s.hyb_part(part, slot), (int) n, 0, stm);
    if (last_part) {
        return hipStreamWriteValue32(stm, s.flag_dev + s.hybh_off(slot), (uint32_t) seq, 0) == hipSuccess;
    }
    return true;
}

bool FastReduce::hyb_combine(const float * data, size_t n, void * stream, uint64_t k) {
    auto & s = *p_;
    if (s.failed.load(std::memory_order_relaxed)) return true;
    if (!s.hyb_host || n == 0 || n > (size_t) s.hyb_elems) return false;
    static const int zc_bf16 = fr_env_on("LUCE_CLUSTER_FAST_REDUCE_BF16") ? 1 : 0;
    const uint64_t seq = hyb_seq_for(s, k, n);
    const int slot = (int) (seq % (uint64_t) s.cfg.slots);
    const int peer = 1 - s.cfg.rank;
    float * own_row = s.data_dev + (size_t) slot * s.slot_stride() + (size_t) s.cfg.rank * s.cfg.max_elems;
    const float * peer_row = s.data_dev + (size_t) slot * s.slot_stride() + (size_t) peer * s.cfg.max_elems;
    auto stm = (hipStream_t) stream;
    if (hipStreamWaitValue32(stm, s.flag_dev + s.hybh_off(slot), (uint32_t) seq,
                             hipStreamWaitValueEq, 0xffffffffu) != hipSuccess) return false;
    static const bool one_export = fr_env_on("LUCE_CLUSTER_HYBRID_ONE_EXPORT");
    if (fast_reduce_kpub()) {
        uint32_t * ctr_pub = s.pub_counter(stm, slot, 2);
        uint32_t * ctr_fin = s.pub_counter(stm, slot, 3);
        if (!ctr_pub || !ctr_fin) return false;
        fast_reduce_launch_hyb_combine_pub(data, s.hyb_part(0, slot), one_export ? nullptr : s.hyb_part(1, slot),
                                           own_row, (int) n, zc_bf16,
                                           s.flag_dev + s.pub_off(slot), (uint32_t) seq, ctr_pub, stm);
        if (hipStreamWaitValue32(stm, s.flag_dev + s.visible_off(slot, peer), (uint32_t) seq,
                                 hipStreamWaitValueEq, 0xffffffffu) != hipSuccess) return false;
        fast_reduce_launch_hyb_final_pub(own_row, peer_row, s.hyb_part(2, slot), (int) n, zc_bf16,
                                         s.flag_dev + s.hybf_off(slot), (uint32_t) seq, ctr_fin, stm);
        return true;
    }
    fast_reduce_launch_hyb_combine(data, s.hyb_part(0, slot), one_export ? nullptr : s.hyb_part(1, slot), own_row, (int) n, zc_bf16, stm);
    if (hipStreamWriteValue32(stm, s.flag_dev + s.pub_off(slot), (uint32_t) seq, 0) != hipSuccess) return false;
    if (hipStreamWaitValue32(stm, s.flag_dev + s.visible_off(slot, peer), (uint32_t) seq,
                             hipStreamWaitValueEq, 0xffffffffu) != hipSuccess) return false;
    fast_reduce_launch_hyb_final(own_row, peer_row, s.hyb_part(2, slot), (int) n, zc_bf16, stm);
    return hipStreamWriteValue32(stm, s.flag_dev + s.hybf_off(slot), (uint32_t) seq, 0) == hipSuccess;
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
    static const bool import_dma = std::getenv("LUCE_CLUSTER_HYBRID_IMPORT_KERNEL") == nullptr;
    if (import_dma) {
        if (hipMemcpyAsync(data, s.hyb_part(2, slot), n * sizeof(float), hipMemcpyHostToDevice, stm) != hipSuccess) return false;
    } else {
        fast_reduce_launch_pack(s.hyb_part(2, slot), data, (int) n, 0, stm);
    }
    if (fast_reduce_lean()) return true;
    return hipStreamWriteValue32(stm, s.flag_dev + s.done_off(slot), (uint32_t) seq, 0) == hipSuccess;
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

    if (s.peer_flags) { hipHostFree(s.peer_flags); s.peer_flags = nullptr; }
    if (s.scratch) { hipFree(s.scratch); s.scratch = nullptr; }
    if (s.data_host) { hipHostFree(s.data_host); s.data_host = nullptr; }
    if (s.flag_host) { hipHostFree(s.flag_host); s.flag_host = nullptr; }
    if (s.timed_out_host) { hipHostFree(s.timed_out_host); s.timed_out_host = nullptr; }
    if (s.progress_host) { hipHostFree(s.progress_host); s.progress_host = nullptr; }
    if (s.hyb_host) { hipHostFree(s.hyb_host); s.hyb_host = nullptr; }
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
bool FastReduce::hyb_export(const float *, size_t, void *, int, uint64_t) { return false; }
bool FastReduce::hyb_combine(const float *, size_t, void *, uint64_t) { return false; }
bool FastReduce::hyb_import(float *, size_t, void *, uint64_t) { return false; }
uint64_t FastReduce::submitted() const { return 0; }
uint64_t FastReduce::timed_out() const { return 0; }

}  // namespace luce::cluster

#endif  // LUCE_CLUSTER
