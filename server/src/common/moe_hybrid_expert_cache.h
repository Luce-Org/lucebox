// Device cache for streamed MoE experts.
//
// Experts that neither owner stack holds are read from the model file on
// demand. This cache keeps a fixed pool of expert slots on one GPU: gate, up
// and down are three stacked [.., .., n_slots] tensors, so a layer's streamed
// routes run as one MUL_MAT_ID graph over slot ids, exactly like an owner
// stack. Slots are recycled least recently used first.
//
// Misses are loaded by a small pool of loader threads: each copies an expert
// out of the file mapping into its own pinned staging buffer and uploads it on
// its own stream, so reading one expert overlaps uploading another. Callers
// can stage a layer's misses early and prefetch predicted experts of a later
// layer; both only change when bytes move, never which experts are computed.
// So does the warm start, which fills the empty pool after load with the
// experts a usage profile ranks highest. A long prefill loads in bulk mode
// (set_bulk): direct reads that bypass the page cache, recycled within half
// the pool, so the decode working set and the page cache survive it.

#pragma once

#include "platform_io.h"
#include "moe_hybrid_routing_stats.h"
#include "moe_hybrid_storage.h"
#include "moe_hybrid_types.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <atomic>
#include <map>
#include <memory>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace luce::common {

struct MoeExpertCacheOptions {
    int    device      = 0;   // GPU that holds the slots and computes
    size_t pool_bytes  = 0;   // 0 = free memory on `device` minus reserve_bytes
    size_t reserve_bytes = (size_t) 2 << 30;
    int    n_loaders   = 4;
    // The file the storage maps from its first byte; bulk loads read it with
    // O_DIRECT. Empty: bulk loads read through the mapping.
    std::string direct_path;
    // Every load, not only bulk ones, reads with O_DIRECT and skips the page
    // cache: for hosts whose RAM cannot cache the streamed experts anyway.
    bool direct_all = false;
};

class MoeStreamedExpertCache;

// Byte stride between expert slots of one role (gate, up or down). Every layer
// reads the shared slots through a view whose expert stride is this value, and
// the matvec kernels index experts in whole quant blocks (stride / type size),
// so it must be a multiple of the block size of every type that shares the
// role: the largest expert rounded up to the least common multiple of those
// sizes. With one type this is the largest expert itself.
size_t moe_expert_slot_stride(size_t largest_expert_bytes, const std::vector<size_t> & type_sizes);

// Resolves a device graph's streamed routes without returning control to the
// host. Per layer the graph posts its route ids to host-mapped memory
// (ggml_host_mailbox_post) and later waits for the answer
// (ggml_host_mailbox_wait): the slot lookup rows of the streamed experts. A
// resolver thread answers the layers of each launch in order, pinning and
// loading through the cache, so the device waits only where a load still
// runs, and the rest of the graph keeps its devices busy meanwhile.
class MoeStreamedMailbox {
public:
    static constexpr int kMaxTokens = 8;

    // Host-mapped words of one layer, for building the graph.
    struct Channel {
        const uint32_t * step = nullptr;
        uint32_t * posted = nullptr;
        uint32_t * answered = nullptr;
        int32_t * ids = nullptr;    // [n_expert_used * n_tokens]
        int32_t * lut = nullptr;    // [n_expert * n_tokens] slot or invalid_route
        float * valid = nullptr;    // [n_expert * n_tokens] 1 for a resident slot
        // Optional: this layer's routes as predicted one layer early; the
        // resolver prefetches them when it answers the previous layer.
        uint32_t * predicted = nullptr;
        int32_t * predicted_ids = nullptr;  // [n_expert_used * n_tokens]
    };
    struct Job {
        int layer = -1;
        int n_routes = 0;
        int n_tokens = 0;
    };

    MoeStreamedMailbox() = default;
    ~MoeStreamedMailbox();
    MoeStreamedMailbox(const MoeStreamedMailbox &) = delete;
    MoeStreamedMailbox & operator=(const MoeStreamedMailbox &) = delete;

    bool init(MoeStreamedExpertCache * cache, int n_layers, int n_expert,
              int n_expert_used, std::string * err);
    void destroy();
    bool ready() const { return base_ != nullptr; }
    const Channel * channel(int layer) const;

    // Starts answering one launch: `jobs` in the graph's execution order.
    // Rows of experts that are not resident read `invalid_route`.
    void begin(const std::vector<Job> & jobs, int32_t invalid_route);
    // After the launch completed (or failed): waits for the resolver and
    // releases the launch's slots. False with *err when a layer could not be
    // answered exactly (a load failed or its experts exceed the pool).
    bool end(std::string * err);
    // A launch was begun and not ended yet.
    bool busy() const;

private:
    void resolver_main();
    bool answer(const Job & job, uint32_t step, std::string * err);

    MoeStreamedExpertCache * cache_ = nullptr;
    int n_expert_ = 0;
    int n_expert_used_ = 0;
    void * base_ = nullptr;
    uint32_t * step_ = nullptr;
    std::vector<Channel> channels_;
    std::vector<int32_t> slot_of_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::thread thread_;
    std::vector<Job> jobs_;
    int32_t invalid_route_ = -1;
    bool pending_ = false;     // a launch is being answered
    bool launch_done_ = false; // end() was called for it
    bool open_ = false;        // begin() was called, end() not yet
    bool stopping_ = false;
    std::string error_;
};

// One compute thread drives the cache at a time (stage, prefetch, eval,
// acquire / release_acquired): a backend steps its model on one thread, and
// paged serving puts every sequence into that one step. The loaders, the
// warm start and the mailbox resolver run in the background under mu_. A
// second concurrent eval or acquire fails instead of sharing the staging,
// pins and graphs of the first.
class MoeStreamedExpertCache {
public:
    MoeStreamedExpertCache() = default;
    ~MoeStreamedExpertCache();
    MoeStreamedExpertCache(const MoeStreamedExpertCache &) = delete;
    MoeStreamedExpertCache & operator=(const MoeStreamedExpertCache &) = delete;

    // `storage` provides the file mapping, the per-layer file regions and the
    // streamed sets; it must outlive the cache. `descs` give each layer's
    // weight types; slots are sized for the largest expert of any layer.
    bool init(const MoeHybridConfig & cfg,
              const std::vector<MoeLayerDesc> & descs,
              const MoeHybridStorage & storage,
              const MoeExpertCacheOptions & opts,
              std::string * err = nullptr);
    void destroy();
    bool ready() const { return backend_ != nullptr; }
    int  device() const { return device_; }
    int  n_slots() const { return n_slots_; }
    size_t slot_bytes() const { return slot_bytes_; }

    // Queue loads for the streamed experts routed in `selected`
    // ([n_used, n_tokens]) that are not cached yet, ahead of any prefetch.
    // Returns immediately; eval() waits for them.
    void stage(int layer, const int32_t * selected, int n_routes);

    // Queue loads for experts predicted for `layer`, behind staged loads.
    // Never evicts a slot a pending eval needs. Records the prediction so
    // eval() can report its accuracy.
    void prefetch(int layer, const int32_t * experts, int n);

    // Fills empty slots in the background with the streamed experts `usage`
    // counts most often, most used first. Loaders take these only while no
    // other load waits, and a warm load never evicts a slot, so a request
    // that starts meanwhile is only ever sped up. Returns the number queued.
    int warm(const MoeHybridRoutingStats & usage);

    // Bulk mode, for a long prefill that routes to most streamed experts.
    // Its loads read the file with O_DIRECT (they neither go through nor
    // evict the page cache the decode that follows relies on), and once bulk
    // slots hold half the pool they recycle among themselves, so the slots
    // decode used survive. Leaving bulk mode reloads, in the background and
    // under the warm-start rules, the decode slots it evicted anyway (only
    // into empty or bulk slots). Loads only move bytes: outputs never change.
    void set_bulk(bool bulk);

    // Pauses the background work, for a caller that is about to release
    // device state the cache's users depend on (DS4 partial eviction): the
    // loaders take no new job or warm load, and the loads already running
    // finish their copies. The slots keep their experts, queued loads stay
    // queued (their slots stay Loading) and the warm start keeps its place, so
    // resume() continues exactly where suspend() stopped. Fails, changing
    // nothing, while an eval/acquire runs, slots are pinned or a mailbox
    // launch has not ended: finish those first, since they wait on the
    // loaders. Idempotent; a cache that is not ready suspends trivially.
    bool suspend(std::string * err = nullptr);
    void resume();
    bool suspended() const;
    // Drops the cached compute graphs (rebuilt on demand). They are built
    // over the caller's layer descriptors, so a caller that reloads the
    // tensors behind those drops them first. Only while no call runs.
    void release_graphs();

    // Adds the weighted output of the streamed routes to out
    // ([n_embd, n_tokens], host). `selected` / `weights` are [n_used,
    // n_tokens]; routes the storage does not stream are skipped.
    bool eval(int layer, const MoeLayerDesc & desc,
              const float * inp, const int32_t * selected, const float * weights,
              int n_used, int n_tokens, float * out, std::string * err = nullptr);

    // For a graph that computes the streamed routes itself, with the slot
    // stacks as one more expert owner: makes the streamed experts routed in
    // `selected` resident, pinned until release_acquired(), waits for their
    // loads and writes each one's slot to slot_of_expert ([n_expert]; other
    // entries are left alone). Fails when they do not fit the pool at once.
    bool acquire(int layer, const int32_t * selected, int n_routes,
                 std::vector<int32_t> & slot_of_expert, std::string * err = nullptr);
    void release_acquired();

    // The slot stacks of a layer's weight types (gate/up or gate_up, down);
    // all null for a layer without streamed experts.
    struct Stacks {
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        ggml_tensor * down = nullptr;
        ggml_tensor * gate_up = nullptr;
    };
    Stacks stacks(int layer) const;

    // Answers graphs that resolve their streamed routes in the device graph;
    // null until init() succeeded.
    MoeStreamedMailbox * mailbox() { return mailbox_.ready() ? &mailbox_ : nullptr; }

    struct Stats {
        uint64_t experts       = 0;  // streamed experts computed
        uint64_t hits          = 0;  // ... already cached or prefetched when needed
        uint64_t prefetch_hits = 0;  // ... of which a prefetch loaded
        uint64_t warm_hits     = 0;  // ... of which the warm start loaded
        uint64_t loads         = 0;  // experts loaded into a slot
        uint64_t bytes         = 0;  // bytes loaded
        uint64_t prefetched    = 0;  // loads issued by prefetch()
        uint64_t predicted_used = 0; // used experts that the prediction named
        uint64_t predicted_of  = 0;  // used experts in layers with a prediction
        uint64_t missed        = 0;  // needed experts not cached or loading yet
        uint64_t late          = 0;  // needed experts whose prefetch was still loading
        uint64_t read_us       = 0;  // loader time copying out of the mapping
        uint64_t upload_us     = 0;  // loader time uploading
        uint64_t wait_us       = 0;  // eval time blocked on loads
        uint64_t compute_us    = 0;  // eval graph time incl. readback
    };
    Stats stats() const;
    void reset_stats();

private:
    enum class SlotState : uint8_t { Empty, Loading, Ready };
    struct Slot {
        int32_t   layer = -1;
        int32_t   expert = -1;
        SlotState state = SlotState::Empty;
        bool      demand = false;      // loaded because a layer needed it, unused
        bool      prefetched = false;  // loaded by prefetch, unused
        bool      warm = false;        // loaded by the warm start, unused
        bool      bulk = false;        // loaded in bulk mode, not used outside it
        int       pins = 0;
        uint64_t  last_use = 0;
    };
    struct Loader;
    struct Graph {
        ggml_context * ctx = nullptr;
        ggml_cgraph * gf = nullptr;
        ggml_gallocr_t alloc = nullptr;
        ggml_tensor * inp = nullptr;
        ggml_tensor * sel = nullptr;
        ggml_tensor * wts = nullptr;
        ggml_tensor * out = nullptr;
        void free();
    };
    // One weight-type signature: tensors over the pool viewed with its types.
    struct PoolView {
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        ggml_tensor * down = nullptr;
        ggml_tensor * gate_up = nullptr;
    };

    static uint64_t key(int layer, int expert) {
        return ((uint64_t) (uint32_t) layer << 32) | (uint32_t) expert;
    }
    // Caller holds mu_. Returns the slot of (layer, expert), starting a load
    // when it is not cached, or -1 when every slot is pinned or loading.
    int  lookup_or_load_locked(int layer, int expert, bool front, bool * hit);
    // Caller holds lk on mu_. Pins a slot for each of `experts` (loading
    // misses ahead of prefetches) and waits until they are ready; false on a
    // load failure (slots stay pinned).
    bool pin_ready_locked(std::unique_lock<std::mutex> & lk, int layer,
                          const int32_t * experts, size_t n, std::vector<int> & slots);
    // Drops the pins stage() took for `layer`.
    void release_staged(int layer);
    // `refill`: only an empty slot or a bulk slot nobody uses (warm loads).
    int  evict_locked(bool refill = false);
    // Caller holds mu_. Marks a slot used; outside bulk mode it joins the
    // decode working set.
    void touch_locked(int slot);
    // Caller holds mu_. Drops one pin; wakes a refill waiting for a slot.
    void unpin_locked(int slot);
    // Caller holds mu_. Claims an empty slot for the next warm expert and
    // marks it loading; -1 when the warm list is done or no slot is empty.
    int  next_warm_locked();
    void loader_main(Loader * self);
    bool load_slot(Loader & loader, int slot, bool bulk, uint64_t * read_us, uint64_t * upload_us);
    Graph * graph_for(int view, const MoeLayerDesc & desc, int n_routes, int n_tokens,
                      std::string * err);
    bool build_graph(Graph & g, int view, const MoeLayerDesc & desc, int n_routes,
                     int n_tokens, std::string * err);

    MoeHybridConfig cfg_;
    const MoeHybridStorage * storage_ = nullptr;
    int device_ = -1;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t pool_buf_ = nullptr;
    ggml_context * pool_ctx_ = nullptr;
    int n_slots_ = 0;
    size_t slot_bytes_ = 0;
    // Byte stride of one slot in each role's stack (gate, up, down).
    size_t stride_gate_ = 0, stride_up_ = 0, stride_down_ = 0;
    uint8_t * base_gate_ = nullptr;
    uint8_t * base_up_ = nullptr;
    uint8_t * base_down_ = nullptr;
    std::vector<PoolView> views_;
    std::vector<int> view_of_layer_;

    mutable std::mutex mu_;
    std::condition_variable cv_;       // slot became ready or a job arrived
    std::vector<Slot> slots_;
    std::unordered_map<uint64_t, int> slot_of_;
    std::deque<int> jobs_;
    std::vector<uint64_t> warm_;       // warm start keys, most used first
    size_t warm_next_ = 0;
    bool warm_blocked_ = false;        // a refill waits for a slot to be unpinned
    size_t refill_from_ = SIZE_MAX;    // warm_ entries from here reload evicted decode slots
    std::atomic<bool> in_call_{false}; // an eval or acquire is running
    bool bulk_ = false;
    bool direct_all_ = false;          // every load reads with O_DIRECT
    ReadOnlyFile direct_file_;         // the model file opened for direct reads, if supported
    std::vector<uint64_t> evicted_hot_; // decode slots bulk loads evicted, oldest first
    int warm_loading_ = 0;
    uint64_t warm_loads_ = 0, warm_bytes_ = 0;
    std::chrono::steady_clock::time_point warm_t0_;
    uint64_t tick_ = 0;
    bool stopping_ = false;
    bool suspended_ = false;           // loaders take no work (suspend())
    int loads_in_flight_ = 0;          // loads a loader has taken and not finished
    std::vector<std::vector<int32_t>> predicted_;  // per layer, until its eval
    std::vector<int> staged_;          // slots stage() pinned for staged_layer_
    std::vector<int> acquired_;        // slots acquire() pinned
    int staged_layer_ = -1;
    std::string failed_;               // first load failure
    std::vector<Loader *> loaders_;
    std::vector<std::thread> threads_;
    Stats stats_;

    std::map<std::tuple<int, int, int>, Graph> graphs_;
    MoeStreamedMailbox mailbox_;
};

}  // namespace luce::common
