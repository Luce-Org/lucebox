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
// experts a usage profile ranks highest.

#pragma once

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
#include <functional>
#include <atomic>
#include <map>
#include <memory>
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
};

class MoeStreamedExpertCache;

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
    // Called by the resolver thread with each layer's posted routes
    // ([n_expert_used * n_tokens] global ids), e.g. to learn expert heat.
    // Set while no launch is being answered.
    void set_route_observer(std::function<void(int layer, const int32_t * ids, int n)> fn) {
        observer_ = std::move(fn);
    }
    // After the launch completed (or failed): waits for the resolver and
    // releases the launch's slots. False with *err when a layer could not be
    // answered exactly (a load failed or its experts exceed the pool).
    bool end(std::string * err);

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

    std::mutex mu_;
    std::condition_variable cv_;
    std::thread thread_;
    std::vector<Job> jobs_;
    int32_t invalid_route_ = -1;
    bool pending_ = false;     // a launch is being answered
    bool launch_done_ = false; // end() was called for it
    bool stopping_ = false;
    std::string error_;
    std::function<void(int, const int32_t *, int)> observer_;
};

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
    int  evict_locked();
    // Caller holds mu_. Claims an empty slot for the next warm expert and
    // marks it loading; -1 when the warm list is done or no slot is empty.
    int  next_warm_locked();
    void loader_main(Loader * self);
    bool load_slot(Loader & loader, int slot, uint64_t * read_us, uint64_t * upload_us);
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
    int warm_loading_ = 0;
    uint64_t warm_loads_ = 0, warm_bytes_ = 0;
    std::chrono::steady_clock::time_point warm_t0_;
    uint64_t tick_ = 0;
    bool stopping_ = false;
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

// Moves experts that became hot onto the primary owner's spare rows (the
// owner stack is allocated with `cache_slots` spare entries per layer, see
// MoeHybridLayerStorage) and demotes ones that cooled down, in the
// background. Heat is a decayed count of each expert's routes. A copy runs on
// its own low-priority stream from the model file, rate limited; placement
// changes (the owner maps) are applied only between graph launches
// (apply_pending), so a launch always sees one consistent placement: the
// routes a token takes never change, only which owner computes an expert.
// Pinned experts (the calibration-placed primary set) never move.
struct MoeExpertPromoterOptions {
    int    device = 0;                 // GPU that holds the primary stacks
    double max_bytes_per_s = 1.0e9;    // background copy budget
    double decay = 0.97;               // heat kept per rebalance tick
    int    tick_ms = 50;
    double min_heat = 4.0;             // an expert must be at least this hot
    double hysteresis = 1.5;           // challenger heat / resident heat to swap
};

class MoeExpertPromoter {
public:
    MoeExpertPromoter() = default;
    ~MoeExpertPromoter();
    MoeExpertPromoter(const MoeExpertPromoter &) = delete;
    MoeExpertPromoter & operator=(const MoeExpertPromoter &) = delete;

    // `storage` must have spare rows (cache_slots) on the primary stacks and
    // the model file mapping; it must outlive the promoter.
    bool init(MoeHybridStorage & storage, int n_expert, const MoeExpertPromoterOptions & opts,
              std::string * err = nullptr);
    void destroy();
    bool ready() const { return storage_ != nullptr; }

    // Any thread: routes one layer took (global expert ids, negatives skipped).
    void observe(int layer, const int32_t * ids, int n);
    // Launching thread, between graph launches: applies finished promotions
    // and requested demotions to the owner maps. Returns the placement
    // generation, which changes whenever the maps did.
    uint64_t apply_pending();
    uint64_t generation() const { return generation_; }

    struct Stats {
        uint64_t promoted = 0;
        uint64_t demoted = 0;
        uint64_t bytes = 0;
        int      resident = 0;   // spare rows holding an expert
    };
    Stats stats() const;

private:
    enum class RowState : uint8_t { Free, Loading, Ready, Active, Leaving };
    struct Row {
        int32_t expert = -1;
        RowState state = RowState::Free;
    };
    struct Change {
        int layer = -1;
        int row = -1;
        bool promote = false;
    };
    void worker();
    bool copy_expert(int layer, int row, int expert);

    MoeHybridStorage * storage_ = nullptr;
    MoeExpertPromoterOptions opts_;
    int n_layer_ = 0;
    int n_expert_ = 0;
    std::unique_ptr<std::atomic<uint32_t>[]> hits_;   // [layer * n_expert]
    std::vector<float> heat_;                         // worker only
    std::vector<uint8_t> pinned_;                     // [layer * n_expert] calibration-placed
    void * staging_ = nullptr;                        // pinned, one expert
    size_t staging_bytes_ = 0;
    void * stream_ = nullptr;                         // cudaStream_t

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::vector<Row>> rows_;              // [layer][spare row]
    std::vector<Change> pending_;
    uint64_t generation_ = 0;
    Stats stats_;
    bool stopping_ = false;
    std::thread thread_;
};

}  // namespace luce::common
