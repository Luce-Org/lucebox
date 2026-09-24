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

#pragma once

#include "moe_hybrid_storage.h"
#include "moe_hybrid_types.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
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

    // Adds the weighted output of the streamed routes to out
    // ([n_embd, n_tokens], host). `selected` / `weights` are [n_used,
    // n_tokens]; routes the storage does not stream are skipped.
    bool eval(int layer, const MoeLayerDesc & desc,
              const float * inp, const int32_t * selected, const float * weights,
              int n_used, int n_tokens, float * out, std::string * err = nullptr);

    struct Stats {
        uint64_t experts       = 0;  // streamed experts computed
        uint64_t hits          = 0;  // ... already cached or prefetched when needed
        uint64_t prefetch_hits = 0;  // ... of which a prefetch loaded
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
    int  evict_locked();
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
    uint64_t tick_ = 0;
    bool stopping_ = false;
    std::vector<std::vector<int32_t>> predicted_;  // per layer, until its eval
    std::vector<int> staged_;          // slots stage() pinned for staged_layer_
    int staged_layer_ = -1;
    std::string failed_;               // first load failure
    std::vector<Loader *> loaders_;
    std::vector<std::thread> threads_;
    Stats stats_;

    std::map<std::tuple<int, int, int>, Graph> graphs_;
};

}  // namespace luce::common
