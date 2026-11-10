// Split mode: the experts the target computes beside the expert device (hot lanes, qwen4exp_graph.cpp
// build_moe_route). Slots on the target hold copies of experts and a per-layer table maps an expert to its slot.
// The target counts every decode and verify pick (ggml_moe_hot_routes), and every kTickGraphs graphs tick() moves
// the most-picked experts into the slots of the least-picked ones, copying them in the background. A pick computes
// the same in a lane as on the expert device and the expert device sums every pick in route order, so the text does
// not depend on what the slots hold: the set follows the requests being served.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace luce::common {

struct MoeHybridRoutingStats;
struct Qwen4ExpWeights;

class Qwen4ExpHotSet {
public:
    ~Qwen4ExpHotSet();

    // Slots for `budget` bytes of experts on `target`, the same number in every split layer, and their tables and
    // pick counters. stats (may be null): the slots start with each layer's most-picked experts there; without it
    // they start empty and fill as requests run. False: no room for one slot per layer (or allocation failed).
    bool init(ggml_backend_t target, const Qwen4ExpWeights & w, uint64_t budget, const MoeHybridRoutingStats * stats);

    // On the thread that runs the graphs, before a split graph: tables of finished copies go live, and every
    // kTickGraphs decode or verify graphs the counts are read and the next swaps start. prompt_rows > 0: before a
    // prompt chunk of that many rows; the previous chunk's picks count at once (scaled to about kPromptPicks a layer)
    // and more copies may start, so the slots hold the prompt's experts when its reply begins.
    void tick(ggml_backend_t target, const Qwen4ExpWeights & w, int prompt_rows = 0);

    // Per layer (null: the layer's experts are not split): the slot stacks, [ne0, ne1, slots] like the expert
    // device's, the expert -> slot table [n_expert] (I32, -1: none) and the pick counter [n_expert] (I32).
    std::vector<ggml_tensor *> gate, up, down, lut, hist;
    int lanes = 0;     // a token's hot picks computed in lanes (the rest of its picks run on the expert device)
    int slots = 0;     // per split layer
    // The experts the slots hold at start, per layer (for the expert device's read-ahead of the others).
    std::vector<std::vector<int32_t>> initial;

private:
    struct Job {
        int layer, slot, expert;
    };
    struct Batch {
        ggml_backend_event_t ready = nullptr;   // after the tables stopped naming the slots
        std::vector<Job> jobs;
    };
    void copier();
    void copy_expert(const Qwen4ExpWeights & w, const Job & j, uint8_t * staging);

    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;
    ggml_tensor * lut_all_ = nullptr;    // [n_expert, n_layer]
    ggml_tensor * hist_all_ = nullptr;   // [n_expert, n_layer]
    int n_layer_ = 0, n_expert_ = 0;
    std::vector<int32_t> host_lut_;      // [n_layer][n_expert], what the device table says (or will at the next graph)
    std::vector<int32_t> slot_expert_;   // [n_layer][slots], -1: empty or being filled
    std::vector<uint8_t> filling_;       // [n_layer][slots], a copy into the slot is running
    std::vector<uint8_t> incoming_;      // [n_layer][n_expert], the expert is being copied into a slot
    std::vector<uint32_t> seen_;         // [n_layer][n_expert], the counters at the last read
    std::vector<uint32_t> read_;
    std::vector<float> score_;           // [n_layer][n_expert], decayed pick counts
    long long graphs_ = 0;
    int rows_ = 0;                       // the rows of the prompt chunk that ran last (0: a decode or verify graph)

    const Qwen4ExpWeights * w_ = nullptr;
    ggml_backend_buffer_t staging_buf_ = nullptr;
    size_t staging_bytes_ = 0;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Batch> todo_;
    std::vector<Job> done_;
    bool stop_ = false;
};

}  // namespace luce::common
