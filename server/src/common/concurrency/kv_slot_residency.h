#pragma once

#include "common/concurrency/paged_kv_offload.h"
#include "common/concurrency/paged_kv_pool.h"
#include "common/kvflash_prefill_reselect.h"
#include "common/pinned_transfer.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luce::common {

// Per-slot KVFlash 2.0 bounded residency manager composed cleanly over
// PagedKvPool + PinnedTransfer. Splits each sequence into 64-token logical
// chunks (4 x B=16 paged blocks) with a hard per-slot GPU resident budget,
// pinned preamble + sink + tail protection, host-RAM paged archive with
// clean-chunk D2H skip & #1079 pinned-shared zero-copy preservation, and a
// dense O(budget) active block-table builder requiring zero kernel changes.
class KvSlotResidency {
public:
    struct Config {
        int    per_slot_tokens         = 16384;
        int    chunk_tokens            = 64;
        int    preamble_reserve_tokens = 1152;
        int    recall_chunks           = 64;
        int    tail_window_chunks      = 4;
        size_t max_host_bytes          = 0;  // 0 = unlimited
    };

    struct Stats {
        uint64_t evictions     = 0;
        uint64_t d2h_skipped   = 0;
        uint64_t recalls       = 0;
        uint64_t h2d_skipped   = 0;
        uint64_t shared_reused = 0;
    };

    // CoW chunk archive for KVFlash-backed prefix cache checkpoints > per_slot_tokens.
    // Ancestor chunks are shared zero-copy via shared_ptr across multi-turn snapshots.
    struct Snapshot {
        int tokens = 0;
        uint64_t generation = 0;
        int pinned_chunks = 0;
        std::vector<std::shared_ptr<std::vector<uint8_t>>> chunks;
        std::vector<uint8_t> resident;
        std::vector<float> scores;
        std::vector<uint8_t> has_score;
        bool valid() const { return tokens > 0 && !chunks.empty(); }
        size_t total_bytes() const {
            size_t n = 0;
            for (const auto & p : chunks) if (p) n += p->size();
            return n;
        }
        size_t delta_bytes = 0;  // newly D2H'd bytes not shared with prior checkpoints
    };

    KvSlotResidency(PagedKvPool & pool, ggml_backend_t backend,
                    std::vector<PagedKvTensor> tensors, Config cfg);

    const Config & config() const { return cfg_; }
    int chunk_tokens() const { return cfg_.chunk_tokens; }
    int blocks_per_chunk() const { return blocks_per_chunk_; }
    int budget_chunks() const { return budget_chunks_; }
    size_t chunk_host_bytes() const { return chunk_bytes_; }
    size_t host_bytes_used() const { return host_bytes_used_; }

    void init_slot(int slot, PagedKvSequenceHandle handle, int preamble_tokens,
                   const uint32_t * shared_blocks = nullptr,
                   uint32_t shared_block_count = 0, int shared_tokens = 0);

    void on_tokens_written(int slot, int old_pos, int new_pos);

    bool ensure_capacity(int slot, int cur_pos, int add_tokens,
                         std::string & error);

    bool apply_recall(int slot, int r_pos, int prompt_end,
                      const std::vector<float> & chunk_scores,
                      int recall_unit, std::vector<int> * picked_out,
                      std::string & error);

    bool build_active_block_table(int slot, int cur_pos,
                                  std::vector<int32_t> & out_blocks,
                                  int & out_res_tokens,
                                  int & out_prefix_res_tokens,
                                  int chunk_start_pos = -1) const;

    bool capture_snapshot(int slot, int tokens, Snapshot & out, std::string & error);
    void build_restore_resident_mask(const Snapshot & snap, int preamble_tokens,
                                     std::vector<uint8_t> & out_block_mask,
                                     std::vector<uint8_t> & out_chunk_mask) const;
    bool restore_snapshot(int slot, PagedKvSequenceHandle handle,
                          const Snapshot & snap, int preamble_tokens,
                          const std::vector<uint8_t> & chunk_mask,
                          const std::vector<uint32_t> & block_table,
                          std::string & error);

    bool truncate(int slot, int new_pos);
    void clear(int slot);

    uint64_t epoch(int slot) const;
    int resident_chunks(int slot) const;
    int pinned_chunks(int slot) const;
    int effective_pin_reserve_chunks(int slot) const;
    int total_chunks(int slot) const;
    bool is_chunk_resident(int slot, int chunk) const;
    const uint8_t * host_chunk_data(int slot, int chunk) const;
    const Stats & stats(int slot) const;

private:
    struct Chunk {
        bool resident = false;
        bool dirty    = false;
        bool pinned   = false;
        uint64_t last_use = 0;
        float score = -1e30f;
        bool has_score = false;
        std::vector<uint32_t> shared_pins;
        std::shared_ptr<std::vector<uint8_t>> host;
    };

    struct SlotState {
        bool active = false;
        PagedKvSequenceHandle handle;
        int pinned_chunks = 0;
        int resident_chunks = 0;
        uint64_t epoch = 0;
        uint64_t clock = 0;
        std::vector<Chunk> chunks;
        Stats stats;
    };

    bool evict_chunks(SlotState & st, const std::vector<int> & chunk_indices,
                      const std::vector<uint32_t> & block_table,
                      std::string & error);
    bool restore_chunks(SlotState & st, const std::vector<int> & chunk_indices,
                        std::string & error);
    int pick_victim(const SlotState & st, int sealed_chunks,
                    int cur_chunk, const std::vector<uint8_t> * keep_mask) const;
    bool all_pins_still_live(const std::vector<uint32_t> & pins) const;

    PagedKvPool & pool_;
    ggml_backend_t backend_ = nullptr;
    std::vector<PagedKvTensor> tensors_;
    Config cfg_;
    int blocks_per_chunk_ = 4;
    int budget_chunks_ = 0;
    int pin_reserve_chunks_ = 0;
    size_t chunk_bytes_ = 0;
    size_t host_bytes_used_ = 0;
    std::vector<SlotState> slots_;
    PinnedTransfer transfer_;
};

}  // namespace luce::common
