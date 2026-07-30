// CPU-only ownership state for KVFlash's bounded logical-to-physical mapping.
//
// This class deliberately knows nothing about ggml, GPUs, DMA, or host
// backing buffers.  A caller supplies a prepare_page_out callback before a
// resident mapping is released; the state transition is committed only when
// that callback succeeds.  KvFlashPager owns the corresponding data movement.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace dflash::common {

struct KvFlashConfig {
    int chunk_tokens       = 64; // logical tokens per page
    int pool_tokens        = 0;  // physical capacity; whole pages only
    int sink_chunks        = 1;  // leading pages protected from victim choice
    int tail_window_chunks = 4;  // recent pages protected from victim choice
    // Bounds sparse logical-state growth independently of the resident pool.
    // Backends may lower this to their configured context. The conservative
    // default is far above supported production contexts while still
    // rejecting pathological positions before a multi-gigabyte resize.
    int max_logical_tokens = 16 * 1024 * 1024;
};

class KvFlashResidencyMap {
public:
    // Non-owning callback views keep the formal core free of std::function's
    // allocation and type-erasure machinery. Context lifetime is owned by
    // the caller and need only cover the synchronous method invocation.
    struct Score {
        using Function = float (*)(const void *, int /* logical chunk */);
        const void * context;
        Function evaluate;

        constexpr Score(const void * ctx = nullptr, Function fn = nullptr)
            : context(ctx), evaluate(fn) {}

        explicit operator bool() const { return evaluate != nullptr; }
        float operator()(int chunk) const { return evaluate(context, chunk); }
    };

    struct PreparePageOut {
        using Function =
            bool (*)(void *, int /* logical chunk */, int /* physical block */);
        void * context;
        Function run;

        constexpr PreparePageOut(void * ctx = nullptr, Function fn = nullptr)
            : context(ctx), run(fn) {}

        explicit operator bool() const { return run != nullptr; }
        bool operator()(int chunk, int block) const {
            return run(context, chunk, block);
        }
    };

    struct AcquireResult {
        int  slot          = -1;
        int  chunk         = -1;
        int  block         = -1;
        int  evicted_chunk = -1;
        bool recalled      = false;
        bool changed       = false;

        explicit operator bool() const { return slot >= 0; }
    };

    static bool valid_config(const KvFlashConfig & cfg) {
        if (cfg.chunk_tokens <= 0 || cfg.pool_tokens <= 0 ||
            cfg.sink_chunks < 0 || cfg.tail_window_chunks < 0 ||
            cfg.max_logical_tokens < cfg.pool_tokens ||
            cfg.pool_tokens % cfg.chunk_tokens != 0) {
            return false;
        }
        const int64_t min_chunks =
            (int64_t)cfg.sink_chunks + cfg.tail_window_chunks + 2;
        const int64_t min_tokens = min_chunks * cfg.chunk_tokens;
        return min_tokens <= std::numeric_limits<int>::max() &&
               cfg.pool_tokens >= min_tokens;
    }

    static int min_pool_tokens(const KvFlashConfig & cfg) {
        if (cfg.chunk_tokens <= 0 || cfg.sink_chunks < 0 ||
            cfg.tail_window_chunks < 0) {
            return -1;
        }
        const int64_t tokens =
            ((int64_t)cfg.sink_chunks + cfg.tail_window_chunks + 2) *
            cfg.chunk_tokens;
        return tokens <= std::numeric_limits<int>::max() ? (int)tokens : -1;
    }

    bool configure(const KvFlashConfig & cfg) {
        if (!valid_config(cfg)) return false;
        cfg_ = cfg;
        n_blocks_ = cfg.pool_tokens / cfg.chunk_tokens;
        configured_ = true;
        initialize_empty_(false);
        return true;
    }

    bool attached() const { return configured_; }
    int pool_tokens() const { return configured_ ? cfg_.pool_tokens : 0; }
    int chunk_tokens() const { return configured_ ? cfg_.chunk_tokens : 0; }
    int n_chunks() const { return (int)chunks_.size(); }
    int resident_blocks() const {
        return configured_ ? n_blocks_ - (int)free_blocks_.size() : 0;
    }
    uint64_t epoch() const { return epoch_; }

    // Drop request-local mappings while retaining the validated config.
    void reset() {
        if (!configured_) return;
        initialize_empty_(true);
    }

    // Optional deterministic placement order. A malformed or partial order is
    // rejected without changing the current free-block stack.
    bool set_block_order(const std::vector<int> & order) {
        if (!configured_ || resident_blocks() != 0 ||
            (int)order.size() != n_blocks_) {
            return false;
        }
        std::vector<uint8_t> seen((int)n_blocks_);
        for (int block : order) {
            if (block < 0 || block >= n_blocks_ || seen[(size_t)block]) {
                return false;
            }
            seen[(size_t)block] = 1;
        }
        free_blocks_.clear();
        for (int index = n_blocks_ - 1; index >= 0; --index) {
            free_blocks_.push_back(order[(size_t)index]);
        }
        return true;
    }

    // Acquire the slot for a non-negative logical position. If eviction is
    // necessary, prepare_page_out runs before ownership changes. Failure
    // leaves the append head, mappings, free complement, clock, and epoch
    // unchanged.
    AcquireResult acquire(
            int64_t pos,
            Score score = Score{},
            PreparePageOut prepare_page_out = PreparePageOut{}) {
        AcquireResult result;
        if (!configured_ || pos < 0 || pos >= cfg_.max_logical_tokens) {
            return result;
        }

        const int64_t chunk64 = pos / cfg_.chunk_tokens;
        if (chunk64 > std::numeric_limits<int>::max()) return result;
        const int chunk = (int)chunk64;
        const int offset = (int)(pos % cfg_.chunk_tokens);
        result.chunk = chunk;

        if (chunk < (int)chunks_.size() && chunks_[(size_t)chunk].block >= 0) {
            ChunkState & state = chunks_[(size_t)chunk];
            state.last_use = ++clock_;
            result.block = state.block;
            result.slot = state.block * cfg_.chunk_tokens + offset;
            return result;
        }

        const int prospective_head = std::max(cur_chunk_, chunk);
        int block = -1;
        int victim = -1;
        if (!free_blocks_.empty()) {
            block = free_blocks_.back();
        } else {
            victim = choose_victim_(prospective_head, score);
            if (victim < 0 || !prepare_page_out) return result;
            block = chunks_[(size_t)victim].block;
        }

        const size_t old_size = chunks_.size();
        if ((size_t)chunk >= chunks_.size()) {
            chunks_.resize((size_t)chunk + 1);
        }

        // All potentially-throwing state growth is complete. A rejected
        // external page-out preparation restores the exact prior map shape.
        // Prepare callbacks must report failure before producing externally
        // visible DMA/zero/stat side effects; KvFlashPager follows that rule.
        if (victim >= 0 && !prepare_page_out(victim, block)) {
            chunks_.resize(old_size);
            return result;
        }

        if (victim >= 0) {
            chunks_[(size_t)victim].block = -1;
            chunks_[(size_t)victim].on_host = true;
            result.evicted_chunk = victim;
            ++epoch_;
        } else {
            free_blocks_.pop_back();
        }

        ChunkState & state = chunks_[(size_t)chunk];
        result.recalled = state.on_host;
        state.block = block;
        state.last_use = ++clock_;
        cur_chunk_ = prospective_head;
        ++epoch_;

        result.block = block;
        result.slot = block * cfg_.chunk_tokens + offset;
        result.changed = true;
        return result;
    }

    // Explicitly release a resident chunk. This is intentionally permitted
    // for protected chunks: policy protection constrains victim selection,
    // while administrative reset/reselect code may explicitly page any chunk.
    bool page_out(int chunk, PreparePageOut prepare_page_out) {
        if (!configured_ || chunk < 0 || chunk >= (int)chunks_.size()) return false;
        ChunkState & state = chunks_[(size_t)chunk];
        if (state.block < 0 || !prepare_page_out ||
            !prepare_page_out(chunk, state.block)) {
            return false;
        }
        free_blocks_.push_back(state.block);
        state.block = -1;
        state.on_host = true;
        ++epoch_;
        return true;
    }

    bool is_resident(int chunk) const {
        return chunk >= 0 && chunk < (int)chunks_.size() &&
               chunks_[(size_t)chunk].block >= 0;
    }

    bool is_host_backed(int chunk) const {
        return chunk >= 0 && chunk < (int)chunks_.size() &&
               chunks_[(size_t)chunk].on_host;
    }

    int block_of(int chunk) const {
        return chunk >= 0 && chunk < (int)chunks_.size()
            ? chunks_[(size_t)chunk].block : -1;
    }

    int slot_of(int64_t pos) const {
        if (!configured_ || pos < 0 || pos >= cfg_.max_logical_tokens) return -1;
        const int64_t chunk64 = pos / cfg_.chunk_tokens;
        if (chunk64 > std::numeric_limits<int>::max()) return -1;
        const int chunk = (int)chunk64;
        if (!is_resident(chunk)) return -1;
        return chunks_[(size_t)chunk].block * cfg_.chunk_tokens +
               (int)(pos % cfg_.chunk_tokens);
    }

    bool is_identity() const {
        for (int chunk = 0; chunk < (int)chunks_.size(); ++chunk) {
            const ChunkState & state = chunks_[(size_t)chunk];
            if (state.block >= 0 && state.block != chunk) return false;
            if (state.block < 0 && state.on_host) return false;
        }
        return true;
    }

    bool identity_prefix_covers(int n_tok) const {
        if (!configured_ || n_tok < 0) return false;
        if (n_tok == 0) return true;
        if (n_tok > cfg_.max_logical_tokens) return false;
        const int64_t n_chunks =
            ((int64_t)n_tok + cfg_.chunk_tokens - 1) / cfg_.chunk_tokens;
        if (n_chunks > (int64_t)chunks_.size()) return false;
        for (int chunk = 0; chunk < n_chunks; ++chunk) {
            if (chunks_[(size_t)chunk].block != chunk) return false;
        }
        return true;
    }

    void fill_slot_pos(int32_t * dst) const {
        if (!configured_ || !dst) return;
        std::fill(dst, dst + cfg_.pool_tokens, (int32_t)-1);
        for (int chunk = 0; chunk < (int)chunks_.size(); ++chunk) {
            const int block = chunks_[(size_t)chunk].block;
            if (block < 0) continue;
            int32_t * out = dst + (size_t)block * cfg_.chunk_tokens;
            for (int i = 0; i < cfg_.chunk_tokens; ++i) {
                const int64_t pos = (int64_t)chunk * cfg_.chunk_tokens + i;
                out[i] = pos <= std::numeric_limits<int32_t>::max()
                    ? (int32_t)pos : -1;
            }
        }
    }

    void fill_slot_mask(uint16_t * dst) const {
        if (!configured_ || !dst) return;
        constexpr uint16_t F16_ZERO = 0x0000;
        constexpr uint16_t F16_NEG_INF = 0xFC00;
        std::fill(dst, dst + cfg_.pool_tokens, F16_NEG_INF);
        for (const ChunkState & state : chunks_) {
            if (state.block < 0) continue;
            uint16_t * out = dst + (size_t)state.block * cfg_.chunk_tokens;
            std::fill(out, out + cfg_.chunk_tokens, F16_ZERO);
        }
    }

    const std::vector<int> & free_blocks() const { return free_blocks_; }

    // Desired resident set for score-driven reselect. Sinks and the current
    // tail window sort ahead of scored candidates and therefore stay pinned.
    std::vector<uint8_t> desired_residency(Score score) const {
        std::vector<uint8_t> wanted((int)chunks_.size());
        if (!configured_ || !score) return wanted;
        struct Candidate {
            int chunk;
            float score;
            bool protected_chunk;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(chunks_.size());
        for (int chunk = 0; chunk < (int)chunks_.size(); ++chunk) {
            const ChunkState & state = chunks_[(size_t)chunk];
            if (state.block < 0 && !state.on_host) continue;
            const bool keep = protected_(chunk, cur_chunk_);
            const float value = keep ? 0.0f : normalized_score_(score(chunk));
            candidates.push_back({chunk, value, keep});
        }
        std::stable_sort(
            candidates.begin(), candidates.end(),
            [](const Candidate & lhs, const Candidate & rhs) {
                if (lhs.protected_chunk != rhs.protected_chunk) {
                    return lhs.protected_chunk;
                }
                return lhs.score > rhs.score;
            });
        for (int i = 0; i < (int)candidates.size() && i < n_blocks_; ++i) {
            wanted[(size_t)candidates[(size_t)i].chunk] = 1;
        }
        return wanted;
    }

    // Runtime-checkable representation invariant, also useful to formal and
    // dependency-free native harnesses.
    bool invariant_holds() const {
        if (!configured_) {
            return n_blocks_ == 0 && chunks_.empty() && free_blocks_.empty();
        }
        if (!valid_config(cfg_) || n_blocks_ != cfg_.pool_tokens / cfg_.chunk_tokens) {
            return false;
        }
        std::vector<int> owner((int)n_blocks_);
        for (int block = 0; block < n_blocks_; ++block) {
            owner[(size_t)block] = -1;
        }
        for (int chunk = 0; chunk < (int)chunks_.size(); ++chunk) {
            const int block = chunks_[(size_t)chunk].block;
            if (block < 0) continue;
            if (block >= n_blocks_ || owner[(size_t)block] >= 0) return false;
            owner[(size_t)block] = chunk;
        }
        std::vector<uint8_t> free((int)n_blocks_);
        for (int block : free_blocks_) {
            if (block < 0 || block >= n_blocks_ || free[(size_t)block]) return false;
            if (owner[(size_t)block] >= 0) return false;
            free[(size_t)block] = 1;
        }
        for (int block = 0; block < n_blocks_; ++block) {
            if ((owner[(size_t)block] >= 0) == (free[(size_t)block] != 0)) {
                return false;
            }
        }
        return resident_blocks() >= 0 && resident_blocks() <= n_blocks_;
    }

private:
    struct ChunkState {
        int block = -1;
        bool on_host = false;
        uint64_t last_use = 0;
    };

    static float normalized_score_(float score) {
        return std::isnan(score) ? -std::numeric_limits<float>::infinity() : score;
    }

    bool protected_(int chunk, int append_head) const {
        return chunk < cfg_.sink_chunks ||
               chunk > append_head - 1 - cfg_.tail_window_chunks;
    }

    int choose_victim_(int append_head, Score score) const {
        int victim = -1;
        float victim_score = 0.0f;
        uint64_t victim_use = 0;
        for (int chunk = 0; chunk < (int)chunks_.size(); ++chunk) {
            const ChunkState & state = chunks_[(size_t)chunk];
            if (state.block < 0 || protected_(chunk, append_head)) continue;
            if (score) {
                const float value = normalized_score_(score(chunk));
                if (victim < 0 || value < victim_score) {
                    victim = chunk;
                    victim_score = value;
                }
            } else if (victim < 0 || state.last_use < victim_use) {
                victim = chunk;
                victim_use = state.last_use;
            }
        }
        return victim;
    }

    void initialize_empty_(bool advance_epoch) {
        chunks_.clear();
        free_blocks_.clear();
        free_blocks_.reserve((size_t)n_blocks_);
        for (int block = n_blocks_ - 1; block >= 0; --block) {
            free_blocks_.push_back(block);
        }
        cur_chunk_ = 0;
        clock_ = 0;
        if (advance_epoch) ++epoch_;
    }

    KvFlashConfig cfg_;
    std::vector<ChunkState> chunks_;
    std::vector<int> free_blocks_;
    int n_blocks_ = 0;
    int cur_chunk_ = 0;
    uint64_t clock_ = 0;
    uint64_t epoch_ = 0;
    bool configured_ = false;
};

} // namespace dflash::common
