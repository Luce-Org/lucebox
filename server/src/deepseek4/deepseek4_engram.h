// DeepSeek V4.1 Engram: the n-gram hash memory applied to the residual copies
// at the two Engram layers (released checkpoint: 1 and 14).
//
// Addressing and the table read are tested bit-exact against the reference
// (test/test_ds4_engram.cpp), the apply against the reference math on the
// released weights (tests/test_deepseek4_unit.cpp); see docs/DS41.md.
//
// Three pieces, kept independent of the graph so each can be tested alone:
//   1. addressing: the last four compressed token ids hash to `cols` rows per
//      layer (released: 3 n-gram orders x 8 heads = 24), integer math only, so
//      every row a position needs is known before its forward pass;
//   2. the table read: rows live inside the GGUF as `blk.N.engram_embd`
//      (I8 [264, rows]: 256 E4M3 values + 8 E8M0 block scales) and are pread
//      on demand, never mapped; a row decodes to 256 floats, rounded to bf16
//      as the reference does;
//   3. the apply: wkv projects the concatenated rows to four keys (one per
//      hyper-connection copy) and one shared value; each copy is gated by a
//      normalized dot product against its key and receives gate * value.
//
// Reference: DeepSeek-V4.1-Flash inference/model.py (class Engram, 328-368),
// engram.py (NgramHashState); antirez/ds4 ds4_engram.c as a second reading.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
struct ggml_gallocr;

namespace luce::common {

struct DeepSeek4Weights;
struct DeepSeek4Layer;

// ── 1. addressing ─────────────────────────────────────────────────────────

// The n-gram tail of one sequence, most recent first, compressed ids.
// DEAD marks a slot that carries no token (before the start of the sequence,
// or a non-text position). Copying the struct snapshots the state, which is
// how speculative rollback restores it.
struct DeepSeek4EngramHistory {
    static constexpr int kMaxTail = 7;     // max_ngram - 1, max_ngram <= 8
    static constexpr int32_t kDead = -1;
    int32_t tail[kMaxTail] = { kDead, kDead, kDead, kDead, kDead, kDead, kDead };
    void reset() { for (int32_t & t : tail) t = kDead; }
};

class DeepSeek4EngramHasher {
public:
    // Reads layer ids, geometry and the hash constants out of the loaded
    // weights; false (with a reason) when the model has no Engram block or the
    // constants disagree with the geometry.
    bool init(const DeepSeek4Weights & w, std::string * err);
    // The same from the raw constants (tests, tools). Arrays are flat per
    // layer: multipliers [n_layers * max_ngram], primes and offsets
    // [n_layers * (max_ngram - 1) * n_heads], rows [n_layers].
    bool init_raw(const std::vector<int> & layer_ids, int n_heads, int max_ngram, int32_t pad_id,
                  std::vector<int32_t> token_map,
                  std::vector<uint64_t> multipliers, std::vector<uint64_t> primes,
                  std::vector<uint64_t> offsets, std::vector<uint64_t> rows,
                  std::string * err);

    bool present() const { return n_layers_ > 0; }
    int n_layers() const { return n_layers_; }
    int n_heads() const { return n_heads_; }
    int max_ngram() const { return max_ngram_; }
    int cols() const { return cols_; }            // rows per token per layer
    int layer_id(int i) const { return layer_ids_[(size_t) i]; }
    // -1 when `il` is not an Engram layer.
    int layer_index(int il) const;
    uint64_t rows(int i) const { return rows_[(size_t) i]; }

    int32_t compress(int32_t token) const {
        if (token < 0 || (size_t) token >= token_map_.size()) return pad_id_;
        return token_map_[(size_t) token];
    }

    // Row ids for `count` consecutive tokens, advancing `history`.
    // `dead` (optional, one byte per token) marks positions that carry no
    // n-gram (image spans); text-only callers pass nullptr.
    // out: [count][n_layers][cols], every row id already offset into its
    // layer's table.
    void hash(DeepSeek4EngramHistory & history,
              const int32_t * tokens, const uint8_t * dead, size_t count,
              uint32_t * out) const;

private:
    int n_layers_ = 0, n_heads_ = 0, max_ngram_ = 0, cols_ = 0;
    int32_t pad_id_ = 0;
    std::vector<int> layer_ids_;
    std::vector<uint64_t> rows_;
    std::vector<int32_t> token_map_;
    std::vector<uint64_t> multipliers_;   // [n_layers][max_ngram]
    std::vector<uint64_t> primes_;        // [n_layers][cols]
    std::vector<uint64_t> offsets_;       // [n_layers][cols]
};

// ── 2. the table read ─────────────────────────────────────────────────────

// E4M3 value times 2^(E8M0 - 127), then round-to-nearest-even to bf16.
float deepseek4_engram_decode_value(uint8_t e4m3, uint8_t e8m0);

class DeepSeek4EngramTable {
public:
    static constexpr int kDim = 256;
    static constexpr int kRowBytes = 264;   // 256 values + 8 block scales

    DeepSeek4EngramTable() = default;
    DeepSeek4EngramTable(DeepSeek4EngramTable && o) noexcept : fd_(o.fd_), offset_(o.offset_), rows_(o.rows_) { o.fd_ = -1; }
    DeepSeek4EngramTable & operator=(DeepSeek4EngramTable && o) noexcept {
        if (this != &o) { close(); fd_ = o.fd_; offset_ = o.offset_; rows_ = o.rows_; o.fd_ = -1; }
        return *this;
    }
    DeepSeek4EngramTable(const DeepSeek4EngramTable &) = delete;
    DeepSeek4EngramTable & operator=(const DeepSeek4EngramTable &) = delete;
    ~DeepSeek4EngramTable() { close(); }
    // A separate, unbuffered descriptor on the GGUF; `offset` is the absolute
    // byte offset of the table, `rows` its row count.
    bool open(const std::string & path, uint64_t offset, uint64_t rows, std::string * err);
    void close();
    bool is_open() const { return fd_ >= 0; }
    uint64_t rows() const { return rows_; }

    // Decodes `count` rows into out[count][kDim]. Rows are fetched in sorted
    // order with duplicates read once; batches of 256 rows or more use
    // `threads` readers. False (errno kept) on a short read or a bad code.
    bool read(const uint32_t * row_ids, size_t count, float * out, int threads = 16) const;
    // Starts reading `count` rows into the page cache without waiting
    // (POSIX_FADV_WILLNEED), so a later `read` of them does not wait on the
    // drive one row at a time.
    void prefetch(const uint32_t * row_ids, size_t count) const;

private:
    int fd_ = -1;
    uint64_t offset_ = 0;
    uint64_t rows_ = 0;
};


// ── host-side runtime ─────────────────────────────────────────────────────

// The raw token ids at the most recent positions of one sequence: the n-gram
// context the hash of the next position reads. Entries are keyed by position,
// so rewinding a sequence (a rejected speculative suffix) needs no undo: the
// positions before the rewind point still hold the tokens that were fed.
struct DeepSeek4EngramTokens {
    static constexpr int kSize = 64;
    int32_t token[kSize];
    int32_t pos[kSize];
    DeepSeek4EngramTokens() { reset(); }
    void reset() { for (int i = 0; i < kSize; ++i) { token[i] = -1; pos[i] = -1; } }
    void put(int p, int32_t t) { token[p % kSize] = t; pos[p % kSize] = p; }
    // The token at position p, or -1 when this context does not hold it.
    int32_t at(int p) const { return p >= 0 && pos[p % kSize] == p ? token[p % kSize] : -1; }
};

// Owns the hasher and one table per Engram layer. `prepare` turns a run of
// token ids into the decoded key rows the apply consumes:
// keys[layer][token][cols * kDim] floats, exactly the `keys` input of
// deepseek4_build_engram_apply for that layer. It keeps no per-sequence state:
// the n-gram context comes from the caller's DeepSeek4EngramTokens.
class DeepSeek4EngramRuntime {
public:
    // Opens every embedded table recorded by the loader (w.engram.tables) on
    // `gguf_path`. False when the model has Engram layers but a table is
    // missing (the PR-dialect files without tables refuse here).
    bool init(const DeepSeek4Weights & w, const std::string & gguf_path, std::string * err);
    bool present() const { return hasher_.present(); }
    const DeepSeek4EngramHasher & hasher() const { return hasher_; }
    int n_layers() const { return hasher_.n_layers(); }
    int layer_id(int i) const { return hasher_.layer_id(i); }
    int layer_index(int il) const { return hasher_.layer_index(il); }
    // floats per token per layer
    size_t key_floats() const { return (size_t) hasher_.cols() * DeepSeek4EngramTable::kDim; }

    // Hashes `count` tokens at positions first_pos.. of the sequence whose
    // context is `ctx` (recording them there), reads the rows of every Engram
    // layer and decodes them into keys[n_layers][count][key_floats()]. Fails
    // when `ctx` does not hold the tokens before first_pos.
    bool prepare(DeepSeek4EngramTokens & ctx, const int32_t * tokens, int first_pos,
                 size_t count, float * keys, std::string * err) const;
    // Starts reading the rows `prepare` will need for the same arguments,
    // without waiting and without recording the tokens.
    void prefetch(const DeepSeek4EngramTokens & ctx, const int32_t * tokens, int first_pos,
                  size_t count) const;

    // Rows read from the tables so far.
    uint64_t rows_read() const { return rows_read_.load(std::memory_order_relaxed); }

private:
    bool row_ids(const DeepSeek4EngramTokens & ctx, const int32_t * tokens, int first_pos,
                 size_t count, std::vector<uint32_t> & ids, std::string * err) const;

    DeepSeek4EngramHasher hasher_;
    std::vector<DeepSeek4EngramTable> tables_;   // one per Engram layer, hasher order
    mutable std::atomic<uint64_t> rows_read_{0};
    int threads_ = 16;
};

// ── 3. the apply ──────────────────────────────────────────────────────────

// Builds the Engram update for one layer as a ggml subgraph.
//   h:    [n_embd, n_hc, n_tokens] F32, the residual copies entering the layer
//   keys: [cols * key_len, n_tokens] F32, the decoded rows of every token,
//         concatenated in column order (what `DeepSeek4EngramTable::read`
//         produces for the `cols` rows of one token, back to back)
//   L:    layer weights (engram_wkv [cols*key_len, n_embd*(n_hc+1)],
//         engram_q, engram_k [n_embd, n_hc])
// Returns the updated copies, same shape as `h`, or nullptr when a weight is
// missing. Math (model.py 350-368): per copy c,
//   dot_c  = sum_d rms_norm(h_c)[d] * q[c,d] * k[c,d] * rms_norm(key_c)[d] * n_embd^-0.5
//   gate_c = sigmoid(sign(dot_c) * sqrt(max(|dot_c|, 1e-6)))
//   h_c   += gate_c * value
// The reference's rsqrt(mean(x^2) + eps) factors are exactly ggml_rms_norm.
ggml_tensor * deepseek4_build_engram_apply(ggml_context * ctx,
                                           ggml_tensor * h,
                                           ggml_tensor * keys,
                                           const DeepSeek4Layer & L,
                                           int n_embd, int n_hc,
                                           float rms_eps,
                                           ggml_tensor ** gate_out = nullptr);

// Runs the apply on residual copies kept in host memory (the host
// hyper-connection paths): uploads them with their keys, computes on
// `backend`, where the layer's Engram weights live, and reads them back in
// place. Batches of more than kMaxTokens tokens run in parts.
class DeepSeek4EngramApplyRunner {
public:
    static constexpr int kMaxTokens = 256;
    DeepSeek4EngramApplyRunner() = default;
    DeepSeek4EngramApplyRunner(const DeepSeek4EngramApplyRunner &) = delete;
    DeepSeek4EngramApplyRunner & operator=(const DeepSeek4EngramApplyRunner &) = delete;
    ~DeepSeek4EngramApplyRunner() { release(); }
    void release();

    // hc: [n_tokens][n_hc * n_embd], keys: [n_tokens][cols * key_len].
    bool run(ggml_backend * backend, const DeepSeek4Layer & L, int n_embd, int n_hc, float rms_eps,
             float * hc, const float * keys, int n_tokens);

private:
    ggml_gallocr * alloc_ = nullptr;
    ggml_backend * alloc_backend_ = nullptr;
    std::vector<uint8_t> meta_;
};

}  // namespace luce::common
