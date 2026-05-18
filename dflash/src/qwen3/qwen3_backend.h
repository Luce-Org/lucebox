// Qwen3Backend — ModelBackend for the Qwen3-0.6B model used as a standalone
// inference backend (not just as a pflash drafter).
//
// Architecture: 28-layer transformer, 16 heads (8 KV), hidden=1024, vocab=151936.
// Sliding-window attention (FA_WINDOW=512), standard RoPE.
//
// This backend reuses the Qwen3DrafterWeights loader but adds:
//   - Persistent KV cache for incremental decode
//   - Step-based forward (prefill chunks + single-token decode)
//   - Logits output via out_norm + lm_head

#pragma once

#include "common/model_backend.h"
#include "common/device_placement.h"
#include "qwen3_drafter_model.h"
#include "qwen3_drafter.h"
#include "common/sampler.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <random>
#include <string>
#include <vector>

namespace dflash27b {

struct Qwen3BackendConfig {
    const char *    model_path = nullptr;
    DevicePlacement device;
    int             stream_fd  = -1;
    int             chunk      = 512;
};

// Persistent KV cache for incremental decode.
struct Qwen3Cache {
    int cur_pos  = 0;
    int max_ctx  = 0;
    int n_layer  = 0;

    // Per-layer K/V: [head_dim, n_head_kv, max_ctx] in half precision.
    // Allocated once at init, filled incrementally during prefill/decode.
    std::vector<ggml_tensor *> k;   // n_layer entries
    std::vector<ggml_tensor *> v;   // n_layer entries

    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

bool  create_qwen3_cache(ggml_backend_t backend, const Qwen3DrafterWeights & w,
                          int max_ctx, Qwen3Cache & out);
void  free_qwen3_cache(Qwen3Cache & c);

// Snapshot for prefix caching.
struct Qwen3Snapshot {
    int  cur_pos = 0;
    std::vector<ggml_tensor *> k_snap;
    std::vector<ggml_tensor *> v_snap;
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

void free_qwen3_snapshot(Qwen3Snapshot & s);

class Qwen3Backend : public ModelBackend {
public:
    explicit Qwen3Backend(const Qwen3BackendConfig & cfg);
    ~Qwen3Backend() override;

    Qwen3Backend(const Qwen3Backend &) = delete;
    Qwen3Backend & operator=(const Qwen3Backend &) = delete;

    bool init();

    // ModelBackend interface
    void print_ready_banner() const override;

    bool park(const std::string & what) override;
    bool unpark(const std::string & what) override;
    bool is_target_parked() const override { return parked_; }

    GenerateResult generate(const GenerateRequest & req,
                            const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;

    GenerateResult restore_and_generate(int slot,
                                        const GenerateRequest & req,
                                        const DaemonIO & io) override;

    CompressResult compress(const CompressRequest & req) override;
    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override;

    bool try_handle_command(const std::string & line,
                            const DaemonIO & io) override;

    void shutdown() override;

private:
    Qwen3BackendConfig    cfg_;
    ggml_backend_t        backend_ = nullptr;
    Qwen3DrafterWeights   w_;
    Qwen3Cache            cache_;
    bool                  parked_ = false;

    // Pflash drafter (lazy-loaded, reuses the same model for compress)
    DrafterContext         drafter_ctx_;
    bool                  drafter_loaded_ = false;

    // Sampler
    SamplerCfg            sampler_;
    std::mt19937_64       sampler_rng_{std::random_device{}()};

    // Snapshots
    static constexpr int PREFIX_SLOTS = 8;
    Qwen3Snapshot         snapshots_[PREFIX_SLOTS];

    // Step forward: run n_tokens through all layers, write K/V into cache,
    // return logits for the last token. embed is [n_tokens * hidden] f32.
    bool do_step(const float * embed, int n_tokens, int kv_start,
                 std::vector<float> & out_logits);

    // Prefill prompt tokens in chunks, return committed position.
    // Saves last-chunk logits in last_logits_.
    int do_prefill(const std::vector<int32_t> & tokens, const DaemonIO & io,
                   int kv_offset = 0);

    // Autoregressive decode loop.
    bool do_decode(int committed, int n_gen,
                   std::vector<int32_t> & out_tokens,
                   const DaemonIO & io);

    std::vector<float> last_logits_;  // logits from last prefill chunk
};

}  // namespace dflash27b
