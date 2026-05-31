// Generic server-facing backend for target layer split.
//
// Model-specific layer-split details live behind LayerSplitAdapter. This keeps
// server placement, request flow, and compatibility policy in one place while
// allowing each architecture to provide only its partial-load/forward/cache
// implementation.

#pragma once

#include "model_backend.h"

#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

class LayerSplitAdapter {
public:
    virtual ~LayerSplitAdapter() = default;

    virtual const char * name() const = 0;
    virtual bool init() = 0;
    virtual int max_context() const = 0;

    virtual void begin_request(const GenerateRequest & req) { (void)req; }
    virtual void reset_request_state() = 0;
    virtual bool prefill(const std::vector<int32_t> & prompt,
                         int base_pos, int & last_tok) = 0;
    virtual bool decode_ar(int last_tok, int committed, int n_gen,
                           std::vector<int32_t> & out_tokens,
                           const DaemonIO & io) = 0;

    virtual bool can_dflash_decode() const { return false; }
    virtual bool decode_dflash(const std::vector<int32_t> & prompt,
                               int base_pos, int last_tok, int n_gen,
                               std::vector<int32_t> & out_tokens,
                               const DaemonIO & io) {
        (void)prompt; (void)base_pos; (void)last_tok; (void)n_gen;
        (void)out_tokens; (void)io;
        return false;
    }

    virtual bool supports_dflash_spec_decode() const { return false; }
    virtual DFlashTarget * dflash_target() { return nullptr; }
    virtual bool supports_remote_draft() const { return false; }

    virtual const char * default_compress_drafter_path() const { return ""; }
    virtual ModelBackend::CompressResult
    compress(const ModelBackend::CompressRequest & req) {
        (void)req;
        return {};
    }
    virtual void free_drafter() = 0;

    virtual bool snapshot_save(int slot) { (void)slot; return false; }
    virtual void snapshot_free(int slot) { (void)slot; }
    virtual bool snapshot_used(int slot) const { (void)slot; return false; }
    virtual int snapshot_cur_pos(int slot) const { (void)slot; return 0; }
    virtual bool snapshot_restore(int slot) { (void)slot; return false; }
    virtual int current_last_token() const { return -1; }

    virtual void shutdown() = 0;
};

class LayerSplitBackend : public ModelBackend {
public:
    explicit LayerSplitBackend(std::unique_ptr<LayerSplitAdapter> adapter);
    ~LayerSplitBackend() override;

    LayerSplitBackend(const LayerSplitBackend &) = delete;
    LayerSplitBackend & operator=(const LayerSplitBackend &) = delete;

    bool init();

    void print_ready_banner() const override;
    bool park(const std::string & what) override;
    bool unpark(const std::string & what) override;
    bool is_target_parked() const override { return false; }

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;

    bool snapshot_save(int slot) override;
    void snapshot_free(int slot) override;
    bool snapshot_used(int slot) const override;
    int  snapshot_cur_pos(int slot) const override;
    GenerateResult restore_and_generate_impl(int slot,
                                             const GenerateRequest & req,
                                             const DaemonIO & io) override;

    CompressResult compress(const CompressRequest & req) override;
    bool handle_compress(const std::string & line,
                         const DaemonIO & io) override;
    void free_drafter() override;

    bool supports_dflash_spec_decode() const override;
    DFlashTarget * dflash_target() override;
    bool supports_remote_draft() const override;

    void shutdown() override;

private:
    GenerateResult run_from_state(const GenerateRequest & req,
                                  const DaemonIO & io,
                                  int base_pos,
                                  bool reset_state);

    std::unique_ptr<LayerSplitAdapter> adapter_;
    bool shutdown_done_ = false;
};

}  // namespace dflash::common
