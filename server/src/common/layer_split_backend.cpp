// Generic server-facing backend for target layer split.

#include "layer_split_backend.h"

#include "io_utils.h"

#include <chrono>
#include <cstdio>
#include <cmath>
#include <utility>

namespace dflash::common {

LayerSplitBackend::LayerSplitBackend(std::unique_ptr<LayerSplitAdapter> adapter)
    : adapter_(std::move(adapter)) {}

LayerSplitBackend::~LayerSplitBackend() { shutdown(); }

bool LayerSplitBackend::init() {
    if (!adapter_) {
        std::fprintf(stderr, "[target-split] missing model adapter\n");
        return false;
    }
    shutdown_done_ = false;
    return adapter_->init();
}

void LayerSplitBackend::print_ready_banner() const {
    std::printf("[daemon] ready\n");
    std::fflush(stdout);
}

bool LayerSplitBackend::park(const std::string & what) {
    std::fprintf(stderr, "[target-split] park is not supported yet (%s)\n",
                 what.c_str());
    return false;
}

bool LayerSplitBackend::unpark(const std::string & what) {
    std::fprintf(stderr, "[target-split] unpark is not supported yet (%s)\n",
                 what.c_str());
    return false;
}

GenerateResult LayerSplitBackend::run_from_state(const GenerateRequest & req,
                                                 const DaemonIO & io,
                                                 int base_pos,
                                                 bool reset_state) {
    GenerateResult result;
    if (!adapter_) {
        result.error = "adapter";
        return result;
    }

    DaemonIO out_io = io.with_token_callback(req.on_token);
    if (base_pos + (int)req.prompt.size() + req.n_gen + 1 > adapter_->max_context()) {
        result.error = "context";
        return result;
    }
    if (req.do_sample && req.sampler.temp > 0.0f) {
        result.error = "sampling_unsupported";
        return result;
    }

    adapter_->begin_request(req);
    if (reset_state) adapter_->reset_request_state();

    const int prompt_len = (int)req.prompt.size();
    int last_tok = (base_pos > 0 && prompt_len == 0)
        ? adapter_->current_last_token()
        : -1;
    int consumed = 0;
    auto t_prefill_start = std::chrono::steady_clock::now();
    while (consumed < prompt_len) {
        int n_tokens = prompt_len - consumed;
        if (req.snap_pos >= 0 && req.snap_slot >= 0 &&
            req.snap_pos > base_pos + consumed &&
            req.snap_pos < base_pos + consumed + n_tokens) {
            n_tokens = req.snap_pos - (base_pos + consumed);
        }
        std::vector<int32_t> chunk(req.prompt.begin() + consumed,
                                   req.prompt.begin() + consumed + n_tokens);
        if (!adapter_->prefill(chunk, base_pos + consumed, last_tok)) {
            result.error = "prefill";
            return result;
        }
        consumed += n_tokens;
        if (req.snap_pos >= 0 && req.snap_slot >= 0 &&
            base_pos + consumed == req.snap_pos) {
            if (adapter_->snapshot_save(req.snap_slot)) {
                std::printf("[snap] inline slot=%d cur_pos=%d\n",
                            req.snap_slot, req.snap_pos);
                std::fflush(stdout);
            }
        }
    }
    result.prefill_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_prefill_start).count();

    if (req.n_gen > 0) {
        if (last_tok < 0) {
            result.error = "decode_seed";
            return result;
        }
        auto t_decode_start = std::chrono::steady_clock::now();
        const bool ok = adapter_->can_dflash_decode()
            ? adapter_->decode_dflash(req.prompt, base_pos, last_tok, req.n_gen,
                                      result.tokens, out_io)
            : adapter_->decode_ar(last_tok, base_pos + (int)req.prompt.size(), req.n_gen,
                                  result.tokens, out_io);
        if (!ok) {
            result.error = "decode";
            return result;
        }
        result.decode_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_decode_start).count();
    }

    result.ok = true;
    return result;
}

GenerateResult LayerSplitBackend::generate(const GenerateRequest & req,
                                           const DaemonIO & io) {
    return run_from_state(req, io, /*base_pos=*/0, /*reset_state=*/true);
}

bool LayerSplitBackend::snapshot_save(int slot) {
    return adapter_ && adapter_->snapshot_save(slot);
}

void LayerSplitBackend::snapshot_free(int slot) {
    if (adapter_) adapter_->snapshot_free(slot);
}

bool LayerSplitBackend::snapshot_used(int slot) const {
    return adapter_ && adapter_->snapshot_used(slot);
}

int LayerSplitBackend::snapshot_cur_pos(int slot) const {
    return adapter_ ? adapter_->snapshot_cur_pos(slot) : 0;
}

GenerateResult LayerSplitBackend::restore_and_generate(
        int slot, const GenerateRequest & req, const DaemonIO & io) {
    GenerateResult result;
    if (!adapter_ || !adapter_->snapshot_restore(slot)) {
        result.error = "bad slot";
        io.emit(-1);
        return result;
    }
    const int snap_pos = adapter_->snapshot_cur_pos(slot);
    if ((int)req.prompt.size() < snap_pos) {
        result.error = "snapshot_longer_than_prompt";
        io.emit(-1);
        return result;
    }
    GenerateRequest delta_req = req;
    delta_req.prompt = std::vector<int32_t>(
        req.prompt.begin() + snap_pos, req.prompt.end());
    return run_from_state(delta_req, io, snap_pos, /*reset_state=*/false);
}

ModelBackend::CompressResult
LayerSplitBackend::compress(const CompressRequest & req) {
    return adapter_ ? adapter_->compress(req) : CompressResult{};
}

bool LayerSplitBackend::handle_compress(const std::string & line,
                                        const DaemonIO & io) {
    std::string args = line.size() > 9 ? line.substr(9) : std::string{};
    bool skip_park = false;
    const std::string suffix = " nopark";
    if (args.size() >= suffix.size() &&
        args.compare(args.size() - suffix.size(), suffix.size(), suffix) == 0) {
        skip_park = true;
        args.resize(args.size() - suffix.size());
    }

    char ppath[1024];
    int keep_x1000 = 0;
    char drafter_path[1024] = {0};
    const int n = std::sscanf(args.c_str(), "%1023s %d %1023s",
                              ppath, &keep_x1000, drafter_path);
    if (n < 2) {
        std::fprintf(stderr, "[target-split][compress] bad args\n");
        io.emit(-1);
        return false;
    }

    CompressRequest req;
    req.input_ids = read_int32_file(ppath);
    req.keep_ratio = (float)keep_x1000 / 1000.0f;
    if (!std::isfinite(req.keep_ratio) ||
        req.keep_ratio < 0.0f || req.keep_ratio > 1.0f) {
        std::fprintf(stderr,
            "[target-split][compress] keep ratio must be in [0,1], got %d/1000\n",
            keep_x1000);
        io.emit(-1);
        return false;
    }
    if (n >= 3 && drafter_path[0]) {
        req.drafter_path = drafter_path;
    } else if (adapter_) {
        req.drafter_path = adapter_->default_compress_drafter_path();
    }
    req.skip_park = skip_park;

    CompressResult result = compress(req);
    for (int32_t t : result.compressed_ids) io.emit(t);
    io.emit(-1);
    return result.ok;
}

void LayerSplitBackend::free_drafter() {
    if (adapter_) adapter_->free_drafter();
}

bool LayerSplitBackend::supports_dflash_spec_decode() const {
    return adapter_ && adapter_->supports_dflash_spec_decode();
}

DFlashTarget * LayerSplitBackend::dflash_target() {
    return adapter_ ? adapter_->dflash_target() : nullptr;
}

bool LayerSplitBackend::supports_remote_draft() const {
    return adapter_ && adapter_->supports_remote_draft();
}

void LayerSplitBackend::shutdown() {
    if (shutdown_done_) return;
    shutdown_done_ = true;
    if (adapter_) adapter_->shutdown();
}

}  // namespace dflash::common
