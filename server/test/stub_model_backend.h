// StubModelBackend — deterministic ModelBackend driven by a ScenarioStore.
//
// generate() decodes the request prompt back to text using the real
// tokenizer, looks up the matching scenario (longest prompt_suffix wins),
// then streams the scenario's tokens through the production callbacks
// (req.on_token / io.on_token). Streaming behavior is inherited from the
// production code path — no separate streaming machine.
//
// Token-stream construction:
//   - Plain text tokens are BPE-encoded by the real tokenizer, so a
//     scripted "Let me think. " expands to the same Qwen3.6 token IDs
//     the real model would have emitted.
//   - Special tokens (e.g. </think>) are looked up via token_to_id()
//     and emitted as a single ID. This matters: Qwen3.6's </think>
//     is one added token (id 248069); for SseEmitter to recognize it
//     as a channel-close, it must arrive as the single special ID,
//     not as the BPE-decomposed sequence.

#pragma once

#include "common/model_backend.h"
#include "scenario_store.h"

namespace dflash::common {
class Tokenizer;
}

namespace dflash::common::test {

class StubModelBackend : public ModelBackend {
public:
    StubModelBackend(const ScenarioStore & store, const Tokenizer & tokenizer)
        : store_(store), tokenizer_(tokenizer) {}

    void print_ready_banner() const override {}
    bool park(const std::string &) override { return true; }
    bool unpark(const std::string &) override { return true; }
    bool is_target_parked() const override { return false; }

    GenerateResult generate_impl(const GenerateRequest & req,
                                 const DaemonIO & io) override;

    bool snapshot_save(int) override { return false; }
    void snapshot_free(int) override {}
    bool snapshot_used(int) const override { return false; }
    int  snapshot_cur_pos(int) const override { return 0; }
    GenerateResult restore_and_generate_impl(int, const GenerateRequest & req,
                                             const DaemonIO & io) override {
        return generate_impl(req, io);
    }

    bool handle_compress(const std::string &, const DaemonIO &) override {
        return false;
    }
    void free_drafter() override {}
    void shutdown() override {}

private:
    const ScenarioStore & store_;
    const Tokenizer &     tokenizer_;
};

}  // namespace dflash::common::test
