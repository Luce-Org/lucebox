// Unit test driving the extraction of MTP orchestration from
// qwen35_backend.cpp into dflash/src/common/mtp_orchestrator.
//
// Howard's PR #214 review (CHANGES_REQUESTED) asked for MTP logic to live
// in /common so any ModelBackend that supports MTP can leverage it. This
// test pins the public surface of the new common helper and proves it
// handles the trivial guard cases — null backend, no MTP support — before
// any real backend is wired through it.
//
// Plain int main(), assert-based, mirrors test_kv_quant.cpp style.

#include "common/mtp_orchestrator.h"
#include "common/model_backend.h"
#include "common/mtp_interface.h"
#include "common/dflash_target.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Minimal stub backend: implements every pure-virtual but every operation
// is a no-op or returns false. Lets us exercise the orchestrator's guard
// paths without loading any GPU weights.
struct StubBackend : public dflash27b::ModelBackend {
    bool supports_mtp_value = false;

    void print_ready_banner() const override {}
    bool park(const std::string &)   override { return true; }
    bool unpark(const std::string &) override { return true; }
    bool is_target_parked() const override { return false; }
    dflash27b::GenerateResult generate(const dflash27b::GenerateRequest &,
                                       const dflash27b::DaemonIO &) override {
        return {};
    }
    bool snapshot_save(int) override { return false; }
    void snapshot_free(int) override {}
    bool snapshot_used(int) const override { return false; }
    int  snapshot_cur_pos(int) const override { return 0; }
    dflash27b::GenerateResult restore_and_generate(int,
                                                   const dflash27b::GenerateRequest &,
                                                   const dflash27b::DaemonIO &) override {
        return {};
    }
    bool handle_compress(const std::string &,
                         const dflash27b::DaemonIO &) override { return false; }
    void free_drafter() override {}
    bool supports_mtp() const override { return supports_mtp_value; }
    dflash27b::mtp::IMtpModule * mtp() override { return nullptr; }
    void shutdown() override {}
};

// Mock NativeHeads MTP module for the "interface is generic" test (T4).
// Records calls to prove the orchestrator invokes the right virtuals.
struct StubMtpModule : public dflash27b::mtp::INativeMtp {
    int reset_chain_calls = 0;
    int set_initial_hidden_calls = 0;
    int max_gamma() const override { return 3; }
    int effective_gamma_value = 3;
    int  effective_gamma() const override { return effective_gamma_value; }
    void set_effective_gamma(int g) override { effective_gamma_value = g; }
    int hidden_size() const override { return 4; }
    bool attach(dflash27b::DFlashTarget *) override { return true; }
    void reset_chain() override { reset_chain_calls++; }
    void shutdown() override {}
    int num_heads() const override { return 3; }
    bool step_batch(int32_t, int,
                    std::vector<dflash27b::mtp::StepOutput> &) override { return true; }
    void set_initial_hidden(const float *, int) override { set_initial_hidden_calls++; }
};

// Mock DFlashTarget that fails verify_batch — proves the orchestrator reached
// the prefill loop using only abstract DFlashTarget methods.
struct StubTarget : public dflash27b::DFlashTarget {
    int verify_batch_calls = 0;
    int hidden_size() const override { return 4; }
    bool verify_batch(const std::vector<int32_t> &, int, int &,
                      std::vector<int32_t> *) override {
        verify_batch_calls++;
        return false;
    }
    bool verify_tree(const std::vector<int32_t> &, const dflash27b::DDTree &,
                     int, std::vector<int32_t> &, std::vector<float> *) override { return false; }
    bool snapshot_kv() override { return false; }
    bool restore_kv() override { return false; }
    bool restore_kv_at_dfs(const std::vector<int> &) override { return false; }
    bool restore_kv_at_chain(int) override { return false; }
    void capture_topology_for_chain(int, int) override {}
    bool is_eos(int) const override { return false; }
    bool embed_tokens(const int32_t *, int, float *) const override { return false; }
    bool project_hidden_to_tokens(const float *, int,
                                   std::vector<int32_t> &) override { return false; }
    bool project_hidden_to_logits(const float *, int,
                                   std::vector<float> &, int &) override { return false; }
    int  mask_token_id() const override { return 0; }
    const std::vector<int> & capture_layer_ids() const override {
        static std::vector<int> empty;
        return empty;
    }
    ggml_backend * backend() const override { return nullptr; }
    ggml_tensor * lm_head_weight() const override { return nullptr; }
    int fa_window() const override { return 0; }
};

// Richer stub backend that returns valid mtp() + dflash_target() pointers,
// proving the orchestrator drives a generic ModelBackend without any
// Qwen35-specific cast or include.
struct FullStubBackend : public StubBackend {
    StubMtpModule mtp_module;
    StubTarget    target;
    FullStubBackend() { supports_mtp_value = true; }
    dflash27b::mtp::IMtpModule * mtp() override { return &mtp_module; }
    dflash27b::DFlashTarget *    dflash_target() override { return &target; }
};

}  // namespace

// ─── T1: null backend pointer ───────────────────────────────────────────────

static void t1_null_backend() {
    dflash27b::GenerateRequest req;
    dflash27b::DaemonIO io;
    auto res = dflash27b::common::mtp::warm_and_decode(nullptr, req, io);
    assert(!res.ok);
    assert(res.error.find("backend") != std::string::npos);
    std::puts("T1 null_backend PASS");
}

// ─── T2: backend that does NOT support MTP — orchestrator declines cleanly ──

static void t2_backend_without_mtp() {
    StubBackend b;
    b.supports_mtp_value = false;
    dflash27b::GenerateRequest req;
    dflash27b::DaemonIO io;
    auto res = dflash27b::common::mtp::warm_and_decode(&b, req, io);
    assert(!res.ok);
    assert(res.error.find("mtp") != std::string::npos
        || res.error.find("MTP") != std::string::npos);
    std::puts("T2 backend_without_mtp PASS");
}

// ─── T3: empty prompt — orchestrator declines with an explicit error ────────

static void t3_empty_prompt() {
    StubBackend b;
    b.supports_mtp_value = true;
    dflash27b::GenerateRequest req;
    req.n_gen = 8;
    dflash27b::DaemonIO io;
    auto res = dflash27b::common::mtp::warm_and_decode(&b, req, io);
    assert(!res.ok);
    assert(res.error.find("prompt") != std::string::npos);
    std::puts("T3 empty_prompt PASS");
}

// ─── T4: orchestrator drives a generic ModelBackend through abstract
//        interfaces only (proves logic in common/ is not Qwen35-specific).
//        Module needs to expose dflash_target() in the abstract — currently
//        ModelBackend::dflash_target() lives on the base. ───────────────

static void t4_generic_backend_dispatch() {
    FullStubBackend b;
    dflash27b::GenerateRequest req;
    req.prompt = {1, 2, 3, 4};
    req.n_gen = 4;
    dflash27b::DaemonIO io;
    auto res = dflash27b::common::mtp::warm_and_decode(&b, req, io);
    assert(!res.ok);
    // Reached verify_batch (which stub fails) — proves orchestrator depends
    // only on ModelBackend / DFlashTarget / IMtpModule abstractions.
    assert(res.error.find("verify_batch") != std::string::npos);
    assert(b.target.verify_batch_calls >= 1);
    std::puts("T4 generic_backend_dispatch PASS");
}

int main() {
    t1_null_backend();
    t2_backend_without_mtp();
    t3_empty_prompt();
    t4_generic_backend_dispatch();
    std::puts("ALL PASS");
    return 0;
}
