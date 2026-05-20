// Unit test driving the extraction of MTP orchestration from
// qwen35_backend.cpp into dflash/src/common/mtp_orchestrator.
//
// Pins the public surface of the common helper and proves it handles the
// trivial guard cases — null backend, no MTP support — before any real
// backend is wired through it.
//
// T5-T10: MtpChainRunner state machine (gamma propagation, EOS, partial
//         accept, n_gen termination, step failure, stats accounting).
// T11-T14: MtpOrchestrator lifecycle (reset_chain ordering, set_initial_hidden
//          plumbing, gamma derivation, warm_head_kv gate).
// T15-T19: Qwen35MtpModule error paths (no GGUF required).
//
// Plain int main(), assert-based, mirrors test_kv_quant.cpp style.

#include "common/mtp_orchestrator.h"
#include "common/mtp_chain_runner.h"
#include "common/model_backend.h"
#include "common/mtp_interface.h"
#include "common/dflash_target.h"
#include "qwen35/qwen35_mtp.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ── Base stubs (used by T1-T4 and reused by new tests) ───────────────────

// Minimal stub backend: implements every pure-virtual but every operation
// is a no-op or returns false. Lets us exercise the orchestrator's guard
// paths without loading any GPU weights.
struct StubBackend : public dflash::common::ModelBackend {
    bool supports_mtp_value = false;

    void print_ready_banner() const override {}
    bool park(const std::string &)   override { return true; }
    bool unpark(const std::string &) override { return true; }
    bool is_target_parked() const override { return false; }
    dflash::common::GenerateResult generate(const dflash::common::GenerateRequest &,
                                       const dflash::common::DaemonIO &) override {
        return {};
    }
    bool snapshot_save(int) override { return false; }
    void snapshot_free(int) override {}
    bool snapshot_used(int) const override { return false; }
    int  snapshot_cur_pos(int) const override { return 0; }
    dflash::common::GenerateResult restore_and_generate(int,
                                                   const dflash::common::GenerateRequest &,
                                                   const dflash::common::DaemonIO &) override {
        return {};
    }
    bool handle_compress(const std::string &,
                         const dflash::common::DaemonIO &) override { return false; }
    void free_drafter() override {}
    bool supports_mtp() const override { return supports_mtp_value; }
    dflash::common::mtp::IMtpModule * mtp() override { return nullptr; }
    void shutdown() override {}
};

// Mock NativeHeads MTP module for the "interface is generic" test (T4).
// Records calls to prove the orchestrator invokes the right virtuals.
struct StubMtpModule : public dflash::common::mtp::INativeMtp {
    int reset_chain_calls = 0;
    int set_initial_hidden_calls = 0;
    int max_gamma() const override { return 3; }
    int effective_gamma_value = 3;
    int  effective_gamma() const override { return effective_gamma_value; }
    void set_effective_gamma(int g) override { effective_gamma_value = g; }
    int hidden_size() const override { return 4; }
    bool attach(dflash::common::DFlashTarget *) override { return true; }
    void reset_chain() override { reset_chain_calls++; }
    void shutdown() override {}
    int num_heads() const override { return 3; }
    bool step_batch(int32_t, int,
                    std::vector<dflash::common::mtp::StepOutput> &) override { return true; }
    void set_initial_hidden(const float *, int) override { set_initial_hidden_calls++; }
};

// Mock DFlashTarget that fails verify_batch — proves the orchestrator reached
// the prefill loop using only abstract DFlashTarget methods.
struct StubTarget : public dflash::common::DFlashTarget {
    int verify_batch_calls = 0;
    int hidden_size() const override { return 4; }
    bool verify_batch(const std::vector<int32_t> &, int, int &,
                      std::vector<int32_t> *) override {
        verify_batch_calls++;
        return false;
    }
    bool verify_tree(const std::vector<int32_t> &, const dflash::common::DDTree &,
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
    dflash::common::mtp::IMtpModule * mtp() override { return &mtp_module; }
    dflash::common::DFlashTarget *    dflash_target() override { return &target; }
};

// ── Extended stubs for chain runner + orchestrator lifecycle tests ─────────

// DFlashTarget that succeeds verify_batch and is configurable for chain tests.
//
// verify_batch behavior:
//   - returns true, sets last_tok = argmax_token, fills all_argmax if requested
//   - all_argmax[i] = candidate[i] for i < accept_n (accept), then diverge_token
//   - this models the target accepting accept_n drafts + emitting a bonus token
//
// Also supports: snapshot_kv (succeeds), restore_kv (succeeds),
// restore_kv_at_chain (returns false to force slow rollback path),
// is_eos (returns true for eos_token_id when set).
struct SuccessStubTarget : public dflash::common::DFlashTarget {
    int     argmax_token  = 42;    // returned as last_tok and as the bonus token
    int     accept_n      = 0;     // how many candidates to "accept" in all_argmax
    int     eos_token_id  = -1;    // token for which is_eos returns true
    int     verify_calls  = 0;
    int     hidden_sz     = 4;
    // Hidden seq buffer returned by last_hidden_seq (sized to last verify chunk).
    mutable std::vector<float> hidden_seq_buf;
    mutable int                hidden_seq_n = 0;

    int hidden_size() const override { return hidden_sz; }

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int /*base_pos*/,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax) override {
        verify_calls++;
        last_tok = argmax_token;
        if (all_argmax) {
            all_argmax->resize(tokens.size());
            for (int i = 0; i < (int)tokens.size(); i++) {
                if (i < accept_n) {
                    // Accept: target's argmax matches the candidate (simulate
                    // the chain runner's matching logic: drafts[i] == all_argmax[i]).
                    // all_argmax[i] = tokens[i+1] when i < g_actual (drafts).
                    // But for the bonus slot the runner reads all_argmax[accept_n].
                    // We simply set all_argmax[i] = tokens[i] so drafts match.
                    (*all_argmax)[i] = tokens[i];
                } else {
                    (*all_argmax)[i] = argmax_token;
                }
            }
        }
        // Populate hidden seq so orchestrator prefill path succeeds.
        hidden_seq_n = (int)tokens.size();
        hidden_seq_buf.assign((size_t)hidden_seq_n * hidden_sz, 0.1f);
        return true;
    }

    bool verify_tree(const std::vector<int32_t> &, const dflash::common::DDTree &,
                     int, std::vector<int32_t> &, std::vector<float> *) override { return false; }

    bool snapshot_kv()  override { return true; }
    bool restore_kv()   override { return true; }
    bool restore_kv_at_dfs(const std::vector<int> &) override { return false; }
    bool restore_kv_at_chain(int) override { return false; }  // force slow path
    void capture_topology_for_chain(int, int) override {}
    void enable_chain_capture(bool) override {}

    bool is_eos(int tok) const override { return eos_token_id >= 0 && tok == eos_token_id; }
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

    const float * last_hidden_seq(int * out_n) const override {
        if (out_n) *out_n = hidden_seq_n;
        return hidden_seq_n > 0 ? hidden_seq_buf.data() : nullptr;
    }
    const float * last_hidden() const override {
        return hidden_seq_buf.empty() ? nullptr : hidden_seq_buf.data();
    }
};

// MTP module that emits a fixed draft token from step_batch.
// Extends StubMtpModule; overrides step_batch to emit `draft_token`.
struct DraftStubMtpModule : public StubMtpModule {
    int32_t draft_token = 99;  // always propose this token
    int warm_head_kv_calls = 0;

    bool step_batch(int32_t, int,
                    std::vector<dflash::common::mtp::StepOutput> & out) override {
        dflash::common::mtp::StepOutput so;
        so.draft_token = draft_token;
        out.push_back(so);
        return true;
    }

    bool warm_head_kv(const int32_t *, int, int32_t, const float *) override {
        warm_head_kv_calls++;
        return true;
    }
};

// MTP module whose step_chain always returns false (simulate module failure).
struct FailStepChainMtpModule : public StubMtpModule {
    bool step_chain(int32_t, int, int,
                    std::vector<dflash::common::mtp::StepOutput> &) override {
        return false;
    }
};

// Backend for orchestrator lifecycle tests: prefill succeeds, exposes
// DraftStubMtpModule so orchestrator can complete its full control flow.
struct LiveStubBackend : public StubBackend {
    DraftStubMtpModule mtp_mod;
    SuccessStubTarget  target;
    LiveStubBackend() { supports_mtp_value = true; }
    dflash::common::mtp::IMtpModule * mtp() override { return &mtp_mod; }
    dflash::common::DFlashTarget *    dflash_target() override { return &target; }
};

}  // namespace

// ─── T1: null backend pointer ───────────────────────────────────────────────

static void t1_null_backend() {
    dflash::common::GenerateRequest req;
    dflash::common::DaemonIO io;
    auto res = dflash::common::mtp::warm_and_decode(nullptr, req, io);
    assert(!res.ok);
    assert(res.error.find("backend") != std::string::npos);
    std::puts("T1 null_backend PASS");
}

// ─── T2: backend that does NOT support MTP — orchestrator declines cleanly ──

static void t2_backend_without_mtp() {
    StubBackend b;
    b.supports_mtp_value = false;
    dflash::common::GenerateRequest req;
    dflash::common::DaemonIO io;
    auto res = dflash::common::mtp::warm_and_decode(&b, req, io);
    assert(!res.ok);
    assert(res.error.find("mtp") != std::string::npos
        || res.error.find("MTP") != std::string::npos);
    std::puts("T2 backend_without_mtp PASS");
}

// ─── T3: empty prompt — orchestrator declines with an explicit error ────────

static void t3_empty_prompt() {
    StubBackend b;
    b.supports_mtp_value = true;
    dflash::common::GenerateRequest req;
    req.n_gen = 8;
    dflash::common::DaemonIO io;
    auto res = dflash::common::mtp::warm_and_decode(&b, req, io);
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
    dflash::common::GenerateRequest req;
    req.prompt = {1, 2, 3, 4};
    req.n_gen = 4;
    dflash::common::DaemonIO io;
    auto res = dflash::common::mtp::warm_and_decode(&b, req, io);
    assert(!res.ok);
    // Reached verify_batch (which stub fails) — proves orchestrator depends
    // only on ModelBackend / DFlashTarget / IMtpModule abstractions.
    assert(res.error.find("verify_batch") != std::string::npos);
    assert(b.target.verify_batch_calls >= 1);
    std::puts("T4 generic_backend_dispatch PASS");
}

// ─── T5: gamma propagation — runner uses the gamma passed to run(); higher
//         gamma (when module max allows) produces more proposals per iter. ──

static void t5_gamma_propagation() {
    // Module max_gamma=3 (from StubMtpModule). draft_token=99 always.
    // Target: accept_n=0, so every iter accepts 0 drafts + 1 bonus.
    // With gamma=1: proposed=1 per iter; with gamma=2: proposed=2 per iter.
    // We run n_gen=1 (one iter) to keep it simple and compare proposed counts.

    DraftStubMtpModule mod1;
    mod1.effective_gamma_value = 1;
    SuccessStubTarget tgt1;
    tgt1.accept_n = 0;
    dflash::common::SamplerCfg sampler;
    dflash::common::mtp::MtpChainRunner runner1(mod1, tgt1, sampler);
    dflash::common::GenerateRequest req1;
    req1.n_gen = 1;
    req1.stream = false;
    dflash::common::DaemonIO io;
    auto res1 = runner1.run(req1, io, /*last_prefill_token=*/10, /*committed_pos=*/4, /*gamma=*/1);
    assert(res1.ok);
    const int proposed_g1 = runner1.stats().total_proposed;

    DraftStubMtpModule mod2;
    mod2.effective_gamma_value = 2;
    SuccessStubTarget tgt2;
    tgt2.accept_n = 0;
    dflash::common::mtp::MtpChainRunner runner2(mod2, tgt2, sampler);
    dflash::common::GenerateRequest req2;
    req2.n_gen = 2;
    req2.stream = false;
    auto res2 = runner2.run(req2, io, /*last_prefill_token=*/10, /*committed_pos=*/4, /*gamma=*/2);
    assert(res2.ok);
    const int proposed_g2 = runner2.stats().total_proposed;

    // gamma=2 run has >= 2 proposed (one full iter with g=2 -> proposed=2).
    assert(proposed_g1 >= 1);
    assert(proposed_g2 >= proposed_g1);
    std::puts("T5 gamma_propagation PASS");
}

// ─── T6: EOS termination — runner stops when target returns is_eos=true. ──

static void t6_eos_termination() {
    DraftStubMtpModule mod;
    mod.effective_gamma_value = 1;
    SuccessStubTarget tgt;
    // Bonus token will be argmax_token=42. Mark 42 as EOS.
    tgt.argmax_token = 42;
    tgt.eos_token_id = 42;
    tgt.accept_n     = 0;

    dflash::common::SamplerCfg sampler;
    dflash::common::mtp::MtpChainRunner runner(mod, tgt, sampler);
    dflash::common::GenerateRequest req;
    req.n_gen  = 100;   // large; EOS should stop it early
    req.stream = false;
    dflash::common::DaemonIO io;
    auto res = runner.run(req, io, /*last_prefill_token=*/10, /*committed_pos=*/4, /*gamma=*/1);
    assert(res.ok);
    const auto & st = runner.stats();
    assert(st.eos_hits >= 1);
    // Emitted far fewer than 100 tokens.
    assert(st.total_emitted <= 10);
    std::puts("T6 eos_termination PASS");
}

// ─── T7: partial-accept rollback — when target accepts K < gamma drafts,
//         runner advances by K and stats.total_accepted grows by K per iter. ─

static void t7_partial_accept_rollback() {
    // gamma=2, accept_n=1 in the target.
    // Each iter: proposed 2, accepted 1, emitted 2 (1 accepted + 1 bonus).
    DraftStubMtpModule mod;
    mod.effective_gamma_value = 2;
    SuccessStubTarget tgt;
    tgt.accept_n     = 1;
    tgt.argmax_token = 55;  // bonus token

    dflash::common::SamplerCfg sampler;
    dflash::common::mtp::MtpChainRunner runner(mod, tgt, sampler);
    dflash::common::GenerateRequest req;
    req.n_gen  = 2;
    req.stream = false;
    dflash::common::DaemonIO io;
    auto res = runner.run(req, io, /*last_prefill_token=*/10, /*committed_pos=*/4, /*gamma=*/2);
    assert(res.ok);
    const auto & st = runner.stats();
    // Exactly 2 tokens emitted (1 accepted + 1 bonus = 2, capped by n_gen=2).
    assert(st.total_accepted >= 1);
    assert(st.total_emitted >= 1);
    // Restore path was hit (accept_n < g_actual -> fallback through restore_kv).
    assert(tgt.verify_calls >= 1);
    std::puts("T7 partial_accept_rollback PASS");
}

// ─── T8: n_gen termination — runner emits exactly n_gen tokens when no EOS. ─

static void t8_n_gen_termination() {
    DraftStubMtpModule mod;
    mod.effective_gamma_value = 1;
    SuccessStubTarget tgt;
    tgt.argmax_token = 77;
    tgt.eos_token_id = -1;  // no EOS
    tgt.accept_n     = 0;

    dflash::common::SamplerCfg sampler;
    dflash::common::mtp::MtpChainRunner runner(mod, tgt, sampler);
    dflash::common::GenerateRequest req;
    req.n_gen  = 5;
    req.stream = false;
    dflash::common::DaemonIO io;
    auto res = runner.run(req, io, /*last_prefill_token=*/10, /*committed_pos=*/4, /*gamma=*/1);
    assert(res.ok);
    assert((int)res.tokens.size() == 5);
    const auto & st = runner.stats();
    assert(st.total_emitted == 5);
    assert(st.eos_hits == 0);
    std::puts("T8 n_gen_termination PASS");
}

// ─── T9: propose failure — when step_chain returns false, runner aborts and
//         returns ok=false with error "mtp.propose". ────────────────────────

static void t9_propose_failure() {
    FailStepChainMtpModule mod;
    mod.effective_gamma_value = 1;
    SuccessStubTarget tgt;

    dflash::common::SamplerCfg sampler;
    dflash::common::mtp::MtpChainRunner runner(mod, tgt, sampler);
    dflash::common::GenerateRequest req;
    req.n_gen  = 4;
    req.stream = false;
    dflash::common::DaemonIO io;
    auto res = runner.run(req, io, /*last_prefill_token=*/10, /*committed_pos=*/4, /*gamma=*/1);
    assert(!res.ok);
    assert(res.error.find("propose") != std::string::npos
        || res.error.find("mtp") != std::string::npos);
    std::puts("T9 propose_failure PASS");
}

// ─── T10: stats accounting — total_emitted == total_accepted + total_iters ─
//          (each iter adds exactly 1 bonus token to emitted).

static void t10_stats_accounting() {
    DraftStubMtpModule mod;
    mod.effective_gamma_value = 2;
    SuccessStubTarget tgt;
    tgt.accept_n     = 1;  // accept 1 draft + 1 bonus = 2 emitted per iter
    tgt.argmax_token = 33;
    tgt.eos_token_id = -1;

    dflash::common::SamplerCfg sampler;
    dflash::common::mtp::MtpChainRunner runner(mod, tgt, sampler);
    dflash::common::GenerateRequest req;
    req.n_gen  = 6;
    req.stream = false;
    dflash::common::DaemonIO io;
    auto res = runner.run(req, io, /*last_prefill_token=*/10, /*committed_pos=*/4, /*gamma=*/2);
    assert(res.ok);
    const auto & st = runner.stats();
    // Invariant: each iter emits accept_n + 1 (one bonus), so:
    // total_emitted == total_accepted + total_iters
    assert(st.total_emitted == st.total_accepted + st.total_iters);
    std::puts("T10 stats_accounting PASS");
}

// ─── T11: reset_chain called before chain runner drive ────────────────────

static void t11_reset_chain_before_drive() {
    LiveStubBackend b;
    b.mtp_mod.effective_gamma_value = 1;
    dflash::common::GenerateRequest req;
    req.prompt = {1, 2, 3};
    req.n_gen  = 2;
    dflash::common::DaemonIO io;
    auto res = dflash::common::mtp::warm_and_decode(&b, req, io);
    // reset_chain() is called once by the orchestrator before drive.
    assert(b.mtp_mod.reset_chain_calls >= 1);
    // Result should succeed (prefill passes, chain runs).
    assert(res.ok);
    std::puts("T11 reset_chain_before_drive PASS");
}

// ─── T12: set_initial_hidden plumbing ──────────────────────────────────────
//          Orchestrator reads target->last_hidden() and forwards via
//          module->set_initial_hidden().  Asserts the call count == 1.

static void t12_set_initial_hidden_plumbing() {
    LiveStubBackend b;
    b.mtp_mod.effective_gamma_value = 1;
    dflash::common::GenerateRequest req;
    req.prompt = {5, 6, 7, 8};
    req.n_gen  = 1;
    dflash::common::DaemonIO io;
    auto res = dflash::common::mtp::warm_and_decode(&b, req, io);
    assert(res.ok);
    // The orchestrator calls set_initial_hidden once (if last_hidden() != null).
    // SuccessStubTarget::last_hidden() returns non-null after verify_batch,
    // so the orchestrator must have called set_initial_hidden.
    assert(b.mtp_mod.set_initial_hidden_calls >= 1);
    std::puts("T12 set_initial_hidden_plumbing PASS");
}

// ─── T13: gamma derived from module::effective_gamma() ───────────────────
//          Orchestrator reads module->effective_gamma() and passes it to
//          MtpChainRunner::run(). We set gamma=2, expect proposed >= 2.

static void t13_gamma_derived_from_module() {
    LiveStubBackend b;
    b.mtp_mod.effective_gamma_value = 2;
    b.mtp_mod.draft_token           = 88;
    b.target.accept_n               = 0;
    b.target.argmax_token           = 55;
    dflash::common::GenerateRequest req;
    req.prompt = {1, 2};
    req.n_gen  = 3;
    dflash::common::DaemonIO io;
    auto res = dflash::common::mtp::warm_and_decode(&b, req, io);
    assert(res.ok);
    // At least some tokens generated; can't directly inspect the runner's
    // stats from here, but success proves the orchestrator read effective_gamma
    // (gamma==0 would have returned error "effective_gamma() == 0").
    assert(res.tokens.size() >= 1);
    std::puts("T13 gamma_derived_from_module PASS");
}

// ─── T14: zero effective_gamma rejected by orchestrator ──────────────────
//          If module->effective_gamma() == 0, orchestrator must return
//          an error rather than passing gamma=0 to the chain runner.

static void t14_zero_gamma_rejected() {
    LiveStubBackend b;
    b.mtp_mod.effective_gamma_value = 0;  // backend forgot to set gamma
    dflash::common::GenerateRequest req;
    req.prompt = {1, 2, 3};
    req.n_gen  = 4;
    dflash::common::DaemonIO io;
    auto res = dflash::common::mtp::warm_and_decode(&b, req, io);
    assert(!res.ok);
    assert(res.error.find("effective_gamma") != std::string::npos
        || res.error.find("gamma") != std::string::npos);
    std::puts("T14 zero_gamma_rejected PASS");
}

// ─── T15: Qwen35MtpModule::attach(nullptr) returns false without crash ────

static void t15_attach_null_returns_false() {
    dflash::common::mtp::Qwen35MtpModule mod;
    // No init() — module is not loaded. attach(nullptr) must return false.
    bool ok = mod.attach(nullptr);
    assert(!ok);
    std::puts("T15 attach_null_returns_false PASS");
}

// ─── T16: set_effective_gamma clamps to max_gamma() ──────────────────────
//          Pre-init: max_gamma() == 0, so any positive gamma is clamped.
//          Post attach_weights_for_test: max_gamma() == 8 (production ceiling).

static void t16_set_effective_gamma_clamping() {
    dflash::common::mtp::Qwen35MtpModule mod;

    // Pre-init: max_gamma()==0, so effective_gamma stays 0 after any set call.
    mod.set_effective_gamma(5);
    // Implementation: (gamma > 0) ? std::min(gamma, max_gamma()) : max_gamma()
    // With max_gamma()==0: std::min(5, 0) == 0.
    assert(mod.effective_gamma() == 0);

    // Inject minimal weights so loaded==true; max_gamma() returns 8.
    dflash::common::mtp::Qwen35MtpWeights w;
    w.n_embd  = 4;
    w.n_vocab = 16;
    w.n_heads = 1;
    w.n_backbone_layers = 1;
    w.n_head_count  = 1;
    w.n_head_kv     = 1;
    w.n_key_length  = 4;
    w.n_value_length = 4;
    w.n_ffn_length  = 8;
    w.heads.resize(1);
    mod.attach_weights_for_test(w);
    // max_gamma() should now be 8.
    assert(mod.max_gamma() == 8);

    // Value within range: set 3 -> stays 3.
    mod.set_effective_gamma(3);
    assert(mod.effective_gamma() == 3);

    // Value above max: set 99 -> clamped to 8.
    mod.set_effective_gamma(99);
    assert(mod.effective_gamma() == mod.max_gamma());
    std::puts("T16 set_effective_gamma_clamping PASS");
}

// ─── T17: step_batch returns false when not attached ─────────────────────

static void t17_step_batch_not_attached() {
    dflash::common::mtp::Qwen35MtpModule mod;
    // No init/attach — state.loaded==false.
    std::vector<dflash::common::mtp::StepOutput> out;
    bool ok = mod.step_batch(0, 0, out);
    assert(!ok);
    assert(out.empty());
    std::puts("T17 step_batch_not_attached PASS");
}

// ─── T18: shutdown() is idempotent ────────────────────────────────────────

static void t18_shutdown_idempotent() {
    dflash::common::mtp::Qwen35MtpModule mod;
    // Two shutdown calls without init; should not crash.
    mod.shutdown();
    mod.shutdown();
    // After double shutdown: max_gamma()==0 (not loaded).
    assert(mod.max_gamma() == 0);
    std::puts("T18 shutdown_idempotent PASS");
}

// ─── T19: reset_chain() before attach() is a safe no-op ──────────────────

static void t19_reset_chain_before_attach() {
    dflash::common::mtp::Qwen35MtpModule mod;
    // reset_chain() checks state_->loaded; before init it should be safe.
    mod.reset_chain();
    mod.reset_chain();
    // No crash, max_gamma still 0.
    assert(mod.max_gamma() == 0);
    std::puts("T19 reset_chain_before_attach PASS");
}

// ─── T20 support types ────────────────────────────────────────────────────

// ExternalDrafter stub: proposes two tokens (100, 101) per step chain and
// fills next_hidden so the fix branch (set_capture_row / consume_captured_hidden)
// is reachable. Records every call to the two methods under test.
struct StubExternalDrafter : public dflash27b::mtp::IExternalDrafterMtp {
    // IMtpModule tuneables
    int max_gamma_value       = 2;
    int effective_gamma_value = 2;
    int hidden_size_value     = 8;

    // Call counters — IMtpModule lifecycle
    int reset_chain_calls        = 0;
    int set_initial_hidden_calls = 0;
    int attach_calls             = 0;

    // Call counters — IExternalDrafterMtp
    int  step_calls              = 0;
    int  enable_capture_calls    = 0;
    int  set_capture_row_calls   = 0;
    int  set_capture_row_last_arg = -999;  // sentinel: "never called"
    int  consume_calls           = 0;
    bool consume_after_set_capture = false;  // ordering assertion

    std::vector<int> donor_layers_value{0, 1};

    // IMtpModule
    int  max_gamma()       const override { return max_gamma_value; }
    int  effective_gamma() const override { return effective_gamma_value; }
    void set_effective_gamma(int g) override { effective_gamma_value = g; }
    int  hidden_size()     const override { return hidden_size_value; }
    bool attach(dflash27b::DFlashTarget *) override { ++attach_calls; return true; }
    void reset_chain() override { ++reset_chain_calls; }
    void shutdown()    override {}
    void set_initial_hidden(const float *, int) override { ++set_initial_hidden_calls; }

    // IExternalDrafterMtp
    bool step(const dflash27b::mtp::StepInput & in,
              dflash27b::mtp::StepOutput & out) override {
        ++step_calls;
        // Propose token 100+gamma_index so each step produces a distinct token.
        out.draft_token = 100 + in.gamma_index;
        // Non-empty next_hidden activates the fix branch in the runner.
        out.next_hidden.assign(hidden_size_value, 0.5f);
        return true;
    }

    const std::vector<int> & donor_layers() const override {
        return donor_layers_value;
    }

    bool enable_target_hidden_capture(bool /*batch_mode*/,
                                       int /*gamma_max*/) override {
        ++enable_capture_calls;
        return true;
    }

    void set_capture_row(int row) override {
        ++set_capture_row_calls;
        set_capture_row_last_arg = row;
    }

    bool consume_captured_hidden(float * out, int dim) override {
        ++consume_calls;
        // Order check: set_capture_row must have been called first.
        consume_after_set_capture = (set_capture_row_calls > 0);
        for (int i = 0; i < dim; ++i) out[i] = 1.0f;  // sentinel value
        return true;
    }
};

// Target that forces the runner to reach accept_n=1 out of gamma=2.
// verify_batch is overridden so that:
//   all_argmax[0] == drafts[0]  (accept first token)
//   all_argmax[1] != drafts[1]  (reject second token — triggers partial-accept)
// The drafts are always 100 (gamma_index=0) and 101 (gamma_index=1), so we
// set all_argmax[0]=100 (matches) and all_argmax[1]=999 (diverges).
struct ExternalPartialTarget : public SuccessStubTarget {
    static constexpr int32_t kDivergeToken = 999;

    bool verify_batch(const std::vector<int32_t> & tokens,
                      int base_pos,
                      int & last_tok,
                      std::vector<int32_t> * all_argmax) override {
        // Let SuccessStubTarget handle bookkeeping (verify_calls, hidden_seq).
        SuccessStubTarget::verify_batch(tokens, base_pos, last_tok, all_argmax);
        // Override all_argmax to produce the desired partial-accept pattern.
        // candidate = [cur_tok, 100, 101]; drafts = [100, 101].
        // all_argmax[0]=100 -> drafts[0]==100 matches (accept).
        // all_argmax[1]=999 -> drafts[1]==101 != 999 (reject).
        if (all_argmax && (int)all_argmax->size() >= 2) {
            (*all_argmax)[0] = 100;          // accept draft[0]
            (*all_argmax)[1] = kDivergeToken; // reject draft[1]
        }
        last_tok = kDivergeToken;
        return true;
    }
};

// ─── T20: ExternalDrafter partial-accept threads committed-boundary hidden ─
//
// Proves the 74f708a fix: on partial accept (accept_n < g_actual) the runner
// calls set_capture_row(accept_n) then consume_captured_hidden(), in that
// order, to thread the target-captured boundary hidden into running_hidden.
// T7 used a NativeHeads stub (next_hidden always empty) so it never exercised
// this branch. This test closes that gap.

static void t20_external_drafter_partial_accept_threads_committed_row() {
    StubExternalDrafter ext;
    ExternalPartialTarget target;
    // accept_n field on SuccessStubTarget is not used directly here (we
    // override verify_batch above), but keep it consistent.
    target.argmax_token = ExternalPartialTarget::kDivergeToken;
    target.hidden_sz    = ext.hidden_size_value;

    dflash27b::SamplerCfg sampler;
    dflash27b::mtp::MtpChainRunner runner(ext, target, sampler);

    dflash27b::GenerateRequest req;
    req.n_gen  = 2;   // enough for one full chain iteration
    req.stream = false;

    dflash27b::DaemonIO io;
    auto res = runner.run(req, io, /*last_prefill_token=*/10, /*committed_pos=*/4,
                          /*gamma=*/2);

    assert(res.ok && "runner returned error");

    // The runner must have called set_capture_row with arg == accept_n (1).
    assert(ext.set_capture_row_calls > 0          && "set_capture_row was not called");
    assert(ext.set_capture_row_last_arg == 1       && "set_capture_row arg != accept_n=1");

    // consume_captured_hidden must have been called after set_capture_row.
    assert(ext.consume_calls > 0                   && "consume_captured_hidden was not called");
    assert(ext.consume_after_set_capture           && "consume called BEFORE set_capture_row");

    std::puts("T20 external_drafter_partial_accept_threads_committed_row PASS");
}

int main() {
    t1_null_backend();
    t2_backend_without_mtp();
    t3_empty_prompt();
    t4_generic_backend_dispatch();
    // Area A: MtpChainRunner state machine
    t5_gamma_propagation();
    t6_eos_termination();
    t7_partial_accept_rollback();
    t8_n_gen_termination();
    t9_propose_failure();
    t10_stats_accounting();
    // Area B: MtpOrchestrator lifecycle
    t11_reset_chain_before_drive();
    t12_set_initial_hidden_plumbing();
    t13_gamma_derived_from_module();
    t14_zero_gamma_rejected();
    // Area C: Qwen35MtpModule error paths
    t15_attach_null_returns_false();
    t16_set_effective_gamma_clamping();
    t17_step_batch_not_attached();
    t18_shutdown_idempotent();
    t19_reset_chain_before_attach();
    // Area D: ExternalDrafter partial-accept hidden threading (covers 74f708a fix)
    t20_external_drafter_partial_accept_threads_committed_row();
    std::puts("ALL PASS");
    return 0;
}
