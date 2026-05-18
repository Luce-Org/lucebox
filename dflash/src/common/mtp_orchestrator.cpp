#include "common/mtp_orchestrator.h"
#include "common/dflash_target.h"
#include "common/mtp_chain_runner.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dflash27b {
namespace common {
namespace mtp {

namespace {
constexpr int kDefaultPrefillUbatch = 512;

int env_int(const char * name, int defv) {
    if (const char * s = std::getenv(name)) {
        const int v = std::atoi(s);
        if (v > 0) return v;
    }
    return defv;
}
}

GenerateResult warm_and_decode(ModelBackend * backend,
                                const GenerateRequest & req,
                                const DaemonIO & io) {
    GenerateResult result;
    if (!backend) {
        result.error = "warm_and_decode: backend pointer is null";
        return result;
    }
    if (!backend->supports_mtp()) {
        result.error = "warm_and_decode: backend does not support MTP";
        return result;
    }
    if (req.prompt.empty()) {
        result.error = "warm_and_decode: prompt is empty";
        return result;
    }

    dflash27b::mtp::IMtpModule * module = backend->mtp();
    DFlashTarget * target = backend->dflash_target();
    if (!module || !target) {
        result.error = "warm_and_decode: backend missing mtp() or dflash_target()";
        return result;
    }

    const int hidden = target->hidden_size();
    const int prompt_len = (int)req.prompt.size();
    const int prefill_ubatch = env_int("DFLASH27B_PREFILL_UBATCH", kDefaultPrefillUbatch);

    // Capture state is owned by the target+MTP attachment, not the orchestrator.
    // MTP's attach() already enabled+pinned FULL_SEQ; calling here would be a
    // no-op and an architectural smell (orchestrator reaching into target state).
    target->enable_hidden_seq_capture(true);  // idempotent for MTP-bound target

    std::vector<float> all_prefill_hidden((size_t)prompt_len * hidden);
    int32_t last_tok = -1;

    auto t_prefill0 = std::chrono::steady_clock::now();
    for (int start = 0; start < prompt_len;) {
        const int n = std::min(prefill_ubatch, prompt_len - start);
        std::vector<int32_t> chunk(req.prompt.begin() + start,
                                   req.prompt.begin() + start + n);
        if (!target->verify_batch(chunk, start, last_tok, nullptr)) {
            target->enable_hidden_seq_capture(false);
            result.error = "warm_and_decode: verify_batch failed during prefill";
            io.emit(-1);
            return result;
        }
        int n_chunk = 0;
        const float * h_seq = target->last_hidden_seq(&n_chunk);
        // Invariant: capture is enabled+pinned by MTP attach, so verify_batch
        // must return the full chunk. If it doesn't, fail loud rather than
        // silently mangle all_prefill_hidden — clearing it (the pre-fix
        // behavior) made the next chunk's memcpy write past freed memory.
        if (!h_seq || n_chunk != n) {
            result.error = "warm_and_decode: hidden seq capture invariant violated";
            io.emit(-1);
            return result;
        }
        std::memcpy(all_prefill_hidden.data() + (size_t)start * hidden,
                    h_seq, sizeof(float) * (size_t)n * hidden);
        start += n;
    }
    result.prefill_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_prefill0).count();

    // No scope toggle here: MTP-pinned target stays FULL_SEQ for the chain's
    // whole lifetime (partial-accept iters need the COMMITTED row, not the
    // last-candidate row, so LAST_ROW_ONLY would silently return null).

    if (last_tok < 0) {
        result.error = "warm_and_decode: prefill produced invalid argmax";
        io.emit(-1);
        return result;
    }

    module->reset_chain();
    if (target->last_hidden() != nullptr) {
        module->set_initial_hidden(target->last_hidden(), hidden);
    }
    if (module->flavor() == dflash27b::mtp::MtpFlavor::NativeHeads
        && !all_prefill_hidden.empty()) {
        auto * native = dynamic_cast<dflash27b::mtp::INativeMtp *>(module);
        if (native && !native->warm_head_kv(req.prompt.data(), prompt_len,
                                            last_tok, all_prefill_hidden.data())) {
            result.error = "warm_and_decode: warm_head_kv failed";
            io.emit(-1);
            return result;
        }
    }

    // Emit prefill token, then drive chain runner.
    result.tokens.push_back(last_tok);
    io.emit(last_tok);
    if (target->is_eos(last_tok) || req.n_gen <= 1) {
        io.emit(-1);
        result.ok = true;
        return result;
    }

    SamplerCfg sampler = req.sampler;
    GenerateRequest inner;
    inner.n_gen     = req.n_gen - 1;
    inner.stream    = true;
    inner.do_sample = false;
    inner.sampler   = sampler;

    // Single source of truth: backend must have called set_effective_gamma
    // at attach time. effective_gamma() == 0 means the backend forgot — fail
    // loud rather than silently default to max_gamma (the bug class that
    // tanked accept_rate from 0.41 to 0.04 in the earlier orchestrator).
    const int gamma = module->effective_gamma();
    if (gamma <= 0) {
        result.error = "warm_and_decode: module->effective_gamma() == 0 — backend must call set_effective_gamma() during attach";
        io.emit(-1);
        return result;
    }

    auto t_decode0 = std::chrono::steady_clock::now();
    dflash27b::mtp::MtpChainRunner runner(*module, *target, sampler);
    GenerateResult inner_res = runner.run(inner, io, last_tok, prompt_len, gamma);
    result.decode_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_decode0).count();

    if (!inner_res.ok) {
        result.error = "warm_and_decode: chain runner failed: " + inner_res.error;
        return result;
    }

    for (int32_t t : inner_res.tokens) result.tokens.push_back(t);

    const auto & st = runner.stats();
    if (st.total_iters > 0) {
        std::fprintf(stderr,
            "[mtp_decode] iters=%d proposed=%d accepted=%d emitted=%d accept_rate=%.2f\n",
            st.total_iters, st.total_proposed, st.total_accepted, st.total_emitted,
            st.total_proposed > 0
                ? (double)st.total_accepted / (double)st.total_proposed : 0.0);
    }

    result.ok = true;
    return result;
}

}  // namespace mtp
}  // namespace common
}  // namespace dflash27b
