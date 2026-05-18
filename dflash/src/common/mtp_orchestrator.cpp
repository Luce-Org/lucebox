#include "common/mtp_orchestrator.h"
#include "common/dflash_target.h"
#include "common/mtp_chain_runner.h"
#include "common/prefix_snap.h"

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
                                const DaemonIO & io,
                                std::function<void()> on_warm_done) {
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

    // Resolve the snap boundary. Server.py's prefix_cache picks
    // target_cut = second-to-last <|im_start|> boundary (almost always
    // mid-prompt), so honoring snap_pos < prompt_len is the production path.
    // snap_pos == prompt_len (end-of-prompt) is also supported as a side
    // effect — both go through the same partial-warm logic with snap_at
    // landing where the server asked.
    const int snap_at = (req.snap_slot >= 0 && req.snap_pos > 0 &&
                          req.snap_pos <= prompt_len) ? req.snap_pos : -1;

    // Capture state is owned by the target+MTP attachment, not the orchestrator.
    // MTP's attach() already enabled+pinned FULL_SEQ; calling here would be a
    // no-op and an architectural smell (orchestrator reaching into target state).
    target->enable_hidden_seq_capture(true);  // idempotent for MTP-bound target

    std::vector<float> all_prefill_hidden((size_t)prompt_len * hidden);
    int32_t last_tok = -1;

    auto * native = (module->flavor() == dflash27b::mtp::MtpFlavor::NativeHeads)
                  ? dynamic_cast<dflash27b::mtp::INativeMtp *>(module)
                  : nullptr;
    bool snap_done = false;

    auto t_prefill0 = std::chrono::steady_clock::now();
    for (int start = 0; start < prompt_len;) {
        int n = std::min(prefill_ubatch, prompt_len - start);
        // Clip chunk so we land EXACTLY at snap_at when it falls inside this
        // ubatch — the partial-warm + snap fires between chunks, so the cache
        // state must be at cur_pos == snap_at when we take the snapshot.
        if (snap_at > 0 && start < snap_at && start + n > snap_at) {
            n = snap_at - start;
        }
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

        // Mid-prompt snap: partial-warm head_kv up to snap_at, then snapshot.
        // Slot snap_at gets filled with input prompt[snap_at] (the first
        // post-cut token, may differ across requests); the chain runner
        // overwrites that slot on its first decode step, so it can't corrupt
        // restored decode. Slots [1..snap_at-1] depend only on cached prefix
        // tokens and are byte-identical across calls. The corresponding
        // RESTORE-side partial-WARM is gated to cold-restart in
        // Qwen35Backend::restore_and_generate until range-warm lands.
        if (snap_at > 0 && snap_at < prompt_len && start == snap_at && !snap_done) {
            if (native) {
                const int32_t partial_next = req.prompt[snap_at];
                if (!native->warm_head_kv(req.prompt.data(), snap_at,
                                          partial_next,
                                          all_prefill_hidden.data())) {
                    result.error = "warm_and_decode: partial warm_head_kv failed";
                    io.emit(-1);
                    return result;
                }
            }
            if (on_warm_done) on_warm_done();  // flag flip; idempotent
            if (backend->snapshot_save(req.snap_slot)) {
                emit_inline_snap_ack(req.snap_slot, snap_at);
            }
            snap_done = true;
        }
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
    // Full warm covers [0..prompt_len). When partial-warm landed at snap_at,
    // this overwrites slots [1..snap_at) with byte-identical values (same
    // inputs) and fills [snap_at..prompt_len). Skip only when snap_at ==
    // prompt_len — partial warm already covered everything.
    if (native && !all_prefill_hidden.empty() &&
        !(snap_done && snap_at == prompt_len)) {
        if (!native->warm_head_kv(req.prompt.data(), prompt_len,
                                  last_tok, all_prefill_hidden.data())) {
            result.error = "warm_and_decode: warm_head_kv failed";
            io.emit(-1);
            return result;
        }
    }

    // Post-warm hook fires after head_kv is fully populated and BEFORE chain
    // decode mutates cache_.cur_pos beyond prompt_len. Backends use this to
    // flip the head_kv_warm_ flag; the inline snap is taken inline above
    // (mid-prompt) or right below (end-of-prompt).
    if (on_warm_done) on_warm_done();

    // End-of-prompt snap: snap_at == prompt_len means the server asked for a
    // snapshot at the final position. Less common than mid-prompt (server
    // usually cuts before the current turn) but supported for completeness.
    if (snap_at == prompt_len && !snap_done) {
        if (backend->snapshot_save(req.snap_slot)) {
            emit_inline_snap_ack(req.snap_slot, prompt_len);
        }
        snap_done = true;
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
