// End-to-end smoke + parity + needle + concurrent benchmark tool for Inkling 2 Mini (`inkling2_loop`).
//
// Usage:
//   smoke_inkling2_forward <model.gguf> [--gpu 0] [--expert-gpu -1] [--dspark <path|off>]
//       [--n-gen 128] [--ctx 8192] [--kv q8_0|f16] [--prompt "text"] [--prompt-len N]
//       [--parity] [--ar] [--needle-test] [--needle-pos N] [--concurrency N] [--concurrent] [--contract]

#include "inkling2_backend.h"
#include "inkling2_cache.h"
#include "inkling2_graph.h"
#include "inkling2_internal.h"
#include "common/concurrency/seq_engine.h"
#include "seq_engine_contract.h"
#include "server/tokenizer.h"

#include "ggml-cuda.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace luce::common;

static std::vector<int32_t> build_needle_prompt(
        const Tokenizer & tok,
        int target_len,
        int needle_pos_tok,
        const char * array_id,
        const char * proj_name,
        const char * passcode) {
    const std::string hdr = "<|message_user|><|content_text|>Read the following technical log carefully and answer the question at the very end.\n\n";
    const std::string filler =
        std::string("LOG ENTRY: Routine subsystem heartbeat verified across sensor array ") + array_id + ". "
        "Thermal readings remained within nominal bounds (34.2 C to 36.8 C), power rails stable at 12.01 V, "
        "memory scrubbing completed with zero correctable ECC events, and network ring buffers flushed cleanly.\n";
    const std::string needle =
        std::string("\nCRITICAL VAULT RECORD: The secret launch passcode for ") + proj_name + " is " + passcode + ". "
        "Store this exact passcode " + passcode + " for verification.\n\n";
    const std::string question =
        std::string("\nQuestion: What is the exact secret launch passcode for ") + proj_name +
        " mentioned in the CRITICAL VAULT RECORD above? Reply with the passcode clearly.<|end_message|><|message_model|><|content_text|>";
    auto ids_hdr    = tok.encode(hdr, true);
    auto ids_filler = tok.encode(filler, false);
    auto ids_needle = tok.encode(needle, false);
    auto ids_q      = tok.encode(question, false);
    std::vector<int32_t> out = ids_hdr;
    bool inserted = false;
    while ((int) (out.size() + ids_q.size()) < target_len) {
        if (!inserted && (int) out.size() >= needle_pos_tok) {
            out.insert(out.end(), ids_needle.begin(), ids_needle.end());
            inserted = true;
        }
        out.insert(out.end(), ids_filler.begin(), ids_filler.end());
    }
    if (!inserted) out.insert(out.end(), ids_needle.begin(), ids_needle.end());
    out.insert(out.end(), ids_q.begin(), ids_q.end());
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model.gguf> [--gpu 0] [--expert-gpu -1] [--dspark path|off] [--n-gen 128] [--ctx 8192] [--kv q8_0|f16] [--prompt text] [--prompt-len N] [--parity] [--ar] [--needle-test] [--needle-pos N] [--concurrency N] [--contract]\n", argv[0]);
        return 1;
    }
    std::string model_path = argv[1];
    int gpu = 0;
    int expert_gpu = -1;
    std::optional<std::string> dspark_path;
    int n_gen = 128;
    int max_ctx = 8192;
    int chunk = 0;
    int reps = 1;
    int prompt_len = 0;
    int concurrency = 1;
    bool parity = false;
    bool force_ar_cli = false;
    bool needle_test = false;
    int needle_pos_tok = 1500;
    bool run_contract = false;
    float temp = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    uint32_t seed = 0;
    ggml_type kv_type = GGML_TYPE_Q8_0;
    std::string prompt_text = "Write a concise Python function to compute the nth Fibonacci number using dynamic programming, and explain its time complexity.";

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) gpu = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--expert-gpu") == 0 && i + 1 < argc) expert_gpu = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--dspark") == 0 && i + 1 < argc) dspark_path = std::string(argv[++i]);
        else if (std::strcmp(argv[i], "--n-gen") == 0 && i + 1 < argc) n_gen = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) max_ctx = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) chunk = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--prompt-len") == 0 && i + 1 < argc) prompt_len = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt_text = argv[++i];
        else if ((std::strcmp(argv[i], "--concurrency") == 0 || std::strcmp(argv[i], "--max-concurrency") == 0) && i + 1 < argc) {
            concurrency = std::max(1, std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--concurrent") == 0) {
            if (concurrency < 2) concurrency = 2;
        } else if (std::strcmp(argv[i], "--contract") == 0) run_contract = true;
        else if (std::strcmp(argv[i], "--parity") == 0) parity = true;
        else if (std::strcmp(argv[i], "--ar") == 0) force_ar_cli = true;
        else if (std::strcmp(argv[i], "--needle-test") == 0) needle_test = true;
        else if (std::strcmp(argv[i], "--needle-pos") == 0 && i + 1 < argc) needle_pos_tok = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--temp") == 0 && i + 1 < argc) temp = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) top_p = std::strtof(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t) std::strtoul(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--kv") == 0 && i + 1 < argc) {
            ++i;
            kv_type = (std::strcmp(argv[i], "f16") == 0) ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
        }
    }

    Tokenizer tok;
    if (!tok.load_from_gguf(model_path.c_str())) {
        std::fprintf(stderr, "Failed to load tokenizer from %s\n", model_path.c_str());
        return 1;
    }

    std::vector<int32_t> prompt_ids;
    if (needle_test) {
        const int target_len = prompt_len > 0 ? prompt_len : 12000;
        prompt_ids = build_needle_prompt(tok, target_len, needle_pos_tok, "4B", "Project Cobalt-9", "AURORA-7392-QUARTZ");
    } else {
        prompt_ids = tok.encode(prompt_text, true);
        if (prompt_len > 0) {
            std::vector<int32_t> rep;
            rep.reserve(prompt_len);
            while ((int) rep.size() < prompt_len) {
                for (int32_t id : prompt_ids) {
                    if ((int) rep.size() < prompt_len) rep.push_back(id);
                }
            }
            prompt_ids = std::move(rep);
        }
    }
    std::printf("[smoke] prompt_tokens=%zu n_gen=%d gpu=%d expert_gpu=%d kv=%s\n",
                prompt_ids.size(), n_gen, gpu, expert_gpu, ggml_type_name(kv_type));

    Inkling2BackendConfig cfg;
    cfg.model_path = model_path;
    cfg.draft_path = dspark_path;
    cfg.device.gpu = gpu;
    cfg.device.max_ctx = max_ctx;
    cfg.max_concurrency = (run_contract && concurrency < 2) ? 2 : concurrency;
    if (chunk > 0) cfg.chunk = chunk;
    cfg.cache_type_k = kv_type;
    cfg.cache_type_v = kv_type;
    if (expert_gpu >= 0) {
        DevicePlacement ep;
        ep.gpu = expert_gpu;
        cfg.expert_device = ep;
    }

    Inkling2Backend backend(cfg);
    if (!backend.init()) {
        std::fprintf(stderr, "Inkling2Backend init failed\n");
        return 1;
    }
    backend.print_ready_banner();

    if (run_contract) {
        SeqEngine * eng = backend.seq_engine();
        if (!eng) {
            std::fprintf(stderr, "[CONTRACT] FAIL: seq_engine() returned null\n");
            return 1;
        }
        auto violations = check_seq_engine_contract(*eng);
        std::printf("[CONTRACT] violations=%zu\n", violations.size());
        for (const auto & v : violations) {
            std::printf("[CONTRACT]   - %s\n", v.c_str());
        }
        if (!violations.empty()) return 1;
        if (concurrency <= 1) return 0;
    }

    if (concurrency > 1) {
        SeqEngine * eng = backend.seq_engine();
        if (!eng) {
            std::fprintf(stderr, "[CONC] FAIL: seq_engine() returned null\n");
            return 1;
        }
        const int N = std::min(concurrency, eng->slot_count());
        const char * kPrompts[4] = {
            "Write a concise Python function to compute the nth Fibonacci number using dynamic programming, and explain its time complexity.",
            "Explain how a B-tree differs from an LSM-tree for write-heavy database workloads in three short bullet points.",
            "Implement binary search in C++ with clean loop invariants and show an example call.",
            "Summarize the difference between TCP and QUIC connection establishment and head-of-line blocking."
        };
        const char * kArrays[4]    = { "4B", "7C", "2A", "9F" };
        const char * kProjects[4]  = { "Project Cobalt-9", "Project Obsidian-4", "Project Helios-7", "Project Zenith-2" };
        const char * kPasscodes[4] = { "AURORA-7392-QUARTZ", "TITAN-4819-SAPPHIRE", "SOLARIS-6104-BERYL", "VORTEX-2951-TOPAZ" };

        std::vector<std::vector<int32_t>> prompts((size_t) N);
        for (int s = 0; s < N; ++s) {
            if (needle_test) {
                const int target_len = prompt_len > 0 ? prompt_len : 12000;
                prompts[(size_t) s] = build_needle_prompt(
                    tok, target_len, needle_pos_tok, kArrays[s % 4], kProjects[s % 4], kPasscodes[s % 4]);
            } else {
                prompts[(size_t) s] = tok.encode(kPrompts[s % 4], true);
                if (prompt_len > 0) {
                    std::vector<int32_t> rep;
                    rep.reserve((size_t) prompt_len);
                    while ((int) rep.size() < prompt_len) {
                        for (int32_t id : prompts[(size_t) s]) {
                            if ((int) rep.size() < prompt_len) rep.push_back(id);
                        }
                    }
                    prompts[(size_t) s] = std::move(rep);
                }
            }
        }
        SamplerCfg samp;
        samp.temp = temp;
        samp.top_k = top_k;
        samp.top_p = top_p;
        samp.seed = seed;

        for (int rep = 0; rep < reps; ++rep) {
            std::vector<int> slots((size_t) N, -1);
            std::vector<bool> decoding((size_t) N, false);
            std::vector<bool> done((size_t) N, false);
            std::vector<int32_t> pending_tok((size_t) N, -1);
            std::vector<std::vector<int32_t>> out_toks((size_t) N);

            for (int s = 0; s < N; ++s) {
                auto adm = eng->admit((uint64_t) (rep * 100 + s + 1), prompts[(size_t) s], samp);
                if (adm.status != SeqEngine::AdmitResult::Status::admitted) {
                    std::fprintf(stderr, "[CONC] admit failed for stream %d: %s\n", s, adm.error.c_str());
                    return 1;
                }
                slots[(size_t) s] = adm.slot;
            }

            const auto t_all0 = std::chrono::steady_clock::now();
            auto t_pure_dec0 = t_all0;
            bool all_prefilled = false;
            int steps = 0;
            int pure_steps = 0;
            long long total_toks = 0;
            long long pure_toks = 0;
            while (true) {
                SeqEngine::StepPlan plan;
                for (int s = 0; s < N; ++s) {
                    if (decoding[(size_t) s] && !done[(size_t) s]) {
                        plan.decode.push_back({ slots[(size_t) s], pending_tok[(size_t) s], !force_ar_cli });
                    }
                }
                int next_pre = -1;
                for (int s = 0; s < N; ++s) {
                    if (!decoding[(size_t) s] && !done[(size_t) s]) { next_pre = s; break; }
                }
                if (next_pre >= 0) {
                    auto lim = eng->step_plan_limits((int) plan.decode.size());
                    plan.prefills.push_back({ slots[(size_t) next_pre], lim.max_prefill_tokens_per_sequence });
                } else if (!all_prefilled) {
                    all_prefilled = true;
                    t_pure_dec0 = std::chrono::steady_clock::now();
                }
                if (plan.prefills.empty() && plan.decode.empty()) break;

                auto sr = eng->step(plan);
                std::string val_err = validate_step_result(plan, sr, eng->slot_count());
                if (!sr.ok() || !val_err.empty()) {
                    std::fprintf(stderr, "[CONC] step failed: err=%s val=%s\n",
                                 sr.error.c_str(), val_err.c_str());
                    return 1;
                }
                for (const auto & po : sr.prefills) {
                    int s = -1;
                    for (int i = 0; i < N; ++i) if (slots[(size_t) i] == po.slot) { s = i; break; }
                    if (s >= 0 && po.status == SeqEngine::PrefillOutput::Status::completed) {
                        decoding[(size_t) s] = true;
                        pending_tok[(size_t) s] = po.token;
                        out_toks[(size_t) s].push_back(po.token);
                        ++total_toks;
                        if (eng->token_is_eos(po.token) || (int) out_toks[(size_t) s].size() >= n_gen) {
                            done[(size_t) s] = true;
                        }
                    }
                }
                if (!plan.decode.empty()) {
                    ++steps;
                    if (all_prefilled) ++pure_steps;
                }
                for (const auto & d : sr.decode) {
                    int s = -1;
                    for (int i = 0; i < N; ++i) if (slots[(size_t) i] == d.slot) { s = i; break; }
                    if (s < 0 || d.failed) {
                        std::fprintf(stderr, "[CONC] slot %d decode failed: %s\n", d.slot, d.error.c_str());
                        return 1;
                    }
                    auto push_tok = [&](int32_t tk) {
                        if (done[(size_t) s]) return;
                        out_toks[(size_t) s].push_back(tk);
                        ++total_toks;
                        if (all_prefilled) ++pure_toks;
                        if (eng->token_is_eos(tk) || (int) out_toks[(size_t) s].size() >= n_gen) {
                            done[(size_t) s] = true;
                        }
                    };
                    for (int32_t tk : d.committed_tokens) push_tok(tk);
                    push_tok(d.token);
                    pending_tok[(size_t) s] = d.token;
                }
                for (int s = 0; s < N; ++s) {
                    if (done[(size_t) s] && slots[(size_t) s] >= 0) {
                        eng->retire(slots[(size_t) s]);
                        slots[(size_t) s] = -1;
                    }
                }
            }
            const auto t_dec1 = std::chrono::steady_clock::now();
            const double wall_s = std::chrono::duration<double>(t_dec1 - t_all0).count();
            const double pure_s = all_prefilled ? std::chrono::duration<double>(t_dec1 - t_pure_dec0).count() : wall_s;
            for (int s = 0; s < N; ++s) {
                if (slots[(size_t) s] >= 0) eng->retire(slots[(size_t) s]);
            }

            std::printf("[CONC N=%d rep=%d] total_toks=%lld (pure=%lld in %d steps, %.3fs) wall=%.3fs | pure_agg=%.1f tok/s (%.1f tok/s/stream)\n",
                        N, rep, total_toks, pure_toks, pure_steps, pure_s, wall_s,
                        pure_s > 0 ? pure_toks / pure_s : 0.0,
                        pure_s > 0 ? (pure_toks / pure_s) / N : 0.0);
            if (rep == reps - 1) {
                for (int s = 0; s < N; ++s) {
                    std::string txt = tok.decode(out_toks[(size_t) s]);
                    std::printf("[CONC stream %d (%zu tok)] %.400s\n---\n", s, out_toks[(size_t) s].size(), txt.c_str());
                }
            }
        }
        return 0;
    }

    auto run_one = [&](bool force_ar, const char * label) -> GenerateResult {
        GenerateRequest req;
        req.prompt = prompt_ids;
        req.n_gen = n_gen;
        req.sampler.temp = temp;
        req.sampler.top_k = top_k;
        req.sampler.top_p = top_p;
        req.sampler.seed = seed;
        req.force_ar_decode = force_ar;
        DaemonIO io;
        GenerateResult r = backend.generate(req, io);
        std::printf("[%s] ok=%d tokens=%zu prefill=%.3fs (%.1f tok/s) decode=%.3fs (%.1f tok/s) spec=%d accept=%.1f%%\n",
                    label, r.ok() ? 1 : 0, r.tokens.size(),
                    r.prefill_s, r.prefill_s > 0 ? prompt_ids.size() / r.prefill_s : 0.0,
                    r.decode_s, r.decode_s > 0 ? r.tokens.size() / r.decode_s : 0.0,
                    r.spec_decode_ran ? 1 : 0, 100.0 * r.accept_rate);
        std::string text = tok.decode(r.tokens);
        std::printf("[%s] text:\n%s\n---\n", label, text.substr(0, 600).c_str());
        return r;
    };

    if (parity) {
        GenerateResult ar = run_one(true, "AR");
        GenerateResult sp = run_one(false, "DSPARK");
        if (!ar.ok() || !sp.ok()) return 1;
        size_t match = 0;
        const size_t lim = std::min(ar.tokens.size(), sp.tokens.size());
        while (match < lim && ar.tokens[match] == sp.tokens[match]) ++match;
        std::printf("[PARITY] matched=%zu/%zu (%s)\n", match, lim, (match == lim && ar.tokens.size() == sp.tokens.size()) ? "BIT-EXACT!" : "DIFF");
        if (match < lim) {
            std::printf("[PARITY] first diff at %zu: ar=%d ('%s') vs sp=%d ('%s')\n",
                        match, ar.tokens[match], tok.decode({ar.tokens[match]}).c_str(),
                        sp.tokens[match], tok.decode({sp.tokens[match]}).c_str());
        }
        return (match == lim && ar.tokens.size() == sp.tokens.size()) ? 0 : 2;
    } else {
        GenerateResult r;
        for (int rep = 0; rep < reps; ++rep) {
            char lbl[32];
            std::snprintf(lbl, sizeof(lbl), reps > 1 ? "RUN%d" : "RUN", rep);
            r = run_one(force_ar_cli, lbl);
            if (!r.ok()) return 1;
        }
        return 0;
    }
}
