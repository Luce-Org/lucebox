// End-to-end smoke + parity + benchmark tool for Inkling 2 Mini (`inkling2_loop`).
//
// Usage:
//   smoke_inkling2_forward <model.gguf> [--gpu 0] [--expert-gpu -1] [--dspark <mtp.safetensors|off>]
//       [--n-gen 128] [--ctx 8192] [--kv q8_0|f16] [--prompt "text"] [--prompt-len N] [--parity]

#include "inkling2_backend.h"
#include "inkling2_cache.h"
#include "inkling2_graph.h"
#include "inkling2_internal.h"
#include "server/tokenizer.h"

#include "ggml-cuda.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace luce::common;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model.gguf> [--gpu 0] [--expert-gpu -1] [--dspark path|off] [--n-gen 128] [--ctx 8192] [--kv q8_0|f16] [--prompt text] [--prompt-len N] [--parity]\n", argv[0]);
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
    bool parity = false;
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
        else if (std::strcmp(argv[i], "--parity") == 0) parity = true;
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

    std::vector<int32_t> prompt_ids = tok.encode(prompt_text, true);
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
    std::printf("[smoke] prompt_tokens=%zu n_gen=%d gpu=%d expert_gpu=%d kv=%s\n",
                prompt_ids.size(), n_gen, gpu, expert_gpu, ggml_type_name(kv_type));

    Inkling2BackendConfig cfg;
    cfg.model_path = model_path;
    cfg.draft_path = dspark_path;
    cfg.device.gpu = gpu;
    cfg.device.max_ctx = max_ctx;
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

    auto run_one = [&](bool force_ar, const char * label) -> GenerateResult {
        GenerateRequest req;
        req.prompt = prompt_ids;
        req.n_gen = n_gen;
        req.sampler.temp = 0.0f; // greedy
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
            r = run_one(false, lbl);
            if (!r.ok()) return 1;
        }
        return 0;
    }
}
