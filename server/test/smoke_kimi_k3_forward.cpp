#include "kimi_k3/kimi_k3_backend.h"
#include "server/tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace dflash::common;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <kimi-k3.gguf> [gpu=0] [n_gen=16] [prompt] "
            "[stream_experts=1] [expert_gpu=-1] [draft.gguf] "
            "[draft_gpu=0]\n",
            argv[0]);
        return 2;
    }
    const char * model = argv[1];
    const int gpu = argc > 2 ? std::atoi(argv[2]) : 0;
    const int n_gen = argc > 3 ? std::atoi(argv[3]) : 16;
    const std::string prompt = argc > 4
        ? argv[4]
        : "According to all known laws";

    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(model)) {
        std::fprintf(stderr, "[kimi-k3-smoke] tokenizer load failed\n");
        return 1;
    }
    std::vector<int32_t> prompt_ids = tokenizer.encode(prompt);
    if (prompt_ids.empty()) {
        std::fprintf(stderr, "[kimi-k3-smoke] prompt tokenized to zero IDs\n");
        return 1;
    }

    KimiK3BackendConfig config;
    config.model_path = model;
    config.device.gpu = gpu;
    config.device.max_ctx = 4096;
    config.moe_storage = argc <= 5 || std::atoi(argv[5]) != 0
        ? MoeStoragePolicy::Ssd
        : MoeStoragePolicy::Resident;
    config.expert_gpu = argc > 6 ? std::atoi(argv[6]) : -1;
    config.draft_path = argc > 7 && argv[7][0] != '\0' ? argv[7] : nullptr;
    config.draft_gpu = argc > 8 ? std::atoi(argv[8]) : gpu;
    KimiK3Backend backend(config);
    if (!backend.init()) return 1;

    GenerateRequest request;
    request.prompt = prompt_ids;
    request.n_gen = n_gen;
    request.do_sample = false;
    DaemonIO io;
    GenerateResult result = backend.generate(request, io);
    if (!result.ok()) {
        std::fprintf(stderr, "[kimi-k3-smoke] generation failed: %s (%s)\n",
                     std::string(result.error_code()).c_str(),
                     std::string(result.error_detail()).c_str());
        return 1;
    }

    std::printf("[kimi-k3-smoke] prompt_ids:");
    for (int32_t id : prompt_ids) std::printf(" %d", id);
    std::printf("\n[kimi-k3-smoke] output_ids:");
    for (int32_t id : result.tokens) std::printf(" %d", id);
    std::printf("\n[kimi-k3-smoke] text: %s%s\n",
                prompt.c_str(), tokenizer.decode(result.tokens).c_str());
    const double prefill_rate = result.prefill_s > 0.0
        ? static_cast<double>(prompt_ids.size()) / result.prefill_s : 0.0;
    const double decode_rate = result.decode_s > 0.0
        ? static_cast<double>(result.tokens.size()) / result.decode_s : 0.0;
    std::printf("[kimi-k3-smoke] prefill=%.3fs (%.2f tok/s) "
                "decode=%.3fs (%.2f tok/s)\n",
                result.prefill_s, prefill_rate,
                result.decode_s, decode_rate);
    return 0;
}
