#include "lfm2_backend.h"
#include "qwen2_backend.h"
#include "model_selection.h"
#include "gguf.h"
#include "engine/luce_engine.h"
#include "server/http_server.h"
#include <csignal>
#include <cstdlib>
#include <iostream>
using namespace luce::common;
static HttpServer * server = nullptr;
static void stop(int) { if (server) server->request_stop(); }
int main(int argc, char ** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: lucebox-vulkan MODEL [--port 18080] [--ctx 4096] [--chunk 256] [--flash 0|1]\n";
            return 2;
        }
        int port = 18080, ctx = 4096, chunk = 256;
        bool flash = false;
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::runtime_error("missing option value");
            const std::string key = argv[i];
            size_t end = 0;
            const int value = std::stoi(argv[i + 1], &end);
            if (argv[i + 1][end]) throw std::runtime_error("invalid option value");
            if (key == "--port") port = value;
            else if (key == "--ctx") ctx = value;
            else if (key == "--chunk") chunk = value;
            else if (key == "--flash" && (value == 0 || value == 1)) flash = value;
            else throw std::runtime_error("unknown option");
        }
        if (port < 1024 || port > 65535 || ctx < 128 || ctx > 32768 || chunk < 1 || chunk > 512)
            throw std::runtime_error("option out of bounds");
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(
            gguf_init_from_file(argv[1], {true, nullptr}), gguf_free);
        if (!meta) throw std::runtime_error("GGUF metadata read failed");
        const auto ai = gguf_find_key(meta.get(), "general.architecture");
        if (ai < 0 || gguf_get_kv_type(meta.get(), ai) != GGUF_TYPE_STRING)
            throw std::runtime_error("missing architecture");
        const std::string arch = gguf_get_val_str(meta.get(), ai);
        const bool lfm = vulkan_model(arch) == VulkanModel::Lfm2Moe;
        Tokenizer tokenizer;
        if (!tokenizer.load_from_gguf(argv[1])) throw std::runtime_error("tokenizer load failed");
        auto backend = lfm ? make_lfm2_vulkan(argv[1], ctx, chunk, flash)
                           : make_qwen2_vulkan(argv[1], ctx, chunk, flash);
        luce::engine::LuceEngine engine(std::move(backend));
        ServerConfig cfg;
        cfg.host = "127.0.0.1"; cfg.port = port;
        cfg.model_name = lfm ? "lfm2.5-8b-a1b-vulkan" : "qwen2.5-vulkan";
        cfg.model_path = argv[1]; cfg.arch = arch;
        cfg.runtime_backend = lfm ? "native_lfm2_vulkan" : "native_qwen2_vulkan";
        cfg.target_device = "vulkan:0"; cfg.kv_cache_k = "f16"; cfg.kv_cache_v = "f16";
        cfg.max_ctx = ctx; cfg.chunk = chunk;
        cfg.prefix_cache_cap = 0; cfg.prefill_cache_cap = 0; cfg.ppp_enabled = false;
        cfg.decode_kv_offload_bytes = 0; cfg.default_max_tokens = 1024;
        cfg.max_tokens = 1024; cfg.hard_limit_reply_budget = 0;
        cfg.sampler_defaults.temperature = 0;
        const auto ti = gguf_find_key(meta.get(), "tokenizer.chat_template");
        if (ti >= 0 && gguf_get_kv_type(meta.get(), ti) == GGUF_TYPE_STRING) {
            cfg.chat_template_src = gguf_get_val_str(meta.get(), ti);
            cfg.chat_template_path = "GGUF:tokenizer.chat_template";
        } else if (lfm) throw std::runtime_error("missing GGUF chat template");
        HttpServer http(engine, tokenizer, cfg);
        http.set_chat_format(ChatFormat::QWEN3);
        server = &http;
        std::signal(SIGINT, stop); std::signal(SIGTERM, stop);
        const int result = http.run(); server = nullptr; http.shutdown();
        return result;
    } catch (const std::exception & e) {
        std::cerr << "lucebox-vulkan: " << e.what() << "\n";
        return 1;
    }
}
