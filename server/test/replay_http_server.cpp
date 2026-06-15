// CPU-only test driver for dflash HttpServer.
//
// Boots the server with the real Qwen3.6 tokenizer (loaded from a GGUF —
// vocab/metadata only, never any GPU weights) and a deterministic
// StubModelBackend whose generate() replays scripted token streams from
// JSON scenario files. The chain
//   real chat template renderer → real ParsedRequest plumbing →
//   real SseEmitter wiring → real socket writes
// is fully exercised; only the per-token sample comes from the stub.
//
// Runs without a GPU: link against dflash_common (CUDA TUs included), but
// because no real ModelBackend is instantiated, ggml_cuda_init() is never
// called. CUDA_VISIBLE_DEVICES="" is the supported test configuration.
//
// Usage:
//   CUDA_VISIBLE_DEVICES="" ./replay_http_server \
//       <vocab.gguf> --scenarios <dir> [--port 9999]
//
// See server/test/scenarios/*.json for the scenario file schema.

#include "server/http_server.h"
#include "server/tokenizer.h"
#include "server/chat_template.h"
#include "common/model_backend.h"
#include "scenario_store.h"
#include "stub_model_backend.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace dflash::common;

namespace {

HttpServer * g_server = nullptr;

void signal_handler(int) {
    if (g_server) g_server->request_stop();
}

}  // namespace

int main(int argc, char ** argv) {
    const char * gguf_path     = nullptr;
    const char * scenarios_dir = nullptr;
    int          port          = 9999;
    int          max_ctx       = 4096;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--scenarios") == 0 && i + 1 < argc) {
            scenarios_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--max-ctx") == 0 && i + 1 < argc) {
            max_ctx = std::atoi(argv[++i]);
        } else if (argv[i][0] != '-' && !gguf_path) {
            gguf_path = argv[i];
        } else {
            std::fprintf(stderr,
                "usage: %s <vocab.gguf> --scenarios <dir> [--port N] [--max-ctx N]\n",
                argv[0]);
            return 1;
        }
    }
    if (!gguf_path) {
        std::fprintf(stderr, "[driver] missing positional <vocab.gguf>\n");
        return 1;
    }

    const char * cvd = std::getenv("CUDA_VISIBLE_DEVICES");
    std::fprintf(stderr, "[driver] CUDA_VISIBLE_DEVICES=%s\n",
                 cvd ? cvd : "(unset)");

    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(gguf_path)) {
        std::fprintf(stderr, "[driver] tokenizer load failed: %s\n", gguf_path);
        return 2;
    }
    std::fprintf(stderr, "[driver] tokenizer loaded: vocab=%d\n",
                 tokenizer.vocab_size());

    test::ScenarioStore store;
    if (scenarios_dir) {
        if (!store.load_directory(scenarios_dir)) {
            std::fprintf(stderr,
                "[driver] one or more scenarios failed to load — aborting\n");
            return 3;
        }
        std::fprintf(stderr, "[driver] loaded %zu scenarios from %s\n",
            store.size(), scenarios_dir);
    } else {
        std::fprintf(stderr,
            "[driver] no --scenarios dir given; every request will 500\n");
    }

    test::StubModelBackend backend(store, tokenizer);

    ServerConfig cfg;
    cfg.host       = "127.0.0.1";
    cfg.port       = port;
    cfg.model_name = "dflash";    // matches existing test-suite default
    cfg.max_ctx    = max_ctx;

    HttpServer server(backend, tokenizer, cfg);
    server.set_chat_format(ChatFormat::QWEN3);

    g_server = &server;
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT,  signal_handler);

    std::fprintf(stderr,
        "[driver] HttpServer listening on http://%s:%d (SIGINT/SIGTERM to stop)\n",
        cfg.host.c_str(), cfg.port);
    int rc = server.run();
    std::fprintf(stderr, "[driver] exit rc=%d\n", rc);
    return rc;
}
