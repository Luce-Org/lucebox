// dflash_server — native C++ HTTP server for dflash::common.
//
// Replaces the Python server.py for production use. Owns the target
// ModelBackend directly, while optional draft/PFlash IPC paths can be used
// for mixed-backend placement. The direct target path enables:
//   - Immediate client-disconnect cancellation (via send() failure)
//   - Lower latency (no IPC overhead)
//   - Single binary deployment
//
// Usage:
//   dflash_server <model.gguf> [--draft <draft.gguf>] [--port 8080]
//                              [--host 0.0.0.0] [--max-ctx 131072]
//                              [--max-tokens 4096] [--target-device auto:0]

#include "http_server.h"
#include "chat_template.h"
#include "common/backend_factory.h"
#include "common/gguf_inspect.h"
#include "common/peer_access.h"
#include "placement/pflash_placement.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace dflash::common;

// Global server pointer for signal handling.
static HttpServer * g_server = nullptr;

static void signal_handler(int sig) {
    (void)sig;
    if (g_server) {
        g_server->request_stop();
    }
}

static bool parse_double_list(const char * value, std::vector<double> & out) {
    out.clear();
    if (!value || !*value) return false;
    const char * p = value;
    while (*p) {
        char * end = nullptr;
        double v = std::strtod(p, &end);
        if (end == p) return false;
        out.push_back(v);
        if (*end == '\0') return true;
        if (*end != ',') return false;
        p = end + 1;
        if (!*p) return false;
    }
    return !out.empty();
}

static bool validate_server_placement(const BackendArgs & bargs,
                                      const ServerConfig & sconfig) {
    const PlacementBackend compiled = compiled_placement_backend();
    if (!placement_backend_supported(bargs.device.backend)) {
        std::fprintf(stderr,
            "[server] --target-device=%s is unsupported in this binary "
            "(compiled backend: %s)\n",
            placement_device_name(bargs.device).c_str(),
            placement_backend_name(compiled));
        return false;
    }
    const bool pflash_enabled =
        sconfig.pflash_mode != ServerConfig::PflashMode::OFF;
    const PFlashDrafterPlacement pflash_placement =
        resolve_pflash_drafter_placement(
            bargs.device, bargs.draft_device, bargs.remote_draft,
            pflash_enabled);
    const PlacementBackend target = pflash_placement.target_backend;
    const PlacementBackend draft = pflash_placement.drafter_backend;
    const bool draft_placement_used =
        pflash_drafter_placement_used(pflash_enabled, bargs.draft_path != nullptr);
    if (!bargs.remote_draft.enabled() && bargs.remote_draft.has_aux_options()) {
        std::fprintf(stderr,
            "[server] --draft-ipc-work-dir and --draft-ipc-ring-cap require "
            "--draft-ipc-bin\n");
        return false;
    }
    if (draft_placement_used && target != draft) {
        if (!bargs.remote_draft.enabled()) {
            std::fprintf(stderr,
                "[server] mixed target/draft backends require --draft-ipc-bin "
                "(target=%s draft=%s)\n",
                placement_backend_name(target), placement_backend_name(draft));
            return false;
        }
        if (!bargs.draft_path && !pflash_enabled) {
            std::fprintf(stderr,
                "[server] mixed target/draft backends require --draft <path> "
                "or --prefill-compression with --prefill-drafter\n");
            return false;
        }
    } else if (bargs.remote_draft.enabled()) {
        std::fprintf(stderr,
            "[server] --draft-ipc-bin is only needed for mixed target/draft "
            "backends (target=%s draft=%s)\n",
            placement_backend_name(target), placement_backend_name(draft));
        return false;
    } else if (draft_placement_used &&
               !placement_backend_supported(bargs.draft_device.backend)) {
        std::fprintf(stderr,
            "[server] --draft-device=%s is unsupported in this binary "
            "(compiled backend: %s)\n",
            placement_device_name(bargs.draft_device).c_str(),
            placement_backend_name(compiled));
        return false;
    }
    if (!bargs.device.layer_split_gpus.empty()) {
        std::fprintf(stderr,
            "[server] target layer split is not implemented in the native "
            "server yet (--target-devices was provided)\n");
        return false;
    }
    if (!bargs.device.layer_split_weights.empty()) {
        std::fprintf(stderr,
            "[server] --target-layer-split requires native target layer split "
            "support, which is not implemented yet\n");
        return false;
    }
    return true;
}

static void print_usage(const char * prog) {
    std::fprintf(stderr,
        "Usage: %s <model.gguf> [options]\n"
        "\n"
        "Options:\n"
        "  --draft <path>       Draft model for speculative decode (qwen35 only)\n"
        "  --port <N>           Listen port (default: 8080)\n"
        "  --host <addr>        Bind address (default: 0.0.0.0)\n"
        "  --max-ctx <N>        Max context length (default: 131072)\n"
        "  --max-tokens <N>     Default max output tokens (default: 4096)\n"
        "  --target-device <backend:gpu>  Target device (default: auto:0)\n"
        "  --draft-device <backend:gpu>   Draft device (default: auto:0)\n"
        "  --draft-ipc-bin <path>         Remote backend IPC daemon for mixed backends\n"
        "  --draft-ipc-work-dir <path>    Remote draft IPC scratch directory\n"
        "  --draft-ipc-ring-cap <N>       Remote draft feature ring capacity\n"
        "  --target-devices <list>        Reserved layer-split devices, e.g. cuda:0,cuda:1\n"
        "  --target-layer-split <weights>  Reserved layer-split weights\n"
        "  --peer-access        Enable peer access for multi-GPU placement\n"
        "  --chunk <N>          Chunked-prefill chunk size (default: 512)\n"
        "  --fa-window <N>     Flash-attention sliding window (default: 0=full)\n"
        "  --model-name <name>  Model name for /v1/models (default: dflash)\n"
        "  --ddtree             Enable DDTree speculative decode\n"
        "  --ddtree-budget <N>  DDTree budget (default: 64)\n"
        "  --no-cors            Disable CORS headers\n"
        "\n"
        "KV cache:\n"
        "  --cache-type-k <type>  KV cache K type (f16,bf16,q4_0,q4_1,q5_0,q5_1,q8_0,tq3_0)\n"
        "  --cache-type-v <type>  KV cache V type (same choices as above)\n"
#ifdef GGML_USE_HIP
        "                         Default: q4_0 (HIP builds; tq3_0 fattn unsupported)\n"
#else
        "                         Default: tq3_0 when max_ctx>6144, else q4_0\n"
#endif
        "\n"
        "PFlash (speculative prefill compression):\n"
        "  --prefill-compression off|auto|always  (default: off)\n"
        "  --prefill-threshold <N>     Token threshold for auto mode (default: 2000)\n"
        "  --prefill-keep-ratio <F>    Fraction of tokens to keep (default: 0.05)\n"
        "  --prefill-drafter <path>    Drafter GGUF for compression (Qwen3-0.6B)\n"
        "  --prefill-skip-park         Skip park/unpark (for >=32GB GPUs)\n"
        "  --lazy-draft                Park decode draft when idle to save VRAM\n"
        "\n"
        "Disk KV cache:\n"
        "  --kv-cache-dir <path>       Directory for ondisk KV cache (enables feature)\n"
        "  --kv-cache-budget <MB>      Max disk usage in MB (default: 4096)\n"
        "  --kv-cache-min-tokens <N>   Min tokens to persist (default: 512)\n"
        "  --kv-cache-interval <N>     Continued checkpoint every N tokens (default: 10240)\n"
        "  --kv-cache-cold-max <N>     Cold prefix for prompts longer than N tokens (default: 10240)\n"
        "\n"
        "Chat template (optional, e.g. froggeric Qwen3.6 template for tool-using\n"
        "agents that need the Anthropic tool_use envelope):\n"
        "  --chat-template-file <path>  Load a Jinja chat template file.\n"
        "                               Overrides the hardcoded Qwen3/Laguna\n"
        "                               renderer. Empty or missing falls back\n"
        "                               to the hardcoded template.\n"
        "\n", prog);
}

int main(int argc, char ** argv) {
    if (argc < 2 || argv[1][0] == '-') {
        print_usage(argv[0]);
        return 2;
    }

    // Parse arguments.
    BackendArgs bargs;
    ServerConfig sconfig;
    bargs.model_path = argv[1];
    std::string cache_type_k;  // explicit --cache-type-k override
    std::string cache_type_v;  // explicit --cache-type-v override
    bool target_device_seen = false;
    bool target_devices_seen = false;

    for (int i = 2; i < argc; i++) {
        if (std::strcmp(argv[i], "--draft") == 0 && i + 1 < argc) {
            bargs.draft_path = argv[++i];
        } else if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            sconfig.port = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            sconfig.host = argv[++i];
        } else if (std::strcmp(argv[i], "--max-ctx") == 0 && i + 1 < argc) {
            int v = std::atoi(argv[++i]);
            sconfig.max_ctx = v;
            bargs.device.max_ctx = v;
        } else if (std::strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            sconfig.max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--target-device") == 0 && i + 1 < argc) {
            if (target_devices_seen) {
                std::fprintf(stderr, "[server] --target-device conflicts with --target-devices\n");
                return 2;
            }
            target_device_seen = true;
            if (!parse_placement_device(argv[++i], bargs.device)) {
                std::fprintf(stderr, "[server] bad --target-device value (expected backend:gpu)\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--draft-device") == 0 && i + 1 < argc) {
            if (!parse_placement_device(argv[++i], bargs.draft_device)) {
                std::fprintf(stderr, "[server] bad --draft-device value (expected backend:gpu)\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--draft-ipc-bin") == 0 && i + 1 < argc) {
            bargs.remote_draft.ipc_bin = argv[++i];
        } else if (std::strcmp(argv[i], "--draft-ipc-work-dir") == 0 && i + 1 < argc) {
            bargs.remote_draft.work_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--draft-ipc-ring-cap") == 0 && i + 1 < argc) {
            bargs.remote_draft.ring_cap = std::atoi(argv[++i]);
            if (bargs.remote_draft.ring_cap <= 0) {
                std::fprintf(stderr, "[server] bad --draft-ipc-ring-cap value\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--target-devices") == 0 && i + 1 < argc) {
            if (target_device_seen) {
                std::fprintf(stderr, "[server] --target-devices conflicts with --target-device\n");
                return 2;
            }
            target_devices_seen = true;
            if (!parse_placement_device_list(argv[++i], bargs.device)) {
                std::fprintf(stderr, "[server] bad --target-devices value (expected backend:gpu[,backend:gpu...])\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--target-layer-split") == 0 && i + 1 < argc) {
            if (!parse_double_list(argv[++i], bargs.device.layer_split_weights)) {
                std::fprintf(stderr, "[server] bad --target-layer-split value\n");
                return 2;
            }
        } else if (std::strcmp(argv[i], "--peer-access") == 0) {
            bargs.device.peer_access = true;
        } else if (std::strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) {
            bargs.chunk = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--fa-window") == 0 && i + 1 < argc) {
            bargs.fa_window = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--model-name") == 0 && i + 1 < argc) {
            sconfig.model_name = argv[++i];
        } else if (std::strcmp(argv[i], "--ddtree") == 0) {
            bargs.ddtree_mode = true;
            bargs.fast_rollback = true;
        } else if (std::strcmp(argv[i], "--ddtree-budget") == 0 && i + 1 < argc) {
            bargs.ddtree_budget = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-cors") == 0) {
            sconfig.enable_cors = false;
        } else if (std::strcmp(argv[i], "--prefill-compression") == 0 && i + 1 < argc) {
            const char * mode = argv[++i];
            if (std::strcmp(mode, "auto") == 0)
                sconfig.pflash_mode = ServerConfig::PflashMode::AUTO;
            else if (std::strcmp(mode, "always") == 0)
                sconfig.pflash_mode = ServerConfig::PflashMode::ALWAYS;
            else if (std::strcmp(mode, "off") == 0)
                sconfig.pflash_mode = ServerConfig::PflashMode::OFF;
            else {
                std::fprintf(stderr, "[server] unknown --prefill-compression mode: '%s' (expected: auto, always, off)\n", mode);
                print_usage(argv[0]);
                return 1;
            }
        } else if (std::strcmp(argv[i], "--prefill-threshold") == 0 && i + 1 < argc) {
            sconfig.pflash_threshold = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--prefill-keep-ratio") == 0 && i + 1 < argc) {
            sconfig.pflash_keep_ratio = (float)std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--prefill-drafter") == 0 && i + 1 < argc) {
            sconfig.pflash_drafter_path = argv[++i];
        } else if (std::strcmp(argv[i], "--prefill-skip-park") == 0) {
            sconfig.pflash_skip_park = true;
        } else if (std::strcmp(argv[i], "--lazy-draft") == 0) {
            sconfig.lazy_draft = true;
        } else if (std::strcmp(argv[i], "--chat-template-file") == 0 && i + 1 < argc) {
            const char * path = argv[++i];
            std::FILE * f = std::fopen(path, "rb");
            if (!f) {
                std::fprintf(stderr, "[server] --chat-template-file: cannot open '%s'\n", path);
                return 1;
            }
            std::fseek(f, 0, SEEK_END);
            long n = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            if (n <= 0) {
                // The usage text promises "Empty or missing falls back to the
                // hardcoded template." Honor that: log a warning and leave
                // chat_template_src empty so http_server.cpp falls through to
                // the hardcoded QWEN3/LAGUNA renderer, instead of aborting
                // startup.
                std::fclose(f);
                std::fprintf(stderr, "[server] --chat-template-file: '%s' is empty, "
                                     "falling back to hardcoded template\n", path);
            } else {
                sconfig.chat_template_src.resize((size_t)n);
                size_t got = std::fread(sconfig.chat_template_src.data(), 1, (size_t)n, f);
                std::fclose(f);
                if (got != (size_t)n) {
                    std::fprintf(stderr, "[server] --chat-template-file: short read on '%s'\n", path);
                    return 1;
                }
                sconfig.chat_template_path = path;
                std::fprintf(stderr, "[server] loaded chat template from %s (%ld bytes)\n", path, n);
            }
        } else if (std::strcmp(argv[i], "--kv-cache-dir") == 0 && i + 1 < argc) {
            sconfig.disk_cache_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--kv-cache-budget") == 0 && i + 1 < argc) {
            sconfig.disk_cache_budget_mb = (size_t)std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--kv-cache-min-tokens") == 0 && i + 1 < argc) {
            sconfig.disk_cache_min_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--kv-cache-interval") == 0 && i + 1 < argc) {
            sconfig.disk_cache_continued_interval = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--kv-cache-cold-max") == 0 && i + 1 < argc) {
            sconfig.disk_cache_cold_max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--cache-type-k") == 0 && i + 1 < argc) {
            cache_type_k = argv[++i];
        } else if (std::strcmp(argv[i], "--cache-type-v") == 0 && i + 1 < argc) {
            cache_type_v = argv[++i];
        } else {
            std::fprintf(stderr, "[server] unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }

    if (!validate_server_placement(bargs, sconfig)) return 2;

    if (bargs.remote_draft.enabled() && bargs.draft_path) {
        const std::string arch = detect_arch(bargs.model_path);
        if (arch.empty()) {
            std::fprintf(stderr,
                "[server] failed to detect model architecture for remote draft validation\n");
            return 1;
        }
        if (!arch_supports_remote_draft(arch)) {
            std::fprintf(stderr,
                "[server] model architecture '%s' does not support remote draft execution\n",
                arch.c_str());
            return 2;
        }
    }

    // Sync max_ctx: if --max-ctx was not provided, use the backend's default.
    // This prevents the HTTP server from accepting prompts larger than the
    // KV cache the backend actually allocates.
    if (sconfig.max_ctx <= 0) {
        sconfig.max_ctx = bargs.device.max_ctx;
    }
    const PFlashDrafterPlacement pflash_placement =
        resolve_pflash_drafter_placement(
            bargs.device, bargs.draft_device, bargs.remote_draft,
            sconfig.pflash_mode != ServerConfig::PflashMode::OFF);
    sconfig.pflash_drafter_gpu = pflash_placement.drafter_gpu;
    sconfig.pflash_remote_drafter = pflash_placement.remote_drafter;
    sconfig.pflash_remote = pflash_placement.remote;

    // ── Apply environment defaults (mirrors server.py logic) ────────────
    // Explicit --cache-type-k/v override via env vars.
    if (!cache_type_k.empty()) {
        setenv("DFLASH27B_KV_K", cache_type_k.c_str(), 1);
    }
    if (!cache_type_v.empty()) {
        setenv("DFLASH27B_KV_V", cache_type_v.c_str(), 1);
    }

    // Auto-select TQ3_0 KV cache for large contexts (saves ~40% VRAM).
    // Q4_0 remains default for short contexts where quality matters more.
    // HIP build skips this: tq3_0 fattn unsupported (ggml-cuda/fattn.cu).
#ifndef GGML_USE_HIP
    if (sconfig.max_ctx > 6144 && cache_type_k.empty() && cache_type_v.empty()) {
        setenv("DFLASH27B_KV_TQ3", "1", 0);  // don't overwrite user env
    }
#endif

    // PFlash performance defaults: BSA kernel + sparse alpha + full attention window.
    bool pflash_enabled = (sconfig.pflash_mode != ServerConfig::PflashMode::OFF);
    if (pflash_enabled) {
        setenv("DFLASH_FP_USE_BSA", "1", 0);
        setenv("DFLASH_FP_ALPHA", "0.85", 0);
        setenv("DFLASH27B_FA_WINDOW", "0", 0);
    }

    // Lazy-draft requires both prefill-drafter AND decode draft to be useful.
    if (sconfig.lazy_draft && !(pflash_enabled && bargs.draft_path)) {
        std::fprintf(stderr, "[server] --lazy-draft ignored: requires both --prefill-drafter and --draft\n");
        sconfig.lazy_draft = false;
    }

    // Load tokenizer.
    std::fprintf(stderr, "[server] loading tokenizer from %s\n", bargs.model_path);
    Tokenizer tokenizer;
    if (!tokenizer.load_from_gguf(bargs.model_path)) {
        std::fprintf(stderr, "[server] tokenizer load failed\n");
        return 1;
    }

    // Load pflash drafter tokenizer (if pflash enabled).
    Tokenizer drafter_tokenizer;
    if (pflash_enabled) {
        if (sconfig.pflash_drafter_path.empty()) {
            std::fprintf(stderr, "[server] --prefill-compression requires --prefill-drafter\n");
            return 1;
        }
        std::fprintf(stderr, "[server] loading pflash drafter tokenizer from %s\n",
                     sconfig.pflash_drafter_path.c_str());
        if (!drafter_tokenizer.load_from_gguf(sconfig.pflash_drafter_path.c_str())) {
            std::fprintf(stderr, "[server] drafter tokenizer load failed\n");
            return 1;
        }
        if (sconfig.pflash_remote_drafter) {
            if (!bargs.remote_draft.enabled()) {
                std::fprintf(stderr,
                    "[server] mixed-backend PFlash requires --draft-ipc-bin\n");
                return 2;
            }
            const std::string arch = detect_arch(bargs.model_path);
            if (!arch_supports_pflash_compression(arch)) {
                std::fprintf(stderr,
                    "[server] model architecture '%s' does not support PFlash compression\n",
                    arch.c_str());
                return 2;
            }
        }
        std::fprintf(stderr, "[server] pflash: mode=%s threshold=%d keep=%.3f drafter_gpu=%d skip_park=%d\n",
                     sconfig.pflash_mode == ServerConfig::PflashMode::AUTO ? "auto" : "always",
                     sconfig.pflash_threshold, sconfig.pflash_keep_ratio,
                     sconfig.pflash_drafter_gpu,
                     (int)sconfig.pflash_skip_park);
    }

    // Create backend.
    g_peer_access_opt_in = bargs.device.peer_access;
    std::fprintf(stderr, "[server] creating backend...\n");
    const std::string arch = detect_arch(bargs.model_path);
    auto backend = create_backend(bargs);
    if (!backend) {
        std::fprintf(stderr, "[server] backend creation failed\n");
        return 1;
    }
    if (bargs.remote_draft.enabled() && bargs.draft_path &&
        !backend->supports_remote_draft()) {
        std::fprintf(stderr,
            "[server] detected model backend does not support remote draft execution\n");
        backend->shutdown();
        return 2;
    }

    // Start HTTP server.
    std::fprintf(stderr, "\n");
    std::fprintf(stderr, "[server] ╭─── Configuration ───────────────────────────────────╮\n");
    std::fprintf(stderr, "[server] │  host            = %s\n", sconfig.host.c_str());
    std::fprintf(stderr, "[server] │  port            = %d\n", sconfig.port);
    std::fprintf(stderr, "[server] │  model           = %s\n", bargs.model_path);
    std::fprintf(stderr, "[server] │  draft           = %s\n", bargs.draft_path ? bargs.draft_path : "(none)");
    std::fprintf(stderr, "[server] │  model_name      = %s\n", sconfig.model_name.c_str());
    std::fprintf(stderr, "[server] │  max_ctx         = %d\n", sconfig.max_ctx);
    std::fprintf(stderr, "[server] │  max_tokens      = %d\n", sconfig.max_tokens);
    std::fprintf(stderr, "[server] │  target_device   = %s\n",
                 placement_device_name(bargs.device).c_str());
    if (bargs.device.is_layer_split()) {
        std::fprintf(stderr, "[server] │  target_shards   =");
        for (int gpu : bargs.device.layer_split_gpus) {
            std::fprintf(stderr, " %s:%d",
                         placement_backend_name(bargs.device.backend), gpu);
        }
        std::fprintf(stderr, "\n");
    }
    std::fprintf(stderr, "[server] │  draft_device    = %s\n",
                 placement_device_name(bargs.draft_device).c_str());
    std::fprintf(stderr, "[server] │  draft_exec      = %s\n",
                 bargs.remote_draft.enabled() && bargs.draft_path ? "remote-ipc" : "local");
    if (bargs.remote_draft.enabled()) {
        std::fprintf(stderr, "[server] │  draft_ipc_bin  = %s\n",
                     bargs.remote_draft.ipc_bin.c_str());
        if (!bargs.remote_draft.work_dir.empty()) {
            std::fprintf(stderr, "[server] │  draft_ipc_dir  = %s\n",
                         bargs.remote_draft.work_dir.c_str());
        }
        std::fprintf(stderr, "[server] │  draft_ipc_cap  = %d\n",
                     bargs.remote_draft.ring_cap);
    }
    std::fprintf(stderr, "[server] │  peer_access     = %s\n",
                 bargs.device.peer_access ? "ON" : "off");
    std::fprintf(stderr, "[server] │  chunk           = %d\n", bargs.chunk);
    std::fprintf(stderr, "[server] │  fa_window       = %d\n", bargs.fa_window);
    std::fprintf(stderr, "[server] │  ddtree          = %s\n", bargs.ddtree_mode ? "ON" : "off");
    std::fprintf(stderr, "[server] │  ddtree_budget   = %d\n", bargs.ddtree_budget);
    std::fprintf(stderr, "[server] │  cors            = %s\n", sconfig.enable_cors ? "ON" : "off");
    std::fprintf(stderr, "[server] │  cache_type_k    = %s\n",
#ifdef GGML_USE_HIP
        cache_type_k.empty() ? "q4_0 (default, HIP)" : cache_type_k.c_str());
#else
        cache_type_k.empty() ? (sconfig.max_ctx > 6144 ? "tq3_0 (auto)" : "q4_0 (default)") : cache_type_k.c_str());
#endif
    std::fprintf(stderr, "[server] │  cache_type_v    = %s\n",
#ifdef GGML_USE_HIP
        cache_type_v.empty() ? "q4_0 (default, HIP)" : cache_type_v.c_str());
#else
        cache_type_v.empty() ? (sconfig.max_ctx > 6144 ? "tq3_0 (auto)" : "q4_0 (default)") : cache_type_v.c_str());
#endif
    std::fprintf(stderr, "[server] │  pflash          = %s\n",
        sconfig.pflash_mode == ServerConfig::PflashMode::AUTO ? "auto" :
        sconfig.pflash_mode == ServerConfig::PflashMode::ALWAYS ? "always" : "off");
    if (pflash_enabled) {
        std::fprintf(stderr, "[server] │  pflash_threshold= %d\n", sconfig.pflash_threshold);
        std::fprintf(stderr, "[server] │  pflash_keep     = %.3f\n", sconfig.pflash_keep_ratio);
        std::fprintf(stderr, "[server] │  pflash_drafter  = %s\n", sconfig.pflash_drafter_path.c_str());
        std::fprintf(stderr, "[server] │  pflash_drafter_gpu= %d\n", sconfig.pflash_drafter_gpu);
        std::fprintf(stderr, "[server] │  pflash_drafter_exec= %s\n",
                     sconfig.pflash_remote_drafter ? "remote-ipc" : "local");
        std::fprintf(stderr, "[server] │  pflash_skip_park= %s\n", sconfig.pflash_skip_park ? "ON" : "off");
        std::fprintf(stderr, "[server] │  fp_use_bsa      = %s\n", getenv("DFLASH_FP_USE_BSA") ? "ON" : "off");
        std::fprintf(stderr, "[server] │  fp_alpha        = %s\n", getenv("DFLASH_FP_ALPHA") ? getenv("DFLASH_FP_ALPHA") : "0.12 (default)");
    }
    if (bargs.draft_path) {
        std::fprintf(stderr, "[server] │  lazy_draft      = %s\n", sconfig.lazy_draft ? "ON" : "off");
    }
    std::fprintf(stderr, "[server] ╰─────────────────────────────────────────────────────╯\n\n");

    HttpServer server(*backend, tokenizer, sconfig);
    server.set_chat_format(chat_format_for_arch(arch));
    g_server = &server;
    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);
    if (pflash_enabled) {
        server.set_drafter_tokenizer(&drafter_tokenizer);
    }

    // Lazy-draft: park decode draft at startup to free VRAM (~3.3 GB).
    if (sconfig.lazy_draft && bargs.draft_path) {
        backend->park("draft");
    }

    int ret = server.run();

    // Cleanup.
    backend->shutdown();
    return ret;
}
