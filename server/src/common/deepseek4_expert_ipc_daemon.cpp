// deepseek4_expert_ipc_daemon.cpp - DeepSeek4 expert IPC worker boundary.

#include "deepseek4_expert_ipc.h"
#include "../deepseek4/deepseek4_internal.h"
#include "dflash_draft_ipc.h"
#include "io_utils.h"
#include "moe_hybrid_ffn_eval.h"
#include "moe_hybrid_storage.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

namespace dflash::common {

namespace {

struct Ds4ExpertWorker {
    ggml_backend_t backend = nullptr;
    std::string model_path;
    DeepSeek4Weights weights;
    MoeHybridPlacement placement;
    std::shared_ptr<MoeHybridStorage> storage;
    MoeHybridConfig cfg;

    ~Ds4ExpertWorker() {
        storage.reset();
        free_deepseek4_weights(weights);
        if (backend) {
            ggml_backend_free(backend);
            backend = nullptr;
        }
    }
};

static uint64_t ds4_layer_expert_bytes(const DeepSeek4Layer & layer, int n_expert) {
    if (n_expert <= 0) return 0;
    uint64_t bytes = 0;
    if (layer.ffn_gate_exps) bytes += ggml_nbytes(layer.ffn_gate_exps) / (uint64_t)n_expert;
    if (layer.ffn_up_exps) bytes += ggml_nbytes(layer.ffn_up_exps) / (uint64_t)n_expert;
    if (layer.ffn_down_exps) bytes += ggml_nbytes(layer.ffn_down_exps) / (uint64_t)n_expert;
    return bytes;
}

static MoeHybridConfig make_worker_cfg(const DeepSeek4Weights & w) {
    MoeHybridConfig cfg;
    cfg.n_embd = w.n_embd;
    cfg.n_expert = w.n_expert;
    cfg.n_expert_used = w.n_expert_used;
    cfg.n_ff_exp = w.n_ff_exp;
    cfg.n_ff_shexp = 0;
    cfg.n_layer = w.n_layer;
    cfg.first_moe_layer = 0;
    cfg.swiglu_clamp = w.swiglu_clamp_exp;
    cfg.materialize_cold_experts = false;
    return cfg;
}

static MoeLayerDesc make_worker_desc(const DeepSeek4Layer & L) {
    MoeLayerDesc desc;
    desc.ffn_gate_exps = L.ffn_gate_exps;
    desc.ffn_up_exps = L.ffn_up_exps;
    desc.ffn_down_exps = L.ffn_down_exps;
    return desc;
}

static MoeHybridPlacement make_single_layer_placement(const DeepSeek4Weights & w,
                                                      int layer,
                                                      const std::vector<int32_t> & experts) {
    MoeHybridPlacement placement;
    placement.n_layer = w.n_layer;
    placement.n_expert = w.n_expert;
    placement.n_expert_used = w.n_expert_used;
    placement.total_hot = (int)experts.size();
    placement.hot_counts.assign((size_t)w.n_layer, 0);
    placement.hot_expert_ids.resize((size_t)w.n_layer);
    placement.hot_counts[(size_t)layer] = (int)experts.size();
    placement.hot_expert_ids[(size_t)layer] = experts;
    return placement;
}

static bool build_worker_placement(const DeepSeek4Weights & w,
                                   int worker_gpu,
                                   MoeHybridPlacement & out,
                                   std::string & err) {
    size_t gpu_free = 0;
    size_t gpu_total = 0;
    ggml_backend_cuda_get_device_memory(worker_gpu, &gpu_free, &gpu_total);
    if (gpu_total == 0) {
        err = "could not query worker GPU memory";
        return false;
    }
    uint64_t round_bytes = 0;
    uint64_t total_expert_bytes = 0;
    for (int il = 0; il < w.n_layer; ++il) {
        const uint64_t b = ds4_layer_expert_bytes(w.layers[(size_t)il], w.n_expert);
        round_bytes += b;
        total_expert_bytes += b * (uint64_t)w.n_expert;
    }
    if (round_bytes == 0) {
        err = "expert metadata missing";
        return false;
    }

    uint64_t budget = gpu_free > (768ULL << 20) ? gpu_free - (768ULL << 20) : 0;
    if (budget > total_expert_bytes) budget = total_expert_bytes;
    if (const char * env = std::getenv("DFLASH_DS4_EXPERT_WORKER_BUDGET_MB")) {
        const uint64_t cap = (uint64_t)std::max(0, std::atoi(env)) << 20;
        if (cap > 0 && cap < budget) budget = cap;
    } else if (const char * env = std::getenv("DFLASH_EXPERT_BUDGET_MB")) {
        const uint64_t cap = (uint64_t)std::max(0, std::atoi(env)) << 20;
        if (cap > 0 && cap < budget) budget = cap;
    }
    const int hot_per_layer = std::min(w.n_expert, (int)(budget / round_bytes));
    if (hot_per_layer <= 0) {
        err = "worker expert budget is smaller than one uniform expert round";
        return false;
    }
    int first_expert = 0;
    if (const char * env = std::getenv("DFLASH_DS4_EXPERT_WORKER_OFFSET")) {
        first_expert = std::max(0, std::atoi(env));
    }
    if (first_expert >= w.n_expert) first_expert = 0;
    const int resident_per_layer = std::min(hot_per_layer, w.n_expert - first_expert);
    if (resident_per_layer <= 0) {
        err = "worker expert offset leaves no resident experts";
        return false;
    }

    out = {};
    out.n_layer = w.n_layer;
    out.n_expert = w.n_expert;
    out.n_expert_used = w.n_expert_used;
    out.total_hot = resident_per_layer * w.n_layer;
    out.hot_counts.assign((size_t)w.n_layer, resident_per_layer);
    out.hot_expert_ids.resize((size_t)w.n_layer);
    for (int il = 0; il < w.n_layer; ++il) {
        auto & ids = out.hot_expert_ids[(size_t)il];
        ids.reserve((size_t)resident_per_layer);
        for (int ie = first_expert; ie < first_expert + resident_per_layer; ++ie) {
            ids.push_back((int32_t)ie);
        }
    }
    std::fprintf(stderr,
                 "[deepseek4-expert-ipc-daemon] placement gpu_total=%.2f GiB gpu_free=%.2f GiB budget=%.2f GiB hot/layer=%d first=%d total_hot=%d\n",
                 (double)gpu_total / 1024.0 / 1024.0 / 1024.0,
                 (double)gpu_free / 1024.0 / 1024.0 / 1024.0,
                 (double)budget / 1024.0 / 1024.0 / 1024.0,
                 resident_per_layer, first_expert, out.total_hot);
    return true;
}

static bool init_worker(const char * model_path, int worker_gpu, Ds4ExpertWorker & worker) {
    worker.model_path = model_path;
    worker.backend = ggml_backend_cuda_init(std::max(0, worker_gpu));
    if (!worker.backend) {
        std::fprintf(stderr, "[deepseek4-expert-ipc-daemon] backend init failed gpu=%d\n",
                     std::max(0, worker_gpu));
        return false;
    }
    TargetLoadPlan plan;
    plan.expert_metadata_only = true;
    if (!load_deepseek4_gguf_partial(model_path, worker.backend, plan, worker.weights)) {
        std::fprintf(stderr, "[deepseek4-expert-ipc-daemon] partial model load failed: %s\n",
                     model_path);
        return false;
    }
    std::string err;
    if (!build_worker_placement(worker.weights, std::max(0, worker_gpu), worker.placement, err)) {
        std::fprintf(stderr, "[deepseek4-expert-ipc-daemon] placement failed: %s\n", err.c_str());
        return false;
    }
    worker.cfg = make_worker_cfg(worker.weights);
    std::vector<MoeLayerDesc> descs((size_t)worker.weights.n_layer);
    for (int il = 0; il < worker.weights.n_layer; ++il) {
        descs[(size_t)il] = make_worker_desc(worker.weights.layers[(size_t)il]);
    }
    worker.storage = std::make_shared<MoeHybridStorage>();
    if (!build_deepseek4_moe_hybrid_storage_from_file(
            model_path, worker.backend, worker.weights, worker.placement,
            &worker.cfg, *worker.storage, &err)) {
        std::fprintf(stderr, "[deepseek4-expert-ipc-daemon] expert storage failed: %s\n", err.c_str());
        return false;
    }
    const int total_cold = worker.weights.n_layer * worker.weights.n_expert - worker.placement.total_hot;
    std::fprintf(stderr,
                 "[deepseek4-expert-ipc-daemon] expert storage ready expert_only=yes hot=%d cold=%d cold_materialized=%s\n",
                 worker.placement.total_hot, total_cold,
                 worker.storage->materialized_cold_experts ? "yes" : "no");
    return true;
}

static bool eval_worker_selected(ggml_backend_t backend,
                                 const MoeHybridConfig & cfg,
                                 const MoeLayerDesc & desc,
                                 MoeHybridLayerStorage & storage,
                                 ggml_backend_t cpu_backend,
                                 const float * activation,
                                 const std::vector<int32_t> & selected_ids,
                                 const std::vector<float> & selected_weights,
                                 std::vector<float> & out,
                                 std::string & err) {
    if (selected_ids.empty()) {
        out.assign((size_t)cfg.n_embd, 0.0f);
        return true;
    }
    return eval_moe_hybrid_ffn_single(
        backend, cfg, desc, storage, cpu_backend, activation,
        selected_ids.data(), selected_weights.data(), (int)selected_ids.size(),
        out, nullptr, &err);
}

static bool eval_worker_token(Ds4ExpertWorker & worker,
                              int layer,
                              const float * activation,
                              const int32_t * selected_ids,
                              const float * selected_weights,
                              int n_selected,
                              std::vector<float> & out,
                              std::string & err) {
    auto & resident = worker.storage->layers[(size_t)layer];
    std::vector<int32_t> resident_ids;
    std::vector<float> resident_weights;
    std::vector<int32_t> miss_ids;
    std::vector<float> miss_weights;
    for (int i = 0; i < n_selected; ++i) {
        const int32_t gid = selected_ids[i];
        if (gid < 0 || gid >= worker.weights.n_expert) {
            err = "selected id out of range";
            return false;
        }
        if (gid < (int32_t)resident.hot_local_by_global.size() &&
            resident.hot_local_by_global[(size_t)gid] >= 0) {
            resident_ids.push_back(gid);
            resident_weights.push_back(selected_weights[i]);
        } else {
            miss_ids.push_back(gid);
            miss_weights.push_back(selected_weights[i]);
        }
    }

    out.assign((size_t)worker.cfg.n_embd, 0.0f);
    MoeLayerDesc desc = make_worker_desc(worker.weights.layers[(size_t)layer]);
    std::vector<float> partial;
    if (!eval_worker_selected(worker.backend, worker.cfg, desc, resident,
                              worker.storage->cpu_backend, activation,
                              resident_ids, resident_weights, partial, err)) {
        return false;
    }
    for (int i = 0; i < worker.cfg.n_embd; ++i) {
        out[(size_t)i] += partial[(size_t)i];
    }

    if (miss_ids.empty()) return true;

    MoeHybridStorage miss_storage;
    MoeHybridPlacement miss_placement =
        make_single_layer_placement(worker.weights, layer, miss_ids);
    if (!build_deepseek4_moe_hybrid_storage_from_file(
            worker.model_path, worker.backend, worker.weights, miss_placement,
            &worker.cfg, miss_storage, &err)) {
        return false;
    }
    std::fprintf(stderr,
                 "[deepseek4-expert-ipc-daemon] file-backed miss layer=%d experts=%zu\n",
                 layer, miss_ids.size());
    if (!eval_worker_selected(worker.backend, worker.cfg, desc,
                              miss_storage.layers[(size_t)layer],
                              miss_storage.cpu_backend, activation,
                              miss_ids, miss_weights, partial, err)) {
        return false;
    }
    for (int i = 0; i < worker.cfg.n_embd; ++i) {
        out[(size_t)i] += partial[(size_t)i];
    }
    return true;
}

static bool read_eval_request(const std::string & path,
                              DeepSeek4ExpertIpcRequestHeader & hdr,
                              std::vector<float> & activations,
                              std::vector<int32_t> & ids,
                              std::vector<float> & weights) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read((char *)&hdr, sizeof(hdr));
    if (!f || hdr.magic != 0x44533445u || hdr.version != 1 ||
        hdr.layer < 0 || hdr.n_tokens <= 0 || hdr.n_embd <= 0 || hdr.n_selected <= 0) {
        return false;
    }
    const size_t n_act = (size_t)hdr.n_tokens * (size_t)hdr.n_embd;
    const size_t n_sel = (size_t)hdr.n_tokens * (size_t)hdr.n_selected;
    activations.assign(n_act, 0.0f);
    ids.assign(n_sel, 0);
    weights.assign(n_sel, 0.0f);
    f.read((char *)activations.data(), (std::streamsize)(n_act * sizeof(float)));
    f.read((char *)ids.data(), (std::streamsize)(n_sel * sizeof(int32_t)));
    f.read((char *)weights.data(), (std::streamsize)(n_sel * sizeof(float)));
    return (bool)f;
}

static bool write_eval_response(int stream_fd,
                                int status,
                                int n_tokens,
                                int n_embd,
                                const std::vector<float> * out) {
    DeepSeek4ExpertIpcResponseHeader resp;
    resp.status = status;
    resp.n_tokens = n_tokens;
    resp.n_embd = n_embd;
    if (!write_exact_fd(stream_fd, &resp, sizeof(resp))) return false;
    if (status == 0 && out && !out->empty()) {
        return write_exact_fd(stream_fd, out->data(), out->size() * sizeof(float));
    }
    return true;
}

} // namespace

int run_deepseek4_expert_ipc_daemon(const char * model_path,
                                    int worker_gpu,
                                    int stream_fd,
                                    int payload_fd) {
#if defined(_WIN32)
    (void)model_path; (void)worker_gpu; (void)stream_fd; (void)payload_fd;
    std::fprintf(stderr, "DeepSeek4 expert IPC daemon is only implemented on POSIX hosts\n");
    return 2;
#else
    (void)payload_fd;
    if (!model_path || stream_fd < 0) {
        std::fprintf(stderr,
            "usage: backend_ipc_daemon --backend-ipc-mode=deepseek4-expert <model.gguf> "
            "--stream-fd=FD [--payload-fd=FD] [--draft-gpu=N]\n");
        return 2;
    }

    Ds4ExpertWorker worker;
    if (!init_worker(model_path, worker_gpu, worker)) {
        stream_status(stream_fd, -1);
        return 1;
    }

    std::fprintf(stderr, "[deepseek4-expert-ipc-daemon] ready gpu=%d model=%s\n",
                 std::max(0, worker_gpu), model_path);
    stream_status(stream_fd, 0);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd == "quit" || cmd == "exit") break;
        if (cmd == "ping") {
            stream_status(stream_fd, 0);
            continue;
        }
        if (cmd == "eval") {
            std::string path = read_line_tail(iss);
            DeepSeek4ExpertIpcRequestHeader hdr;
            std::vector<float> activations;
            std::vector<int32_t> ids;
            std::vector<float> weights;
            if (path.empty() || !read_eval_request(path, hdr, activations, ids, weights) ||
                hdr.layer >= worker.weights.n_layer ||
                hdr.n_embd != worker.weights.n_embd) {
                std::fprintf(stderr, "[deepseek4-expert-ipc-daemon] bad eval request: %s\n",
                             line.c_str());
                write_eval_response(stream_fd, -1, 0, 0, nullptr);
                continue;
            }

            std::vector<float> out((size_t)hdr.n_tokens * (size_t)hdr.n_embd, 0.0f);
            std::vector<float> one_out;
            bool ok = true;
            for (int t = 0; t < hdr.n_tokens; ++t) {
                std::string err;
                if (!eval_worker_token(
                        worker, hdr.layer,
                        activations.data() + (size_t)t * (size_t)hdr.n_embd,
                        ids.data() + (size_t)t * (size_t)hdr.n_selected,
                        weights.data() + (size_t)t * (size_t)hdr.n_selected,
                        hdr.n_selected, one_out, err)) {
                    std::fprintf(stderr, "[deepseek4-expert-ipc-daemon] eval failed layer=%d: %s\n",
                                 hdr.layer, err.c_str());
                    ok = false;
                    break;
                }
                std::copy(one_out.begin(), one_out.end(),
                          out.begin() + (size_t)t * (size_t)hdr.n_embd);
            }
            write_eval_response(stream_fd, ok ? 0 : -2, hdr.n_tokens, hdr.n_embd,
                                ok ? &out : nullptr);
            continue;
        }
        std::fprintf(stderr, "[deepseek4-expert-ipc-daemon] unknown command: %s\n", line.c_str());
        stream_status(stream_fd, -1);
    }

    std::fprintf(stderr, "[deepseek4-expert-ipc-daemon] stopped\n");
    return 0;
#endif
}

} // namespace dflash::common
