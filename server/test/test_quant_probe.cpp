// Diagnostic: run real quantized weight bytes (extracted from a GGUF) through a
// MUL_MAT on CPU and CUDA and report the divergence per tensor. Used to locate
// which quantization type a heterogeneous GGUF mishandles.
//
// Usage: test_quant_probe <manifest> [T]
// Manifest lines: <name> <type_id> <K> <M> <file-with-row*M-bytes>
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const bool g_cast = getenv("PROBE_CAST") != nullptr;

static uint32_t g_state = 0x12345678u;
static float rnd() {
    g_state = g_state * 1664525u + 1013904223u;
    return (float) ((g_state >> 8) & 0xFFFFFF) / (float) 0x1000000 * 2.0f - 1.0f;
}

static bool run(ggml_backend_t backend, ggml_type wtype,
                const std::vector<uint8_t> & wq, const std::vector<float> & xf,
                int64_t M, int64_t K, int64_t T, std::vector<float> & out) {
    ggml_init_params ip{};
    ip.mem_size = 64 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor * w = ggml_new_tensor_2d(ctx, wtype, K, M);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    ggml_set_input(w); ggml_set_input(x);
    ggml_tensor * dst = g_cast ? ggml_cast(ctx, w, GGML_TYPE_F32) : ggml_mul_mat(ctx, w, x);
    ggml_set_output(dst);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(buf); ggml_free(ctx); return false; }
    out.resize((size_t) M * (g_cast ? K : T));
    ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static bool run_id(ggml_backend_t backend, ggml_type wtype,
                   const std::vector<uint8_t> & wq, const std::vector<float> & xf,
                   const std::vector<int32_t> & ids_in,
                   int64_t M, int64_t K, int64_t E, int64_t n_used, int64_t T,
                   std::vector<float> & out) {
    ggml_init_params ip{};
    ip.mem_size = 64 * 1024 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    ggml_tensor * w = ggml_new_tensor_3d(ctx, wtype, K, M, E);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, n_used, T);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, T);
    ggml_set_input(w); ggml_set_input(x); ggml_set_input(ids);
    ggml_tensor * dst = ggml_mul_mat_id(ctx, w, x, ids);
    ggml_set_output(dst);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_in.data(), 0, ids_in.size() * sizeof(int32_t));
    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(buf); ggml_free(ctx); return false; }
    out.resize((size_t) M * n_used * T);
    ggml_backend_tensor_get(dst, out.data(), 0, out.size() * sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <manifest> [T]\n", argv[0]); return 2; }
    const int64_t T = argc > 2 ? atoll(argv[2]) : 512;
    FILE * mf = std::fopen(argv[1], "r");
    if (!mf) { std::fprintf(stderr, "cannot open manifest\n"); return 1; }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t gpu = ggml_backend_cuda_init(0);
    if (!cpu || !gpu) { std::fprintf(stderr, "backend init failed\n"); return 1; }

    char line[2048];
    bool allok = true;
    while (std::fgets(line, sizeof(line), mf)) {
        char name[256], file[1024];
        long long K, M, E = 0;
        int type_id;
        const int nf = std::sscanf(line, "%255s %d %lld %lld %lld %1023s", name, &type_id, &K, &M, &E, file);
        if (nf != 6 && nf != 5) continue;
        if (nf == 5) { std::sscanf(line, "%255s %d %lld %lld %1023s", name, &type_id, &K, &M, file); E = 0; }
        const ggml_type wt = (ggml_type) type_id;

        std::vector<float> ref, out;
        bool a, b;
        if (E > 0) {
            const int64_t n_used = std::min<int64_t>(E, 10);
            const size_t row = ggml_row_size(wt, K);
            std::vector<uint8_t> wq(row * (size_t) M * (size_t) E);
            FILE * wf = std::fopen(file, "rb");
            if (!wf) { std::printf("PROBE %-30s OPENFAIL\n", name); continue; }
            const size_t got = std::fread(wq.data(), 1, wq.size(), wf);
            std::fclose(wf);
            if (got != wq.size()) { std::printf("PROBE %-30s SHORT %zu/%zu\n", name, got, wq.size()); continue; }
            std::vector<float> xf((size_t) K * n_used * T);
            for (auto & v : xf) v = rnd();
            std::vector<int32_t> ids((size_t) n_used * T);
            for (int64_t t = 0; t < T; ++t) for (int64_t u = 0; u < n_used; ++u) ids[u + n_used * t] = (int32_t) ((t + u) % E);

            // Unambiguous reference: dequantize each selected expert with the
            // type's own to_float and do the dot in f32.
            const ggml_type_traits * tr = ggml_get_type_traits(wt);
            if (!tr || !tr->to_float) { std::printf("PROBE %-30s NO_TO_FLOAT\n", name); continue; }
            std::vector<float> wdec((size_t) M * K * E);
            for (int64_t e = 0; e < E; ++e)
                for (int64_t m = 0; m < M; ++m)
                    tr->to_float(wq.data() + (e * M + m) * row, wdec.data() + (e * M + m) * K, K);
            ref.assign((size_t) M * n_used * T, 0.0f);
            for (int64_t t = 0; t < T; ++t)
                for (int64_t u = 0; u < n_used; ++u) {
                    const int64_t e = ids[u + n_used * t];
                    for (int64_t mm = 0; mm < M; ++mm) {
                        double s = 0;
                        for (int64_t k = 0; k < K; ++k)
                            s += (double) xf[k + K * (u + n_used * t)] * wdec[(e * M + mm) * K + k];
                        ref[(mm + M * (u + n_used * t))] = (float) s;
                    }
                }
            a = true;
            b = run_id(gpu, wt, wq, xf, ids, M, K, E, n_used, T, out);
        } else {
            const size_t row = ggml_row_size(wt, K);
            std::vector<uint8_t> wq(row * (size_t) M);
            FILE * wf = std::fopen(file, "rb");
            if (!wf) { std::printf("PROBE %-30s OPENFAIL\n", name); continue; }
            const size_t got = std::fread(wq.data(), 1, wq.size(), wf);
            std::fclose(wf);
            if (got != wq.size()) { std::printf("PROBE %-30s SHORT %zu/%zu\n", name, got, wq.size()); continue; }
            std::vector<float> xf((size_t) K * T);
            for (auto & v : xf) v = rnd();
            a = run(cpu, wt, wq, xf, M, K, T, ref);
            b = run(gpu, wt, wq, xf, M, K, T, out);
        }
        if (!a || !b) { std::printf("PROBE %-30s COMPUTE_FAIL\n", name); allok = false; continue; }
        double md = 0, norm = 0;
        for (size_t i = 0; i < out.size(); ++i) {
            md = std::fmax(md, std::fabs((double) out[i] - ref[i]));
            norm += (double) ref[i] * ref[i];
        }
        const double rms = std::sqrt(norm / out.size());
        const double rel = rms > 0 ? md / rms : 0;
        const bool ok = rel < 0.15;
        allok = allok && ok;
        std::printf("PROBE %-30s type=%-10s id=%-3d K=%-6lld M=%-3lld %s rel=%.4g %s\n",
            name, ggml_type_name(wt), type_id, K, M, E ? "MMID" : "MM  ", rel, ok ? "ok" : "SUSPECT");
    }
    std::fclose(mf);
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    std::printf("PROBE %s\n", allok ? "ALL-OK" : "SUSPECTS-FOUND");
    return allok ? 0 : 1;
}
