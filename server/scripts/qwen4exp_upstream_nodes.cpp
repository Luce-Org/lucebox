#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    if (ask) return true;
    if (!t->name) return true;
    const char * flt = (const char *) ud;
    if (t->type == GGML_TYPE_I32) {
        // ffn_moe_topk is a non-contiguous view of the argsort; read it row by row.
        if (strstr(t->name, "ffn_moe_topk")) {
            const size_t n = (size_t) ggml_nelements(t);
            const int64_t ne0 = t->ne[0];
            std::vector<int32_t> iv(n);
            for (int64_t r = 0; r < (int64_t) (n / (size_t) ne0); ++r) {
                ggml_backend_tensor_get(t, iv.data() + r * ne0, (size_t) r * t->nb[1],
                                        (size_t) ne0 * sizeof(int32_t));
            }
            printf("IDS %-40s n=%-8zu", t->name, n);
            for (size_t i = 0; i < n; ++i) printf(" %d", iv[i]);
            printf("\n");
        }
        return true;
    }
    if (t->type != GGML_TYPE_F32) return true;
    if (flt && flt[0] && !strstr(t->name, flt)) return true;
    const size_t n = (size_t) ggml_nelements(t);
    std::vector<float> v(n);
    if (ggml_backend_buffer_is_host(t->buffer)) memcpy(v.data(), t->data, n*sizeof(float));
    else ggml_backend_tensor_get(t, v.data(), 0, n*sizeof(float));
    const char * bindump = getenv("QWEN4EXP_UP_DUMP_BIN");
    if (bindump && bindump[0]) {
        std::string names = bindump, want;
        size_t pos = 0;
        while (pos <= names.size()) {
            size_t comma = names.find(',', pos);
            want = names.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (!want.empty() && want == t->name) {
                std::string safe = t->name;
                for (char & ch : safe) if (ch == '/' || ch == '(' || ch == ')' || ch == ' ') ch = '_';
                char path[256];
                snprintf(path, sizeof path, "/tmp/up_%s.bin", safe.c_str());
                FILE * f = fopen(path, "wb");
                if (f) { fwrite(v.data(), sizeof(float), n, f); fclose(f); }
                break;
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    }
    double am=0, sum=0, sumsq=0; bool fin=true;
    for (float x : v) { if(!std::isfinite(x)) fin=false; double a=fabs((double)x); if(a>am)am=a; sum+=x; sumsq+=(double)x*(double)x; }
    printf("NODE %-40s n=%-8zu finite=%d absmax=%-12.5g mean=%-12.5g sumsq=%-18.10g\n", t->name, n, (int)fin, am, n?sum/n:0.0, sumsq);
    return true;
}
int main(int argc, char ** argv) {
    const char * path = argv[1]; const int S = argc>2?atoi(argv[2]):16;
    const char * flt = argc>3?argv[3]:nullptr;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 99;
    llama_model * m = llama_load_model_from_file(path, mp);
    if (!m) { printf("MODEL_LOAD_FAIL\n"); return 1; }
    const llama_vocab * v = llama_model_get_vocab(m); const int n_vocab = llama_vocab_n_tokens(v);
    std::vector<llama_token> tok((size_t)S);
    for (int i=0;i<S;i++) tok[(size_t)i]=(llama_token)(((int64_t)i*7919+13)%n_vocab);
    if (const char * file = getenv("QWEN4EXP_TOKEN_FILE")) {
        std::ifstream input(file);
        for (int i = 0; i < S; ++i) {
            if (!(input >> tok[i]) || tok[i] < 0 || tok[i] >= n_vocab) {
                std::fprintf(stderr, "invalid QWEN4EXP_TOKEN_FILE\n");
                return 2;
            }
        }
        int extra;
        if (input >> extra) { std::fprintf(stderr, "too many input tokens\n"); return 2; }
    }
    if (S>0) printf("TOKENS n=%d n_vocab=%d first=%d last=%d\n", S, n_vocab, (int)tok[0], (int)tok[(size_t)S-1]);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx=S+8; cp.n_batch=S+8; cp.n_ubatch=S+8;
    cp.cb_eval = cb; cp.cb_eval_user_data = (void*)flt;
    llama_context * ctx = llama_new_context_with_model(m, cp);
    if (!ctx) { printf("CTX_FAIL\n"); return 1; }
    if (llama_decode(ctx, llama_batch_get_one(tok.data(), S))) { printf("DECODE_FAIL\n"); return 1; }
    return 0;
}
