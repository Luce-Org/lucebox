// Temporary P0 width-numerics probe. N=1 and row 0 of N=4 share bit-identical
// inputs; three other rows are deterministic decoys. No server path changes.
#include "qwen4exp_internal.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace luce::common;

static uint32_t rng_state = 0x714acafeu;
static float rnd() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return ((rng_state >> 8) & 0xffffffu) * (2.0f / 16777216.0f) - 1.0f;
}
static void fill_row(float * p, size_t n, int row) {
    const uint32_t old = rng_state;
    rng_state += 0x9e3779b9u * (uint32_t)(row + 1);
    for (size_t i = 0; i < n; ++i) p[i] = rnd() * 0.05f;
    rng_state = old;
}

static bool run_dense(ggml_backend_t be, ggml_tensor * wt, int T,
                      const std::vector<float> & xdata, std::vector<float> & out) {
    ggml_init_params ip{}; ip.mem_size = 32u << 20; ip.no_alloc = true;
    ggml_context * c = ggml_init(ip); if (!c) return false;
    ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, wt->ne[0], T);
    ggml_set_input(x);
    ggml_tensor * y = ggml_mul_mat(c, wt, x);
    ggml_set_output(y);
    ggml_cgraph * g = ggml_new_graph(c); ggml_build_forward_expand(g, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
    if (!buf) { ggml_free(c); return false; }
    ggml_backend_tensor_set(x, xdata.data(), 0, xdata.size()*sizeof(float));
    const bool ok = ggml_backend_graph_compute(be, g) == GGML_STATUS_SUCCESS;
    if (ok) { out.resize((size_t)wt->ne[1]*T); ggml_backend_tensor_get(y,out.data(),0,out.size()*sizeof(float)); }
    ggml_backend_buffer_free(buf); ggml_free(c); return ok;
}

static bool run_mmid(ggml_backend_t be, ggml_tensor * wt, int T, int n_used,
                     const std::vector<float> & xdata, const std::vector<int32_t> & idsdata,
                     std::vector<float> & out) {
    ggml_init_params ip{}; ip.mem_size = 32u << 20; ip.no_alloc = true;
    ggml_context * c = ggml_init(ip); if (!c) return false;
    ggml_tensor * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, wt->ne[0], 1, T);
    ggml_tensor * ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, n_used, T);
    ggml_set_input(x); ggml_set_input(ids);
    ggml_tensor * y = ggml_mul_mat_id(c, wt, x, ids);
    ggml_set_output(y);
    ggml_cgraph * g = ggml_new_graph(c); ggml_build_forward_expand(g, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
    if (!buf) { ggml_free(c); return false; }
    ggml_backend_tensor_set(x, xdata.data(), 0, xdata.size()*sizeof(float));
    ggml_backend_tensor_set(ids, idsdata.data(), 0, idsdata.size()*sizeof(int32_t));
    const bool ok = ggml_backend_graph_compute(be, g) == GGML_STATUS_SUCCESS;
    if (ok) { out.resize((size_t)wt->ne[1]*n_used*T); ggml_backend_tensor_get(y,out.data(),0,out.size()*sizeof(float)); }
    ggml_backend_buffer_free(buf); ggml_free(c); return ok;
}

static bool run_hc(ggml_backend_t be, ggml_tensor * gamma, int D, int H, int T,
                   const std::vector<float> & inject, const std::vector<float> & residual,
                   const std::vector<float> & block, std::vector<float> & out) {
    ggml_init_params ip{}; ip.mem_size = 32u << 20; ip.no_alloc = true;
    ggml_context * c = ggml_init(ip); if (!c) return false;
    auto * i = ggml_new_tensor_2d(c, GGML_TYPE_F32, H, T);
    auto * r = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, H, T);
    auto * b = ggml_new_tensor_3d(c, GGML_TYPE_F32, D, 1, T);
    ggml_set_input(i); ggml_set_input(r); ggml_set_input(b);
    auto * y = ggml_hc_combine_norm(c, i, r, b, gamma, 1.f/H, 0.f, 2.f, 0.f, 1e-6f);
    ggml_set_output(y);
    ggml_cgraph * g = ggml_new_graph(c); ggml_build_forward_expand(g, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(c, be);
    if (!buf) { ggml_free(c); return false; }
    ggml_backend_tensor_set(i,inject.data(),0,inject.size()*sizeof(float));
    ggml_backend_tensor_set(r,residual.data(),0,residual.size()*sizeof(float));
    ggml_backend_tensor_set(b,block.data(),0,block.size()*sizeof(float));
    const bool ok = ggml_backend_graph_compute(be,g)==GGML_STATUS_SUCCESS;
    if (ok) { out.resize((size_t)D*H*T*2); ggml_backend_tensor_get(y,out.data(),0,out.size()*sizeof(float)); }
    ggml_backend_buffer_free(buf); ggml_free(c); return ok;
}

static void compare(const char * label, const std::vector<float> & a,
                    const std::vector<float> & b, size_t stride, size_t n) {
    (void)stride; // row 0 occupies the first contiguous output slice
    float md=0; bool exact=true; size_t arg_a=0,arg_b=0;
    for(size_t i=0;i<n;++i) {
        const float x=a[i], y=b[i];
        md=std::max(md,std::fabs(x-y)); exact &= std::memcmp(&x,&y,sizeof(float))==0;
        if(x>a[arg_a]) arg_a=i; if(y>b[arg_b]) arg_b=i;
    }
    std::printf("[delta] %s max_abs=%.9g bitwise=%d argmax_ref=%zu argmax_batch=%zu argmax_same=%d\n",
        label,md,exact,arg_a,arg_b,arg_a==arg_b);
}

static std::vector<size_t> topk(const float * p,size_t n,size_t k) {
    std::vector<size_t> ix(n); for(size_t i=0;i<n;++i) ix[i]=i;
    std::partial_sort(ix.begin(),ix.begin()+k,ix.end(),[&](size_t a,size_t b){return p[a]==p[b]?a<b:p[a]>p[b];});
    ix.resize(k); return ix;
}

int main(int argc,char **argv) {
    if(argc!=2) return 2;
    ggml_backend_t be=ggml_backend_cuda_init(0); if(!be) return 3;
    Qwen4ExpWeights w; if(!load_qwen4exp_gguf(argv[1],be,w)) return 4;
    std::printf("[meta] layers=%d n_embd=%d experts=%d topk=%d vocab=%d\n",w.n_layer,w.n_embd,w.n_expert,w.n_expert_used,w.n_vocab);

    // Enumerate all 2-D projection/router types, then test one real weight per type.
    std::map<int,ggml_tensor*> dense;
    for(auto & L:w.layers) {
        ggml_tensor * cand[]={L.hc_attn_down,L.hc_attn_up,L.hc_attn_inject,L.hc_ffn_down,L.hc_ffn_up,L.hc_ffn_inject,
            L.attn_qkv,L.attn_gate,L.wq,L.wk,L.wv,L.wo,L.indexer_q_proj,L.indexer_k_proj,
            L.ffn_gate_inp,L.ffn_gate_inp_shexp,L.ffn_gate_shexp,L.ffn_up_shexp,L.ffn_down_shexp};
        for(auto * t:cand) if(t && t->ne[1]>1 && t->ne[2]==1) dense.emplace((int)t->type,t);
    }
    for(auto [type,wt]:dense) {
        const int T1=1,T4=4; const size_t K=(size_t)wt->ne[0], M=(size_t)wt->ne[1];
        std::vector<float>x1(K),x4(K*T4),o1,o4; fill_row(x1.data(),K,0);
        std::copy(x1.begin(),x1.end(),x4.begin());
        for(int t=1;t<T4;++t) fill_row(x4.data()+K*t,K,t);
        const bool ok=run_dense(be,wt,T1,x1,o1)&&run_dense(be,wt,T4,x4,o4);
        std::printf("[dense] type=%s tensor=%s K=%zu M=%zu ok=%d\n",ggml_type_name(wt->type),wt->name,K,M,ok);
        if(ok) compare("dense.row0",o1,o4,M,M);
        // Preserve the real router's top-10 decision for this identical row.
        if(wt==w.layers[0].ffn_gate_inp && ok) {
            auto a=topk(o1.data(),M,w.n_expert_used); auto b=topk(o4.data(),M,w.n_expert_used);
            std::printf("[router] top%d_ref=",w.n_expert_used); for(auto x:a)std::printf("%zu,",x);
            std::printf(" batch=");for(auto x:b)std::printf("%zu,",x);std::printf(" same=%d\n",a==b);
        }
    }

    // The first model layer is a real routed-expert tensor (typically IQ4_NL).
    ggml_tensor * ex=nullptr;
    for(auto & L:w.layers) if(L.ffn_gate_exps && L.ffn_gate_exps->ne[2]==w.n_expert){ex=L.ffn_gate_exps;break;}
    if(ex) {
        const int U=w.n_expert_used,T=4; const size_t K=ex->ne[0],M=ex->ne[1];
        std::vector<float>x1(K),x4(K*T),a,b; fill_row(x1.data(),K,0); std::copy(x1.begin(),x1.end(),x4.begin());
        for(int t=1;t<T;++t) fill_row(x4.data()+K*t,K,t);
        std::vector<int32_t> ids1(U),ids4(U*T);
        std::vector<float> route1,route4;
        auto * router=w.layers[0].ffn_gate_inp;
        const bool routeok=run_dense(be,router,1,x1,route1)&&run_dense(be,router,T,x4,route4);
        if(!routeok) return 7;
        auto selected1=topk(route1.data(),router->ne[1],U);
        auto selected4=topk(route4.data(),router->ne[1],U);
        for(int u=0;u<U;++u) ids1[u]=ids4[u]=(int32_t)selected1[u];
        bool selection_same=true;
        for(int t=1;t<T;++t) {
            auto selected=topk(route4.data()+router->ne[1]*t,router->ne[1],U);
            for(int u=0;u<U;++u) ids4[t*U+u]=(int32_t)selected[u];
        }
        selection_same &= selected1==selected4;
        const bool ok=run_mmid(be,ex,1,U,x1,ids1,a)&&run_mmid(be,ex,T,U,x4,ids4,b);
        std::printf("[moe] type=%s tensor=%s K=%zu M=%zu E=%lld routes=%d ok=%d router_selected_experts_same=%d\n",
            ggml_type_name(ex->type),ex->name,K,M,(long long)ex->ne[2],U,ok,selection_same);
        if(ok) compare("moe.row0",a,b,M*U,M*U);
    }

    // Exercise the actual HC combine+norm op using the model's F32 HC gamma.
    ggml_tensor * gamma=w.layers[0].hc_ffn_norm; const int D=w.n_embd,H=w.n_hc,T=4;
    std::vector<float> i1(H),r1((size_t)D*H),b1(D),io((size_t)H*T),ro((size_t)D*H*T),bo((size_t)D*T),oa,ob;
    fill_row(i1.data(),i1.size(),0);fill_row(r1.data(),r1.size(),2);fill_row(b1.data(),b1.size(),3);
    std::copy(i1.begin(),i1.end(),io.begin());std::copy(r1.begin(),r1.end(),ro.begin());std::copy(b1.begin(),b1.end(),bo.begin());
    for(int t=1;t<T;++t){fill_row(io.data()+H*t,H,t);fill_row(ro.data()+(size_t)D*H*t,(size_t)D*H,t+4);fill_row(bo.data()+(size_t)D*t,D,t+8);}
    const bool hcok=run_hc(be,gamma,D,H,1,i1,r1,b1,oa)&&run_hc(be,gamma,D,H,T,io,ro,bo,ob);
    std::printf("[hc] tensor=%s type=%s elems=%zu ok=%d\n",gamma?gamma->name:"null",gamma?ggml_type_name(gamma->type):"null",gamma?ggml_nelements(gamma):0,hcok);
    if(hcok) {
        const size_t slice=(size_t)D*H;
        std::vector<float> a0(oa.begin(),oa.begin()+slice), b0(ob.begin(),ob.begin()+slice);
        std::vector<float> a1(oa.begin()+slice,oa.end()), b1(ob.begin()+slice*T,ob.begin()+slice*T+slice);
        compare("hc.row0.residual",a0,b0,slice,slice);
        compare("hc.row0.normalized",a1,b1,slice,slice);
    }

    free_qwen4exp_weights(w); ggml_backend_free(be); return 0;
}
