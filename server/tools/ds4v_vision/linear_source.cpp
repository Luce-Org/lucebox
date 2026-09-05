// Exact frozen original-source fixtures through the production vision helper.
#include "deepseek4/deepseek4_vision.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#ifdef DS4V_VISION_HIP
#include "ggml-cuda.h"
#endif
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <unistd.h>
static_assert(GGML_OP_PAGED_ATTN==104 && GGML_OP_MUL_MAT_BIAS_BF16==105 && GGML_OP_COUNT==106,"operation ABI changed unexpectedly");
static void check(bool b,const char *s) { if(!b) throw std::runtime_error(s); }
static uint32_t bits(float v) { uint32_t b; std::memcpy(&b,&v,4); return b; }
static std::vector<float> load(const std::filesystem::path&p,size_t n) {
    check(std::filesystem::file_size(p)==n*4,"fixture size mismatch");
    std::vector<float> v(n); std::ifstream f(p,std::ios::binary); f.read((char*)v.data(),n*4);
    check(bool(f),"fixture read failed");
    for(float x:v) check(std::isfinite(x) && !(bits(x)&65535),"fixture not exact finite BF16");
    return v;
}
static std::vector<ggml_bf16_t> pack(const std::vector<float>&v) {
    std::vector<ggml_bf16_t> r; for(float x:v) r.push_back({uint16_t(bits(x)>>16)}); return r;
}
struct Graph {
    ggml_context *c=ggml_init({1024*1024,nullptr,true});
    ggml_gallocr_t a;
    explicit Graph(ggml_backend_t b):a(ggml_gallocr_new(ggml_backend_get_default_buffer_type(b))) { check(c&&a,"graph creation failed"); }
    ~Graph() { ggml_gallocr_free(a); ggml_free(c); }
};
int main(int argc,char**argv) {
    std::cout<<std::unitbuf<<"pid="<<getpid()<<'\n';
    if(argc!=6) { std::cerr<<"usage: ds4v_linear_source cpu|hip:0 tiny|patch|qkv FIXTURE_DIR REFERENCE_F32 NEW_OUTPUT_DIR\n"; return 2; }
    ggml_backend_t backend=nullptr;
    try {
        const std::string dev=argv[1],shape=argv[2];
        check(shape=="tiny"||shape=="patch"||shape=="qkv","unrecognized fixed shape");
        const int k=shape=="tiny"?64:shape=="patch"?588:1024;
        const int m=shape=="tiny"?32:shape=="patch"?1024:3072,n=shape=="tiny"?32:782;
        std::filesystem::path in=argv[3],out=argv[5]; check(!std::filesystem::exists(out),"output exists");
        auto wf=load(in/"weights.f32",size_t(k)*m),xv=load(in/"inputs.f32",size_t(k)*n),bf=load(in/"bias.f32",m);
        auto ref=load(argv[4],size_t(m)*n); auto wv=pack(wf),bv=pack(bf);
        if(dev=="cpu") { backend=ggml_backend_cpu_init(); if(backend) ggml_backend_cpu_set_n_threads(backend,2); }
#ifdef DS4V_VISION_HIP
        else if(dev=="hip:0") { check(ggml_backend_cuda_get_device_count()==1,"one visible GPU required"); backend=ggml_backend_cuda_init(0); }
#endif
        check(backend,"requested backend unavailable");
        std::cout<<"backend="<<ggml_backend_name(backend)<<" shape="<<shape<<" k="<<k<<" m="<<m<<" n="<<n<<'\n';
        int status;
        {
            Graph owner(backend); auto c=owner.c;
            auto w=ggml_new_tensor_2d(c,GGML_TYPE_BF16,k,m),x=ggml_new_tensor_2d(c,GGML_TYPE_F32,k,n),b=ggml_new_tensor_1d(c,GGML_TYPE_BF16,m);
            for(auto t:{w,x,b}) ggml_set_input(t);
            auto device=ggml_backend_get_device(backend);
            bool preserve=device&&(ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_GPU||ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_IGPU);
            auto y=dflash::vision::detail::linear(c,w,x,b,preserve,backend);
            // CPU retains its raw-dot/final-cast contract. HIP unbiased output is diagnostic here;
            // original-source MLP fixtures are checked by ds4v_linear_unbiased_source.
            auto u=dflash::vision::detail::linear(c,w,x,nullptr,preserve,backend);
            ggml_tensor * expected_u=nullptr;
            if(dev=="cpu") {
                auto raw=ggml_mul_mat(c,w,x); ggml_mul_mat_set_prec(raw,GGML_PREC_F32);
                expected_u=ggml_cast(c,ggml_cast(c,raw,GGML_TYPE_BF16),GGML_TYPE_F32);
            }
            auto g=ggml_new_graph(c);
            for(auto t:{y,u,expected_u}) if(t) { ggml_set_output(t); ggml_build_forward_expand(g,t); }
            const size_t external=dflash::vision::detail::hip_bias_workspace(backend);
            size_t ops=0;
            for(int i=0;i<ggml_graph_n_nodes(g);++i) ops+=ggml_graph_node(g,i)->op==GGML_OP_MUL_MAT_BIAS_BF16;
            check(ops==(dev=="hip:0"?2u:0u),"wrong actual fused-op graph dispatch");
            check((external!=0)==(dev=="hip:0"),"wrong HIP-only capability");
            const size_t before=dflash::vision::detail::hip_bias_launches(backend);
            size_t scratch=0; ggml_gallocr_reserve_n_size(owner.a,g,nullptr,nullptr,&scratch);
            constexpr size_t limit=128ULL*1024*1024;
            check(external<=limit && scratch<=limit-external,"graph arena plus workspace exceeds fixed bound");
            for(int i=0;i<ggml_graph_n_nodes(g);++i) check(ggml_backend_supports_op(backend,ggml_graph_node(g,i)),"unsupported node");
            check(ggml_gallocr_reserve(owner.a,g),"graph reservation failed");
            check(ggml_gallocr_alloc_graph(owner.a,g),"graph allocation failed");
            ggml_backend_tensor_set(w,wv.data(),0,wv.size()*2); ggml_backend_tensor_set(b,bv.data(),0,bv.size()*2); ggml_backend_tensor_set(x,xv.data(),0,xv.size()*4);
            check(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS,"compute failed");
            const size_t launches=dflash::vision::detail::hip_bias_launches(backend)-before;
            check(launches==ops,"actual Lt launch count differs from explicit op count");
            std::cout<<"fused_graph_ops="<<ops<<" actual_lt_launches="<<launches<<" retained_workspace_bytes="<<external<<'\n';
            std::vector<float> actual(ref.size()),unbiased(ref.size()),expected(ref.size());
            ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4); ggml_backend_tensor_get(u,unbiased.data(),0,unbiased.size()*4); if(expected_u) ggml_backend_tensor_get(expected_u,expected.data(),0,expected.size()*4);
            size_t bad=0,ubad=0; float maxabs=0;
            for(size_t i=0;i<actual.size();++i) { check(std::isfinite(actual[i])&&std::isfinite(unbiased[i]),"nonfinite output"); bad+=bits(actual[i])!=bits(ref[i]); if(expected_u) ubad+=bits(unbiased[i])!=bits(expected[i]); maxabs=std::max(maxabs,std::abs(actual[i]-ref[i])); }
            std::filesystem::create_directories(out);
            for(auto item:{std::make_pair("biased.f32",&actual),std::make_pair("unbiased.f32",&unbiased)}) { std::ofstream f(out/item.first,std::ios::binary); f.write((char*)item.second->data(),item.second->size()*4); check(bool(f),"output write failed"); }
            std::cout<<"elements="<<actual.size()<<" source_bitwise_mismatches="<<bad<<" maxabs="<<maxabs<<" unbiased_validation="<<(expected_u?"CPU-generic-preservation":"HIP-diagnostic-only")
                     <<" unbiased_contract_mismatches="<<(expected_u?std::to_string(ubad):"not_applicable")<<" graph_scratch_bytes="<<scratch<<'\n';
            status=(bad||ubad)?3:0;
        }
        ggml_backend_free(backend); return status;
    } catch(const std::exception&e) { std::cerr<<"ERROR: "<<e.what()<<'\n'; if(backend)ggml_backend_free(backend); return 1; }
}
