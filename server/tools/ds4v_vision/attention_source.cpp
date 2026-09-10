// Production attention against frozen original-source outputs.
#include "deepseek4/deepseek4_vision.h"
#include "ggml-alloc.h"
#ifdef DS4V_VISION_HIP
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unistd.h>

namespace fs=std::filesystem;
constexpr int ROWS=782, WIDTH=1024, HEADS=16, DIM=64;
constexpr size_t LIMIT=256ULL*1024*1024;
static void check(bool ok,const char *why) { if(!ok) throw std::runtime_error(why); }
static uint32_t bits(float x) { uint32_t out; std::memcpy(&out,&x,4); return out; }
static std::vector<float> load(const fs::path &path,size_t count,bool bf16=false) {
    check(fs::is_regular_file(path) && fs::file_size(path)==count*4,"wrong fixture size");
    std::vector<float> out(count); std::ifstream f(path,std::ios::binary);
    f.read(reinterpret_cast<char *>(out.data()),count*4); check(bool(f),"fixture read failed");
    for(float x:out) check(std::isfinite(x) && (!bf16 || !(bits(x)&65535)),"invalid fixture values");
    return out;
}
static size_t compare(const std::vector<float> &a,const std::vector<float> &b) {
    check(a.size()==b.size(),"comparison size mismatch"); size_t count=0;
    for(size_t i=0;i<a.size();++i) { check(std::isfinite(a[i]),"nonfinite output"); count+=bits(a[i])!=bits(b[i]); }
    return count;
}
static void save(const fs::path &path,const std::vector<float> &values) {
    std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<const char *>(values.data()),values.size()*4); check(bool(f),"output write failed");
}
struct Backend {
    ggml_backend_t value=nullptr;
    ~Backend() { if(value) ggml_backend_free(value); }
};
struct Graph {
    ggml_backend_t backend;
    ggml_context *ctx=nullptr; ggml_cgraph *graph=nullptr; ggml_gallocr_t allocator=nullptr;
    std::map<std::string,ggml_tensor *> outputs;
    std::vector<std::pair<ggml_tensor *,const std::vector<float> *>> inputs;
    explicit Graph(ggml_backend_t b):backend(b) {
        ctx=ggml_init({1024*1024,nullptr,true}); check(ctx,"metadata allocation failed");
        graph=ggml_new_graph(ctx);
    }
    ~Graph() {
        ggml_backend_synchronize(backend);
        if(allocator) ggml_gallocr_free(allocator);
        if(ctx) ggml_free(ctx);
    }
    ggml_tensor *input(const std::vector<float> &values,int64_t n0,int64_t n1,int64_t n2=1) {
        check(values.size()==size_t(n0*n1*n2),"input shape mismatch");
        auto t=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,n0,n1,n2);
        ggml_set_input(t); inputs.emplace_back(t,&values); return t;
    }
    void output(const std::string &name,ggml_tensor *t) {
        auto copy=ggml_dup(ctx,t); ggml_set_output(copy);
        check(outputs.emplace(name,copy).second,"duplicate output");
        ggml_build_forward_expand(graph,copy);
    }
    std::map<std::string,std::vector<float>> execute(const fs::path &path,const std::string &prefix) {
        for(int i=0;i<ggml_graph_n_nodes(graph);++i)
            check(ggml_backend_supports_op(backend,ggml_graph_node(graph,i)),"unsupported diagnostic graph");
        allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)); check(allocator,"allocator unavailable");
        size_t arena=0; ggml_gallocr_reserve_n_size(allocator,graph,nullptr,nullptr,&arena);
        const size_t external=dflash::vision::detail::hip_bias_workspace(backend);
        check(external<=LIMIT && arena<=LIMIT-external,"attention arena and workspace exceed 256 MiB");
        check(ggml_gallocr_reserve(allocator,graph) && ggml_gallocr_alloc_graph(allocator,graph),"allocation failed");
        check(ggml_gallocr_get_buffer_size(allocator,0)<=arena,"allocation exceeds reservation");
        for(auto &[t,values]:inputs) ggml_backend_tensor_set(t,values->data(),0,values->size()*4);
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"graph execution failed");
        ggml_backend_synchronize(backend);
        std::map<std::string,std::vector<float>> result;
        for(auto &[name,t]:outputs) {
            std::vector<float> values(ggml_nelements(t));
            ggml_backend_tensor_get(t,values.data(),0,values.size()*4);
            for(float x:values) check(std::isfinite(x),"nonfinite graph result");
            save(path/(prefix+name+".f32"),values); result.emplace(name,std::move(values));
        }
        std::cout<<prefix<<"graph_arena_bytes="<<arena<<" accounted_scratch_bytes="<<arena+external<<'\n';
        return result;
    }
};

int main(int argc,char **argv) {
    using namespace dflash::vision;
    std::cout<<std::unitbuf<<"pid="<<getpid()<<'\n';
    try {
        check(argc==4,"usage: ds4v_attention_source softmax|attention SOURCE_FIXTURES NEW_OUTPUT_DIR");
        const std::string mode=argv[1]; const fs::path source=argv[2],out=argv[3];
        check(mode=="softmax" || mode=="attention","unknown source test");
        check(!fs::exists(out),"fresh output required");
        Backend backend;
#ifdef DS4V_VISION_HIP
        check(ggml_backend_cuda_get_device_count()==1,"exactly one visible GPU required");
        backend.value=ggml_backend_cuda_init(0);
#endif
        check(backend.value && detail::hip_softmax_capable(backend.value) && detail::hip_av_capable(backend.value),
              "HIP source attention unavailable");
        std::cout<<"backend="<<ggml_backend_name(backend.value)<<" requested=hip:0 case="<<mode<<'\n';
        fs::create_directory(out);
        if(mode=="softmax") {
            for(int width:{16,128,2048,2049,2560,2562}) {
                const std::string name="n"+std::to_string(width)+"_";
                const auto scores=load(source/(name+"scores.f32"),size_t(width)*16);
                const auto expected=load(source/(name+"probabilities.f32"),scores.size());
                Graph g(backend.value);
                auto probabilities=ggml_soft_max_vision_f32(g.ctx,g.input(scores,width,16));
                g.output("probabilities",probabilities);
                const auto actual=g.execute(out,name);
                const size_t different=compare(actual.at("probabilities"),expected);
                std::cout<<"width="<<width<<" source_bitwise_mismatches="<<different<<'\n';
                check(different==0,"production softmax differs from actual Torch source");
            }
            check(detail::hip_softmax_launches(backend.value)==6 && detail::hip_av_launches(backend.value)==0 &&
                  detail::hip_rotary_launches(backend.value)==0,"softmax dispatch count changed");
            std::cout<<"actual_softmax_launches=6 actual_av_launches=0 actual_rotary_launches=0\n";
        } else {
            const auto qkv=load(source/"qkv.f32",size_t(ROWS)*WIDTH*3,true);
            std::vector<float> cosine,sine;
            detail::rotary_tables({23,34},cosine,sine,backend.value);
            check(compare(cosine,load(source/"cos.f32",cosine.size()))==0 &&
                  compare(sine,load(source/"sin.f32",sine.size()))==0,"rotary source differs");
            save(out/"cos.f32",cosine); save(out/"sin.f32",sine);
            Graph g(backend.value);
            auto input=g.input(qkv,3072,ROWS);
            auto cos=g.input(cosine,32,1,ROWS),sin=g.input(sine,32,1,ROWS);
            auto slice=[&](int offset) {
                return ggml_cont(g.ctx,ggml_view_3d(g.ctx,input,DIM,HEADS,ROWS,DIM*4,3072*4,offset*WIDTH*4));
            };
            auto q=detail::rotate(g.ctx,slice(0),cos,sin),k=detail::rotate(g.ctx,slice(1),cos,sin),v=slice(2);
            auto attention=detail::attention(g.ctx,q,k,v,backend.value);
            g.output("attention",attention); g.output("q",q); g.output("k",k); g.output("v",v);
            ggml_tensor *probabilities=nullptr,*precast=nullptr;
            int softmax_count=0,av_count=0;
            for(int i=0;i<ggml_graph_n_nodes(g.graph);++i) {
                auto node=ggml_graph_node(g.graph,i);
                if(node->op==GGML_OP_SOFT_MAX_VISION_F32) { probabilities=node; ++softmax_count; }
                if(node->op==GGML_OP_MUL_MAT_VISION_AV_F32) { precast=node; ++av_count; }
                check(node->op!=GGML_OP_SOFT_MAX,"generic softmax in HIP attention");
            }
            check(softmax_count==1 && av_count==1 && probabilities && precast,"attention operation count changed");
            g.output("scores",probabilities->src[0]); g.output("probabilities",probabilities); g.output("precast_av",precast);
            const auto actual=g.execute(out,"");
            for(const auto &[name,values]:actual) {
                const size_t different=compare(values,load(source/(name+".f32"),values.size()));
                std::cout<<name<<"_source_bitwise_mismatches="<<different<<'\n';
                check(different==0,"combined production attention differs from source");
            }
            check(detail::hip_softmax_launches(backend.value)==1 && detail::hip_av_launches(backend.value)==1 &&
                  detail::hip_rotary_launches(backend.value)==1,"combined attention dispatch count changed");
            std::cout<<"actual_softmax_launches=1 actual_av_launches=1 actual_rotary_launches=1\n";
        }
        check(detail::hip_bias_launches(backend.value)==0 && detail::hip_norm_launches(backend.value)==0,
              "unexpected linear or normalization dispatch");
        std::cout<<"actual_lt_launches=0 actual_norm_launches=0\n";
        std::cout<<"PASS: production "<<mode<<" matches original source bitwise\n";
        return 0;
    } catch(const std::exception &error) { std::cerr<<"FAIL: "<<error.what()<<'\n'; return 1; }
}
