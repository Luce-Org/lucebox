// Original-source fixtures through the production normalization helper.
#include "deepseek4/deepseek4_vision.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#ifdef DS4V_VISION_HIP
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace fs=std::filesystem;
constexpr int columns=1024,source_rows=782;
static void check(bool ok,const char *why) { if(!ok) throw std::runtime_error(why); }
static uint32_t bits(float value) { uint32_t out; std::memcpy(&out,&value,4); return out; }
static std::vector<float> read(const fs::path &path,size_t count,bool bf16=false) {
    check(fs::is_regular_file(path) && fs::file_size(path)==count*4,"wrong fixture size");
    std::vector<float> values(count); std::ifstream file(path,std::ios::binary);
    file.read(reinterpret_cast<char *>(values.data()),values.size()*4); check(bool(file),"fixture read failed");
    for(float value:values) check(std::isfinite(value) && (!bf16 || !(bits(value)&65535)),"invalid fixture values");
    return values;
}
static std::vector<float> tile(const std::vector<float> &source,int rows) {
    check(source.size()==size_t(source_rows)*columns,"wrong fixed source shape");
    std::vector<float> result(size_t(rows)*columns);
    for(int row=0;row<rows;++row)
        std::copy_n(source.data()+size_t(row%source_rows)*columns,columns,result.data()+size_t(row)*columns);
    return result;
}
struct Resources {
    ggml_backend_t backend=nullptr; ggml_context *context=nullptr; ggml_gallocr_t allocator=nullptr;
    ~Resources() {
        if(backend) ggml_backend_synchronize(backend);
        if(allocator) ggml_gallocr_free(allocator);
        if(context) ggml_free(context);
        if(backend) ggml_backend_free(backend);
    }
};
int main(int argc,char **argv) {
    std::cout<<std::unitbuf<<"pid="<<getpid()<<'\n';
    if(argc!=5) { std::cerr<<"usage: ds4v_norm_source hip:0 782|2562 SOURCE_FIXTURES NEW_OUTPUT_DIR\n"; return 1; }
    try {
        check(std::string(argv[1])=="hip:0","HIP source qualification only");
        const int rows=std::stoi(argv[2]); check(rows==782 || rows==2562,"fixed qualification row count required");
        const fs::path fixtures=argv[3],out=argv[4]; check(!fs::exists(out),"fresh output required");
        auto inputs=tile(read(fixtures/"inputs.f32",size_t(source_rows)*columns,true),rows);
        auto weights=read(fixtures/"weights.f32",columns,true);
        std::vector<ggml_bf16_t> packed; for(float x:weights) packed.push_back({uint16_t(bits(x)>>16)});
        std::vector<std::vector<float>> expected;
        for(const char *name:{"scaled","weighted","output"}) expected.push_back(tile(read(fixtures/(std::string(name)+".f32"),size_t(source_rows)*columns),rows));
        Resources owner;
#ifdef DS4V_VISION_HIP
        check(ggml_backend_cuda_get_device_count()==1,"exactly one visible GPU required");
        owner.backend=ggml_backend_cuda_init(0);
#endif
        check(owner.backend && dflash::vision::detail::hip_norm_capable(owner.backend),"source-order HIP normalization unavailable");
        std::cout<<"backend="<<ggml_backend_name(owner.backend)<<" requested=hip:0 rows="<<rows<<" columns="<<columns<<'\n';
        owner.context=ggml_init({1024*1024,nullptr,true}); check(owner.context,"metadata allocation failed");
        auto x=ggml_new_tensor_2d(owner.context,GGML_TYPE_F32,columns,rows);
        auto w=ggml_new_tensor_1d(owner.context,GGML_TYPE_BF16,columns);
        ggml_set_input(x); ggml_set_input(w);
        auto scaled=dflash::vision::detail::rms_norm(owner.context,x,1e-6f,owner.backend);
        auto weighted=ggml_mul(owner.context,scaled,ggml_cast(owner.context,w,GGML_TYPE_F32));
        auto rounded=ggml_cast(owner.context,ggml_cast(owner.context,weighted,GGML_TYPE_BF16),GGML_TYPE_F32);
        ggml_tensor *outputs[]={scaled,weighted,rounded};
        auto graph=ggml_new_graph(owner.context);
        for(auto tensor:outputs) { ggml_set_output(tensor); ggml_build_forward_expand(graph,tensor); }
        size_t explicit_ops=0;
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto node=ggml_graph_node(graph,i);
            check(ggml_backend_supports_op(owner.backend,node),"unsupported normalization graph operation");
            explicit_ops+=node->op==GGML_OP_RMS_NORM_VISION_F32;
            check(node->op!=GGML_OP_RMS_NORM,"generic norm appeared in source graph");
        }
        check(explicit_ops==1,"one explicit normalization operation required");
        owner.allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.backend));
        check(owner.allocator,"allocator unavailable"); size_t arena=0;
        ggml_gallocr_reserve_n_size(owner.allocator,graph,nullptr,nullptr,&arena);
        check(arena<=128ULL*1024*1024,"normalization graph exceeds 128 MiB");
        check(ggml_gallocr_reserve(owner.allocator,graph) && ggml_gallocr_alloc_graph(owner.allocator,graph),"allocation failed");
        check(ggml_gallocr_get_buffer_size(owner.allocator,0)<=arena,"allocation exceeds reservation");
        ggml_backend_tensor_set(x,inputs.data(),0,inputs.size()*4);
        ggml_backend_tensor_set(w,packed.data(),0,packed.size()*2);
        const size_t before=dflash::vision::detail::hip_norm_launches(owner.backend);
        check(ggml_backend_graph_compute(owner.backend,graph)==GGML_STATUS_SUCCESS,"normalization graph failed");
        ggml_backend_synchronize(owner.backend);
        const size_t after=dflash::vision::detail::hip_norm_launches(owner.backend);
        check(after>=before && after-before==1,"actual source normalization dispatch mismatch");
        const size_t lt_launches=dflash::vision::detail::hip_bias_launches(owner.backend);
        check(lt_launches==0,"normalization unexpectedly submitted Lt");
        fs::create_directory(out); size_t total=0;
        const char *names[]={"scaled","weighted","output"};
        for(int field=0;field<3;++field) {
            std::vector<float> actual(inputs.size());
            ggml_backend_tensor_get(outputs[field],actual.data(),0,actual.size()*4);
            size_t mismatches=0;
            for(size_t i=0;i<actual.size();++i) {
                check(std::isfinite(actual[i]),"nonfinite normalization output");
                mismatches+=bits(actual[i])!=bits(expected[field][i]);
            }
            std::ofstream file(out/(std::string(names[field])+".f32"),std::ios::binary);
            file.write(reinterpret_cast<const char *>(actual.data()),actual.size()*4); check(bool(file),"output write failed");
            std::cout<<names[field]<<"_source_mismatches="<<mismatches<<'\n'; total+=mismatches;
        }
        std::cout<<"explicit_norm_ops=1 actual_norm_launches=1 actual_lt_launches="<<lt_launches<<" graph_arena_bytes="<<arena<<'\n';
        std::cout<<"source_bitwise_mismatches="<<total<<" reference_layout="<<(rows==source_rows?"original":"rowwise-tiled-original")<<'\n';
        if(total) return 3;
        std::cout<<"PASS: production normalization matches original source intermediates\n";
        return 0;
    } catch(const std::exception &e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
