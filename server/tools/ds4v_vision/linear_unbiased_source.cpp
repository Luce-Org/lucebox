// Frozen original-source MLP fixtures through the production unbiased helper.
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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

static_assert(GGML_OP_PAGED_ATTN==104 && GGML_OP_MUL_MAT_BIAS_BF16==105 && GGML_OP_RMS_NORM_VISION_F32==106 && GGML_OP_COUNT==107,
              "operation ABI changed unexpectedly");
static void check(bool ok,const char *why) { if(!ok) throw std::runtime_error(why); }
static size_t product(size_t a,size_t b) {
    check(b==0 || a<=std::numeric_limits<size_t>::max()/b,"size overflow");
    return a*b;
}
static uint32_t bits(float value) { uint32_t result; std::memcpy(&result,&value,4); return result; }
static std::vector<float> load(const std::filesystem::path &path,size_t count) {
    const size_t bytes=product(count,sizeof(float));
    check(bytes<=size_t(std::numeric_limits<std::streamsize>::max()),"fixture exceeds stream limit");
    check(std::filesystem::is_regular_file(path) && std::filesystem::file_size(path)==bytes,"fixture size mismatch");
    std::vector<float> result(count);
    std::ifstream file(path,std::ios::binary);
    file.read(reinterpret_cast<char *>(result.data()),std::streamsize(bytes));
    check(bool(file),"fixture read failed");
    for(float value:result)
        check(std::isfinite(value) && !(bits(value)&65535),"fixture must contain finite exact BF16 values");
    return result;
}
static std::vector<ggml_bf16_t> pack(const std::vector<float> &values) {
    std::vector<ggml_bf16_t> result; result.reserve(values.size());
    for(float value:values) result.push_back({uint16_t(bits(value)>>16)});
    return result;
}
struct Resources {
    ggml_backend_t backend=nullptr;
    ggml_context *context=nullptr;
    ggml_gallocr_t allocator=nullptr;
    ~Resources() {
        if(backend) ggml_backend_synchronize(backend);
        if(allocator) ggml_gallocr_free(allocator);
        if(context) ggml_free(context);
        if(backend) ggml_backend_free(backend);
    }
};
int main(int argc,char **argv) {
    std::cout<<std::unitbuf<<"pid="<<getpid()<<'\n';
    if(argc!=6) {
        std::cerr<<"usage: ds4v_linear_unbiased_source cpu|hip:0 mlp_w1|mlp_w2 FIXTURE_DIR REFERENCE_F32 NEW_OUTPUT_DIR\n";
        return 1;
    }
    try {
        const std::string device=argv[1],selected=argv[2];
        check(device=="cpu" || device=="hip:0","unrecognized backend");
        check(selected=="mlp_w1" || selected=="mlp_w2","unrecognized fixed MLP shape");
        const int k=selected=="mlp_w1"?1024:2816,m=selected=="mlp_w1"?5632:1024,n=782;
        const size_t elements=product(size_t(m),size_t(n)),output_bytes=product(elements,sizeof(float));
        const std::filesystem::path input=argv[3],output=argv[5];
        check(!std::filesystem::exists(output),"output directory already exists");
        auto weights=pack(load(input/"weights.f32",product(size_t(k),size_t(m))));
        auto inputs=load(input/"inputs.f32",product(size_t(k),size_t(n)));
        auto reference=load(argv[4],elements);
        Resources owner;
        if(device=="cpu") {
            owner.backend=ggml_backend_cpu_init();
            if(owner.backend) ggml_backend_cpu_set_n_threads(owner.backend,2);
        }
#ifdef DS4V_VISION_HIP
        else {
            check(ggml_backend_cuda_get_device_count()==1,"exactly one visible GPU required");
            owner.backend=ggml_backend_cuda_init(0);
        }
#endif
        check(owner.backend,"requested backend unavailable");
        std::cout<<"backend="<<ggml_backend_name(owner.backend)<<" requested="<<device
                 <<" case="<<selected<<" k="<<k<<" m="<<m<<" n="<<n<<'\n';
        owner.context=ggml_init({1024*1024,nullptr,true});
        check(owner.context,"metadata allocation failed");
        auto w=ggml_new_tensor_2d(owner.context,GGML_TYPE_BF16,k,m);
        auto x=ggml_new_tensor_2d(owner.context,GGML_TYPE_F32,k,n);
        ggml_set_input(w); ggml_set_input(x);
        auto y=dflash::vision::detail::linear(owner.context,w,x,nullptr,device=="hip:0",owner.backend);
        check(y && y->type==GGML_TYPE_F32 && size_t(ggml_nelements(y))==elements,"unexpected output layout");
        ggml_set_output(y);
        auto graph=ggml_new_graph(owner.context);
        ggml_build_forward_expand(graph,y);
        size_t explicit_ops=0;
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto node=ggml_graph_node(graph,i);
            check(ggml_backend_supports_op(owner.backend,node),"backend does not support graph; fallback forbidden");
            if(node->op==GGML_OP_MUL_MAT_BIAS_BF16) {
                ++explicit_ops;
                check(node->src[2]==nullptr,"unbiased graph unexpectedly has a bias operand");
            }
        }
        check(explicit_ops==(device=="hip:0"?1u:0u),"wrong explicit unbiased graph dispatch");
        const size_t external=dflash::vision::detail::hip_bias_workspace(owner.backend);
        check(external==(device=="hip:0"?76ULL*1024*1024:0),"wrong fixed HIP workspace capability");
        owner.allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(owner.backend));
        check(owner.allocator,"graph allocator creation failed");
        size_t arena=0;
        ggml_gallocr_reserve_n_size(owner.allocator,graph,nullptr,nullptr,&arena);
        constexpr size_t limit=128ULL*1024*1024;
        check(external<=limit && arena<=limit-external,"graph arena plus workspace exceeds 128 MiB");
        check(ggml_gallocr_reserve(owner.allocator,graph),"graph reservation failed");
        check(ggml_gallocr_alloc_graph(owner.allocator,graph),"graph allocation failed");
        check(ggml_gallocr_get_buffer_size(owner.allocator,0)<=arena,"actual arena exceeds reservation estimate");
        ggml_backend_tensor_set(w,weights.data(),0,product(weights.size(),sizeof(ggml_bf16_t)));
        ggml_backend_tensor_set(x,inputs.data(),0,product(inputs.size(),sizeof(float)));
        const size_t before=dflash::vision::detail::hip_bias_launches(owner.backend);
        check(ggml_backend_graph_compute(owner.backend,graph)==GGML_STATUS_SUCCESS,"graph execution failed");
        ggml_backend_synchronize(owner.backend);
        const size_t after=dflash::vision::detail::hip_bias_launches(owner.backend);
        check(after>=before && after-before==explicit_ops,"wrong actual Lt submission count");
        const size_t norm_launches=dflash::vision::detail::hip_norm_launches(owner.backend);
        check(norm_launches==0,"linear graph unexpectedly submitted vision normalization");
        std::vector<float> actual(elements);
        ggml_backend_tensor_get(y,actual.data(),0,output_bytes);
        size_t mismatches=0; double max_abs=0;
        for(size_t i=0;i<elements;++i) {
            check(std::isfinite(actual[i]) && !(bits(actual[i])&65535),"output must contain finite exact BF16 values");
            mismatches+=bits(actual[i])!=bits(reference[i]);
            max_abs=std::max(max_abs,std::abs(double(actual[i])-reference[i]));
        }
        check(std::filesystem::create_directories(output),"output directory creation failed");
        std::ofstream file(output/"unbiased.f32",std::ios::binary);
        file.write(reinterpret_cast<const char *>(actual.data()),std::streamsize(output_bytes));
        file.close(); check(bool(file),"output write failed");
        std::cout<<"explicit_unbiased_ops="<<explicit_ops<<" actual_lt_launches="<<after-before
                 <<" actual_norm_launches="<<norm_launches
                 <<" retained_workspace_bytes="<<external<<" graph_arena_bytes="<<arena
                 <<" accounted_scratch_bytes="<<arena+external<<'\n';
        std::cout<<"elements="<<elements<<" source_bitwise_mismatches="<<mismatches<<" max_abs="<<max_abs<<'\n';
        std::cout<<(mismatches?"FAIL":"PASS")<<": frozen original-source unbiased MLP comparison\n";
        return mismatches?3:0;
    } catch(const std::exception &error) {
        std::cerr<<"ERROR: "<<error.what()<<'\n';
        return 1;
    }
}
