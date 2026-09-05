#include "deepseek4/deepseek4_vision.h"
#include "ggml.h"
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

static void check(bool ok,const char * message) { if(!ok) throw std::runtime_error(message); }
static uint32_t bits(float value) { uint32_t b; std::memcpy(&b,&value,4); return b; }
static float from_bits(uint32_t b) { float value; std::memcpy(&value,&b,4); return value; }
// Independent nearest-even BF16 oracle. The fixture is finite and every F64
// dot/sum is exactly representable in F32, so accumulation order cannot affect it.
static float round_bf16(double exact) {
    float value=static_cast<float>(exact);
    check(double(value)==exact,"oracle value is not exactly representable in F32");
    uint32_t b=bits(value), high=b>>16, low=b&65535;
    if(low>32768 || (low==32768 && (high&1))) ++high;
    return from_bits(high<<16);
}
static std::vector<ggml_bf16_t> pack(const std::vector<float> & values) {
    std::vector<ggml_bf16_t> out;
    for(float value:values) { check((bits(value)&65535)==0,"fixture input is not exact BF16"); out.push_back({uint16_t(bits(value)>>16)}); }
    return out;
}
static void save(const std::filesystem::path & path,const std::vector<float> & values) {
    std::ofstream file(path,std::ios::binary); file.write(reinterpret_cast<const char *>(values.data()),values.size()*4);
    check(bool(file),"output write failed");
}
static std::vector<float> read(ggml_tensor * tensor) {
    std::vector<float> values(ggml_nelements(tensor)); ggml_backend_tensor_get(tensor,values.data(),0,values.size()*4); return values;
}
struct Graph {
    ggml_context * context=ggml_init({1024*1024,nullptr,true});
    ggml_gallocr_t allocator;
    explicit Graph(ggml_backend_t backend):allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {
        check(context && allocator,"graph creation failed");
    }
    ~Graph() { ggml_gallocr_free(allocator); ggml_free(context); }
};
int main(int argc,char ** argv) {
    if(argc!=3) { std::cerr<<"usage: ds4v_linear_rounding cpu|hip:0 NEW_OUTPUT_DIR\n"; return 2; }
    ggml_backend_t backend=nullptr;
    const std::string device=argv[1];
    if(device=="cpu") { backend=ggml_backend_cpu_init(); if(backend) ggml_backend_cpu_set_n_threads(backend,2); }
#ifdef DS4V_VISION_HIP
    else if(device=="hip:0" && ggml_backend_cuda_get_device_count()>0) backend=ggml_backend_cuda_init(0);
#endif
    if(!backend) { std::cerr<<"requested backend unavailable\n"; return 1; }
    int status=1;
    try {
        const std::filesystem::path directory=argv[2];
        check(!std::filesystem::exists(directory),"output directory already exists");
        std::filesystem::create_directories(directory);
        std::cout<<"backend="<<ggml_backend_name(backend)<<" requested="<<device<<'\n';
        const auto backend_device=ggml_backend_get_device(backend);
        const bool preserve=backend_device && (ggml_backend_dev_type(backend_device)==GGML_BACKEND_DEVICE_TYPE_GPU ||
                                              ggml_backend_dev_type(backend_device)==GGML_BACKEND_DEVICE_TYPE_IGPU);
        std::cout<<"preserve_biased_product="<<preserve<<'\n';
        constexpr int k=64,m=32,n=32; // N>16: exercises ROCm BLAS instead of small-N MMF.
        std::vector<float> weights(k*m,0),inputs(k*n,0),bias(m);
        for(int row=0;row<m;++row) {
            const float sign=row%2 ? -1.f : 1.f;
            weights[row*k]=sign;
            weights[row*k+1]=sign*(row%4<2 ? 1.f/256 : 3.f/512);
            bias[row]=sign*(row%4<2 ? 1.f/256 : -1.f/512);
        }
        for(int column=0;column<n;++column) {
            inputs[column*k]=column%3==0 ? -1.f : 1.f;
            inputs[column*k+1]=column%3==2 ? -1.f : 1.f;
        }
        auto wbits=pack(weights),bbits=pack(bias); pack(inputs);
        std::vector<float> expected(m*n),premature(m*n),dot(m*n),rounded_dot(m*n);
        size_t sensitive=0;
        for(int column=0;column<n;++column) for(int row=0;row<m;++row) {
            double sum=0; for(int j=0;j<k;++j) sum+=double(weights[row*k+j])*inputs[column*k+j];
            const size_t i=column*m+row;
            dot[i]=float(sum); check(double(dot[i])==sum,"dot is not exact F32");
            rounded_dot[i]=round_bf16(sum);
            expected[i]=round_bf16(sum+double(bias[row]));
            premature[i]=round_bf16(double(rounded_dot[i])+double(bias[row]));
            sensitive+=expected[i]!=premature[i];
        }
        check(sensitive>=256,"fixture fails to distinguish intermediate rounding");
        check(expected[m]==1.0078125f && premature[m]==1.f,"positive tie oracle changed");
        check(expected[m+1]==-1.0078125f && premature[m+1]==-1.f,"negative tie oracle changed");
        Graph owner(backend); auto c=owner.context;
        auto w=ggml_new_tensor_2d(c,GGML_TYPE_BF16,k,m);
        auto x=ggml_new_tensor_2d(c,GGML_TYPE_F32,k,n);
        auto b=ggml_new_tensor_1d(c,GGML_TYPE_BF16,m);
        for(auto t:{w,x,b}) ggml_set_input(t);
        auto raw_dot=ggml_mul_mat(c,w,x); ggml_mul_mat_set_prec(raw_dot,GGML_PREC_F32);
        auto actual=dflash::vision::detail::linear(c,w,x,b,preserve);
        auto unbiased=dflash::vision::detail::linear(c,w,x,nullptr,preserve);
        auto graph=ggml_new_graph(c);
        for(auto t:{raw_dot,actual,unbiased}) { ggml_set_output(t); ggml_build_forward_expand(graph,t); }
        size_t required=0; ggml_gallocr_reserve_n_size(owner.allocator,graph,nullptr,nullptr,&required);
        check(required<16*1024*1024,"tiny graph scratch unexpectedly large");
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) check(ggml_backend_supports_op(backend,ggml_graph_node(graph,i)),"unsupported test graph operation");
        check(ggml_gallocr_reserve(owner.allocator,graph),"graph reservation failed");
        check(ggml_gallocr_alloc_graph(owner.allocator,graph),"graph allocation failed");
        ggml_backend_tensor_set(w,wbits.data(),0,wbits.size()*sizeof(ggml_bf16_t));
        ggml_backend_tensor_set(x,inputs.data(),0,inputs.size()*4);
        ggml_backend_tensor_set(b,bbits.data(),0,bbits.size()*sizeof(ggml_bf16_t));
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"graph execution failed");
        auto values=read(actual),dots=read(raw_dot),no_bias=read(unbiased);
        size_t mismatch=0,early_match=0,dot_exact=0,dot_rounded=0,no_bias_mismatch=0,no_bias_raw=0,no_bias_round_raw=0;
        float max_abs=0;
        for(size_t i=0;i<values.size();++i) {
            check(std::isfinite(values[i]) && std::isfinite(dots[i]) && std::isfinite(no_bias[i]),"non-finite output");
            mismatch+=values[i]!=expected[i]; early_match+=values[i]==premature[i];
            dot_exact+=dots[i]==dot[i]; dot_rounded+=dots[i]==rounded_dot[i];
            no_bias_mismatch+=no_bias[i]!=rounded_dot[i]; no_bias_raw+=no_bias[i]!=dots[i];
            // Original Torch HIP BF16 GEMM shares the native raw-product tie
            // behavior. Gate the explicit final boundary, not universal GEMM RNE.
            no_bias_round_raw+=no_bias[i]!=round_bf16(double(dots[i]));
            max_abs=std::max(max_abs,std::abs(values[i]-expected[i]));
        }
        for(const auto & item:std::vector<std::pair<const char *,const std::vector<float> *>>{
                {"weights.f32",&weights},{"inputs.f32",&inputs},{"bias.f32",&bias},{"expected.f32",&expected},
                {"premature.f32",&premature},{"actual.f32",&values},{"raw-dot.f32",&dots},{"exact-dot.f32",&dot},
                {"unbiased.f32",&no_bias}}) save(directory/item.first,*item.second);
        std::cout<<"shape="<<k<<","<<m<<","<<n<<" elements="<<values.size()<<" sensitive="<<sensitive
                 <<" mismatches="<<mismatch<<" max_abs="<<max_abs<<" premature_matches="<<early_match
                 <<" dot_exact_matches="<<dot_exact<<" dot_rounded_matches="<<dot_rounded
                 <<" unbiased_rne_mismatches="<<no_bias_mismatch<<" unbiased_raw_mismatches="<<no_bias_raw
                 <<" unbiased_round_raw_mismatches="<<no_bias_round_raw<<" scratch_bytes="<<required<<'\n';
        std::cout<<"example expected="<<expected[m]<<" premature="<<premature[m]<<" actual="<<values[m]<<" raw_dot="<<dots[m]<<'\n';
        status=(mismatch || no_bias_round_raw) ? 3 : 0;
        std::cout<<(status==0 ? "PASS" : "ISSUES")<<": biased linear must round only after bias\n";
    } catch(const std::exception & error) { std::cerr<<"ERROR: "<<error.what()<<'\n'; }
    ggml_backend_free(backend);
    return status;
}
