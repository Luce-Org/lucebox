#include "deepseek4/deepseek4_norm.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
static void check(bool ok,const char * msg) { if(!ok) throw std::runtime_error(msg); }
static float decode(uint16_t v) { uint32_t b=uint32_t(v)<<16;float f;std::memcpy(&f,&b,4);return f; }
int main(int argc,char ** argv) {
    if(argc!=3) { std::cerr<<"usage: ds4_bf16_affine cpu|hip:0 contract|execute\n";return 2; }
    const bool gpu=std::string(argv[1])=="hip:0",contract=std::string(argv[2])=="contract";
    if(!gpu && std::string(argv[1])!="cpu") return 2;
    if(!contract && std::string(argv[2])!="execute") return 2;
    ggml_backend_t backend=gpu?ggml_backend_cuda_init(0):ggml_backend_cpu_init();
    if(!backend) return 1;
    if(!gpu) ggml_backend_cpu_set_n_threads(backend,2);
    std::cout<<"backend="<<ggml_backend_name(backend)<<" requested="<<argv[1]<<std::endl;
    int status=0;
    try {
        for(int n:{1,24}) {
            constexpr int k=4096;constexpr float eps=1e-6f;
            auto ctx=ggml_init({2*1024*1024,nullptr,true});check(ctx,"context");
            auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,k,n);
            auto w=ggml_new_tensor_1d(ctx,GGML_TYPE_BF16,k);
            auto control=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,k);
            auto f16=ggml_new_tensor_1d(ctx,GGML_TYPE_F16,k);
            auto y=dflash::common::detail::build_rms_norm(ctx,x,w,eps);
            auto z=dflash::common::detail::build_rms_norm(ctx,x,control,eps);
            auto half_graph=dflash::common::detail::build_rms_norm(ctx,x,f16,eps);
            check(z->src[1]==control && half_graph->src[1]==f16,"F32/F16 graph changed");
            const bool affine_f32=y->src[1]->type==GGML_TYPE_F32;
            std::cout<<"n="<<n<<" affine_type="<<ggml_type_name(y->src[1]->type)<<" F16_F32_graph_unchanged=1"<<std::endl;
            if(contract) { status|=!affine_f32;ggml_free(ctx);continue; }
            if(affine_f32) check(ggml_nbytes(y->src[1])==4*k && y->src[1]->src[0]==w,"not vector-only cast");
            std::vector<float> input(k*n),weight(k);std::vector<uint16_t> raw(k),after(k);
            for(int i=0;i<k;++i) { raw[i]=uint16_t(0x3f00+(i%128));if(i%7==0)raw[i]|=0x8000;weight[i]=decode(raw[i]); }
            for(size_t i=0;i<input.size();++i) input[i]=float(int(i%31)-15)/16;
            auto g=ggml_new_graph(ctx);ggml_set_output(y);ggml_set_output(z);
            ggml_build_forward_expand(g,y);ggml_build_forward_expand(g,z);
            auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            size_t bytes=0;ggml_gallocr_reserve_n_size(alloc,g,nullptr,nullptr,&bytes);
            check(bytes<2*1024*1024,"scratch limit");check(ggml_gallocr_reserve(alloc,g),"reserve");check(ggml_gallocr_alloc_graph(alloc,g),"allocate");
            ggml_backend_tensor_set(x,input.data(),0,input.size()*4);ggml_backend_tensor_set(w,raw.data(),0,raw.size()*2);ggml_backend_tensor_set(control,weight.data(),0,weight.size()*4);
            check(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS,"compute");
            std::vector<float> a(k*n),b(k*n);ggml_backend_tensor_get(y,a.data(),0,a.size()*4);ggml_backend_tensor_get(z,b.data(),0,b.size()*4);ggml_backend_tensor_get(w,after.data(),0,after.size()*2);
            check(raw==after,"BF16 payload mutated");double maxerr=0;size_t different=0;
            for(int row=0;row<n;++row) {
                double sum=0;for(int j=0;j<k;++j)sum+=double(input[row*k+j])*input[row*k+j];
                for(int j=0;j<k;++j) {size_t i=row*k+j;check(std::isfinite(a[i]),"nonfinite");double expected=double(input[i])*weight[j]/std::sqrt(sum/k+double(eps));maxerr=std::max(maxerr,std::abs(double(a[i])-expected));different+=a[i]!=b[i];}
            }
            std::cout<<"max_abs_oracle="<<maxerr<<" decoded_F32_differences="<<different<<" scratch_bytes="<<bytes<<" payload_unchanged=1"<<std::endl;
            check(maxerr<2e-6 && different==0,"numerical regression");
            ggml_gallocr_free(alloc);ggml_free(ctx);
        }
    } catch(const std::exception & e) { std::cerr<<e.what()<<std::endl;status=1; }
    ggml_backend_free(backend);return status?3:0;
}
