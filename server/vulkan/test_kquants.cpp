#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
#include <chrono>
static void check(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int main(){try{
 auto backend=ggml_backend_vk_init(0);check(backend,"Vulkan init");
 struct Shape{int k,m,n;};
 for(auto type:{GGML_TYPE_Q4_K,GGML_TYPE_Q6_K})for(Shape s:std::vector<Shape>{{256,7,1},{3584,513,1},{3584,18944,1},{18944,3584,1},{3584,152064,1},{256,9,2}}){
  auto ctx=ggml_init({2*1024*1024,nullptr,true});
  auto *a=ggml_new_tensor_2d(ctx,type,s.k,s.m),*b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,s.k,s.n),*bias=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,s.m);
  auto *out=ggml_add(ctx,ggml_mul_mat(ctx,a,b),bias);auto *g=ggml_new_graph(ctx);ggml_build_forward_expand(g,out);
  auto buf=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buf,"allocation");
  std::mt19937 gen(2026);std::uniform_real_distribution<float> dist(-0.5,0.5);
  std::vector<float> row(s.k),x(s.k*s.n),off(s.m),gpu(s.m*s.n);for(auto&v:x)v=dist(gen);for(auto&v:off)v=dist(gen);
  const size_t rowbytes=ggml_row_size(type,s.k);std::vector<unsigned char> quant(rowbytes*s.m);std::vector<double> ref(s.m*s.n);
  const auto *traits=ggml_get_type_traits(type);check(traits->to_float,"dequant reference");
  for(int j=0;j<s.m;j++){
   for(auto&v:row)v=dist(gen);ggml_quantize_chunk(type,row.data(),quant.data()+j*rowbytes,0,1,s.k,nullptr);
   traits->to_float(quant.data()+j*rowbytes,row.data(),s.k);
   for(int n=0;n<s.n;n++){double sum=off[j];for(int k=0;k<s.k;k++)sum+=double(row[k])*x[n*s.k+k];ref[n*s.m+j]=sum;}
  }
  ggml_backend_tensor_set(a,quant.data(),0,quant.size());ggml_backend_tensor_set(b,x.data(),0,x.size()*4);ggml_backend_tensor_set(bias,off.data(),0,off.size()*4);
  check(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS,"compute");ggml_backend_tensor_get(out,gpu.data(),0,gpu.size()*4);
  double maxerr=0,maxref=0,err2=0,ref2=0;
  for(size_t i=0;i<gpu.size();i++){check(std::isfinite(gpu[i]),"nonfinite");double e=gpu[i]-ref[i];maxerr=std::max(maxerr,std::abs(e));maxref=std::max(maxref,std::abs(ref[i]));err2+=e*e;ref2+=ref[i]*ref[i];}
  double rel=std::sqrt(err2/std::max(1e-30,ref2));bool pass=rel<1e-5&&maxerr<5e-4*std::max(1.0,maxref);
  printf("{\"type\":\"%s\",\"k\":%d,\"m\":%d,\"n\":%d,\"relative_l2\":%.12g,\"max_abs_error\":%.12g,\"pass\":%s}\n",ggml_type_name(type),s.k,s.m,s.n,rel,maxerr,pass?"true":"false");fflush(stdout);
  ggml_backend_buffer_free(buf);ggml_free(ctx);check(pass,"numerical tolerance failed");
 }
 ggml_backend_free(backend);return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}}
