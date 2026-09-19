
#include <map>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <cassert>
#ifdef HIP_TEST
#include <hip/hip_runtime_api.h>
using cudaError_t=hipError_t;
constexpr auto cudaSuccess=hipSuccess,cudaErrorMemoryAllocation=hipErrorOutOfMemory;
#else
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaErrorMemoryAllocation=2;
#endif
size_t used=0,capacity=8ull<<20;
std::map<void*,size_t> live;
cudaError_t ggml_cuda_device_malloc(void** p,size_t n,int) {
 if(used+n>capacity)return cudaErrorMemoryAllocation;
#ifdef HIP_TEST
 auto error=hipMalloc(p,n);if(error!=hipSuccess)return error;
#else
 *p=malloc(1);
#endif
 live[*p]=n;used+=n;return cudaSuccess;
}
cudaError_t cudaFree(void* p){
#ifdef HIP_TEST
 auto error=hipFree(p);if(error!=hipSuccess)return error;
#else
 free(p);
#endif
 used-=live.at(p);live.erase(p);return cudaSuccess;
}
cudaError_t cudaGetLastError(){
#ifdef HIP_TEST
 return hipGetLastError();
#else
 return cudaSuccess;
#endif
}
void ggml_cuda_set_device(int){}
#define CUDA_CHECK(x) do {if((x)!=0)throw std::runtime_error("OOM");} while(0)
#define GGML_ASSERT(x) assert(x)
#define GGML_LOG_DEBUG(...) ((void)0)
#define GGML_LOG_WARN(...) ((void)0)
#define GGML_CUDA_NAME "MOCK"
struct ggml_cuda_pool {virtual ~ggml_cuda_pool()=default;virtual void* alloc(size_t,size_t*)=0;virtual void free(void*,size_t)=0;virtual bool is_legacy()const=0;virtual size_t trim()=0;};
