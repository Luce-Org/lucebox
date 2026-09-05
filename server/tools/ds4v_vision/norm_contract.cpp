#include "deepseek4/deepseek4_vision.h"
#include "ggml-cpu.h"
#include <climits>
#include <cmath>
#include <csignal>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

static void check(bool ok,const char *why) { if(!ok) throw std::runtime_error(why); }
static_assert(GGML_OP_PAGED_ATTN==104 && GGML_OP_MUL_MAT_BIAS_BF16==105 &&
              GGML_OP_RMS_NORM_VISION_F32==106 && GGML_OP_COUNT==107,"operation ABI changed");
static void rejected(int mode) {
    const pid_t pid=fork(); check(pid>=0,"fork failed");
    if(pid==0) {
        auto c=ggml_init({1024*1024,nullptr,true});
        auto x=ggml_new_tensor_2d(c,mode==1?GGML_TYPE_BF16:GGML_TYPE_F32,mode==2?512:1024,mode==3?15:16);
        if(mode==0) x=nullptr;
        if(mode==4) x=ggml_transpose(c,ggml_new_tensor_2d(c,GGML_TYPE_F32,16,1024));
        if(mode==5) x=ggml_new_tensor_3d(c,GGML_TYPE_F32,1024,16,2);
        if(mode==9) x=ggml_new_tensor_2d(c,GGML_TYPE_F32,1024,int64_t(INT_MAX)/1024+1);
        const float eps=mode==6?-1.f:mode==7?std::numeric_limits<float>::infinity():
                        mode==8?std::numeric_limits<float>::quiet_NaN():1e-6f;
        (void)ggml_rms_norm_vision_f32(c,x,eps);
        _exit(0);
    }
    int status=0; check(waitpid(pid,&status,0)==pid,"wait failed");
    check(WIFSIGNALED(status) && WTERMSIG(status)==SIGABRT,"invalid source-order norm was accepted");
}
int main() {
    auto backend=ggml_backend_cpu_init(); auto c=ggml_init({1024*1024,nullptr,true});
    try {
        check(backend && c,"initialization failed");
        check(!dflash::vision::detail::hip_norm_capable(nullptr) && !dflash::vision::detail::hip_norm_capable(backend),
              "CPU/null advertised HIP source normalization");
        check(dflash::vision::detail::hip_norm_launches(nullptr)==0 && dflash::vision::detail::hip_norm_launches(backend)==0,
              "CPU/null reported HIP normalization launches");
        for(int rows:{16,782,2562}) {
            auto x=ggml_new_tensor_2d(c,GGML_TYPE_F32,1024,rows);
            auto y=ggml_rms_norm_vision_f32(c,x,1e-6f);
            check(y->op==GGML_OP_RMS_NORM_VISION_F32 && y->type==GGML_TYPE_F32 &&
                  y->src[0]==x && y->src[1]==nullptr && y->ne[0]==1024 && y->ne[1]==rows,
                  "source normalization constructor contract changed");
            check(!ggml_backend_supports_op(backend,y),"CPU advertised source-order HIP operation");
        }
        for(int rows:{1,16,782}) {
            auto x=ggml_new_tensor_2d(c,GGML_TYPE_F32,1024,rows);
            for(auto selected:{static_cast<ggml_backend_t>(nullptr),backend}) {
                auto y=dflash::vision::detail::rms_norm(c,x,1e-6f,selected);
                check(y->op==GGML_OP_RMS_NORM && y->src[0]==x && y->type==GGML_TYPE_F32,
                      "generic CPU/null normalization path changed");
            }
        }
        for(int mode=0;mode<10;++mode) rejected(mode);
        ggml_free(c); ggml_backend_free(backend);
        std::cout<<"PASS: explicit HIP norm contract, CPU/null preservation, invalid input rejection\n";
        return 0;
    } catch(const std::exception &e) {
        std::cerr<<e.what()<<'\n'; ggml_free(c); ggml_backend_free(backend); return 1;
    }
}
