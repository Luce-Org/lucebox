#include "deepseek4/deepseek4_vision.h"
#include "ggml-cpu.h"
#include <climits>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace dflash::vision;
static void check(bool ok,const char *why) { if(!ok) throw std::runtime_error(why); }
static_assert(GGML_OP_PAGED_ATTN==104 && GGML_OP_MUL_MAT_BIAS_BF16==105 &&
              GGML_OP_RMS_NORM_VISION_F32==106 && GGML_OP_SOFT_MAX_VISION_F32==107 &&
              GGML_OP_MUL_MAT_VISION_AV_F32==108 && GGML_OP_COUNT==109,"operation ABI changed");

static void reject(bool av,int mode) {
    const pid_t pid=fork(); check(pid>=0,"fork failed");
    if(pid==0) {
        const rlimit limit={0,0}; setrlimit(RLIMIT_CORE,&limit);
        auto c=ggml_init({1024*1024,nullptr,true});
        if(!av) {
            auto x=ggml_new_tensor_2d(c,mode==1?GGML_TYPE_BF16:GGML_TYPE_F32,16,16);
            if(mode==0) x=nullptr;
            if(mode==2) x=ggml_new_tensor_2d(c,GGML_TYPE_F32,15,16);
            if(mode==3) x=ggml_new_tensor_2d(c,GGML_TYPE_F32,4097,16);
            if(mode==4) x=ggml_transpose(c,x);
            if(mode==5) x=ggml_new_tensor_2d(c,GGML_TYPE_F32,16,int64_t(INT_MAX)/64+1);
            (void)ggml_soft_max_vision_f32(c,x);
        } else {
            int n=mode==5?15:mode==6?4097:16;
            auto v=ggml_new_tensor_3d(c,mode==2?GGML_TYPE_BF16:GGML_TYPE_F32,mode==3?32:64,mode==4?8:16,n);
            auto p=ggml_new_tensor_3d(c,mode==7?GGML_TYPE_BF16:GGML_TYPE_F32,n,n,16);
            if(mode==0) v=nullptr;
            if(mode==1) p=nullptr;
            if(mode==8) p=ggml_new_tensor_3d(c,GGML_TYPE_F32,n,n+1,16);
            if(mode==9) p=ggml_new_tensor_3d(c,GGML_TYPE_F32,n,n,8);
            if(mode==10) p=ggml_transpose(c,p);
            if(mode==11) v=ggml_permute(c,ggml_new_tensor_3d(c,GGML_TYPE_F32,16,64,n),1,0,2,3);
            if(mode==12) v=ggml_new_tensor_4d(c,GGML_TYPE_F32,64,16,n,2);
            if(mode==13) p=ggml_new_tensor_4d(c,GGML_TYPE_F32,n,n,16,2);
            (void)ggml_mul_mat_vision_av_f32(c,v,p);
        }
        _exit(0);
    }
    int status=0; check(waitpid(pid,&status,0)==pid,"wait failed");
    check(WIFSIGNALED(status) && WTERMSIG(status)==SIGABRT,"invalid attention constructor accepted");
}

int main() {
    auto backend=ggml_backend_cpu_init(); auto c=ggml_init({2*1024*1024,nullptr,true});
    try {
        check(backend && c,"initialization failed");
        for(auto selected:{static_cast<ggml_backend_t>(nullptr),backend}) {
            check(!detail::hip_softmax_capable(selected) && !detail::hip_av_capable(selected),"CPU/null advertised HIP attention");
            check(detail::hip_softmax_launches(selected)==0 && detail::hip_av_launches(selected)==0,"CPU/null reported HIP launches");
            auto q=ggml_new_tensor_3d(c,GGML_TYPE_F32,4,2,3);
            auto g=ggml_new_graph(c); ggml_build_forward_expand(g,detail::attention(c,q,q,q,selected));
            int softmax=0,matmul=0;
            for(int i=0;i<ggml_graph_n_nodes(g);++i) {
                const auto op=ggml_graph_node(g,i)->op;
                check(op!=GGML_OP_SOFT_MAX_VISION_F32 && op!=GGML_OP_MUL_MAT_VISION_AV_F32,"CPU generic attention changed");
                softmax+=op==GGML_OP_SOFT_MAX; matmul+=op==GGML_OP_MUL_MAT;
            }
            check(softmax==1 && matmul==2,"CPU attention graph operation count changed");
        }
        for(int n:{16,128,782,2048,2049,2560,2562,4096}) {
            auto v=ggml_new_tensor_3d(c,GGML_TYPE_F32,64,16,n);
            auto x=ggml_new_tensor_3d(c,GGML_TYPE_F32,n,n,16);
            auto p=ggml_soft_max_vision_f32(c,x);
            auto y=ggml_mul_mat_vision_av_f32(c,v,p);
            check(p->op==GGML_OP_SOFT_MAX_VISION_F32 && p->src[0]==x && p->type==GGML_TYPE_F32 &&
                  ggml_are_same_shape(p,x),"softmax constructor shape changed");
            check(y->op==GGML_OP_MUL_MAT_VISION_AV_F32 && y->src[0]==v && y->src[1]==p &&
                  y->type==GGML_TYPE_F32 && y->ne[0]==64 && y->ne[1]==n && y->ne[2]==16 && y->ne[3]==1,
                  "AV constructor layout changed");
            check(!ggml_backend_supports_op(backend,p) && !ggml_backend_supports_op(backend,y),"CPU advertised HIP operations");
        }
        for(int mode=0;mode<6;++mode) reject(false,mode);
        for(int mode=0;mode<14;++mode) reject(true,mode);
        ggml_free(c); ggml_backend_free(backend);
        std::cout<<"PASS: HIP attention ABI, bounded shapes, CPU/null preservation, invalid input rejection\n";
        return 0;
    } catch(const std::exception &error) {
        std::cerr<<"FAIL: "<<error.what()<<'\n'; ggml_free(c); ggml_backend_free(backend); return 1;
    }
}
