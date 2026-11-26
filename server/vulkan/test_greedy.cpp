#include "greedy.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <random>
#include <vector>
using luce::common::finite_argmax;
int main() {
    size_t cases=0;
    auto check=[&](const std::vector<float>&x) {
        const auto expected=std::max_element(x.begin(),x.end())-x.begin();
        if(finite_argmax(x.data(),x.size())!=expected) throw std::runtime_error("argmax mismatch");
        ++cases;
    };
    std::mt19937 rng(7141);
    for(int n: {1,2,3,4,5,7,31,64,127,256,3584,18944,152064}) {
        std::vector<float>x(n);
        for(int repeat=0;repeat<30;++repeat) {
            for(auto &v:x) v=float(int(rng()%2001)-1000)/8;
            check(x);
        }
        for(float v: {-std::numeric_limits<float>::max(),-0.0f,0.0f,1.0f,std::numeric_limits<float>::max()}) {
            std::fill(x.begin(),x.end(),v);check(x);
        }
        for(int k=0;k<std::min(n,16);++k) {
            std::fill(x.begin(),x.end(),-1.0f);x[n-1]=2;x[k]=2;check(x);
        }
        for(float v:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
            for(int k:{0,n/2,n-1}) {
                std::fill(x.begin(),x.end(),1.0f);x[k]=v;
                bool threw=false;try{finite_argmax(x.data(),n);}catch(const std::runtime_error&){threw=true;}
                if(!threw)throw std::runtime_error("nonfinite accepted");++cases;
            }
        }
    }
    bool threw=false;try{finite_argmax(nullptr,0);}catch(const std::runtime_error&){threw=true;}
    if(!threw)throw std::runtime_error("empty accepted");++cases;
    std::vector<float>x(152064);for(auto &v:x)v=float(int(rng()%2001)-1000)/8;
    volatile int sink=0;
#ifdef __SSE2__
    const char * fused_label = "fused-SSE2";
#else
    const char * fused_label = "fused-scalar";
#endif
    for(int mode=0;mode<2;++mode) {
        auto start=std::chrono::steady_clock::now();
        for(int r=0;r<2000;++r) {
            if(mode==0) {
                for(float v:x)if(!std::isfinite(v))throw std::runtime_error("nonfinite");
                sink=int(std::max_element(x.begin(),x.end())-x.begin());
            } else sink=finite_argmax(x.data(),x.size());
        }
        printf("%s us/call %.3f token %d\n",mode?fused_label:"old-two-pass",std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/2000,int(sink));
    }
    printf("PASS %zu exact reference/edge/nonfinite cases\n",cases);
}
