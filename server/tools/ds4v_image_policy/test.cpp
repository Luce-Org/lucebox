#include "deepseek4/deepseek4_image_policy.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace dflash::vision;
static void check(bool ok, const char * reason) { if (!ok) throw std::runtime_error(reason); }
static bool visible(int64_t q, int64_t k, int64_t w, int64_t begin=-1, int64_t end=-1) {
    bool result = false;
    check(raw_key_visible(q,k,w,begin,end,result), "valid visibility rejected");
    return result;
}
int main() {
    try {
        ImageExpertSelection result;
        std::string error;
        float scores[] = {1,2,3,4}, bias[] = {0,8,0,0};
        check(select_image_experts(scores,bias,4,2,result,error), "valid selection rejected");
        check(result.count==2 && result.indices[0]==1 && result.indices[1]==3, "bias selection");
        check(result.weights[0]==0.5f && result.weights[1]==1.0f, "unbiased weights");
        float ties[] = {3,2,1,0};
        check(select_image_experts(scores,ties,4,4,result,error), "ties rejected");
        for (int i=0;i<4;++i) check(result.indices[i]==i, "lower-index tie handling");
        auto reject = [&](const float * s,const float * b,size_t n,size_t k,float scale=1.5f) {
            result.count=9;
            check(!select_image_experts(s,b,n,k,result,error,scale), "invalid selection accepted");
            check(result.count==0 && !error.empty(), "failure result/error contract");
        };
        reject(nullptr,bias,4,2); reject(scores,nullptr,4,2);
        reject(scores,bias,0,0); reject(scores,bias,257,2);
        reject(scores,bias,4,0); reject(scores,bias,4,5);
        reject(scores,bias,4,2,0); reject(scores,bias,4,2,-1);
        reject(scores,bias,4,2,std::numeric_limits<float>::infinity());
        float invalid[] = {-1,2,3,4}; reject(invalid,bias,4,2);
        invalid[0]=std::numeric_limits<float>::quiet_NaN(); reject(invalid,bias,4,2);
        invalid[0]=std::numeric_limits<float>::infinity(); reject(scores,invalid,4,2);
        float huge[] = {std::numeric_limits<float>::max(),1,1,1}; reject(huge,huge,4,2);
        float zero[] = {0,0,0,0}; reject(zero,bias,4,2);
        float sum_overflow[] = {std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),0,0};
        reject(sum_overflow,zero,4,2);
        check(visible(0,0,1) && !visible(0,1,1), "initial causal row");
        check(visible(10,7,4) && !visible(10,6,4) && !visible(10,11,4), "window boundaries");
        check(visible(10,20,4,8,21) && visible(10,7,4,8,21), "image union");
        check(visible(20,8,4,8,21) && !visible(20,7,4,8,21), "long image");
        check(!visible(7,8,4,8,21) && !visible(21,8,4,8,21), "outside image isolation");
        check(!visible(10,30,4,8,21), "other future image isolation");
        const int64_t max=std::numeric_limits<int64_t>::max();
        check(visible(max,max,1) && visible(max,max-2,3) && !visible(max,max-3,3), "near integer limit");
        check(visible(max-2,max-1,3,max-3,max), "near limit image");
        bool value=true;
        for (auto args : {std::array<int64_t,5>{-1,0,1,-1,-1}, {0,-1,1,-1,-1},
                          {0,0,0,-1,-1}, {0,0,1,-1,2}, {0,0,1,2,2}, {0,0,1,3,2}}) {
            check(!raw_key_visible(args[0],args[1],args[2],args[3],args[4],value) && !value,
                  "invalid visibility accepted");
        }
        std::cout << "PASS routing selection/weights/ties, invalid contracts, raw visibility boundaries and int64 limits\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
