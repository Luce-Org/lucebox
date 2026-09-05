#include "deepseek4/deepseek4_vision.h"
#include "ggml-cpu.h"
#include <iostream>
#include <stdexcept>

using namespace dflash::vision;
static void check(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }
int main() {
    auto backend=ggml_backend_cpu_init();
    try {
        check(backend,"CPU backend unavailable");
        check(!detail::hip_rotary_capable(nullptr) && !detail::hip_rotary_capable(backend),"CPU reports HIP rotary capability");
        check(detail::hip_rotary_launches(nullptr)==0 && detail::hip_rotary_launches(backend)==0,"CPU reports HIP rotary dispatch");
        std::vector<float> generic_cos,generic_sin,cpu_cos,cpu_sin;
        detail::rotary_tables({23,34},generic_cos,generic_sin);
        detail::rotary_tables({23,34},cpu_cos,cpu_sin,backend);
        check(generic_cos==cpu_cos && generic_sin==cpu_sin,"CPU backend changed generic rotary tables");
        check(cpu_cos.size()==782*32 && cpu_sin.size()==cpu_cos.size(),"wrong rotary table size");
        for(int i=0;i<32;++i) check(cpu_cos[i]==1.f && cpu_sin[i]==0.f,"zero-position rotary changed");
        for(PatchGrid grid:std::vector<PatchGrid>{{0,1},{1,0},{-1,1},{1,-1},{1153,1},{1,1153}}) {
            bool rejected=false;
            try { detail::rotary_tables(grid,cpu_cos,cpu_sin,backend); }
            catch(const std::runtime_error &) { rejected=true; }
            check(rejected,"invalid rotary grid accepted");
        }
        ggml_backend_free(backend);
        std::cout<<"PASS: rotary CPU portability and grid contract\n";
        return 0;
    } catch(const std::exception &error) {
        if(backend) ggml_backend_free(backend);
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
