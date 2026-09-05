#include "deepseek4/deepseek4_vision.h"
#ifdef DS4V_VISION_HIP
#include "ggml-cuda.h"
#endif
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

using namespace dflash::vision;
namespace fs=std::filesystem;
static void check(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }
static std::vector<float> read(const fs::path &path) {
    constexpr size_t count=782*32;
    check(fs::is_regular_file(path) && fs::file_size(path)==count*4,"invalid rotary source file");
    std::vector<float> values(count); std::ifstream f(path,std::ios::binary);
    f.read(reinterpret_cast<char *>(values.data()),count*4); check(bool(f),"rotary fixture read failed");
    for(float value:values) check(std::isfinite(value),"nonfinite rotary source");
    return values;
}
static void save(const fs::path &path,const std::vector<float> &values) {
    std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<const char *>(values.data()),values.size()*4);
    check(bool(f),"rotary output write failed");
}
int main(int argc,char **argv) {
    std::cout<<std::unitbuf<<"pid="<<getpid()<<'\n';
    ggml_backend_t backend=nullptr;
    try {
        check(argc==3,"usage: ds4v_rotary_source SOURCE_ATTENTION_FIXTURES NEW_OUTPUT_DIR");
        const fs::path source=argv[1],out=argv[2];
        check(!fs::exists(out),"fresh rotary output required");
        const auto expected_cos=read(source/"cos.f32"),expected_sin=read(source/"sin.f32");
#ifdef DS4V_VISION_HIP
        check(ggml_backend_cuda_get_device_count()==1,"one visible HIP device required");
        backend=ggml_backend_cuda_init(0);
#endif
        check(backend && detail::hip_rotary_capable(backend),"HIP rotary capability unavailable");
        std::cout<<"backend="<<ggml_backend_name(backend)<<" requested=hip:0 case=rotary\n";
        using Fill=bool (*)(ggml_backend_t,int64_t,int64_t,float *,float *);
        auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto fill=reinterpret_cast<Fill>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_hip_vision_rotary_f32"));
        check(fill!=nullptr,"HIP rotary fill unavailable");
        float a=123.f,b=456.f;
        check(!fill(nullptr,23,34,&a,&b),"null backend accepted");
        for(PatchGrid grid:std::vector<PatchGrid>{{0,1},{1,0},{-1,1},{1,-1},{1153,1},{1,1153},{1152,1152}})
            check(!fill(backend,grid.height,grid.width,&a,&b),"invalid or oversized direct rotary grid accepted");
        check(!fill(backend,23,34,nullptr,&b) && !fill(backend,23,34,&a,nullptr),"null rotary output accepted");
        check(!fill(backend,23,34,&a,&a),"aliased rotary outputs accepted");
        check(a==123.f && b==456.f && detail::hip_rotary_launches(backend)==0,"rejected rotary call had side effects");
        std::vector<float> cosine,sine;
        detail::rotary_tables({23,34},cosine,sine,backend);
        check(cosine.size()==expected_cos.size() && sine.size()==expected_sin.size(),"rotary output shape changed");
        check(std::memcmp(cosine.data(),expected_cos.data(),cosine.size()*4)==0
            && std::memcmp(sine.data(),expected_sin.data(),sine.size()*4)==0,"rotary source differs bitwise");
        check(detail::hip_rotary_launches(backend)==1,"wrong rotary preparation count");
        check(detail::hip_bias_launches(backend)==0 && detail::hip_norm_launches(backend)==0,"unexpected graph operation");
        fs::create_directory(out); save(out/"cos.f32",cosine); save(out/"sin.f32",sine);
        std::cout<<"source_bitwise_mismatches=0 actual_rotary_launches=1 actual_lt_launches=0 actual_norm_launches=0\n";
        std::cout<<"PASS: HIP rotary tables match original source\n";
        ggml_backend_free(backend); return 0;
    } catch(const std::exception &error) {
        if(backend) ggml_backend_free(backend);
        std::cerr<<"FAIL: "<<error.what()<<'\n'; return 1;
    }
}
