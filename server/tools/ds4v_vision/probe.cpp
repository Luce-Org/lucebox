#include "deepseek4/deepseek4_vision.h"
#include "ggml-cpu.h"
#ifdef DS4V_VISION_HIP
#include "ggml-cuda.h"
#endif
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace dflash::vision;
static std::vector<float> read_file(const std::string & path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()<0 || size_t(f.tellg())%4) throw std::runtime_error("bad patch file");
    std::vector<float> out(size_t(f.tellg())/4);
    f.seekg(0); f.read(reinterpret_cast<char *>(out.data()),out.size()*4);
    if(!f) throw std::runtime_error("patch file read failed");
    return out;
}
static void save(const std::string & path,const std::vector<float> & values) {
    std::ofstream f(path,std::ios::binary);
    f.write(reinterpret_cast<const char *>(values.data()),values.size()*4);
    if(!f) throw std::runtime_error("output write failed: "+path);
}
int main(int argc,char ** argv) {
    const bool load_only=argc>=3 && std::string(argv[2])=="--load-only";
    if((load_only && argc!=5 && argc!=6) || (!load_only && argc!=8 && argc!=9)) {
        std::cerr<<"usage: ds4v_vision_probe mmproj patches.f32 height width output-dir label stages(0|1) [cpu|hip:0|hip:1]\n"
                 <<"       ds4v_vision_probe mmproj --load-only dimension vocabulary [cpu|hip:0|hip:1]\n"; return 2;
    }
    const std::string device=(load_only?argc==6:argc==9)?argv[argc-1]:"cpu";
    ggml_backend_t backend=nullptr;
    if(device=="cpu") {
        backend=ggml_backend_cpu_init();
        if(backend) ggml_backend_cpu_set_n_threads(backend,2);
    }
#ifdef DS4V_VISION_HIP
    else if(device=="hip:0" || device=="hip:1") {
        const int index=device.back()-'0';
        if(index<ggml_backend_cuda_get_device_count()) backend=ggml_backend_cuda_init(index);
    }
#endif
    if(!backend) { std::cerr<<"ERROR: requested qualification backend is unavailable\n"; return 1; }
    std::cout<<"backend="<<ggml_backend_name(backend)<<" requested="<<device<<"\n";
    int status=0;
    try {
        VisionRuntime runtime;
        std::string error;
        int dimension=load_only?std::stoi(argv[3]):4096,vocabulary=load_only?std::stoi(argv[4]):129280;
        if(!runtime.load(argv[1],backend,dimension,vocabulary,error)) throw std::runtime_error(error);
        std::cout<<"weights_bytes="<<runtime.weight_bytes()<<"\n";
        if(!load_only) {
            for(const auto & check:std::vector<std::pair<PatchGrid,bool>>{
                    {{48,72},false}, {{3,564},false}, {{6,564},false},
                    {{3,3},true}, {{3,561},true}, {{6,561},true}, {{564,3},true}}) {
                VisionOutput rejected;
                if(runtime.encode({},check.first,rejected,error)) throw std::runtime_error("empty patches accepted");
                const auto expected=check.second?"patch count/shape mismatch":"patch grid exceeds image token budget";
                if(error!=expected) throw std::runtime_error("grid budget check: expected "+std::string(expected)+", got "+error);
            }
            if(runtime.load(argv[1],backend,4095,129280,error)) throw std::runtime_error("incompatible reload accepted");
            if(!runtime.config() || runtime.weight_bytes()==0) throw std::runtime_error("failed reload destroyed runtime");
            std::vector<float> sentinel_after_reload;
            if(!runtime.sentinel(Sentinel::Start,sentinel_after_reload,error)) throw std::runtime_error("failed reload lost sentinels");
            if(runtime.sentinel(static_cast<Sentinel>(99),sentinel_after_reload,error)) throw std::runtime_error("invalid sentinel accepted");
            const auto patches=read_file(argv[2]);
            PatchGrid grid{std::stoi(argv[3]),std::stoi(argv[4])};
            const std::string output_dir=argv[5],label=argv[6];
            const bool stages=std::stoi(argv[7])!=0;
            std::filesystem::create_directories(output_dir);
            StageObserver observer;
            if(stages) observer=[&](const std::string & name,const std::vector<int64_t> & shape,const std::vector<float> & values) {
                save(output_dir+"/"+label+"-"+name+".f32",values);
                std::cout<<"stage="<<name<<" shape="<<shape[0]<<","<<shape[1]<<","<<shape[2]<<","<<shape[3]<<std::endl;
            };
            VisionOutput output;
            const auto lt_before=detail::hip_bias_launches(backend);
            const auto norm_before=detail::hip_norm_launches(backend);
            const auto rotary_before=detail::hip_rotary_launches(backend);
            auto started=std::chrono::steady_clock::now();
            if(!runtime.encode(patches,grid,output,error,true,observer)) throw std::runtime_error(error);
            const auto lt_launches=detail::hip_bias_launches(backend)-lt_before;
            const auto norm_launches=detail::hip_norm_launches(backend)-norm_before;
            const auto rotary_launches=detail::hip_rotary_launches(backend)-rotary_before;
            const auto external=detail::hip_bias_workspace(backend);
            const bool hip_requested=device=="hip:0" || device=="hip:1";
            if((external!=0)!=hip_requested) throw std::runtime_error("requested HIP backend lacks BF16 linear capability");
            if(lt_launches!=(hip_requested ? 131u : 0u)) throw std::runtime_error("unexpected actual HIP BF16 linear dispatch count");
            if(norm_launches!=(hip_requested ? 65u : 0u)) throw std::runtime_error("unexpected actual HIP vision normalization dispatch count");
            if(rotary_launches!=(hip_requested ? 1u : 0u)) throw std::runtime_error("unexpected actual HIP vision rotary table preparation count");
            std::cout<<"hip_vision_linear_launches="<<lt_launches<<" retained_workspace_bytes="<<external<<"\n";
            std::cout<<"hip_vision_norm_launches="<<norm_launches<<"\n";
            std::cout<<"hip_vision_rotary_launches="<<rotary_launches<<"\n";
            save(output_dir+"/"+label+"-features.f32",output.features);
            save(output_dir+"/"+label+"-embeddings.f32",output.embeddings);
            for(auto identity:{Sentinel::Start,Sentinel::Pad,Sentinel::Newline,Sentinel::End}) {
                std::vector<float> sentinel;
                if(!runtime.sentinel(identity,sentinel,error) || sentinel.size()!=4096) throw std::runtime_error("sentinel failure");
            }
            std::cout<<"output_shape="<<output.rows<<","<<output.columns
                     <<" scratch_bytes="<<runtime.scratch_bytes()
                     <<" elapsed_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"\n";
            // Releasing scratch must leave the loaded projector and sentinels usable.
            runtime.release_scratch();
            if(runtime.scratch_bytes()!=external) throw std::runtime_error("graph scratch release or external workspace accounting failed");
            VisionOutput invalid;
            if(runtime.encode({},grid,invalid,error)) throw std::runtime_error("invalid patch input accepted");
            if(runtime.encode({}, {0,1},invalid,error)) throw std::runtime_error("invalid grid accepted");
            if(runtime.encode({}, {1152,1152},invalid,error)) throw std::runtime_error("oversized grid accepted");
        }
    } catch(const std::exception & e) { std::cerr<<"ERROR: "<<e.what()<<"\n"; status=1; }
    ggml_backend_free(backend);
    return status;
}
