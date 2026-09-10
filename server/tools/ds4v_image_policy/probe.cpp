#include "deepseek4/deepseek4_image_policy.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace dflash::vision;
template<class T> static std::vector<T> read(const char * path,size_t count) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if (!file || file.tellg()!=std::streamoff(count*sizeof(T))) throw std::runtime_error("input byte size mismatch");
    std::vector<T> data(count);
    file.seekg(0); file.read(reinterpret_cast<char *>(data.data()),count*sizeof(T));
    if (!file) throw std::runtime_error("input read failed");
    return data;
}
template<class T> static void save(const std::string & path,const std::vector<T> & values) {
    std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(T));
    if (!file) throw std::runtime_error("output write failed");
}
int main(int argc,char ** argv) {
    try {
        if (argc==8 && std::string(argv[1])=="route") {
            size_t rows=std::stoul(argv[5]),experts=std::stoul(argv[6]),topk=std::stoul(argv[7]);
            if (rows>1024 || experts>256 || topk>experts) throw std::runtime_error("probe dimensions out of range");
            auto scores=read<float>(argv[2],rows*experts),bias=read<float>(argv[3],experts);
            std::vector<int32_t> indices;
            std::vector<float> weights;
            for (size_t row=0;row<rows;++row) {
                ImageExpertSelection selection; std::string error;
                if (!select_image_experts(scores.data()+row*experts,bias.data(),experts,topk,selection,error)) throw std::runtime_error(error);
                indices.insert(indices.end(),selection.indices.begin(),selection.indices.begin()+topk);
                weights.insert(weights.end(),selection.weights.begin(),selection.weights.begin()+topk);
            }
            save(std::string(argv[4])+"-indices.i32",indices);
            save(std::string(argv[4])+"-weights.f32",weights);
        } else if (argc==6 && std::string(argv[1])=="mask") {
            size_t count=std::stoul(argv[3]); int64_t window=std::stoll(argv[4]);
            if (count>4096) throw std::runtime_error("probe mask too large");
            auto ranges=read<int64_t>(argv[2],count*2);
            std::vector<uint8_t> mask(count*count);
            for (size_t q=0;q<count;++q) for (size_t k=0;k<count;++k) {
                bool visible=false;
                if (!raw_key_visible(q,k,window,ranges[q*2],ranges[q*2+1],visible)) throw std::runtime_error("invalid visibility arguments");
                mask[q*count+k]=visible;
            }
            save(argv[5],mask);
        } else throw std::runtime_error("usage: probe route scores bias output-prefix rows experts topk | probe mask ranges count window output");
        return 0;
    } catch (const std::exception & e) { std::cerr<<e.what()<<'\n'; return 1; }
}
