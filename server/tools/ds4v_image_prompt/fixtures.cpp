#include "deepseek4_image_prompt.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace dflash::vision;
namespace fs=std::filesystem;
static void check(bool value,const char * why) { if (!value) throw std::runtime_error(why); }
template<class T> static std::vector<T> read(const fs::path & path) {
    std::ifstream stream(path,std::ios::binary|std::ios::ate);
    check(bool(stream),"fixture open failed");
    const auto size=stream.tellg();
    check(size>=0 && size<16*1024*1024 && size%sizeof(T)==0,"fixture byte size invalid");
    std::vector<T> values(static_cast<size_t>(size)/sizeof(T));
    stream.seekg(0); stream.read(reinterpret_cast<char *>(values.data()),size);
    check(bool(stream),"fixture read failed");
    return values;
}
static ImagePatchInput image(const fs::path & root,const std::string & label) {
    ImagePatchInput value;
    value.plan=label=="corn" ? ResizePlan{476,322,23,34,8,12,false} : ResizePlan{854,588,42,61,14,21,false};
    value.patches_bf16=read<uint16_t>(root/label/"patches.bf16");
    return value;
}
static void compare(const PreparedImagePrompt & result,size_t index,const ImagePatchInput & input,
                    const fs::path & root,const std::string & label,uint64_t start,int fixture_start) {
    const auto types=read<int64_t>(root/label/("types-"+std::to_string(fixture_start)+".i64"));
    const auto permutation=read<int64_t>(root/label/("permutation-"+std::to_string(fixture_start)+".i64"));
    check(bool(result) && result.images.size()>index,"fixture preparation failed");
    const auto & item=result.images[index];
    check(item.input.patches_bf16==input.patches_bf16,"patch bytes changed");
    check(item.layout.types.size()==types.size() && item.layout.permutation==permutation,"source layout size/permutation mismatch");
    for (size_t i=0;i<types.size();++i) {
        check(static_cast<int64_t>(item.layout.types[i])==types[i],"source layout kind mismatch");
        check(result.tokens[start+i]==129280+types[i],"source expanded token mismatch");
    }
    const auto first=std::find(types.begin(),types.end(),int64_t(ImageTokenType::Start));
    const auto last=std::find(types.begin(),types.end(),int64_t(ImageTokenType::End));
    check(first!=types.end() && last!=types.end(),"bad source sentinel fixture");
    check(item.layout.span.block_begin==start && item.layout.span.block_end==start+types.size() &&
          item.layout.span.visible_begin==start+static_cast<uint64_t>(first-types.begin()) &&
          item.layout.span.visible_end==start+static_cast<uint64_t>(last-types.begin())+1,"source absolute spans mismatch");
}
int main(int argc,char ** argv) {
    try {
        check(argc==2,"usage: fixture_probe fixture_directory");
        const fs::path root=argv[1];
        for (const std::string label:{"corn","carrots"}) {
            const auto source=image(root,label);
            for (int start:{0,1,2,3,127}) {
                std::vector<int32_t> tokens(start,42);
                tokens.push_back(129264); tokens.push_back(77);
                auto result=prepare_image_prompt(tokens,{source});
                compare(result,0,source,root,label,start,start);
                check(result.tokens.back()==77,"text suffix changed");
                for (int i=0;i<start;++i) check(result.tokens[i]==42,"text prefix changed");
                std::cout<<label<<" start="<<start<<" PASS tokens="<<result.tokens.size()<<'\n';
            }
        }
        for (const std::string first:{"corn","carrots"}) {
            const std::string second=first=="corn" ? "carrots" : "corn";
            const auto first_types=read<int64_t>(root/first/"types-0.i64");
            const auto a=image(root,first),b=image(root,second);
            auto result=prepare_image_prompt({129264,129264},{a,b});
            compare(result,0,a,root,first,0,0);
            compare(result,1,b,root,second,first_types.size(),static_cast<int>(first_types.size()%4));
            std::cout<<first<<" then "<<second<<" PASS second_start="<<first_types.size()<<'\n';
        }
        return 0;
    } catch (const std::exception & e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
