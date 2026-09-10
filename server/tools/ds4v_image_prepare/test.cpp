#include "compose.h"
#include "server/chat_template.h"
#include "lodepng.h"
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace dflash::common;
using namespace dflash::vision;
using json=nlohmann::json;
namespace fs=std::filesystem;
static void check(bool ok,const char * why) { if (!ok) throw std::runtime_error(why); }
template<class T> static std::vector<T> read(const fs::path & path) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    check(bool(file),"fixture open failed");
    auto bytes=file.tellg();
    check(bytes>=0 && bytes<32*1024*1024 && bytes%sizeof(T)==0,"fixture size invalid");
    std::vector<T> value(static_cast<size_t>(bytes)/sizeof(T));
    file.seekg(0); file.read(reinterpret_cast<char *>(value.data()),bytes);
    check(bool(file),"fixture read failed"); return value;
}
static std::string data_url(const std::vector<uint8_t> & bytes,const char * mime="image/jpeg") {
    constexpr char chars[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result=std::string("data:")+mime+";base64,";
    for (size_t i=0;i<bytes.size();i+=3) {
        const uint32_t word=(uint32_t(bytes[i])<<16) | (i+1<bytes.size()?uint32_t(bytes[i+1])<<8:0) | (i+2<bytes.size()?bytes[i+2]:0);
        result+=chars[(word>>18)&63]; result+=chars[(word>>12)&63];
        result+=i+1<bytes.size()?chars[(word>>6)&63]:'=';
        result+=i+2<bytes.size()?chars[word&63]:'=';
    }
    return result;
}
static json messages(const std::vector<std::string> & urls) {
    json parts=json::array({{{"type","text"},{"text","Describe these: "}}});
    for (const auto & url:urls) {
        parts.push_back({{"type","image_url"},{"image_url",{{"url",url}}}});
        parts.push_back({{"type","text"},{"text"," then "}});
    }
    parts.push_back({{"type","text"},{"text","Explain the difference."}});
    return json::array({{{"role","system"},{"content","Be concise."}},{{"role","user"},{"content",parts}}});
}
static void compare(const Composition & result,size_t index,const fs::path & root,const std::string & label) {
    const auto & item=result.prepared.images[index];
    check(result.decoded[index].pixels==read<uint8_t>(root/label/"input.rgb"),"decoded RGB differs from source");
    check(item.input.patches_bf16==read<uint16_t>(root/label/"patches.bf16"),"BF16 patches differ from source");
    check(result.decoded[index].width==(label=="corn"?450U:1024U) &&
          result.decoded[index].height==(label=="corn"?308U:701U),"source decoded dimensions");
    check(item.input.plan.vit_rows==(label=="corn"?23U:42U) &&
          item.input.plan.vit_cols==(label=="corn"?34U:61U) &&
          item.input.plan.aligner_rows==(label=="corn"?8U:14U) &&
          item.input.plan.aligner_cols==(label=="corn"?12U:21U),"source patch/aligner dimensions");
    const auto start=item.layout.span.block_begin;
    const int residue=static_cast<int>(start%4);
    const auto types=read<int64_t>(root/label/("types-"+std::to_string(residue)+".i64"));
    check(item.layout.types.size()==types.size(),"type count mismatch");
    check(item.layout.permutation==read<int64_t>(root/label/("permutation-"+std::to_string(residue)+".i64")),"permutation mismatch");
    for (size_t i=0;i<types.size();++i) {
        check(static_cast<int64_t>(item.layout.types[i])==types[i],"source type mismatch");
        check(result.prepared.tokens[start+i]==129280+types[i],"generated ID mismatch");
    }
    const auto first=std::find(types.begin(),types.end(),int64_t(0));
    const auto last=std::find(types.begin(),types.end(),int64_t(4));
    check(first!=types.end() && last!=types.end(),"source sentinels absent");
    check(item.layout.span.visible_begin==start+static_cast<size_t>(first-types.begin()) &&
          item.layout.span.visible_end==start+static_cast<size_t>(last-types.begin())+1 &&
          item.layout.span.block_end==start+types.size(),"source span mismatch");
    std::cout<<label<<" decoded="<<result.decoded[index].width<<'x'<<result.decoded[index].height
             <<" patches="<<item.input.plan.vit_rows<<'x'<<item.input.plan.vit_cols
             <<" aligner="<<item.input.plan.aligner_rows<<'x'<<item.input.plan.aligner_cols
             <<" block=["<<start<<','<<item.layout.span.block_end<<") visible=["
             <<item.layout.span.visible_begin<<','<<item.layout.span.visible_end<<")\n";
}
static void failure(const Composition & result,const char * category) {
    check(!result && result.error.find(category)!=std::string::npos,"wrong failure category");
    check(result.prepared.tokens.empty() && result.prepared.images.empty() && result.decoded.empty(),"partial composition escaped");
}
static std::string solid_png(uint8_t pixel) {
    std::vector<uint8_t> rgb(19*11*3,pixel);
    unsigned char * bytes=nullptr; size_t size=0;
    check(lodepng_encode24(&bytes,&size,rgb.data(),19,11)==0,"synthetic PNG encode failed");
    std::vector<uint8_t> data(bytes,bytes+size); std::free(bytes);
    return data_url(data,"image/png");
}
int main(int argc,char ** argv) {
    try {
        check(argc==3,"usage: composition_probe tokenizer_gguf fixtures");
        Tokenizer tokenizer; check(tokenizer.load_from_gguf(argv[1]),"tokenizer load failed");
        check(tokenizer.vocab_size()==129280 && tokenizer.token_to_id(DS4_IMAGE_PLACEHOLDER)==129264,"tokenizer contract");
        const auto text_messages=json::array({{{"role","user"},{"content","Hello."}}});
        auto text=compose(text_messages,tokenizer);
        check(bool(text),"text composition failed");
        const auto expected=tokenizer.encode(render_chat_template({{"user","Hello."}},ChatFormat::DEEPSEEK4));
        check(text.prepared.tokens==expected && text.prepared.images.empty(),"text tokens changed");
        const fs::path root=argv[2];
        const auto corn=data_url(read<uint8_t>(root/"corn/encoded.bin"));
        const auto carrots=data_url(read<uint8_t>(root/"carrots/encoded.bin"));
        for (bool reverse:{false,true}) {
            const std::string first=reverse?"carrots":"corn",second=reverse?"corn":"carrots";
            const auto input=messages(reverse?std::vector<std::string>{carrots,corn}:std::vector<std::string>{corn,carrots});
            auto result=compose(input,tokenizer);
            check(bool(result) && result.prepared.images.size()==2,"real image composition failed");
            compare(result,0,root,first); compare(result,1,root,second);
            size_t position=0,index=0;
            for (int32_t token:result.rendered_tokens) {
                if (token==129264) {
                    check(result.prepared.images[index].layout.span.block_begin==position,"ordered expanded image position");
                    position+=result.prepared.images[index++].layout.types.size();
                } else check(result.prepared.tokens[position++]==token,"surrounding text changed");
            }
            check(position==result.prepared.tokens.size() && index==2,"expanded coverage");
            const uint64_t total=result.prepared.tokens.size();
            check(bool(compose(input,tokenizer,{total+1,1,total})),"exact fit rejected");
            failure(compose(input,tokenizer,{total,1,total}),"prompt");
            std::cout<<first<<" then "<<second<<" PASS tokens="<<total
                     <<" first_start="<<result.prepared.images[0].layout.span.block_begin
                     <<" second_start="<<result.prepared.images[1].layout.span.block_begin<<'\n';
        }
        const auto one=messages({corn});
        const std::string preserved="{% for message in messages %}{{ message.content }}{% endfor %}";
        check(bool(compose(one,tokenizer,{},"",preserved)),"Jinja preserved marker failed");
        failure(compose(one,tokenizer,{},"","ordinary text"),"cardinality");
        failure(compose(one,tokenizer,{},"","{% for message in messages %}{{ message.content }}{{ message.content }}{% endfor %}"),"cardinality");
        failure(compose(text_messages,tokenizer,{},"",DS4_IMAGE_PLACEHOLDER),"cardinality");
        json schema={{"type","function"},{"function",{{"name","tool"},{"parameters",{{"type","object"},{"properties",{{DS4_IMAGE_PLACEHOLDER,{{"type","string"}}}}}}}}}};
        failure(compose(text_messages,tokenizer,{},json::array({schema}).dump()),"cardinality");
        auto malformed=read<uint8_t>(root/"corn/encoded.bin"); malformed.resize(3);
        failure(compose(messages({corn,data_url(malformed)}),tokenizer),"decode image 1");
        json redacted=messages({corn,carrots}); redact_image_urls(redacted);
        const auto logged=redacted.dump();
        check(logged.find("base64,")==std::string::npos && logged.find(corn.substr(0,80))==std::string::npos &&
              logged.find("[image omitted]")!=std::string::npos,"redaction before dump failed");
        auto black=compose(messages({solid_png(0)}),tokenizer);
        auto white=compose(messages({solid_png(255)}),tokenizer);
        check(bool(black) && bool(white) && black.prepared.tokens==white.prepared.tokens,"same-layout PNG tokens");
        check(black.prepared.images[0].input.patches_bf16!=white.prepared.images[0].input.patches_bf16,"different pixels collapsed");
        const auto white_first=white.prepared.images[0].input.patches_bf16[0];
        black.prepared.images[0].input.patches_bf16[0]=0;
        check(white.prepared.images[0].input.patches_bf16[0]==white_first,"image storage shared");
        std::cout<<"PASS text preservation, Jinja/cardinality, zero-image tool key, malformed second image, exact context, redaction, distinct PNG storage\n";
        return 0;
    } catch (const std::exception & e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
