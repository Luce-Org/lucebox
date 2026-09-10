#include "compose.h"
#include "server/chat_template.h"
#include <algorithm>
#include <stdexcept>

using namespace dflash::common;
using namespace dflash::vision;
namespace {
Composition fail(const std::string & error) {
    Composition result;
    result.error=error;
    return result;
}
std::vector<ChatMessage> probe_text_adapter(const nlohmann::json & normalized) {
    std::vector<ChatMessage> result;
    for (const auto & message:normalized) {
        ChatMessage item;
        item.role=message.value("role",std::string("user"));
        const auto & content=message.at("content");
        if (content.is_string()) item.content=content.get<std::string>();
        else if (content.is_array()) {
            for (const auto & part:content) {
                const auto type=part.value("type",std::string());
                if (type!="text" && type!="input_text" && type!="output_text")
                    throw std::runtime_error("unsupported probe part");
                item.content+=part.at("text").get<std::string>();
            }
        } else throw std::runtime_error("unsupported probe content");
        result.push_back(std::move(item));
    }
    return result;
}
}
Composition compose(const nlohmann::json & messages,Tokenizer & tokenizer,
                    const ImagePromptLimits & limits,const std::string & tools,const std::string & jinja) {
    try {
        nlohmann::json normalized;
        std::vector<EncodedImage> encoded;
        std::string error;
        if (!extract_chat_images(messages,normalized,encoded,error)) return fail("transport: "+error);
        const auto chat=probe_text_adapter(normalized);
        const auto rendered=jinja.empty()
            ? render_chat_template(chat,ChatFormat::DEEPSEEK4,true,false,tools)
            : render_chat_template_jinja(jinja,chat,"","",true,false,tools);
        auto tokens=tokenizer.encode(rendered);
        if (std::count(tokens.begin(),tokens.end(),129264)!=static_cast<ptrdiff_t>(encoded.size()))
            return fail("cardinality: final image marker count differs from image count");
        Composition result;
        std::vector<ImagePatchInput> patches;
        for (size_t i=0;i<encoded.size();++i) {
            auto decoded=decode_image({encoded[i].bytes.data(),encoded[i].bytes.size()});
            if (!decoded) return fail("decode image "+std::to_string(i)+": "+decode_error_name(decoded.status.code));
            auto processed=preprocess_rgb(decoded.image.view(),0);
            if (!processed) return fail("preprocess image "+std::to_string(i)+": "+preprocess_error_name(processed.status.code));
            patches.push_back({processed.image.plan,std::move(processed.image.patches_bf16)});
            result.decoded.push_back(std::move(decoded.image));
        }
        auto prepared=prepare_image_prompt(tokens,patches,limits,
            {static_cast<uint32_t>(tokenizer.vocab_size()),tokenizer.token_to_id(DS4_IMAGE_PLACEHOLDER)});
        if (!prepared) return fail("prompt: "+prepared.message);
        result.prepared=std::move(prepared);
        result.rendered_tokens=std::move(tokens);
        return result;
    } catch (const std::exception &) {
        return fail("probe adapter or rendering failure");
    }
}
