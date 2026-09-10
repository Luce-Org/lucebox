#include "deepseek4_image_prompt.h"
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace dflash::vision;
constexpr int32_t marker=129264,vocab=129280;
static void check(bool value,const char * why) { if (!value) throw std::runtime_error(why); }
static ImagePatchInput small_image(uint16_t pixel=0) {
    ImagePatchInput image;
    image.plan={42,42,3,3,1,1,false};
    image.patches_bf16.assign(9*588,pixel);
    return image;
}
static void rejected(const PreparedImagePrompt & result,ImagePromptError expected) {
    check(!result && result.error==expected,"wrong failure category");
    check(result.tokens.empty() && result.images.empty() && !result.message.empty(),"nontransactional failure");
}
int main() {
    try {
        auto image=small_image();
        auto text=prepare_image_prompt({1,2,42},{ });
        check(bool(text) && text.tokens==std::vector<int32_t>({1,2,42}) && text.images.empty(),"text path");
        const std::vector<ImageTokenType> body={ImageTokenType::Start,ImageTokenType::Image,
            ImageTokenType::Pad,ImageTokenType::Newline,ImageTokenType::Pad,ImageTokenType::End};
        for (uint64_t start=0;start<4;++start) {
            std::vector<int32_t> input(start,42); input.push_back(marker); input.push_back(7);
            auto result=prepare_image_prompt(input,{image});
            check(bool(result) && result.images.size()==1,"single image");
            const auto & layout=result.images[0].layout;
            std::vector<ImageTokenType> expected(3-start,ImageTokenType::Pad);
            expected.insert(expected.end(),body.begin(),body.end());
            check(layout.types==expected && layout.permutation==std::vector<int64_t>({0}),"source grounded tiny layout");
            check(layout.span.block_begin==start && layout.span.visible_begin==3 &&
                  layout.span.visible_end==9 && layout.span.block_end==9,"span residues");
            check(result.tokens.size()==10 && result.tokens.back()==7,"expanded count/text tail");
            for (size_t i=0;i<expected.size();++i)
                check(result.tokens[start+i]==vocab+static_cast<int32_t>(expected[i]),"generated IDs");
        }
        auto pair=prepare_image_prompt({marker,marker},{image,small_image(0x3f80)});
        check(bool(pair) && pair.tokens.size()==17 && pair.images[1].layout.span.block_begin==9 &&
              pair.images[1].layout.span.visible_begin==11 && pair.images[1].layout.span.block_end==17,"consecutive expanded positions");
        auto spaced=prepare_image_prompt({marker,55,marker,66},{image,image});
        check(bool(spaced) && spaced.images[1].layout.span.block_begin==10 && spaced.tokens[9]==55 &&
              spaced.tokens.back()==66,"text between images");
        auto copy=pair;
        pair.images[0].input.patches_bf16[0]=0x3f80;
        check(copy.images[0].input.patches_bf16[0]==0,"copy owns patches");
        auto moved=std::move(copy);
        check(moved.images[1].input.patches_bf16[0]==0x3f80 && moved.tokens.size()==17,"move lifetime");
        auto independent=prepare_image_prompt({marker},{image});
        image.patches_bf16[0]=0xbf80;
        check(independent.images[0].input.patches_bf16[0]==0,"input lifetime independence");
        auto other_request=prepare_image_prompt({marker},{small_image(0x3f80)});
        check(independent.tokens==other_request.tokens,"equal layout fixture");
        independent.images[0].input.patches_bf16[0]=0xbf80;
        check(other_request.images[0].input.patches_bf16[0]==0x3f80,"distinct requests own image storage");
        check(bool(prepare_image_prompt(std::vector<int32_t>(4,marker),std::vector<ImagePatchInput>(4,image))),"four-image boundary");
        rejected(prepare_image_prompt({marker},{}),ImagePromptError::MarkerCount);
        rejected(prepare_image_prompt({1},{image}),ImagePromptError::MarkerCount);
        rejected(prepare_image_prompt({marker,marker},{image}),ImagePromptError::MarkerCount);
        rejected(prepare_image_prompt({-1},{}),ImagePromptError::InvalidToken);
        for (int kind=0;kind<5;++kind) rejected(prepare_image_prompt({vocab+kind},{}),ImagePromptError::InvalidToken);
        rejected(prepare_image_prompt({marker},{image},{},{129279,marker}),ImagePromptError::InvalidContract);
        rejected(prepare_image_prompt({marker},{image},{},{vocab,marker-1}),ImagePromptError::InvalidContract);
        rejected(prepare_image_prompt(std::vector<int32_t>(5,marker),std::vector<ImagePatchInput>(5,image)),ImagePromptError::ImageCount);
        auto bad=image; bad.plan.resized_width++;
        rejected(prepare_image_prompt({marker,marker},{image,bad}),ImagePromptError::InvalidPlan);
        bad=image; bad.plan.aligner_rows=2;
        rejected(prepare_image_prompt({marker},{bad}),ImagePromptError::InvalidPlan);
        bad=image; bad.plan.vit_rows=0;
        rejected(prepare_image_prompt({marker},{bad}),ImagePromptError::InvalidPlan);
        bad=image; bad.plan.vit_rows=std::numeric_limits<uint32_t>::max();
        rejected(prepare_image_prompt({marker},{bad}),ImagePromptError::InvalidPlan);
        bad=image; bad.plan={1008,672,48,72,16,24,false};
        rejected(prepare_image_prompt({marker},{bad}),ImagePromptError::InvalidPlan);
        bad=image; bad.patches_bf16.pop_back();
        rejected(prepare_image_prompt({marker},{bad}),ImagePromptError::InvalidPatches);
        for (uint16_t word : {uint16_t(0x3f81),uint16_t(0xbf81),uint16_t(0x7f80),uint16_t(0x7fc1)}) {
            bad=image; bad.patches_bf16[0]=word;
            rejected(prepare_image_prompt({marker},{bad}),ImagePromptError::InvalidPatches);
        }
        check(bool(prepare_image_prompt({marker},{image},{10,1,9})),"exact context fit");
        rejected(prepare_image_prompt({marker},{image},{9,1,9}),ImagePromptError::ContextOverflow);
        rejected(prepare_image_prompt({marker},{image},{10,1,8}),ImagePromptError::TokenLimit);
        rejected(prepare_image_prompt({1},{},{0,0,1}),ImagePromptError::InvalidLimits);
        rejected(prepare_image_prompt({1},{},{1,2,1}),ImagePromptError::InvalidLimits);
        rejected(prepare_image_prompt({1},{},{1,0,0}),ImagePromptError::InvalidLimits);
        rejected(prepare_image_prompt({1},{},{2147483648ULL,0,1}),ImagePromptError::InvalidLimits);
        rejected(prepare_image_prompt({1},{},{2147483647,0,2147483647}),ImagePromptError::InvalidLimits);
        rejected(prepare_image_prompt({1},{},{2147483647,2147483647,1}),ImagePromptError::ContextOverflow);
        check(bool(prepare_image_prompt({1},{},{2147483647,2147483646,1})),"wide exact arithmetic");
        rejected(prepare_image_prompt({1},{},{1,std::numeric_limits<uint64_t>::max(),1}),ImagePromptError::InvalidLimits);
        std::cout<<"PASS text, cardinality, source layout residues, ordering, ownership, negative contracts, checked context\n";
        return 0;
    } catch (const std::exception & e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
