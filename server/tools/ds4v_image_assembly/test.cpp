#include "deepseek4/deepseek4_image_assembly.h"
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace dflash::vision;
static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static PromptImage fixture(uint64_t position) {
    PromptImage image;
    image.input.plan.aligner_rows = 2;
    image.input.plan.aligner_cols = 2;
    image.layout.span = {position, position + 1, position + 9, position + 10};
    image.layout.types = {ImageTokenType::Pad, ImageTokenType::Start,
        ImageTokenType::Image, ImageTokenType::Image, ImageTokenType::Newline,
        ImageTokenType::Image, ImageTokenType::Image, ImageTokenType::Pad,
        ImageTokenType::End, ImageTokenType::Pad};
    image.layout.permutation = {2, 0, 3, 1};
    return image;
}

int main() {
    try {
        ImageRequestGate gate;
        auto lease = gate.try_acquire();
        check(bool(lease) && !gate.try_acquire(), "concurrent image request was admitted");
        auto retained_lease = lease;
        lease.reset();
        check(!gate.try_acquire(), "request copy released image admission early");
        retained_lease.reset();
        lease = gate.try_acquire();
        check(bool(lease), "completed request did not release image admission");
        {
            ImageRequestGate transient;
            retained_lease = transient.try_acquire();
        }
        retained_lease.reset();
        const ImageSentinels sentinels{{10,11}, {20,21}, {30,31}, {40,41}};
        const ImageRaster raster{4, 2, {100,101,200,201,300,301,400,401}};
        const std::vector<float> expected{20,21,10,11,300,301,100,101,30,31,
                                          400,401,200,201,20,21,40,41,20,21};
        std::string error;
        std::vector<float> output{999};
        check(assemble_image_rows(fixture(1).layout, raster, sentinels, 2, output, error), "assembly rejected");
        check(output == expected, "sentinel identity or raster permutation applied incorrectly");
        auto bad_assembly = [&](ImageLayout layout, ImageRaster values, ImageSentinels marks) {
            output = {999};
            check(!assemble_image_rows(layout, values, marks, 2, output, error), "malformed assembly accepted");
            check(output == std::vector<float>{999} && !error.empty(), "assembly failure changed output");
        };
        auto layout = fixture(1).layout;
        layout.permutation = {2,0,2,1}; bad_assembly(layout,raster,sentinels);
        layout = fixture(1).layout; layout.permutation[0] = -1; bad_assembly(layout,raster,sentinels);
        layout = fixture(1).layout; layout.permutation[0] = 4; bad_assembly(layout,raster,sentinels);
        layout = fixture(1).layout; layout.types[4] = static_cast<ImageTokenType>(99); bad_assembly(layout,raster,sentinels);
        layout = fixture(1).layout; layout.types[1] = ImageTokenType::End; bad_assembly(layout,raster,sentinels);
        layout = fixture(1).layout; layout.span.visible_end--; bad_assembly(layout,raster,sentinels);
        auto values = raster; values.columns = 3; bad_assembly(fixture(1).layout,values,sentinels);
        values = raster; values.values.pop_back(); bad_assembly(fixture(1).layout,values,sentinels);
        values = raster; values.values[0] = std::numeric_limits<float>::quiet_NaN(); bad_assembly(fixture(1).layout,values,sentinels);
        auto marks = sentinels; marks.pad[0] = std::numeric_limits<float>::infinity(); bad_assembly(fixture(1).layout,raster,marks);
        marks = sentinels; marks.end.pop_back(); bad_assembly(fixture(1).layout,raster,marks);

        std::vector<PromptImage> images{fixture(1),fixture(12)};
        const ImageRows old{{777},{888}};
        ImageRows result = old;
        int calls = 0;
        ImageEncode fail_second = [&](const PromptImage &, ImageRaster & out, std::string & reason) {
            if (++calls == 2) { reason = "second image failed"; return false; }
            out = raster; return true;
        };
        check(!materialize_image_rows(images,sentinels,2,fail_second,{},result,error) &&
              calls == 2 && result == old && error == "second image failed", "second-image failure is not atomic");
        calls = 0;
        bool stop = false;
        ImageEncode cancel_after_first = [&](const PromptImage &, ImageRaster & out, std::string &) {
            ++calls; out = raster; stop = true; return true;
        };
        check(!materialize_image_rows(images,sentinels,2,cancel_after_first,[&] { return stop; },result,error) &&
              calls == 1 && result == old && !error.empty(), "cancellation launched a remaining encode");
        calls = 0;
        check(!materialize_image_rows(images,sentinels,2,cancel_after_first,[] { return true; },result,error) &&
              calls == 0 && result == old, "pre-cancelled materialization launched encode");
        ImageEncode good = [&](const PromptImage &, ImageRaster & out, std::string &) {
            out = raster;
            for (float & v : out.values) v += 1000.0f * calls;
            ++calls; return true;
        };
        check(materialize_image_rows(images,sentinels,2,good,{},result,error) && calls == 2,
              "valid materialization failed");
        check(result[0] == expected && result[1][4] == 1300 && result[1][0] == 20,
              "equal-layout images lost distinct raster values");
        calls = 0;
        auto wrong_plan = images; wrong_plan[1].input.plan.aligner_cols = 3;
        ImageRows retained = result;
        check(!materialize_image_rows(wrong_plan,sentinels,2,good,{},result,error) && calls == 0 &&
              result == retained, "bad second-image shape was not rejected before encoding");
        check(!materialize_image_rows(images,sentinels,2,
              [](const PromptImage &, ImageRaster &, std::string &) -> bool { throw 1; },{},result,error) &&
              result == retained, "callback exception changed output");

        PreparedImagePrompt prompt;
        prompt.images = images;
        prompt.tokens.assign(23, 7);
        for (const auto & image : images) {
            for (size_t r = 0; r < image.layout.types.size(); ++r)
                prompt.tokens[size_t(image.layout.span.block_begin) + r] = 100 + int32_t(image.layout.types[r]);
        }
        size_t text_calls = 0, text_tokens = 0;
        TextEmbed embed = [&](const int32_t * ids, size_t count, float * out) {
            ++text_calls; text_tokens += count;
            for (size_t i = 0; i < count; ++i) {
                check(ids[i] >= 0 && ids[i] < 100, "external ID reached ordinary embedder");
                out[2*i] = float(ids[i]); out[2*i+1] = float(ids[i]+1);
            }
            return true;
        };
        check(embed_image_prompt_chunk(prompt,result,100,2,0,23,embed,output,error), "mixed prompt failed");
        check(text_calls == 3 && text_tokens == 3 && output.size() == 46 &&
              output[0] == 7 && output[6] == 300 && output[28] == 1300 && output[44] == 7,
              "mixed embedding order or offsets are wrong");
        check(embed_image_prompt_chunk(prompt,result,100,2,1,10,{},output,error) && output == expected,
              "exact image block slice failed");
        auto reject_chunk = [&](const PreparedImagePrompt & p, size_t start, size_t count) {
            output = {999}; text_calls = 0;
            check(!embed_image_prompt_chunk(p,result,100,2,start,count,embed,output,error) &&
                  output == std::vector<float>{999} && text_calls == 0 && !error.empty(),
                  "invalid chunk reached embedder or changed output");
        };
        reject_chunk(prompt,2,9); reject_chunk(prompt,0,5);
        auto invalid = prompt; invalid.tokens[0] = 100; reject_chunk(invalid,0,23);
        invalid = prompt; invalid.tokens[3] = 100; reject_chunk(invalid,0,23);
        invalid = prompt; invalid.tokens[22] = -1; reject_chunk(invalid,0,23);
        output = {999};
        check(!embed_image_prompt_chunk(prompt,result,100,2,0,23,
              [](const int32_t *,size_t,float *) { return false; },output,error) &&
              output == std::vector<float>{999}, "text callback failure changed output");
        std::cout << "PASS image assembly identities/permutation, atomic materialization/cancellation, mixed embedding boundaries\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
