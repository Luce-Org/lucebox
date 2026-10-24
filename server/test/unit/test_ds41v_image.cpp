// DeepSeek V4.1 image input: the resize plan, the image span, prompt
// expansion, row assembly without an image_pad row, and image routing over
// 384 experts.
//
// The plans were produced by DeepSeek-V4.1-Flash inference/image_processor.py
// (plan_image_grid with the released config: patch 14, downsample 3, at most
// 1024 tokens, 295,936 minimum pixels, no aspect limit), run on the sizes below.
#include "deepseek4_image_assembly.h"
#include "deepseek4_image_policy.h"
#include "deepseek4_image_prompt.h"
#include "deepseek4_vision_preprocess.h"
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace luce::vision;
namespace {
constexpr int32_t marker = 129264, vocab = 129280;
void check(bool value, const std::string & why) { if (!value) throw std::runtime_error(why); }

struct PlanCase { uint32_t width, height, best_width, best_height, rows, cols, tokens; };
// {original width, height, resized width, height, aligner rows, columns, image tokens}
const PlanCase kPlans[] = {
    {1,1,546,546,13,13,184},          {14,14,546,546,13,13,184},
    {100,100,546,546,13,13,184},      {544,544,546,546,13,13,184},
    {545,545,546,546,13,13,184},      {640,480,644,490,12,16,206},
    {480,640,490,644,16,12,210},      {800,600,812,602,15,20,317},
    {1024,768,1036,770,19,25,496},    {1280,720,1288,728,18,31,578},
    {1920,1080,1708,966,23,41,968},   {1080,1920,966,1708,41,23,986},
    {2048,2048,1302,1302,31,31,994},  {4000,3000,1512,1134,27,36,1001},
    {3000,4000,1134,1512,36,27,1010}, {8000,100,8008,112,3,191,578},
    {100,8000,112,8008,191,3,766},    {20000,30,20006,42,1,477,480},
    {30,20000,42,20006,477,1,956},    {65535,1,42882,42,1,1021,1024},
    {1,65535,42,21462,511,1,1024},    {1537,1,21336,14,1,508,511},
    {37,4096,56,5726,137,2,413},      {4096,37,5726,56,2,137,278},
    {1366,768,1372,770,19,33,648},    {3840,2160,1708,966,23,41,968},
    {1000,1001,1008,1008,24,24,602},  {999,1000,1008,1008,24,24,602},
    {13,7000,28,12628,301,1,604},     {7000,13,12628,28,1,301,304},
    {300,1200,308,1204,29,8,263},     {1200,300,1204,308,8,29,242},
    {588,588,588,588,14,14,212},      {589,587,602,588,14,15,226},
    {4095,4095,1302,1302,31,31,994},  {5000,5000,1302,1302,31,31,994},
    {123,456,294,1050,25,7,202},      {4567,321,4578,322,8,109,882},
};

ImagePatchInput image_of(uint32_t vit_rows, uint32_t vit_cols) {
    ImagePatchInput image;
    image.plan = {vit_cols * 14, vit_rows * 14, vit_rows, vit_cols, (vit_rows + 2) / 3, (vit_cols + 2) / 3, false};
    image.patches_bf16.assign(size_t(vit_rows) * vit_cols * 588, 0);
    return image;
}
}  // namespace

int main() {
    try {
        const PreprocessConfig v41 = v41_preprocess_config();
        check(bool(validate_config(v41)) && bool(validate_config(PreprocessConfig{})), "both recipes are valid");
        for (int field = 0; field < 4; ++field) {
            PreprocessConfig mixed = v41;
            if (field == 0) mixed.max_tokens = 384;
            if (field == 1) mixed.min_pixels = 147456;
            if (field == 2) mixed.max_aspect_ratio = 8;
            if (field == 3) mixed.compress_pad_to = 4;
            check(!validate_config(mixed), "a recipe mixing V4 and V4.1 fields is rejected");
        }

        // 1. The resize plan of every size matches the source's.
        for (const auto & c : kPlans) {
            const std::string where = std::to_string(c.width) + "x" + std::to_string(c.height) + ": ";
            ResizePlan plan;
            const auto status = plan_image(c.width, c.height, plan, v41);
            check(bool(status), where + "plan failed: " + status.message);
            check(plan.resized_width == c.best_width && plan.resized_height == c.best_height, where + "resized size");
            check(plan.vit_rows == c.best_height / 14 && plan.vit_cols == c.best_width / 14, where + "patch grid");
            check(plan.aligner_rows == c.rows && plan.aligner_cols == c.cols, where + "aligner grid");
            check(!plan.direct_resize, where + "V4.1 pads every image (no aspect limit)");
            ImageLayout layout;
            check(bool(build_image_layout(plan.aligner_rows, plan.aligner_cols, 0, layout, v41)), where + "layout");
            check(layout.types.size() == c.tokens, where + "image token count");
        }

        // 2. Reading order, one newline per row, no padding, at any position.
        using T = ImageTokenType;
        ImageLayout layout;
        check(bool(build_image_layout(2, 3, 5, layout, v41)), "2x3 layout");
        const std::vector<T> expected{T::Start, T::Image, T::Image, T::Image, T::Newline,
                                      T::Image, T::Image, T::Image, T::Newline, T::End};
        check(layout.types == expected, "reading-order types");
        check(layout.permutation == std::vector<int64_t>({0, 1, 2, 3, 4, 5}), "aligner rows in reading order");
        check(layout.span.block_begin == 5 && layout.span.visible_begin == 5 &&
              layout.span.visible_end == 15 && layout.span.block_end == 15, "span has no padding");
        check(!build_image_layout(32, 32, 0, layout, v41), "32x32 costs 1058 tokens: rejected");
        check(bool(build_image_layout(31, 31, 0, layout, v41)) && layout.types.size() == 994, "largest square fits");

        // 3. Prompt expansion: ids are the vocabulary plus the type, the
        // block starts where the marker was, whatever that position is.
        const ImagePatchInput tiny = image_of(3, 3);
        for (int lead = 0; lead < 4; ++lead) {
            std::vector<int32_t> rendered(size_t(lead), 42);
            rendered.push_back(marker);
            rendered.push_back(8);
            const auto prompt = prepare_image_prompt(rendered, {tiny}, {}, {}, v41);
            check(bool(prompt) && prompt.images.size() == 1, "one image");
            std::vector<int32_t> ids(size_t(lead), 42);
            for (int32_t kind : {0, 2, 3, 4}) ids.push_back(vocab + kind);
            ids.push_back(8);
            check(prompt.tokens == ids, "expanded ids");
            check(prompt.images[0].layout.span.block_begin == uint64_t(lead) &&
                  prompt.images[0].layout.span.block_end == uint64_t(lead) + 4, "no compression padding");
        }
        const auto pair = prepare_image_prompt({marker, 55, marker}, {tiny, image_of(6, 9)}, {}, {}, v41);
        check(bool(pair) && pair.tokens.size() == 4 + 1 + 10 && pair.tokens[4] == 55 &&
              pair.images[1].layout.span.block_begin == 5 && pair.images[1].layout.span.block_end == 15,
              "text between two images");
        const auto over = prepare_image_prompt({marker}, {image_of(96, 96)}, {}, {}, v41);
        check(!over && over.error == ImagePromptError::InvalidPlan, "a plan over 1024 tokens is rejected");
        const auto v4 = prepare_image_prompt({marker}, {tiny});
        check(bool(v4) && v4.tokens.size() == 9, "the V4 recipe keeps its pads");

        // 4. Row assembly without an image_pad row.
        const ImageSentinels marks{{1, 1}, {}, {3, 3}, {4, 4}};
        const ImageRaster raster{1, 2, {9, 9}};
        std::vector<float> rows;
        std::string error;
        check(assemble_image_rows(pair.images[0].layout, raster, marks, 2, rows, error), "assembly: " + error);
        check(rows == std::vector<float>({1, 1, 9, 9, 3, 3, 4, 4}), "start, image, newline, end rows");
        check(!assemble_image_rows(v4.images[0].layout, raster, marks, 2, rows, error),
              "a padded layout needs the pad row");

        // 5. Image routing over V4.1's 384 experts.
        std::vector<float> scores(385, 1.0f), bias(385, 0.0f);
        for (int e : {383, 300, 257, 5, 129, 200}) bias[size_t(e)] = 10.0f;
        ImageExpertSelection selection;
        check(select_image_experts(scores.data(), bias.data(), 384, 6, selection, error, 1.5f), "384 experts: " + error);
        const int picked[6] = {5, 129, 200, 257, 300, 383};
        for (int k = 0; k < 6; ++k) {
            check(selection.indices[size_t(k)] == picked[k], "biased experts, ties to the lower index");
            check(selection.weights[size_t(k)] == 0.25f, "unbiased weights times the route scale");
        }
        check(!select_image_experts(scores.data(), bias.data(), 385, 6, selection, error), "385 experts rejected");

        std::cout << "PASS V4.1 plans for " << sizeof(kPlans) / sizeof(kPlans[0])
                  << " sizes, reading-order span, prompt expansion, pad-free assembly, 384-expert routing\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
