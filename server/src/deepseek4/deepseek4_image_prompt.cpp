#include "deepseek4_image_prompt.h"
#include <limits>
#include <new>
#include <utility>

namespace dflash::vision {
namespace {
PreparedImagePrompt fail(ImagePromptError error,const std::string & message) {
    PreparedImagePrompt result;
    result.error=error;
    result.message=message;
    return result;
}
}
PreparedImagePrompt prepare_image_prompt(const std::vector<std::int32_t> & tokens,
    const std::vector<ImagePatchInput> & images,const ImagePromptLimits & limits,
    const ImageTokenizerContract & tokenizer) {
    if (tokenizer.vocabulary!=129280 || tokenizer.marker!=129264)
        return fail(ImagePromptError::InvalidContract,"unsupported image tokenizer contract");
    if (limits.context_capacity==0 || limits.context_capacity>std::numeric_limits<int>::max() ||
        limits.output_reserve>limits.context_capacity || limits.max_expanded_tokens==0 ||
        limits.max_expanded_tokens>MAX_PREPARED_PROMPT_TOKENS)
        return fail(ImagePromptError::InvalidLimits,"invalid context, output reserve, or prompt bound");
    if (images.size()>4) return fail(ImagePromptError::ImageCount,"at most four images are supported");
    if (tokens.size()>limits.max_expanded_tokens)
        return fail(ImagePromptError::TokenLimit,"rendered tokens exceed prompt bound");
    size_t markers=0;
    for (int32_t token:tokens) {
        if (token<0 || static_cast<uint32_t>(token)>=tokenizer.vocabulary)
            return fail(ImagePromptError::InvalidToken,"rendered token outside text vocabulary");
        if (token==tokenizer.marker) ++markers;
    }
    if (markers!=images.size()) return fail(ImagePromptError::MarkerCount,"final image marker count differs from image count");

    try {
        for (size_t i=0;i<images.size();++i) {
            const auto & image=images[i];
            const auto & plan=image.plan;
            const std::string where="image "+std::to_string(i)+": ";
            if (plan.vit_rows==0 || plan.vit_cols==0 ||
                uint64_t(plan.vit_cols)*14!=plan.resized_width ||
                uint64_t(plan.vit_rows)*14!=plan.resized_height ||
                (uint64_t(plan.vit_rows)+2)/3!=plan.aligner_rows ||
                (uint64_t(plan.vit_cols)+2)/3!=plan.aligner_cols)
                return fail(ImagePromptError::InvalidPlan,where+"inconsistent patch/grid dimensions");
            // Source safe_resize reserves three leading pads regardless of the
            // eventual position. Reuse the core layout builder for that bound.
            ImageLayout maximum_padding;
            if (!build_image_layout(plan.aligner_rows,plan.aligner_cols,0,maximum_padding))
                return fail(ImagePromptError::InvalidPlan,where+"source image token budget exceeded");
            // The validated layout bounds the aligner grid, hence this product.
            const uint64_t expected=uint64_t(plan.vit_rows)*plan.vit_cols*588;
            if (expected!=image.patches_bf16.size())
                return fail(ImagePromptError::InvalidPatches,where+"patch tensor size mismatch");
            for (uint16_t value:image.patches_bf16) {
                // BF16 magnitude ordering makes this both a finite-value and
                // normalized [-1,1] check, without floating-point conversion.
                if ((value&0x7fffU)>0x3f80U)
                    return fail(ImagePromptError::InvalidPatches,where+"patch value outside finite normalized range");
            }
        }
        std::vector<ImageLayout> layouts;
        layouts.reserve(images.size());
        uint64_t position=0;
        for (int32_t token:tokens) {
            uint64_t added=1;
            if (token==tokenizer.marker) {
                const auto & plan=images[layouts.size()].plan;
                ImageLayout layout;
                if (!build_image_layout(plan.aligner_rows,plan.aligner_cols,position,layout))
                    return fail(ImagePromptError::InvalidPlan,"image layout could not be built");
                added=layout.types.size();
                layouts.push_back(std::move(layout));
            }
            if (added>limits.max_expanded_tokens-position)
                return fail(ImagePromptError::TokenLimit,"expanded tokens exceed prompt bound");
            position+=added;
        }
        if (position>limits.context_capacity-limits.output_reserve)
            return fail(ImagePromptError::ContextOverflow,"expanded prompt and output reserve exceed context capacity");

        PreparedImagePrompt result;
        result.tokens.reserve(static_cast<size_t>(position));
        result.images.reserve(images.size());
        size_t image_index=0;
        for (int32_t token:tokens) {
            if (token!=tokenizer.marker) { result.tokens.push_back(token); continue; }
            auto & layout=layouts[image_index];
            for (ImageTokenType kind:layout.types)
                result.tokens.push_back(static_cast<int32_t>(tokenizer.vocabulary)+static_cast<int32_t>(kind));
            result.images.push_back({images[image_index],std::move(layout)});
            ++image_index;
        }
        return result;
    } catch (const std::bad_alloc &) {
        return fail(ImagePromptError::AllocationFailed,"image prompt allocation failed");
    }
}
}
