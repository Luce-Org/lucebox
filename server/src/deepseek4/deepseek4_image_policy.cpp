#include "deepseek4_image_policy.h"
#include <algorithm>
#include <cmath>

namespace dflash::vision {
bool select_image_experts(const float * scores, const float * bias, size_t experts,
                          size_t topk, ImageExpertSelection & output, std::string & error,
                          float route_scale) {
    output = {}; error.clear();
    if (!scores || !bias || experts==0 || experts>MAX_IMAGE_EXPERTS || topk==0 || topk>experts) {
        error="invalid expert arrays/count/topk"; return false;
    }
    if (!std::isfinite(route_scale) || route_scale<=0) {
        error="route scale must be finite and positive"; return false;
    }
    std::array<float,MAX_IMAGE_EXPERTS> corrected{};
    std::array<int32_t,MAX_IMAGE_EXPERTS> order{};
    for (size_t i=0;i<experts;++i) {
        if (!std::isfinite(scores[i]) || scores[i]<0 || !std::isfinite(bias[i])) {
            error="scores must be finite/nonnegative and bias finite"; return false;
        }
        corrected[i]=scores[i]+bias[i];
        if (!std::isfinite(corrected[i])) { error="corrected score overflow"; return false; }
        order[i]=static_cast<int32_t>(i);
    }
    std::sort(order.begin(),order.begin()+experts,[&](int32_t a,int32_t b) {
        return corrected[a]>corrected[b] || (corrected[a]==corrected[b] && a<b);
    });
    float sum=0;
    for (size_t i=0;i<topk;++i) sum+=scores[order[i]];
    if (!std::isfinite(sum) || sum<=0) { error="selected score sum must be finite and positive"; return false; }
    ImageExpertSelection result;
    result.count=topk;
    for (size_t i=0;i<topk;++i) {
        result.indices[i]=order[i];
        result.weights[i]=(scores[order[i]]/sum)*route_scale;
    }
    output=result;
    return true;
}
bool raw_key_visible(int64_t query,int64_t key,int64_t window,
                     int64_t image_begin,int64_t image_end,bool & visible) {
    visible=false;
    const bool absent=image_begin==-1 && image_end==-1;
    if (query<0 || key<0 || window<=0 || (!absent && (image_begin<0 || image_end<=image_begin))) return false;
    visible=key<=query && query-key<window;
    if (!absent && query>=image_begin && query<image_end)
        visible=visible || (key>=image_begin && key<image_end);
    return true;
}
}
