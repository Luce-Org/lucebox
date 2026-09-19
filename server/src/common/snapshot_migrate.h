#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cstring>
#include <vector>
namespace dflash::common {
// Owns only the unpublished destination. The caller publishes it after success.
struct SnapshotCopy {
    ggml_context *ctx=nullptr;
    ggml_backend_buffer_t buf=nullptr;
    ~SnapshotCopy(){if(buf)ggml_backend_buffer_free(buf);if(ctx)ggml_free(ctx);}
    bool copy(ggml_context *source,ggml_backend_t destination) {
        if(!source || !destination)return false;
        size_t count=0;for(auto*t=ggml_get_first_tensor(source);t;t=ggml_get_next_tensor(source,t))++count;
        ctx=ggml_init({ggml_tensor_overhead()*(count+1),nullptr,true});if(!ctx)return false;
        for(auto*t=ggml_get_first_tensor(source);t;t=ggml_get_next_tensor(source,t)) {
            if(!ggml_is_contiguous(t) || !t->name[0])return false;
            auto *d=ggml_dup_tensor(ctx,t);ggml_set_name(d,t->name);
        }
        buf=ggml_backend_alloc_ctx_tensors(ctx,destination);if(!buf)return false;
        std::vector<unsigned char> a(8*1024*1024),b(a.size());
        for(auto*t=ggml_get_first_tensor(source);t;t=ggml_get_next_tensor(source,t)) {
            auto*d=ggml_get_tensor(ctx,t->name);if(!d || ggml_nbytes(d)!=ggml_nbytes(t))return false;
            for(size_t off=0;off<ggml_nbytes(t);off+=a.size()) {
                size_t n=std::min(a.size(),ggml_nbytes(t)-off);
                ggml_backend_tensor_get(t,a.data(),off,n);
                ggml_backend_tensor_set(d,a.data(),off,n);
                ggml_backend_tensor_get(d,b.data(),off,n);
                if(std::memcmp(a.data(),b.data(),n))return false;
            }
        }
        ggml_backend_synchronize(destination);return true;
    }
};
}
