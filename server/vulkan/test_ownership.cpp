// CPU fault injection through the real backend constructors. No Vulkan loader
// or device is linked: only the device API boundary below is replaced.
#include "qwen2_backend.h"
#include "lfm2_backend.h"
#include "ggml-alloc.h"
#include "ggml-vulkan.h"
#include "gguf.h"
#include "server/tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::set<void *> backends, buffers, allocators;
int buffer_calls=0, allocator_calls=0, init_calls=0, freed=0, fail_buffer=0, fail_allocator=0;
bool throw_alloc=false;
void check(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
void * acquire(std::set<void *> & live) { auto p=new char;live.insert(p);return p; }
void release(std::set<void *> & live, void * p) {
    if (live.erase(p)!=1) std::abort(); // double-free/foreign owner is a test failure
    delete static_cast<char *>(p);
}
}
// All symbols below replace external device allocation operations only. Model
// validation, metadata allocation, mapping, construction/unwinding, and shutdown
// are the unchanged production translation units linked into this test.
extern "C" {
ggml_backend_t __wrap_ggml_backend_vk_init(size_t) { ++init_calls;return reinterpret_cast<ggml_backend_t>(acquire(backends)); }
const char * __wrap_ggml_backend_name(ggml_backend_t) { return "CPU ownership fault harness (not a device)"; }
void __wrap_ggml_backend_free(ggml_backend_t p) {
    if (!buffers.empty() || !allocators.empty()) std::abort();
    release(backends,p);++freed;
}
void __wrap_ggml_backend_synchronize(ggml_backend_t) {}
ggml_backend_buffer_t __wrap_ggml_backend_alloc_ctx_tensors(ggml_context *,ggml_backend_t) {
    if (++buffer_calls==fail_buffer) { if (throw_alloc) throw std::bad_alloc();return nullptr; }
    return reinterpret_cast<ggml_backend_buffer_t>(acquire(buffers));
}
void __wrap_ggml_backend_buffer_free(ggml_backend_buffer_t p) { release(buffers,p); }
void __wrap_ggml_backend_buffer_set_usage(ggml_backend_buffer_t,enum ggml_backend_buffer_usage) {}
size_t __wrap_ggml_backend_buffer_get_size(ggml_backend_buffer_t) { return 1; }
ggml_backend_buffer_type_t __wrap_ggml_backend_get_default_buffer_type(ggml_backend_t) { return nullptr; }
ggml_gallocr_t __wrap_ggml_gallocr_new(ggml_backend_buffer_type_t) {
    if (++allocator_calls==fail_allocator) { if (throw_alloc) throw std::bad_alloc();return nullptr; }
    return reinterpret_cast<ggml_gallocr_t>(acquire(allocators));
}
void __wrap_ggml_gallocr_free(ggml_gallocr_t p) { release(allocators,p); }
void __wrap_ggml_backend_tensor_set(ggml_tensor *, const void *, size_t, size_t) {}
}
namespace luce::common {
// Tokenization is irrelevant to ownership: avoid embedding a real model vocab.
// No generate/forward method is invoked by this harness.
bool Tokenizer::load_from_gguf(const char *) { return true; }
std::string Tokenizer::decode(const std::vector<int32_t> &) const { return {}; }
}

static void write_fixture(const std::string & path, bool lfm) {
    std::unique_ptr<gguf_context,decltype(&gguf_free)> g(gguf_init_empty(),gguf_free);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> c(ggml_init({4*1024*1024,nullptr,true}),ggml_free);
    check(bool(g)&&bool(c),"fixture metadata allocation");
    gguf_set_val_str(g.get(),"general.architecture",lfm?"lfm2moe":"qwen2");
    auto u=[&](const char *key,int n){gguf_set_val_u32(g.get(),key,n);};
    auto tensor=[&](const std::string & name,std::vector<int64_t> shape) {
        auto t=ggml_new_tensor(c.get(),GGML_TYPE_F32,int(shape.size()),shape.data());
        ggml_set_name(t,name.c_str());gguf_add_tensor(g.get(),t);
    };
    if (!lfm) {
        u("qwen2.embedding_length",4);u("qwen2.attention.head_count",1);
        u("qwen2.attention.head_count_kv",1);u("qwen2.block_count",1);
        tensor("token_embd.weight",{4,8});
    } else {
        u("lfm2moe.embedding_length",2048);u("lfm2moe.attention.head_count",32);u("lfm2moe.block_count",24);
        u("lfm2moe.vocab_size",128000);u("lfm2moe.feed_forward_length",7168);u("lfm2moe.expert_count",32);
        u("lfm2moe.expert_used_count",4);u("lfm2moe.expert_feed_forward_length",1792);
        u("lfm2moe.leading_dense_block_count",2);u("lfm2moe.shortconv.l_cache",3);
        uint32_t kv[24]{};gguf_set_arr_data(g.get(),"lfm2moe.attention.head_count_kv",GGUF_TYPE_UINT32,kv,24);
        tensor("token_embd.weight",{2048,128000});tensor("token_embd_norm.weight",{2048});
        for (int i=0;i<24;++i) {
            auto n=[&](const char *s){return "blk."+std::to_string(i)+"."+s;};
            tensor(n("attn_norm.weight"),{2048});tensor(n("ffn_norm.weight"),{2048});
            tensor(n("shortconv.conv.weight"),{3,2048});
            tensor(n("shortconv.in_proj.weight"),{2048,6144});tensor(n("shortconv.out_proj.weight"),{2048,2048});
            if (i<2) {
                tensor(n("ffn_gate.weight"),{2048,7168});tensor(n("ffn_up.weight"),{2048,7168});tensor(n("ffn_down.weight"),{7168,2048});
            } else {
                tensor(n("ffn_gate_inp.weight"),{2048,32});tensor(n("exp_probs_b.bias"),{32});
                tensor(n("ffn_gate_exps.weight"),{2048,1792,32});tensor(n("ffn_up_exps.weight"),{2048,1792,32});tensor(n("ffn_down_exps.weight"),{1792,2048,32});
            }
        }
    }
    check(gguf_write_to_file(g.get(),path.c_str(),true),"write fixture metadata");
    if (!lfm) {
        // Tiny Qwen fixture includes enough payload for bounds checks. Uploads
        // are mocked; no arithmetic or model-inference claim is made.
        std::ofstream out(path,std::ios::binary|std::ios::app);
        std::vector<char> zeros(4096);out.write(zeros.data(),zeros.size());check(bool(out),"write fixture payload");
    }
}
static void reset_faults(int b,int a,bool throws) {
    check(backends.empty()&&buffers.empty()&&allocators.empty(),"resources leaked between cases");
    buffer_calls=allocator_calls=init_calls=freed=0;fail_buffer=b;fail_allocator=a;throw_alloc=throws;
}
static void failure_case(const std::string & name,const std::string & path,bool lfm,int b,int a,bool throws,const std::string & expected) {
    reset_faults(b,a,throws);
    bool caught=false;
    try { auto model=lfm?luce::common::make_lfm2_vulkan(path,4096,256,false):luce::common::make_qwen2_vulkan(path,4096,256,false); }
    catch (const std::bad_alloc &) { caught=throws; }
    catch (const std::runtime_error & e) { caught=!throws && std::string(e.what())==expected; }
    check(caught,"wrong constructor failure");
    check(init_calls==1 && freed==1,"backend not released exactly once");
    check(backends.empty()&&buffers.empty()&&allocators.empty(),"constructor leaked owned resource");
    if(b)check(buffer_calls==b,"did not reach requested buffer failure");
    if(a)check(allocator_calls==a,"did not reach requested allocator failure");
    std::cout<<"PASS "<<name<<" init="<<init_calls<<" freed="<<freed<<" live_resources=0\n";
}
int main(int argc,char **argv) {
    try {
        check(argc==2,"usage: test ownership FIXTURE_DIRECTORY");
        const std::string q=std::string(argv[1])+"/qwen-ownership.gguf", l=std::string(argv[1])+"/lfm-ownership.gguf";
        write_fixture(q,false);write_fixture(l,true);
        failure_case("Qwen missing GGUF after init",std::string(argv[1])+"/missing.gguf",false,0,0,false,"GGUF metadata read failed");
        failure_case("LFM truncated mapping after weight buffer",l,true,0,0,false,"tensor exceeds model shard");
        for(bool throws:{false,true}) {
            failure_case("Qwen weight allocation",q,false,1,0,throws,"Vulkan weight allocation failed");
            failure_case("LFM weight allocation",l,true,1,0,throws,"Vulkan weight allocation failed");
            failure_case("Qwen KV allocation",q,false,2,0,throws,"KV allocation failed");
            failure_case("Qwen decode allocator",q,false,0,1,throws,"decode allocator allocation failed");
            failure_case("Qwen prefill allocator",q,false,0,2,throws,"prefill allocator allocation failed");
        }
        reset_faults(0,0,false);
        { auto model=luce::common::make_qwen2_vulkan(q,4096,256,false);model->shutdown();model->shutdown(); }
        check(init_calls==1&&freed==1&&backends.empty()&&buffers.empty()&&allocators.empty(),"shutdown/destructor not idempotent");
        std::cout<<"PASS repeated shutdown then destructor frees each resource once\n";
        return 0;
    } catch(const std::exception &e) {std::cerr<<"FAIL ownership: "<<e.what()<<'\n';return 1;}
}
