#include "qwen2_backend.h"
#include "greedy.h"
#include "resource_owners.h"
#include "ggml-alloc.h"
#include "ggml-vulkan.h"
#include "gguf.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace luce::common {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
void require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
struct Context { ggml_context *p=nullptr; ~Context(){if(p)ggml_free(p);} };
struct Buffer { ggml_backend_buffer_t p=nullptr; ~Buffer(){if(p)ggml_backend_buffer_free(p);} };
struct Gguf { gguf_context *p=nullptr; ~Gguf(){if(p)gguf_free(p);} };
struct Mapping {
 int fd=-1; size_t size=0; void *p=MAP_FAILED;
 explicit Mapping(const std::string &path){fd=open(path.c_str(),O_RDONLY);require(fd>=0,"open model failed");struct stat s{};require(fstat(fd,&s)==0,"stat model failed");size=s.st_size;p=mmap(nullptr,size,PROT_READ,MAP_PRIVATE,fd,0);require(p!=MAP_FAILED,"mmap model failed");}
 ~Mapping(){if(p!=MAP_FAILED)munmap(p,size);if(fd>=0)close(fd);}
};
class Qwen2Vulkan final : public ModelBackend {
 VulkanBackendOwner backend_;
 GraphAllocatorOwner decode_alloc_, prefill_alloc_;
 struct StepGraph {
  Context context; ggml_cgraph *graph=nullptr;
  ggml_tensor *tokens=nullptr,*positions=nullptr,*mask=nullptr,*logits=nullptr;
  int cache_len=0;
  std::vector<std::pair<ggml_tensor*,size_t>> writes;
 };
 std::unique_ptr<StepGraph> decode_graph_;
 Context weights_,cache_;
 Buffer wbuf_,cbuf_;
 std::map<std::string,ggml_tensor*> tensors_;
 std::vector<ggml_tensor*> keys_,values_;
 int dim_=0,heads_=0,kvheads_=0,layers_=0,head_=0,vocab_=0;
 int ctx_,chunk_,eos_=151643,eot_=151645;
 float eps_=1e-6f,theta_=1000000;
 bool flash_;
 ggml_tensor *w(const std::string &s) const { auto i=tensors_.find(s);require(i!=tensors_.end(),("missing weight "+s).c_str());return i->second; }
 ggml_tensor *norm(ggml_context *c,ggml_tensor *x,ggml_tensor *scale){return ggml_mul(c,ggml_rms_norm(c,x,eps_),scale);}
public:
 Qwen2Vulkan(const std::string &path,int context,int chunk,bool flash):ctx_(context),chunk_(chunk),flash_(flash){
  backend_.reset(ggml_backend_vk_init(0));require(backend_!=nullptr,"Vulkan device initialization failed");
  std::fprintf(stderr,"[lucebox-vulkan] native Qwen2 target on %s; context=%d chunk=%d flash=%d\n",ggml_backend_name(backend_.get()),ctx_,chunk_,flash_);
  weights_.p=ggml_init({16*1024*1024,nullptr,true}); require(weights_.p,"weight metadata allocation failed");
  struct Part { std::string path; std::unique_ptr<Gguf> g; std::unique_ptr<Context> c; }; std::vector<Part> parts;
  int nsplit=1;
  for(int pi=0;pi<nsplit;pi++){
   std::string pp=path;
   if(pi){auto pos=pp.rfind("-00001-of-");require(pos!=std::string::npos,"split model must start at shard 1");char id[16];snprintf(id,sizeof(id),"%05d",pi+1);pp.replace(pos+1,5,id);}
   Part p{pp,std::make_unique<Gguf>(),std::make_unique<Context>()};
   p.g->p=gguf_init_from_file(pp.c_str(),{true,&p.c->p});require(p.g->p,"GGUF metadata read failed");
   auto u=[&](const char*k,int def){int64_t i=gguf_find_key(p.g->p,k);return i<0?def:(int)gguf_get_val_u32(p.g->p,i);};
   auto f=[&](const char*k,float def){int64_t i=gguf_find_key(p.g->p,k);return i<0?def:gguf_get_val_f32(p.g->p,i);};
   if(pi==0){
    auto ai=gguf_find_key(p.g->p,"general.architecture");require(ai>=0&&std::string(gguf_get_val_str(p.g->p,ai))=="qwen2","only qwen2 architecture supported");
    auto si=gguf_find_key(p.g->p,"split.count");if(si>=0)nsplit=gguf_get_val_u16(p.g->p,si);
    dim_=u("qwen2.embedding_length",0);heads_=u("qwen2.attention.head_count",0);kvheads_=u("qwen2.attention.head_count_kv",0);layers_=u("qwen2.block_count",0);
    require(dim_>0&&heads_>0&&kvheads_>0&&layers_>0&&dim_%heads_==0,"invalid dimensions");head_=dim_/heads_;
    eps_=f("qwen2.attention.layer_norm_rms_epsilon",1e-6f);theta_=f("qwen2.rope.freq_base",1000000);
    eos_=u("tokenizer.ggml.eos_token_id",151645);
   }
   for(auto *t=ggml_get_first_tensor(p.c->p);t;t=ggml_get_next_tensor(p.c->p,t)){
    require(!tensors_.count(t->name),"duplicate tensor in shards");auto *dst=ggml_new_tensor(weights_.p,t->type,4,t->ne);ggml_set_name(dst,t->name);tensors_[t->name]=dst;
   }
   parts.push_back(std::move(p));
  }
  require(nsplit>0&&nsplit<=99999,"invalid split count");
  wbuf_.p=ggml_backend_alloc_ctx_tensors(weights_.p,backend_.get());require(wbuf_.p,"Vulkan weight allocation failed");ggml_backend_buffer_set_usage(wbuf_.p,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  for(auto &p:parts){Mapping m(p.path);for(int64_t i=0;i<gguf_get_n_tensors(p.g->p);i++){
   auto *dst=w(gguf_get_tensor_name(p.g->p,i));size_t off=gguf_get_data_offset(p.g->p)+gguf_get_tensor_offset(p.g->p,i),sz=ggml_nbytes(dst);require(off<=m.size&&sz<=m.size-off,"tensor exceeds model shard");ggml_backend_tensor_set(dst,(char*)m.p+off,0,sz);
  }}
  vocab_=w("token_embd.weight")->ne[1];
  cache_.p=ggml_init({2*1024*1024,nullptr,true});require(cache_.p,"cache metadata allocation failed");
  for(int l=0;l<layers_;l++){
   keys_.push_back(ggml_new_tensor_3d(cache_.p,GGML_TYPE_F16,head_,kvheads_,ctx_));
   values_.push_back(flash_ ? ggml_new_tensor_3d(cache_.p,GGML_TYPE_F16,head_,kvheads_,ctx_) : ggml_new_tensor_3d(cache_.p,GGML_TYPE_F16,ctx_,head_,kvheads_));
  }
  cbuf_.p=ggml_backend_alloc_ctx_tensors(cache_.p,backend_.get());require(cbuf_.p,"KV allocation failed");
  decode_alloc_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_.get())));
  require(decode_alloc_!=nullptr,"decode allocator allocation failed");
  prefill_alloc_.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_.get())));
  require(prefill_alloc_!=nullptr,"prefill allocator allocation failed");
  std::fprintf(stderr,"[lucebox-vulkan] loaded %zu tensors, %d layers, %d vocab; weights=%zu KV=%zu bytes\n",tensors_.size(),layers_,vocab_,ggml_backend_buffer_get_size(wbuf_.p),ggml_backend_buffer_get_size(cbuf_.p));
 }
 ~Qwen2Vulkan() override { shutdown(); }
 void forward(const int32_t *ids,int n,int past,std::vector<float> &result){
  require(n>0&&past+n<=ctx_,"context overflow");
  const int cache_len = n==1 ? std::min(ctx_,((past+n+63)/64)*64) : past+n;
  StepGraph prefill_graph;
  if(n==1 && (!decode_graph_ || decode_graph_->cache_len!=cache_len)) decode_graph_=std::make_unique<StepGraph>();
  StepGraph &step=n==1?*decode_graph_:prefill_graph;
  Context &c=step.context;
  auto alloc=n==1?decode_alloc_.get():prefill_alloc_.get();
  if(!step.graph) {
  step.cache_len=cache_len;
  c.p=ggml_init({32*1024*1024,nullptr,true});require(c.p,"graph metadata allocation failed");
  auto *g=ggml_new_graph_custom(c.p,8192,false);
  auto *tokens=ggml_new_tensor_1d(c.p,GGML_TYPE_I32,n);ggml_set_input(tokens);
  auto *pos=ggml_new_tensor_1d(c.p,GGML_TYPE_I32,n);ggml_set_input(pos);
  auto *mask=ggml_new_tensor_2d(c.p,GGML_TYPE_F32,cache_len,n);ggml_set_input(mask);
  auto *x=ggml_get_rows(c.p,w("token_embd.weight"),tokens);
  for(int l=0;l<layers_;l++){
   auto name=[&](const char*s){return "blk."+std::to_string(l)+"."+s;};
   auto *h=norm(c.p,x,w(name("attn_norm.weight")));
   auto *q=ggml_add(c.p,ggml_mul_mat(c.p,w(name("attn_q.weight")),h),w(name("attn_q.bias")));
   auto *k=ggml_add(c.p,ggml_mul_mat(c.p,w(name("attn_k.weight")),h),w(name("attn_k.bias")));
   auto *v=ggml_add(c.p,ggml_mul_mat(c.p,w(name("attn_v.weight")),h),w(name("attn_v.bias")));
   q=ggml_reshape_3d(c.p,q,head_,heads_,n);k=ggml_reshape_3d(c.p,k,head_,kvheads_,n);v=ggml_reshape_3d(c.p,v,head_,kvheads_,n);
   q=ggml_rope_ext(c.p,q,pos,nullptr,head_,GGML_ROPE_TYPE_NEOX,ctx_,theta_,1,0,1,32,1);
   k=ggml_rope_ext(c.p,k,pos,nullptr,head_,GGML_ROPE_TYPE_NEOX,ctx_,theta_,1,0,1,32,1);
   auto view=[&](ggml_tensor*t,int len,int off){return ggml_view_3d(c.p,t,head_,kvheads_,len,t->nb[1],t->nb[2],off*t->nb[2]);};
   auto *kd= view(keys_[l],n,past);step.writes.emplace_back(kd,keys_[l]->nb[2]);
   auto *kc=ggml_cpy(c.p,k,kd);step.writes.emplace_back(kc,keys_[l]->nb[2]);ggml_build_forward_expand(g,kc);
   if(flash_) {auto *vd=view(values_[l],n,past);step.writes.emplace_back(vd,values_[l]->nb[2]);auto *vcp=ggml_cpy(c.p,v,vd);step.writes.emplace_back(vcp,values_[l]->nb[2]);ggml_build_forward_expand(g,vcp);}
   else { auto *vc=values_[l];auto *dst=ggml_view_3d(c.p,vc,n,head_,kvheads_,vc->nb[1],vc->nb[2],past*vc->nb[0]);
     step.writes.emplace_back(dst,vc->nb[0]);auto *vcp=ggml_cpy(c.p,ggml_permute(c.p,v,1,2,0,3),dst);step.writes.emplace_back(vcp,vc->nb[0]);ggml_build_forward_expand(g,vcp); }
   auto *K=ggml_permute(c.p,view(keys_[l],cache_len,0),0,2,1,3);
   auto *V=flash_ ? ggml_permute(c.p,view(values_[l],cache_len,0),0,2,1,3) : ggml_view_3d(c.p,values_[l],cache_len,head_,kvheads_,values_[l]->nb[1],values_[l]->nb[2],0);
   auto *Q=ggml_permute(c.p,q,0,2,1,3);
   ggml_tensor *a;
   if(flash_){a=ggml_flash_attn_ext(c.p,Q,K,V,ggml_cast(c.p,mask,GGML_TYPE_F16),1/std::sqrt(float(head_)),0,0);ggml_flash_attn_ext_set_prec(a,GGML_PREC_F32);}
   else {
    auto *scores=ggml_mul_mat(c.p,K,Q);ggml_mul_mat_set_prec(scores,GGML_PREC_F32);
    auto *p=ggml_soft_max_ext(c.p,scores,mask,1/std::sqrt(float(head_)),0);
    auto *vt=V;
    a=ggml_mul_mat(c.p,vt,p);a=ggml_cont(c.p,ggml_permute(c.p,a,0,2,1,3));
   }
   a=ggml_reshape_2d(c.p,a,dim_,n);
   x=ggml_add(c.p,x,ggml_mul_mat(c.p,w(name("attn_output.weight")),a));
   auto *f=norm(c.p,x,w(name("ffn_norm.weight")));
   auto *gate=ggml_mul_mat(c.p,w(name("ffn_gate.weight")),f);
   auto *up=ggml_mul_mat(c.p,w(name("ffn_up.weight")),f);
   x=ggml_add(c.p,x,ggml_mul_mat(c.p,w(name("ffn_down.weight")),ggml_swiglu_split(c.p,gate,up)));
  }
  x=ggml_view_2d(c.p,x,dim_,1,x->nb[1],(n-1)*x->nb[1]);
  auto *logits=ggml_mul_mat(c.p,w("output.weight"),norm(c.p,x,w("output_norm.weight")));
  ggml_set_output(logits);ggml_build_forward_expand(g,logits);
  for(int i=0;i<ggml_graph_n_nodes(g);i++) { auto *node=ggml_graph_node(g,i); require(ggml_backend_supports_op(backend_.get(),node),("unsupported Vulkan op "+std::string(ggml_op_name(node->op))).c_str()); }
  require(ggml_gallocr_alloc_graph(alloc,g),"graph device allocation failed");
  step.graph=g;step.tokens=tokens;step.positions=pos;step.mask=mask;step.logits=logits;
  }
  for(auto [view,stride]:step.writes) {
   view->view_offs=past*stride;
   view->data=static_cast<char*>(view->view_src->data)+view->view_offs;
  }
  auto *g=step.graph;auto *tokens=step.tokens;auto *pos=step.positions;auto *mask=step.mask;auto *logits=step.logits;
  ggml_backend_tensor_set(tokens,ids,0,n*sizeof(int32_t));
  std::vector<int32_t> positions(n);for(int i=0;i<n;i++)positions[i]=past+i;
  ggml_backend_tensor_set(pos,positions.data(),0,n*sizeof(int32_t));
  std::vector<float> masks(cache_len*n);for(int j=0;j<n;j++)for(int i=0;i<cache_len;i++)masks[j*cache_len+i]=i>past+j?-std::numeric_limits<float>::infinity():0;
  ggml_backend_tensor_set(mask,masks.data(),0,masks.size()*sizeof(float));
  auto status=ggml_backend_graph_compute(backend_.get(),g);
  result.resize(vocab_);if(status==GGML_STATUS_SUCCESS)ggml_backend_tensor_get(logits,result.data(),0,result.size()*sizeof(float));
  require(status==GGML_STATUS_SUCCESS,"Vulkan compute failed");
 }
 GenerateResult generate_impl(const GenerateRequest &req,const DaemonIO &io) override {
  GenerateResult r;if(!backend_){r.fail(GenerateErrorCode::ModelParked);return r;}
  if(req.prompt.empty()||req.n_gen<0||req.prompt.size()+req.n_gen>size_t(ctx_)){r.fail(GenerateErrorCode::ContextOverflow);return r;}
  try {
   ggml_backend_buffer_clear(cbuf_.p,0);auto t=Clock::now();int past=0;std::vector<float> logits;
   while(past<int(req.prompt.size())){if(io.is_cancelled()){r.fail(GenerateErrorCode::Cancelled);return r;}int n=std::min(chunk_,int(req.prompt.size())-past);forward(req.prompt.data()+past,n,past,logits);past+=n;}
   r.prefill_s=seconds(t);t=Clock::now();std::mt19937_64 rng(req.sampler.seed);auto hist=req.prompt;
   for(int i=0;i<req.n_gen;i++){
    if(io.is_cancelled()){r.fail(GenerateErrorCode::Cancelled);return r;}
    int32_t tok;
    if(req.sampler.needs_logit_processing()) {
     for(float v:logits)require(std::isfinite(v),"nonfinite logits");
     tok=sample_logits(logits.data(),vocab_,req.sampler,hist,rng);
    } else tok=finite_argmax(logits.data(),vocab_);
    if(tok==eos_||tok==eot_||tok==151643)break;
    r.tokens.push_back(tok);hist.push_back(tok);io.emit(tok);
    if(req.on_token&&!req.on_token(tok))break;
    if(i+1<req.n_gen)forward(&tok,1,past++,logits);
   }
   r.decode_s=seconds(t);r.succeed();io.emit(-1);
  }catch(const std::exception&e){r.fail(GenerateErrorCode::BackendSpecific,e.what());}
  return r;
 }
 void print_ready_banner() const override {std::fprintf(stderr,"[qwen2-vulkan] ready\n");}
 bool park(ParkTarget)override{return false;}bool unpark(ParkTarget)override{return backend_!=nullptr;}bool is_target_parked()const override{return !backend_;}
 bool snapshot_save(int)override{return false;}void snapshot_free(int)override{}bool snapshot_used(int)const override{return false;}int snapshot_cur_pos(int)const override{return 0;}
 GenerateResult restore_and_generate_impl(int,const GenerateRequest&,const DaemonIO&)override{GenerateResult r;r.fail(GenerateErrorCode::InvalidSnapshotSlot,"Qwen2 Vulkan snapshot cache unsupported");return r;}
 bool handle_compress(const std::string&,const DaemonIO&)override{return false;}void free_drafter()override{}
 void shutdown()override{
  if(backend_){
   ggml_backend_synchronize(backend_.get());
   decode_graph_.reset();
   decode_alloc_.reset();prefill_alloc_.reset();
   if(cbuf_.p){ggml_backend_buffer_free(cbuf_.p);cbuf_.p=nullptr;}
   if(wbuf_.p){ggml_backend_buffer_free(wbuf_.p);wbuf_.p=nullptr;}
   backend_.reset();
  }
 }
};
}
std::unique_ptr<ModelBackend> make_qwen2_vulkan(const std::string &p,int c,int chunk,bool flash){return std::make_unique<Qwen2Vulkan>(p,c,chunk,flash);}
}
