// Native integration adapted from ggml/llama.cpp graph semantics.
// See PROVENANCE.md and LICENSE.ggml for source attribution and MIT notices.
#include "lfm2_backend.h"
#include "greedy.h"
#include "common/moe_router_graph.h"
#include "prefill_partition.h"
#include <nlohmann/json.hpp>
#include <numeric>
#include "server/tokenizer.h"
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
class LFM2Vulkan final : public ModelBackend {
 ggml_backend_t backend_=nullptr;
 ggml_gallocr_t decode_alloc_=nullptr, prefill_alloc_=nullptr;
 struct StepGraph {
  Context context; ggml_cgraph *graph=nullptr;
  ggml_tensor *tokens=nullptr,*positions=nullptr,*mask=nullptr,*logits=nullptr,*ki=nullptr,*vi=nullptr;
  int cache_len=0;
  std::vector<ggml_tensor*> captures;
  std::vector<std::pair<ggml_tensor*,size_t>> writes;
 };
 std::unique_ptr<StepGraph> decode_graph_;
 Context weights_,cache_;
 Buffer wbuf_,cbuf_;
 std::map<std::string,ggml_tensor*> tensors_;
 std::vector<ggml_tensor*> keys_,values_,conv_;
 std::vector<int> kv_;
 uint64_t executions_=0,requests_=0;
 Tokenizer trace_tokenizer_;
 int dim_=0,heads_=0,kvheads_=0,layers_=0,head_=0,vocab_=0;
 int ctx_,chunk_,eos_=124900;
 float eps_=1e-6f,theta_=1000000;
 bool flash_;
 ggml_tensor *w(const std::string &s) const { auto i=tensors_.find(s);require(i!=tensors_.end(),("missing weight "+s).c_str());return i->second; }
 ggml_tensor *norm(ggml_context *c,ggml_tensor *x,ggml_tensor *scale){return ggml_mul(c,ggml_rms_norm(c,x,eps_),scale);}
public:
 LFM2Vulkan(const std::string &path,int context,int chunk,bool flash):ctx_(context),chunk_(chunk),flash_(flash){
  require(trace_tokenizer_.load_from_gguf(path.c_str()),"tokenizer failed");
  require(!flash_,"LFM Vulkan flash path not validated; use --flash 0");
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
    auto ai=gguf_find_key(p.g->p,"general.architecture");require(ai>=0&&std::string(gguf_get_val_str(p.g->p,ai))=="lfm2moe","only lfm2moe architecture supported");
    require(u("split.count",1)==1,"split GGUF unsupported");
    dim_=u("lfm2moe.embedding_length",0);heads_=u("lfm2moe.attention.head_count",0);layers_=u("lfm2moe.block_count",0);
    require(dim_==2048&&heads_==32&&layers_==24&&u("lfm2moe.vocab_size",0)==128000&&u("lfm2moe.feed_forward_length",0)==7168&&u("lfm2moe.expert_count",0)==32&&u("lfm2moe.expert_used_count",0)==4&&u("lfm2moe.expert_feed_forward_length",0)==1792&&u("lfm2moe.leading_dense_block_count",0)==2&&u("lfm2moe.shortconv.l_cache",0)==3,"unsupported LFM geometry");
    head_=64;
    int ki=gguf_find_key(p.g->p,"lfm2moe.attention.head_count_kv");
    require(ki>=0&&gguf_get_kv_type(p.g->p,ki)==GGUF_TYPE_ARRAY&&(gguf_get_arr_type(p.g->p,ki)==GGUF_TYPE_UINT32||gguf_get_arr_type(p.g->p,ki)==GGUF_TYPE_INT32)&&gguf_get_arr_n(p.g->p,ki)==24,"unsupported KV geometry");
    auto *kh=(const uint32_t*)gguf_get_arr_data(p.g->p,ki);for(int i=0;i<24;i++){require(kh[i]==0||kh[i]==8,"unsupported KV head count");kv_.push_back(kh[i]);}
    eps_=f("lfm2moe.attention.layer_norm_rms_epsilon",1e-5f);theta_=f("lfm2moe.rope.freq_base",5000000);
    require(eps_>0&&std::isfinite(eps_)&&theta_>0&&std::isfinite(theta_),"invalid normalization/RoPE parameters");
    require(u("lfm2moe.attention.sliding_window",0)==0,"sliding window unsupported");
    eos_=u("tokenizer.ggml.eos_token_id",124900);
   }
   for(auto *t=ggml_get_first_tensor(p.c->p);t;t=ggml_get_next_tensor(p.c->p,t)){
    require(!tensors_.count(t->name),"duplicate tensor in shards");auto *dst=ggml_new_tensor(weights_.p,t->type,4,t->ne);ggml_set_name(dst,t->name);tensors_[t->name]=dst;
   }
   parts.push_back(std::move(p));
  }
  require(nsplit>0&&nsplit<=99999,"invalid split count");
  // Validate every tensor geometry before any GPU allocation or dispatch.
  auto shape=[&](const std::string&name,std::initializer_list<int64_t> dims){auto*t=w(name);int i=0;for(auto d:dims)require(t->ne[i++]==d,("unsupported tensor geometry: "+name).c_str());for(;i<4;i++)require(t->ne[i]==1,"unsupported tensor rank");};
  shape("token_embd.weight",{2048,128000});shape("token_embd_norm.weight",{2048});
  for(int l=0;l<24;l++){
   auto n=[&](const char*x){return "blk."+std::to_string(l)+"."+x;};
   shape(n("attn_norm.weight"),{2048});shape(n("ffn_norm.weight"),{2048});
   if(kv_[l]){shape(n("attn_q.weight"),{2048,2048});shape(n("attn_k.weight"),{2048,512});shape(n("attn_v.weight"),{2048,512});shape(n("attn_output.weight"),{2048,2048});shape(n("attn_q_norm.weight"),{64});shape(n("attn_k_norm.weight"),{64});}
   else{shape(n("shortconv.conv.weight"),{3,2048});shape(n("shortconv.in_proj.weight"),{2048,6144});shape(n("shortconv.out_proj.weight"),{2048,2048});}
   if(l<2){shape(n("ffn_gate.weight"),{2048,7168});shape(n("ffn_up.weight"),{2048,7168});shape(n("ffn_down.weight"),{7168,2048});}
   else{shape(n("ffn_gate_inp.weight"),{2048,32});shape(n("exp_probs_b.bias"),{32});shape(n("ffn_gate_exps.weight"),{2048,1792,32});shape(n("ffn_up_exps.weight"),{2048,1792,32});shape(n("ffn_down_exps.weight"),{1792,2048,32});}
  }
  backend_=ggml_backend_vk_init(0);require(backend_,"Vulkan device initialization failed");
  std::fprintf(stderr,"[native_lfm2_vulkan] %s full GPU weights; context=%d chunk=%d\n",ggml_backend_name(backend_),ctx_,chunk_);
  wbuf_.p=ggml_backend_alloc_ctx_tensors(weights_.p,backend_);require(wbuf_.p,"Vulkan weight allocation failed");ggml_backend_buffer_set_usage(wbuf_.p,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
  for(auto &p:parts){Mapping m(p.path);for(int64_t i=0;i<gguf_get_n_tensors(p.g->p);i++){
   auto *dst=w(gguf_get_tensor_name(p.g->p,i));size_t off=gguf_get_data_offset(p.g->p)+gguf_get_tensor_offset(p.g->p,i),sz=ggml_nbytes(dst);require(off<=m.size&&sz<=m.size-off,"tensor exceeds model shard");ggml_backend_tensor_set(dst,(char*)m.p+off,0,sz);
  }}
  vocab_=w("token_embd.weight")->ne[1];
  cache_.p=ggml_init({2*1024*1024,nullptr,true});require(cache_.p,"cache metadata allocation failed");
  for(int l=0;l<layers_;l++){
   kvheads_=kv_[l];
   conv_.push_back(kvheads_ ? nullptr : ggml_new_tensor_3d(cache_.p,GGML_TYPE_F32,2,dim_,1));
   if(!kvheads_){keys_.push_back(nullptr);values_.push_back(nullptr);continue;}
   keys_.push_back(ggml_new_tensor_3d(cache_.p,GGML_TYPE_F16,head_,kvheads_,ctx_));
   values_.push_back(flash_ ? ggml_new_tensor_3d(cache_.p,GGML_TYPE_F16,head_,kvheads_,ctx_) : ggml_new_tensor_3d(cache_.p,GGML_TYPE_F16,ctx_,head_,kvheads_));
  }
  cbuf_.p=ggml_backend_alloc_ctx_tensors(cache_.p,backend_);require(cbuf_.p,"KV allocation failed");
  decode_alloc_=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
  prefill_alloc_=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
  std::fprintf(stderr,"[lucebox-vulkan] loaded %zu tensors, %d layers, %d vocab; weights=%zu KV=%zu bytes\n",tensors_.size(),layers_,vocab_,ggml_backend_buffer_get_size(wbuf_.p),ggml_backend_buffer_get_size(cbuf_.p));
 }
 ~LFM2Vulkan() override { shutdown(); }
 void forward(const int32_t *ids,int n,int past,std::vector<float> &result){
  require(n>0&&past+n<=ctx_,"context overflow");
  const int cache_len = std::min(ctx_,((past+n+255)/256)*256);
  StepGraph prefill_graph;
  if(n==1 && (!decode_graph_ || decode_graph_->cache_len!=cache_len)) decode_graph_=std::make_unique<StepGraph>();
  StepGraph &step=n==1?*decode_graph_:prefill_graph;
  Context &c=step.context;
  auto alloc=n==1?decode_alloc_:prefill_alloc_;
  if(!step.graph) {
  step.cache_len=cache_len;
  c.p=ggml_init({32*1024*1024,nullptr,true});require(c.p,"graph metadata allocation failed");
  auto *g=ggml_new_graph_custom(c.p,8192,false);
  auto *tokens=ggml_new_tensor_1d(c.p,GGML_TYPE_I32,n);ggml_set_input(tokens);
  auto *pos=ggml_new_tensor_1d(c.p,GGML_TYPE_I32,n);ggml_set_input(pos);
  auto *mask=ggml_new_tensor_2d(c.p,GGML_TYPE_F32,cache_len,n);ggml_set_input(mask);
  auto *ki=ggml_new_tensor_1d(c.p,GGML_TYPE_I64,n);auto *vi=ggml_new_tensor_1d(c.p,GGML_TYPE_I64,n*512);ggml_set_input(ki);ggml_set_input(vi);step.ki=ki;step.vi=vi;
  auto *x=ggml_get_rows(c.p,w("token_embd.weight"),tokens);
  for(int l=0;l<layers_;l++){
   auto name=[&](const char*s){return "blk."+std::to_string(l)+"."+s;};
   auto *h=norm(c.p,x,w(name("attn_norm.weight")));
   if(kv_[l]) {
   int kvheads_=kv_[l];
   auto *q=ggml_mul_mat(c.p,w(name("attn_q.weight")),h);
   auto *k=ggml_mul_mat(c.p,w(name("attn_k.weight")),h);
   auto *v=ggml_mul_mat(c.p,w(name("attn_v.weight")),h);
   q=ggml_reshape_3d(c.p,q,head_,heads_,n);k=ggml_reshape_3d(c.p,k,head_,kvheads_,n);v=ggml_reshape_3d(c.p,v,head_,kvheads_,n);
   q=norm(c.p,q,w(name("attn_q_norm.weight")));k=norm(c.p,k,w(name("attn_k_norm.weight")));
   q=ggml_rope_ext(c.p,q,pos,nullptr,head_,GGML_ROPE_TYPE_NEOX,ctx_,theta_,1,0,1,32,1);
   k=ggml_rope_ext(c.p,k,pos,nullptr,head_,GGML_ROPE_TYPE_NEOX,ctx_,theta_,1,0,1,32,1);
   auto view=[&](ggml_tensor*t,int len,int off){return ggml_view_3d(c.p,t,head_,kvheads_,len,t->nb[1],t->nb[2],off*t->nb[2]);};
   auto *kc=ggml_set_rows(c.p,ggml_reshape_2d(c.p,keys_[l],512,ctx_),ggml_reshape_2d(c.p,k,512,n),ki);ggml_build_forward_expand(g,kc);
   auto *vc=ggml_set_rows(c.p,ggml_reshape_2d(c.p,values_[l],1,ctx_*512),ggml_reshape_2d(c.p,v,1,n*512),vi);ggml_build_forward_expand(g,vc);
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
   } else {
    auto *bcx=ggml_mul_mat(c.p,w(name("shortconv.in_proj.weight")),ggml_reshape_3d(c.p,h,dim_,n,1));
    auto *b=ggml_view_3d(c.p,bcx,dim_,n,1,bcx->nb[1],bcx->nb[2],0);
    auto *cc=ggml_view_3d(c.p,bcx,dim_,n,1,bcx->nb[1],bcx->nb[2],dim_*bcx->nb[0]);
    auto *xx=ggml_view_3d(c.p,bcx,dim_,n,1,bcx->nb[1],bcx->nb[2],2*dim_*bcx->nb[0]);
    auto *bx=ggml_transpose(c.p,ggml_mul(c.p,b,xx));
    auto *ci=ggml_concat(c.p,conv_[l],bx,0);
    auto *ns=ggml_view_3d(c.p,ci,2,dim_,1,ci->nb[1],ci->nb[2],n*ci->nb[0]);
    ggml_build_forward_expand(g,ggml_cpy(c.p,ns,conv_[l]));
    auto *co=ggml_ssm_conv(c.p,ci,w(name("shortconv.conv.weight")));
    // Preserve the projection -> reshape -> residual boundary. Without this
    // view, Vulkan fuses projection+add and changes the following RMS reduction.
    auto *gated=ggml_mul(c.p,cc,co);
    auto *projected=ggml_mul_mat(c.p,w(name("shortconv.out_proj.weight")),gated);
    x=ggml_add(c.p,x,ggml_reshape_2d(c.p,projected,dim_,n));
   }
   const int nf=l==layers_-1?1:n;
   if(nf!=n)x=ggml_view_2d(c.p,x,dim_,1,x->nb[1],(n-1)*x->nb[1]);
   auto *f=norm(c.p,x,w(name("ffn_norm.weight")));
   if(l<2){
   auto *gate=ggml_mul_mat(c.p,w(name("ffn_gate.weight")),f);
   auto *up=ggml_mul_mat(c.p,w(name("ffn_up.weight")),f);
   x=ggml_add(c.p,x,ggml_mul_mat(c.p,w(name("ffn_down.weight")),ggml_swiglu_split(c.p,gate,up)));
   }else{
    auto *router_logits=ggml_mul_mat(c.p,w(name("ffn_gate_inp.weight")),f);
    auto router=build_sigmoid_topk_moe_router(c.p,g,router_logits,w(name("exp_probs_b.bias")),32,4,nf,true,1.0f,true);
    auto *cur=ggml_reshape_3d(c.p,f,dim_,1,nf);
    auto *gate=ggml_mul_mat_id(c.p,w(name("ffn_gate_exps.weight")),cur,router.selected);
    auto *up=ggml_mul_mat_id(c.p,w(name("ffn_up_exps.weight")),cur,router.selected);
    auto *ex=ggml_mul_mat_id(c.p,w(name("ffn_down_exps.weight")),ggml_swiglu_split(c.p,gate,up),router.selected);
    ex=ggml_mul(c.p,ex,router.weights_3d);
    ggml_build_forward_expand(g,ex);ggml_tensor *rows[4];for(int i=0;i<4;i++){rows[i]=ggml_view_2d(c.p,ex,dim_,nf,ex->nb[2],i*ex->nb[1]);ggml_build_forward_expand(g,rows[i]);}
    auto *sum=rows[0];for(int i=1;i<4;i++){sum=ggml_add(c.p,sum,rows[i]);ggml_build_forward_expand(g,sum);}
    x=ggml_add(c.p,x,sum);
   }
   if(std::getenv("LUCEBOX_LAYER_DUMP")&&n>1){ggml_set_output(x);ggml_build_forward_expand(g,x);step.captures.push_back(x);}
  }
  // Last-layer FFN already selects the final position; no view fence here.
  auto *logits=ggml_mul_mat(c.p,w("token_embd.weight"),norm(c.p,x,w("token_embd_norm.weight")));
  ggml_set_output(logits);ggml_build_forward_expand(g,logits);
  for(int i=0;i<ggml_graph_n_nodes(g);i++) { auto *node=ggml_graph_node(g,i); require(ggml_backend_supports_op(backend_,node),("unsupported Vulkan op "+std::string(ggml_op_name(node->op))).c_str()); }
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
  std::vector<int64_t>ki_host(n),vi_host(n*512);for(int t=0;t<n;t++){ki_host[t]=past+t;for(int j=0;j<512;j++)vi_host[t*512+j]=int64_t(j)*ctx_+past+t;}ggml_backend_tensor_set(step.ki,ki_host.data(),0,n*8);ggml_backend_tensor_set(step.vi,vi_host.data(),0,vi_host.size()*8);
  std::vector<float> masks(cache_len*n);for(int j=0;j<n;j++)for(int i=0;i<cache_len;i++)masks[j*cache_len+i]=i>past+j?-std::numeric_limits<float>::infinity():0;
  ggml_backend_tensor_set(mask,masks.data(),0,masks.size()*sizeof(float));
  auto status=ggml_backend_graph_compute(backend_,g);++executions_;
  result.resize(vocab_);if(status==GGML_STATUS_SUCCESS)ggml_backend_tensor_get(logits,result.data(),0,result.size()*sizeof(float));
  require(status==GGML_STATUS_SUCCESS,"Vulkan compute failed");
  if(auto *dir=std::getenv("LUCEBOX_LAYER_DUMP")){for(size_t i=0;i<step.captures.size();i++){auto*t=step.captures[i];std::vector<float>v(t->ne[0]);ggml_backend_tensor_get(t,v.data(),(t->ne[1]-1)*t->nb[1],v.size()*4);std::ofstream out(std::string(dir)+"/l_out-"+std::to_string(i)+".bin",std::ios::binary);out.write((char*)v.data(),v.size()*4);}}
 }
 GenerateResult generate_impl(const GenerateRequest &req,const DaemonIO &io) override {
  GenerateResult r;if(!backend_){r.fail(GenerateErrorCode::ModelParked);return r;}
  if(req.prompt.empty()||req.n_gen<0||req.prompt.size()+req.n_gen>size_t(ctx_)){r.fail(GenerateErrorCode::ContextOverflow);return r;}
  try {
   for(auto id:req.prompt)require(id>=0&&id<vocab_,"invalid token id");
   ++requests_;nlohmann::json trace={{"request",requests_},{"prompt_tokens",req.prompt},{"backend","native_lfm2_vulkan"},{"steps",nlohmann::json::array()}};
   ggml_backend_buffer_clear(cbuf_.p,0);auto t=Clock::now();int past=0;std::vector<float> logits;
   while(past<int(req.prompt.size())){if(io.is_cancelled()){r.fail(GenerateErrorCode::Cancelled);return r;}int remaining=int(req.prompt.size())-past;
    int n=lfm_prefill_partition(remaining,chunk_);forward(req.prompt.data()+past,n,past,logits);past+=n;}
   r.prefill_s=seconds(t);t=Clock::now();std::mt19937_64 rng(req.sampler.seed);auto hist=req.prompt;
   for(int i=0;i<req.n_gen;i++){
    if(io.is_cancelled()){r.fail(GenerateErrorCode::Cancelled);return r;}
    int32_t tok;
    if(req.sampler.needs_logit_processing()) {
     for(float v:logits)require(std::isfinite(v),"nonfinite logits");
     tok=sample_logits(logits.data(),vocab_,req.sampler,hist,rng);
    } else tok=finite_argmax(logits.data(),vocab_);
    if(std::getenv("LUCEBOX_TRACE")){
     std::vector<int> order(vocab_);std::iota(order.begin(),order.end(),0);std::partial_sort(order.begin(),order.begin()+5,order.end(),[&](int a,int b){return logits[a]>logits[b];});
     double z=0,m=logits[order[0]];for(auto v:logits){require(std::isfinite(v),"nonfinite logits");z+=std::exp(double(v)-m);}double lz=m+std::log(z);
     nlohmann::json top=nlohmann::json::array();for(int j=0;j<5;j++)top.push_back({{"token_id",order[j]},{"logprob",double(logits[order[j]])-lz},{"logit",logits[order[j]]}});
     trace["steps"].push_back({{"position",past},{"token_id",tok},{"top",top}});
    }
    if(tok==eos_)break;
    r.tokens.push_back(tok);hist.push_back(tok);io.emit(tok);
    if(req.on_token&&!req.on_token(tok))break;
    if(i+1<req.n_gen)forward(&tok,1,past++,logits);
   }
   r.decode_s=seconds(t);r.succeed();io.emit(-1);
   trace["output"]=trace_tokenizer_.decode(r.tokens);trace["output_tokens"]=r.tokens;trace["prefill_seconds"]=r.prefill_s;trace["decode_seconds"]=r.decode_s;trace["backend_execution_count"]=executions_;trace["weight_bytes"]=ggml_backend_buffer_get_size(wbuf_.p);trace["weight_buffer"]=ggml_backend_buffer_name(wbuf_.p);trace["weights_gpu_resident"]=!ggml_backend_buft_is_host(ggml_backend_buffer_get_type(wbuf_.p));
   if(auto *path=std::getenv("LUCEBOX_TRACE")){std::ofstream out(path,std::ios::app);out<<trace.dump()<<"\n";out.flush();require(bool(out),"trace write failed");}
  }catch(const std::exception&e){r.fail(GenerateErrorCode::BackendSpecific,e.what());}
  return r;
 }
 void print_ready_banner() const override {std::fprintf(stderr,"[lfm2-vulkan] ready\n");}
 bool park(ParkTarget)override{return false;}bool unpark(ParkTarget)override{return backend_!=nullptr;}bool is_target_parked()const override{return !backend_;}
 bool snapshot_save(int)override{return false;}void snapshot_free(int)override{}bool snapshot_used(int)const override{return false;}int snapshot_cur_pos(int)const override{return 0;}
 GenerateResult restore_and_generate_impl(int,const GenerateRequest&,const DaemonIO&)override{GenerateResult r;r.fail(GenerateErrorCode::InvalidSnapshotSlot,"LFM2 Vulkan snapshot cache unsupported");return r;}
 bool handle_compress(const std::string&,const DaemonIO&)override{return false;}void free_drafter()override{}
 void shutdown()override{
  if(backend_){ggml_backend_synchronize(backend_);if(decode_alloc_){ggml_gallocr_free(decode_alloc_);decode_alloc_=nullptr;}if(prefill_alloc_){ggml_gallocr_free(prefill_alloc_);prefill_alloc_=nullptr;}if(cbuf_.p){ggml_backend_buffer_free(cbuf_.p);cbuf_.p=nullptr;}if(wbuf_.p){ggml_backend_buffer_free(wbuf_.p);wbuf_.p=nullptr;}ggml_backend_free(backend_);backend_=nullptr;}
 }
};
}
std::unique_ptr<ModelBackend> make_lfm2_vulkan(const std::string &p,int c,int chunk,bool flash){return std::make_unique<LFM2Vulkan>(p,c,chunk,flash);}
}
