#include "deepseek4_vision.h"
#include "common/gguf_bounds.h"
#include "common/gguf_mmap.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace dflash::vision {
namespace {
using Tensor = ggml_tensor;
constexpr size_t MAX_SCRATCH = size_t(2) * 1024 * 1024 * 1024;
Tensor * rounded(ggml_context * c, Tensor * x) {
    return ggml_cast(c, ggml_cast(c, x, GGML_TYPE_BF16), GGML_TYPE_F32);
}
void require(bool value, const std::string & error) { if (!value) throw std::runtime_error(error); }
struct Meta {
    gguf_context * g = nullptr;
    ggml_context * c = nullptr;
    ~Meta() { if (g) gguf_free(g); if (c) ggml_free(c); }
};
std::map<std::string, std::array<int64_t,4>> inventory() {
    std::map<std::string, std::array<int64_t,4>> result;
    auto add = [&](const std::string & n, int64_t a, int64_t b = 1) { result[n] = {a,b,1,1}; };
    add("vision.patch_embed.proj.weight",588,1024); add("vision.patch_embed.proj.bias",1024);
    add("vision.norm.weight",1024);
    add("aligner.w1.weight",9216,4096); add("aligner.w1.bias",4096);
    add("aligner.w2.weight",4096,4096); add("aligner.w2.bias",4096);
    for (auto n : {"image_start","image_pad","image_newline","image_end"}) add(n,4096);
    for (int i=0;i<32;++i) {
        const auto p = "vision.blocks." + std::to_string(i);
        add(p+".norm1.weight",1024); add(p+".norm2.weight",1024);
        add(p+".attn.wqkv.weight",1024,3072); add(p+".attn.wqkv.bias",3072);
        add(p+".attn.wo.weight",1024,1024); add(p+".attn.wo.bias",1024);
        add(p+".mlp.w1.weight",1024,5632); add(p+".mlp.w2.weight",2816,1024);
    }
    return result;
}
void validate_metadata(gguf_context * g, int dimension, int vocabulary) {
    auto key = [&](const std::string & n, gguf_type type) {
        auto id = gguf_find_key(g,n.c_str());
        require(id >= 0 && gguf_get_kv_type(g,id)==type,"missing or wrong metadata type: "+n);
        return id;
    };
    auto str = [&](const std::string & n, const char * value) {
        require(std::string(gguf_get_val_str(g,key(n,GGUF_TYPE_STRING)))==value,"unsupported metadata: "+n);
    };
    auto integer = [&](const std::string & n, uint32_t value) {
        require(gguf_get_val_u32(g,key(n,GGUF_TYPE_UINT32))==value,"unsupported metadata: "+n);
    };
    auto real = [&](const std::string & n,float value) {
        require(gguf_get_val_f32(g,key(n,GGUF_TYPE_FLOAT32))==value,"unsupported metadata: "+n);
    };
    str("general.architecture","deepseek4_vision"); str("general.type","mmproj");
    integer("general.alignment",32);
    const std::string p="deepseek4.vision.";
    for (auto entry : std::map<std::string,uint32_t>{{"schema_version",1},{"block_count",32},
            {"embedding_length",1024},{"attention.head_count",16},{"attention.head_dimension",64},
            {"feed_forward_length",2816},{"patch_size",14},{"downsample_ratio",3},{"aligner_input_length",9216},
            {"language_embedding_length",4096},{"vocabulary_size",129280},{"image.max_tokens",384},
            {"image.min_pixels",147456},{"image.layout_version",1},{"image.compression_alignment",4},
            {"image.sentinel_type_count",5}}) integer(p+entry.first,entry.second);
    require(dimension==4096 && vocabulary==129280,"language model dimension/vocabulary mismatch");
    real(p+"rope.freq_base",10000.f); real(p+"attention.layer_norm_rms_epsilon",1e-6f);
    real(p+"image.max_aspect_ratio",8.f);
    for (auto entry : std::map<std::string,const char *>{{"attention.rope_layout","2d-half-split-height-width"},
            {"image.patch_layout","channel-major"},{"image.layout","n"},
            {"image.layout_recipe","row-pair-column-interleave"},{"image.sentinel_types","start,pad,image,newline,end"},
            {"aligner.padding","bottom-right"},{"aligner.patch_layout","channel-first-unfold"},
            {"aligner.activation","gelu-exact"}}) str(p+entry.first,entry.second);
    for (auto n : {"image.normalization_mean","image.normalization_std"}) {
        auto id = key(p+n,GGUF_TYPE_ARRAY);
        require(gguf_get_arr_type(g,id)==GGUF_TYPE_FLOAT32 && gguf_get_arr_n(g,id)==3,"bad normalization array");
        const auto * values = static_cast<const float *>(gguf_get_arr_data(g,id));
        require(values[0]==.5f && values[1]==.5f && values[2]==.5f,"unsupported normalization");
    }
}
struct Graph {
    ggml_context * c;
    ggml_cgraph * graph;
    std::vector<std::pair<std::string,Tensor *>> stages;
    explicit Graph() {
        const size_t nodes=512;
        c=ggml_init({nodes*ggml_tensor_overhead()+ggml_graph_overhead_custom(nodes,false),nullptr,true});
        require(c!=nullptr,"graph metadata allocation failed");
        graph=ggml_new_graph_custom(c,nodes,false);
    }
    ~Graph() { ggml_free(c); }
    void stage(const std::string & name,Tensor * t,bool enabled) {
        if (enabled) {
            // An output flag on a view does not pin its underlying allocation.
            // A dedicated snapshot prevents later in-place operations or allocator
            // reuse from corrupting diagnostic values (notably reshape/unfold).
            auto snapshot=ggml_dup(c,t);
            ggml_set_output(snapshot); stages.emplace_back(name,snapshot);
        }
    }
};
std::vector<float> read(Tensor * t) {
    std::vector<float> result(ggml_nelements(t));
    ggml_backend_tensor_get(t,result.data(),0,result.size()*sizeof(float));
    return result;
}
}
namespace detail {
static size_t hip_size_query(ggml_backend_t backend,const char * name) {
    if(!backend) return 0;
    auto device=ggml_backend_get_device(backend);
    if(!device) return 0;
    auto reg=ggml_backend_dev_backend_reg(device);
    if(!reg) return 0;
    using Query=size_t (*)(ggml_backend_t);
    auto query=reinterpret_cast<Query>(ggml_backend_reg_get_proc_address(reg,name));
    return query ? query(backend) : 0;
}
size_t hip_bias_workspace(ggml_backend_t b) { return hip_size_query(b,"ggml_backend_hip_vision_bias_bf16_workspace"); }
size_t hip_bias_launches(ggml_backend_t b) { return hip_size_query(b,"ggml_backend_hip_vision_bias_bf16_launches"); }
size_t hip_norm_launches(ggml_backend_t b) { return hip_size_query(b,"ggml_backend_hip_vision_norm_f32_launches"); }
size_t hip_rotary_launches(ggml_backend_t b) { return hip_size_query(b,"ggml_backend_hip_vision_rotary_f32_launches"); }
bool hip_rotary_capable(ggml_backend_t backend) {
    if(!backend) return false;
    auto device=ggml_backend_get_device(backend);
    if(!device) return false;
    auto reg=ggml_backend_dev_backend_reg(device);
    if(!reg) return false;
    using Query=bool (*)(ggml_backend_t);
    auto query=reinterpret_cast<Query>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_hip_vision_rotary_f32_capable"));
    return query && query(backend);
}
bool hip_norm_capable(ggml_backend_t backend) {
    if(!backend) return false;
    auto device=ggml_backend_get_device(backend);
    if(!device) return false;
    auto reg=ggml_backend_dev_backend_reg(device);
    if(!reg) return false;
    using Query=bool (*)(ggml_backend_t);
    auto query=reinterpret_cast<Query>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_hip_vision_norm_f32_capable"));
    return query && query(backend);
}
Tensor * rms_norm(ggml_context * c,Tensor * input,float epsilon,ggml_backend_t backend) {
    if(!hip_bias_workspace(backend)) return ggml_rms_norm(c,input,epsilon);
    require(hip_norm_capable(backend),"HIP source-order vision normalization unavailable; fallback forbidden");
    auto y=ggml_rms_norm_vision_f32(c,input,epsilon);
    require(ggml_backend_supports_op(backend,y),"HIP source-order vision normalization unsupported; fallback forbidden");
    return y;
}
Tensor * linear(ggml_context * c,Tensor * weight,Tensor * input,Tensor * bias,bool preserve_biased_product,ggml_backend_t backend) {
    // HIP's explicit capability is required: generic supports_op defaults on
    // other backends are not evidence of this source-specific operation.
    if(hip_bias_workspace(backend)) {
        auto y=ggml_mul_mat_bias_bf16(c,weight,ggml_cast(c,input,GGML_TYPE_BF16),bias);
        require(ggml_backend_supports_op(backend,y),"HIP BF16 vision linear unsupported; fallback forbidden");
        return ggml_cast(c,y,GGML_TYPE_F32);
    }
    // Source biased linears round after the bias. The HIP BF16 BLAS path can
    // round its product to BF16 even with GGML_PREC_F32; an F32 weight operand
    // preserves the product until the explicit final boundary below.
    // The CPU path already retains an F32 product; keep its existing reduction.
    if(bias && preserve_biased_product) weight=ggml_cast(c,weight,GGML_TYPE_F32);
    auto y=ggml_mul_mat(c,weight,input);
    ggml_mul_mat_set_prec(y,GGML_PREC_F32);
    if(bias) y=ggml_add(c,y,ggml_cast(c,bias,GGML_TYPE_F32));
    return rounded(c,y);
}
void rotary_tables(PatchGrid grid,std::vector<float> & cosine,std::vector<float> & sine,ggml_backend_t backend) {
    require(grid.height>0 && grid.width>0 && grid.height<=1152 && grid.width<=1152,"invalid rotary grid");
    const int n=grid.height*grid.width;
    if(hip_bias_workspace(backend)) {
        const size_t temporary=(32+size_t(n)*32*3)*sizeof(float);
        constexpr size_t limit=128ULL*1024*1024;
        const size_t external=hip_bias_workspace(backend);
        require(external<=limit && temporary<=limit-external,"HIP rotary temporary storage exceeds 128 MiB bound");
        require(hip_rotary_capable(backend),"HIP source-order vision rotary tables unavailable; fallback forbidden");
        using Fill=bool (*)(ggml_backend_t,int64_t,int64_t,float *,float *);
        auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto fill=reinterpret_cast<Fill>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_hip_vision_rotary_f32"));
        require(fill!=nullptr,"HIP source-order vision rotary table function unavailable");
        cosine.resize(size_t(n)*32); sine.resize(size_t(n)*32);
        require(fill(backend,grid.height,grid.width,cosine.data(),sine.data()),"HIP source-order vision rotary table preparation failed");
        return;
    }
    cosine.resize(size_t(n)*32); sine.resize(size_t(n)*32);
    for(int i=0;i<n;++i) for(int j=0;j<32;++j) {
        // Source uses pow then reciprocal in float32, height half before width.
        const float inv=1.f/std::pow(10000.f,float(2*(j%16))/32.f);
        const float angle=float(j<16 ? i/grid.width : i%grid.width)*inv;
        cosine[i*32+j]=std::cos(angle); sine[i*32+j]=std::sin(angle);
    }
}
Tensor * rotate(ggml_context * c,Tensor * x,Tensor * cosine,Tensor * sine) {
    // [64, heads, N], tables [32, 1, N]; pairing is x[j] with x[j+32].
    auto a=ggml_view_3d(c,x,32,x->ne[1],x->ne[2],x->nb[1],x->nb[2],0);
    auto b=ggml_view_3d(c,x,32,x->ne[1],x->ne[2],x->nb[1],x->nb[2],32*sizeof(float));
    auto first=ggml_sub(c,ggml_mul(c,a,cosine),ggml_mul(c,b,sine));
    auto second=ggml_add(c,ggml_mul(c,b,cosine),ggml_mul(c,a,sine));
    return rounded(c,ggml_concat(c,first,second,0));
}
Tensor * attention(ggml_context * c,Tensor * q,Tensor * k,Tensor * v) {
    q=ggml_cont(c,ggml_permute(c,q,0,2,1,3));
    k=ggml_cont(c,ggml_permute(c,k,0,2,1,3));
    // PyTorch 2.10 Math SDPA (the source's 3D input dispatch) scales both
    // operands in F32 before GEMM. Preserve this order and its rounding;
    // scaling the scores afterwards is algebraically equal but not bit equal.
    const float scale=float(std::sqrt(1.0/std::sqrt(double(q->ne[0]))));
    auto scores=ggml_mul_mat(c,ggml_scale(c,k,scale),ggml_scale(c,q,scale));
    ggml_mul_mat_set_prec(scores,GGML_PREC_F32);
    auto probabilities=ggml_soft_max(c,scores);
    v=ggml_cont(c,ggml_permute(c,v,1,2,0,3)); // [N, D, heads]
    auto out=ggml_mul_mat(c,v,probabilities);
    ggml_mul_mat_set_prec(out,GGML_PREC_F32);
    out=ggml_cont(c,ggml_permute(c,out,0,2,1,3));
    return rounded(c,ggml_reshape_2d(c,out,out->ne[0]*out->ne[1],out->ne[2]));
}
Tensor * unfold(ggml_context * c,Tensor * x,PatchGrid grid,int channels) {
    x=ggml_reshape_3d(c,x,channels,grid.width,grid.height);
    x=ggml_cont(c,ggml_permute(c,x,2,0,1,3)); // [width,height,channels]
    x=ggml_pad(c,x,(3-grid.width%3)%3,(3-grid.height%3)%3,0,0);
    auto kernel=ggml_new_tensor_4d(c,GGML_TYPE_F32,3,3,channels,1);
    auto out=ggml_im2col(c,kernel,x,3,3,0,0,1,1,true,GGML_TYPE_F32);
    return ggml_reshape_2d(c,out,channels*9,((grid.width+2)/3)*((grid.height+2)/3));
}
}
struct VisionRuntime::Impl {
    VisionConfig config;
    ggml_backend_t backend=nullptr;
    ggml_context * weights=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    ggml_gallocr_t allocator=nullptr;
    ~Impl() {
        if(allocator) ggml_gallocr_free(allocator);
        if(buffer) ggml_backend_buffer_free(buffer);
        if(weights) ggml_free(weights);
    }
    Tensor * weight(const std::string & name) { return ggml_get_tensor(weights,name.c_str()); }
    Tensor * linear(ggml_context * c,Tensor * x,const std::string & name,bool bias=true) {
        const auto device=ggml_backend_get_device(backend);
        const bool preserve=device && (ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_GPU ||
                                       ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_IGPU);
        return detail::linear(c,weight(name+".weight"),x,bias ? weight(name+".bias") : nullptr,preserve,backend);
    }
    Tensor * norm(ggml_context * c,Tensor * x,const std::string & name) {
        return rounded(c,ggml_mul(c,detail::rms_norm(c,x,config.rms_epsilon,backend),
                                    ggml_cast(c,weight(name+".weight"),GGML_TYPE_F32)));
    }
    std::vector<float> execute(Graph & g,Tensor * input,const std::vector<float> & values,Tensor * output,
                              const StageObserver & observer,Tensor * cosine=nullptr,Tensor * sine=nullptr,
                              const std::vector<float> & cos_values={},const std::vector<float> & sin_values={}) {
        ggml_set_input(input); ggml_set_output(output);
        for(auto & stage:g.stages) ggml_build_forward_expand(g.graph,stage.second);
        ggml_build_forward_expand(g.graph,output);
        if(!allocator) allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(allocator!=nullptr,"scratch allocator creation failed");
        // reserve_n_size calculates requirements without allocating a backend buffer.
        size_t required=0;
        ggml_gallocr_reserve_n_size(allocator,g.graph,nullptr,nullptr,&required);
        const size_t external=detail::hip_bias_workspace(backend);
        require(external<=MAX_SCRATCH && required<=MAX_SCRATCH-external,"vision scratch including retained Lt workspace exceeds 2 GiB bound");
        for(int i=0;i<ggml_graph_n_nodes(g.graph);++i)
            require(ggml_backend_supports_op(backend,ggml_graph_node(g.graph,i)),"backend does not support vision graph operation");
        require(ggml_gallocr_reserve(allocator,g.graph),"scratch reservation failed");
        require(ggml_gallocr_alloc_graph(allocator,g.graph),"scratch allocation failed");
        ggml_backend_tensor_set(input,values.data(),0,values.size()*sizeof(float));
        if(cosine) {
            ggml_backend_tensor_set(cosine,cos_values.data(),0,cos_values.size()*sizeof(float));
            ggml_backend_tensor_set(sine,sin_values.data(),0,sin_values.size()*sizeof(float));
        }
        require(ggml_backend_graph_compute(backend,g.graph)==GGML_STATUS_SUCCESS,"vision graph compute failed");
        for(auto & stage:g.stages) {
            auto * t=stage.second;
            observer(stage.first,{t->ne[0],t->ne[1],t->ne[2],t->ne[3]},read(t));
        }
        return read(output);
    }
};
VisionRuntime::VisionRuntime()=default;
VisionRuntime::~VisionRuntime()=default;
bool VisionRuntime::load(const std::string & path,ggml_backend_t backend,int dimension,int vocabulary,std::string & error) {
    error.clear();
    try {
        require(backend!=nullptr,"null vision backend");
        require(!detail::hip_bias_workspace(backend) || detail::hip_norm_capable(backend),
                "HIP source-order vision normalization unavailable");
        require(!detail::hip_bias_workspace(backend) || detail::hip_rotary_capable(backend),
                "HIP source-order vision rotary tables unavailable");
        Meta meta;
        meta.g=gguf_init_from_file(path.c_str(),{true,&meta.c});
        require(meta.g && meta.c,"could not parse vision GGUF");
        validate_metadata(meta.g,dimension,vocabulary);
        auto expected=inventory();
        require(gguf_get_n_tensors(meta.g)==int64_t(expected.size()),"wrong projector tensor count");
        common::GgufMmap mapped;
        if(!mapped.open(path,error)) throw std::runtime_error(error);
        std::vector<std::pair<size_t,size_t>> ranges;
        for(int64_t i=0;i<gguf_get_n_tensors(meta.g);++i) {
            std::string name=gguf_get_tensor_name(meta.g,i);
            auto it=expected.find(name);
            require(it!=expected.end(),"unknown or duplicate tensor: "+name);
            auto * t=ggml_get_tensor(meta.c,name.c_str());
            require(t && t->type==GGML_TYPE_BF16,"wrong tensor dtype: "+name);
            for(int d=0;d<4;++d) require(t->ne[d]==it->second[d],"wrong tensor shape: "+name);
            const size_t offset=gguf_get_tensor_offset(meta.g,i),size=gguf_get_tensor_size(meta.g,i);
            require(common::gguf_tensor_in_file(gguf_get_data_offset(meta.g),offset,size,mapped.size()),"tensor outside file: "+name);
            require(offset%32==0,"unaligned tensor: "+name);
            ranges.emplace_back(offset,offset+size);
            expected.erase(it);
        }
        std::sort(ranges.begin(),ranges.end());
        for(size_t i=1;i<ranges.size();++i) require(ranges[i].first>=ranges[i-1].second,"overlapping tensor data");
        require(expected.empty(),"missing projector tensor");
        auto candidate=std::make_unique<Impl>();
        candidate->backend=backend;
        candidate->weights=meta.c; meta.c=nullptr;
        candidate->buffer=ggml_backend_alloc_ctx_tensors(candidate->weights,backend);
        require(candidate->buffer!=nullptr,"projector weight allocation failed");
        ggml_backend_buffer_set_usage(candidate->buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        const auto * bytes=static_cast<const uint8_t *>(mapped.data());
        for(int64_t i=0;i<gguf_get_n_tensors(meta.g);++i) {
            auto * t=candidate->weight(gguf_get_tensor_name(meta.g,i));
            ggml_backend_tensor_set(t,bytes+gguf_get_data_offset(meta.g)+gguf_get_tensor_offset(meta.g,i),0,ggml_nbytes(t));
        }
        impl_=std::move(candidate); // Failed reload leaves prior runtime intact.
        return true;
    } catch(const std::exception & e) { error=e.what(); return false; }
}
bool VisionRuntime::encode(const std::vector<float> & patches,PatchGrid grid,VisionOutput & output,
                           std::string & error,bool retain,const StageObserver & observer) {
    error.clear(); output={};
    try {
        require(bool(impl_),"vision runtime is not loaded");
        require(grid.height>0 && grid.width>0 && grid.height<=1152 && grid.width<=1152,"invalid patch grid");
        const int64_t n=int64_t(grid.height)*grid.width;
        const int64_t aligner_height=(grid.height+2)/3, aligner_width=(grid.width+2)/3;
        const int64_t rows=aligner_height*aligner_width;
        const int64_t padded_height=aligner_height+aligner_height%2;
        const int64_t row_length=aligner_width+1;
        const int64_t block_tokens=padded_height*row_length+2+(padded_height/2*row_length%2)*2;
        require(block_tokens+3<=impl_->config.max_image_tokens,"patch grid exceeds image token budget");
        require(patches.size()==size_t(n)*588,"patch count/shape mismatch");
        require(!detail::hip_bias_workspace(impl_->backend) || n>=16,
                "HIP source-order vision normalization requires at least 16 patches");
        for(float value:patches) require(std::isfinite(value),"non-finite image patch");
        std::vector<float> x;
        {
            Graph g;
            auto input=ggml_new_tensor_2d(g.c,GGML_TYPE_F32,588,n);
            auto y=impl_->linear(g.c,rounded(g.c,input),"vision.patch_embed.proj");
            g.stage("patch_embed",y,bool(observer));
            x=impl_->execute(g,input,patches,y,observer);
        }
        std::vector<float> cos_values,sin_values;
        // Patch output is already on the host. Release its reusable device arena
        // before the independently bounded rotary temporary allocation.
        if(detail::hip_bias_workspace(impl_->backend)) release_scratch();
        detail::rotary_tables(grid,cos_values,sin_values,impl_->backend);
        for(int i=0;i<32;++i) {
            Graph g;
            const auto p="vision.blocks."+std::to_string(i);
            auto input=ggml_new_tensor_2d(g.c,GGML_TYPE_F32,1024,n);
            auto cosine=ggml_new_tensor_3d(g.c,GGML_TYPE_F32,32,1,n);
            auto sine=ggml_new_tensor_3d(g.c,GGML_TYPE_F32,32,1,n);
            ggml_set_input(cosine); ggml_set_input(sine);
            auto normalized=impl_->norm(g.c,input,p+".norm1");
            auto qkv=impl_->linear(g.c,normalized,p+".attn.wqkv");
            auto slice=[&](int offset) {
                return ggml_cont(g.c,ggml_view_3d(g.c,qkv,64,16,n,64*sizeof(float),3072*sizeof(float),offset*1024*sizeof(float)));
            };
            auto q=detail::rotate(g.c,slice(0),cosine,sine);
            auto k=detail::rotate(g.c,slice(1),cosine,sine);
            auto attention=detail::attention(g.c,q,k,slice(2));
            auto projected=impl_->linear(g.c,attention,p+".attn.wo");
            auto residual=rounded(g.c,ggml_add(g.c,input,projected));
            auto mlp=impl_->linear(g.c,impl_->norm(g.c,residual,p+".norm2"),p+".mlp.w1",false);
            auto gate=ggml_cont(g.c,ggml_view_2d(g.c,mlp,2816,n,5632*sizeof(float),0));
            auto up=ggml_view_2d(g.c,mlp,2816,n,5632*sizeof(float),2816*sizeof(float));
            auto product=rounded(g.c,ggml_mul(g.c,rounded(g.c,ggml_silu(g.c,gate)),up));
            auto y=rounded(g.c,ggml_add(g.c,residual,impl_->linear(g.c,product,p+".mlp.w2",false)));
            if(i==0) {
                g.stage("block0.norm1",normalized,bool(observer)); g.stage("block0.qkv",qkv,bool(observer));
                g.stage("block0.q",q,bool(observer)); g.stage("block0.k",k,bool(observer));
                g.stage("block0.attention",attention,bool(observer));
            }
            g.stage("block"+std::to_string(i),y,bool(observer));
            x=impl_->execute(g,input,x,y,observer,cosine,sine,cos_values,sin_values);
        }
        {
            Graph g;
            auto input=ggml_new_tensor_2d(g.c,GGML_TYPE_F32,1024,n);
            auto y=impl_->norm(g.c,input,"vision.norm");
            g.stage("features",y,bool(observer));
            x=impl_->execute(g,input,x,y,observer);
        }
        VisionOutput result;
        if(retain) result.features=x;
        {
            Graph g;
            auto input=ggml_new_tensor_2d(g.c,GGML_TYPE_F32,1024,n);
            auto unfolded=detail::unfold(g.c,input,grid,1024);
            auto first=impl_->linear(g.c,unfolded,"aligner.w1");
            auto activated=rounded(g.c,ggml_gelu_erf(g.c,first));
            auto y=impl_->linear(g.c,activated,"aligner.w2");
            g.stage("unfold",unfolded,bool(observer)); g.stage("aligner.w1",first,bool(observer));
            g.stage("aligner.gelu",activated,bool(observer)); g.stage("embeddings",y,bool(observer));
            result.embeddings=impl_->execute(g,input,x,y,observer);
        }
        for(float value:result.embeddings) require(std::isfinite(value),"non-finite vision output");
        result.rows=int(rows); result.columns=4096; output=std::move(result);
        return true;
    } catch(const std::exception & e) { error=e.what(); return false; }
}
bool VisionRuntime::sentinel(Sentinel identity,std::vector<float> & output,std::string & error) const {
    error.clear(); output.clear();
    if(!impl_) { error="vision runtime is not loaded"; return false; }
    const char * name=nullptr;
    switch(identity) {
        case Sentinel::Start: name="image_start"; break; case Sentinel::Pad: name="image_pad"; break;
        case Sentinel::Newline: name="image_newline"; break; case Sentinel::End: name="image_end"; break;
        default: error="invalid sentinel identity"; return false;
    }
    std::vector<ggml_bf16_t> raw(4096);
    ggml_backend_tensor_get(impl_->weight(name),raw.data(),0,raw.size()*sizeof(ggml_bf16_t));
    output.resize(4096);
    ggml_bf16_to_fp32_row(raw.data(),output.data(),int64_t(raw.size()));
    return true;
}
void VisionRuntime::release_scratch() { if(impl_ && impl_->allocator) { ggml_gallocr_free(impl_->allocator); impl_->allocator=nullptr; } }
const VisionConfig * VisionRuntime::config() const { return impl_ ? &impl_->config : nullptr; }
size_t VisionRuntime::weight_bytes() const { return impl_ ? ggml_backend_buffer_get_size(impl_->buffer) : 0; }
size_t VisionRuntime::scratch_bytes() const {
    if(!impl_) return 0;
    // This conservative reservation remains after release_scratch(): the HIP
    // context retains its workspace until backend destruction.
    return detail::hip_bias_workspace(impl_->backend) +
        (impl_->allocator ? ggml_gallocr_get_buffer_size(impl_->allocator,0) : 0);
}
} // namespace dflash::vision
