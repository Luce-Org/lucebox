#include "deepseek4/deepseek4_vision.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace dflash::vision;
static void check(bool value,const char * message) { if(!value) throw std::runtime_error(message); }
static float bf16(float value) { return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value)); }
struct Test {
    ggml_backend_t backend=ggml_backend_cpu_init();
    ggml_context * c=ggml_init({1024*1024,nullptr,true});
    ggml_gallocr_t alloc=ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    std::vector<std::pair<ggml_tensor *,std::vector<float>>> inputs;
    Test() { ggml_backend_cpu_set_n_threads(backend,2); }
    ~Test() { ggml_gallocr_free(alloc); ggml_free(c); ggml_backend_free(backend); }
    ggml_tensor * input(int a,int b,int d,std::vector<float> data) {
        auto t=ggml_new_tensor_3d(c,GGML_TYPE_F32,a,b,d);
        ggml_set_input(t); inputs.emplace_back(t,std::move(data)); return t;
    }
    std::vector<float> run(ggml_tensor * t) {
        ggml_set_output(t);
        auto g=ggml_new_graph(c); ggml_build_forward_expand(g,t);
        check(ggml_gallocr_alloc_graph(alloc,g),"test graph allocation failed");
        for(auto & p:inputs) ggml_backend_tensor_set(p.first,p.second.data(),0,p.second.size()*4);
        check(ggml_backend_graph_compute(backend,g)==GGML_STATUS_SUCCESS,"test compute failed");
        std::vector<float> out(ggml_nelements(t)); ggml_backend_tensor_get(t,out.data(),0,out.size()*4); return out;
    }
};
int main() {
    try {
        {
            Test t;
            std::vector<float> cosine,sine; detail::rotary_tables({2,3},cosine,sine);
            std::vector<float> x(64*2*6);
            for(size_t i=0;i<x.size();++i) x[i]=float(int(i%41)-20)*.125f;
            auto out=t.run(detail::rotate(t.c,t.input(64,2,6,x),t.input(32,1,6,cosine),t.input(32,1,6,sine)));
            for(int n=0;n<6;++n) for(int h=0;h<2;++h) for(int j=0;j<32;++j) {
                // Independent scalar formula: axis frequencies reset after16 channels.
                float a=float(j<16?n/3:n%3)/std::pow(10000.f,float(j%16)/16.f);
                int i=(n*2+h)*64+j;
                check(std::abs(out[i]-bf16(x[i]*std::cos(a)-x[i+32]*std::sin(a)))<.001f,"half-split RoPE mismatch");
                check(std::abs(out[i+32]-bf16(x[i+32]*std::cos(a)+x[i]*std::sin(a)))<.001f,"half-split RoPE second half mismatch");
            }
        }
        {
            Test t;
            // Identical zero Q/K gives uniform attention. First query must see
            // the final value; a causal implementation would return only1.
            auto q=t.input(2,1,3,std::vector<float>(6,0));
            auto k=t.input(2,1,3,std::vector<float>(6,0));
            auto v=t.input(2,1,3,{1,2,4,5,10,11});
            auto out=t.run(detail::attention(t.c,q,k,v));
            for(int i=0;i<3;++i) { check(out[2*i]==5,"bidirectional attention mismatch"); check(out[2*i+1]==6,"attention channel mismatch"); }
        }
        {
            Test t;
            const int heads=2,n=3,d=4;
            std::vector<float> q(n*heads*d),k(q.size()),v(q.size());
            for(int p=0;p<n;++p) for(int h=0;h<heads;++h) for(int j=0;j<d;++j) {
                int i=(p*heads+h)*d+j;
                q[i]=float((p+2*h+j)%5-2)*.25f;
                k[i]=float((2*p+h+j)%7-3)*.25f;
                v[i]=float(20*h+3*p+j);
            }
            auto out=t.run(detail::attention(t.c,t.input(d,heads,n,q),t.input(d,heads,n,k),t.input(d,heads,n,v)));
            for(int h=0;h<heads;++h) for(int p=0;p<n;++p) {
                double weights[3],sum=0;
                for(int key=0;key<n;++key) {
                    double dot=0;
                    for(int j=0;j<d;++j) dot+=double(q[(p*heads+h)*d+j])*k[(key*heads+h)*d+j];
                    weights[key]=std::exp(dot/std::sqrt(double(d))); sum+=weights[key];
                }
                for(int j=0;j<d;++j) {
                    double value=0;
                    for(int key=0;key<n;++key) value+=weights[key]/sum*v[(key*heads+h)*d+j];
                    check(out[(p*heads+h)*d+j]==bf16(float(value)),"nonuniform multihead attention mismatch");
                }
            }
        }
        {
            Test t;
            // BF16 halfway values round to nearest even, including negatives.
            auto x=t.input(4,1,1,{1.00390625f,1.01171875f,-1.00390625f,-1.01171875f});
            auto out=t.run(ggml_cast(t.c,ggml_cast(t.c,x,GGML_TYPE_BF16),GGML_TYPE_F32));
            check(out==std::vector<float>({1.f,1.015625f,-1.f,-1.015625f}),"BF16 halfway rounding mismatch");
        }
        {
            Test t;
            const int h=4,w=5,c=2;
            std::vector<float> data(h*w*c);
            for(int y=0;y<h;++y) for(int x=0;x<w;++x) for(int ch=0;ch<c;++ch) data[(y*w+x)*c+ch]=100*ch+10*y+x+1;
            auto out=t.run(detail::unfold(t.c,t.input(c,h*w,1,data),{h,w},c));
            check(out.size()==72,"unfold shape mismatch");
            for(int oy=0;oy<2;++oy) for(int ox=0;ox<2;++ox) for(int ch=0;ch<c;++ch)
                for(int dy=0;dy<3;++dy) for(int dx=0;dx<3;++dx) {
                    int y=oy*3+dy,x=ox*3+dx;
                    float expected=y<h&&x<w?float(100*ch+10*y+x+1):0;
                    check(out[((oy*2+ox)*c+ch)*9+dy*3+dx]==expected,"channel-first bottom/right padded unfold mismatch");
                }
        }
        {
            Test t;
            std::vector<float> x={-3.f,-1.f,-.1f,0.f,.1f,1.f,3.f};
            auto out=t.run(ggml_gelu_erf(t.c,t.input(7,1,1,x)));
            for(int i=0;i<7;++i) check(std::abs(out[i]-.5f*x[i]*(1+std::erf(x[i]/std::sqrt(2.f))))<1e-6f,"exact erf GELU mismatch");
        }
        std::cout<<"PASS: half-split 2D RoPE, full bidirectional/multihead attention, BF16 halfway rounding, padded channel-first unfold, exact erf GELU\n";
        return 0;
    } catch(const std::exception & e) { std::cerr<<e.what()<<"\n"; return 1; }
}
