int main(){
 ggml_cuda_pool_leg pool(0);size_t n=0;
 void* p=pool.alloc(4ull<<20,&n);pool.free(p,n);
 bool failed=false;try{p=pool.alloc(5ull<<20,&n);}catch(const std::runtime_error&){failed=true;}
 if(failed){std::cout<<"Original allocator failed with reusable cached memory; used="<<used<<"\n";pool.trim();p=pool.alloc(5ull<<20,&n);}
 pool.free(p,n);pool.trim();assert(used==0);
#ifdef EXPECT_FIXED
 assert(!failed);
#else
 assert(failed);
#endif
 // Live buffers must remain intact under pressure.
 size_t live_n=0;void* live_p=pool.alloc(4ull<<20,&live_n);
 bool exhausted=false;try{pool.alloc(5ull<<20,&n);}catch(const std::runtime_error&){exhausted=true;}
 assert(exhausted && live.count(live_p)==1);pool.free(live_p,live_n);pool.trim();
 // Requested bytes fit exactly even though the 5% growth allowance does not.
 capacity=5ull<<20;p=pool.alloc(capacity,&n);assert(n==capacity);pool.free(p,n);pool.trim();
 assert(used==0);std::cout<<"PASS: reclaim cached blocks; preserve live blocks; exact-size fallback; real exhaustion remains failure\n";
}
