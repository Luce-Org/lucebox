#include "prefill_partition.h"
#include <iostream>
#include <vector>
using luce::common::lfm_prefill_partition;
int main() {
    auto test = [](int n, const std::vector<int> & expected) {
        std::vector<int> actual;
        while (n) { int step = lfm_prefill_partition(n, 256); actual.push_back(step); n -= step; }
        if (actual != expected) throw std::runtime_error("prefill checkpoint regression");
    };
    test(1, {1}); test(4, {4}); test(5, {1,4});
    test(255, {251,4}); test(256, {252,4}); test(257, {1,252,4});
    test(511, {255,252,4}); test(512, {256,252,4}); test(513, {256,1,252,4});
    for (int chunk : {1,2,4,16,128,256,512}) for (int length=1;length<=4096;++length) {
        int remaining=length;
        while (remaining) {
            int n=lfm_prefill_partition(remaining,chunk);
            if(n<=0 || n>chunk || n>remaining) throw std::runtime_error("invalid partition");
            remaining-=n;
        }
    }
    std::cout << "PASS final-chunk/final-four partitions and exhaustive 1..4096 lengths\n";
}
