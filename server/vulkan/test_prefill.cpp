#include "prefill_partition.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using luce::common::lfm_prefill_partition;
template<class Partition>
static void check_partition(int n, const std::vector<int> & expected, Partition partition) {
    const int length = n;
    std::vector<int> actual;
    while (n) {
        int step = partition(n, 256);
        if (step <= 0 || step > n || step > 256)
            throw std::runtime_error("invalid prefill progress: length="+std::to_string(length)+
                                     " remaining="+std::to_string(n)+" step="+std::to_string(step));
        actual.push_back(step);
        n -= step;
    }
    if (actual != expected) throw std::runtime_error("prefill checkpoint regression: length="+std::to_string(length));
}
int main() {
    try {
        auto test = [](int n, const std::vector<int> & expected) {
            check_partition(n, expected, lfm_prefill_partition);
        };
        test(1, {1}); test(4, {4}); test(5, {1,4});
        test(255, {251,4}); test(256, {252,4}); test(257, {1,252,4});
        test(511, {255,252,4}); test(512, {256,252,4}); test(513, {256,1,252,4});
        // Exercise the same named-case loop under progress faults: it must
        // throw before appending/subtracting, not grow an unbounded vector.
        for (int bad : {0,-1,6,257}) {
            int calls = 0;
            bool rejected = false;
            try { check_partition(5, {}, [&](int,int){++calls; return bad;}); }
            catch (const std::runtime_error & e) {
                rejected = std::string(e.what()).find("invalid prefill progress:") == 0;
                std::cout << "PASS fault step=" << bad << " " << e.what() << '\n';
            }
            if (!rejected || calls != 1) throw std::runtime_error("progress fault did not fail fast");
        }
        for (int chunk : {1,2,4,16,128,256,512}) for (int length=1;length<=4096;++length) {
            int remaining=length;
            while (remaining) {
                int n=lfm_prefill_partition(remaining,chunk);
                if(n<=0 || n>chunk || n>remaining) throw std::runtime_error("invalid partition");
                remaining-=n;
            }
        }
        std::cout << "PASS final-chunk/final-four partitions and exhaustive 1..4096 lengths\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
