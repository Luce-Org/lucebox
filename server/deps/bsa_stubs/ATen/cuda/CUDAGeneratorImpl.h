#pragma once
#include <cstdint>
#include <tuple>
namespace at {
struct PhiloxCudaState {
    uint64_t seed_ = 0;
    uint64_t offset_ = 0;
    uint64_t* seed_extragraph_ = nullptr;
    uint64_t* offset_extragraph_ = nullptr;
    uint32_t offset_intragraph_ = 0;
    bool captured_ = false;
};
struct Generator {};
}  // namespace at
