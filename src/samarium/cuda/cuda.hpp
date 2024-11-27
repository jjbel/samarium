
#pragma once

#include <span>

#include "samarium/core/types.hpp"
#include "samarium/math/Vec2.hpp"

namespace sm::cuda
{
void test();
void thrust_benchmark_1(u64 count = 32 << 20);
void print_device_info();

struct ForcesSettings
{
    u64 count;
    std::span<Vec2f> pos;
    std::span<Vec2f> acc;
    f32* pos_dev; // TODO shd be ptr to ptr to be persistent
    f32* acc_dev;
    f32 strength = 0.0006F;
    // f32 min_distance = 0.01F;
    f32 max_force = 1.0F;
    f32 r_max_sq = 9.0F;
};

void forces(const ForcesSettings& settings);
} // namespace sm::cuda
