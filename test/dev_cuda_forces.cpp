/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include "samarium/cuda/cuda.hpp"
#include "samarium/samarium.hpp"

using namespace sm;

auto main(int argc, char* argv[]) -> i32
{
    u64 count = 4;
    auto pos  = std::vector<Vec2f>{{0.0F, 0.0F}, {1.0F, 0.0F}, {0.0F, 1.0F}, {2.0F, 0.0F}};
    auto acc  = std::vector<Vec2f>(count);

    f32* pos_dev;
    f32* acc_dev;

    cuda::forces({count, pos, acc, pos_dev, acc_dev});

    print(/* pos, */ acc);

    f32 error = 0;
    for (auto i : pos) { error += std::abs(i.x - 1) + std::abs(i.y - 3); }
    // print(pos[0], pos[1]);
    print(error);


    // for (int i = 0; i < count; i++)
    // {
    //     Vec2f force{};
    //     for (int j = 0; j < count; j++)
    //     {
    //         if (i == j) { continue; }
    //         const auto delta = pos[j] - pos[i];
    //         const auto l     = 1.0F / delta.length();
    //         force += 1.0F * l * l * l * delta;
    //     }
    //     acc[i] = force;
    // }
    // print(acc);
}
