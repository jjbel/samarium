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
    u64 count = 2'000'000;
    auto pos  = std::vector<Vector2f>(count);
    auto acc  = std::vector<Vector2f>(count);
    cuda::forces({count, pos, acc});

    f32 error = 0;
    for (auto i : pos) { error += std::abs(i.x - 1) + std::abs(i.y - 3); }
    print(pos[0]);
    print(error);
}
