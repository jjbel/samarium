/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "samarium/core/types.hpp"
#include "samarium/math/Vec2.hpp"

namespace sm::cuda
{
struct HashGrid
{
    const u64 width;
    const u64 height;
    const f32 cell_size = 2.0F;

    u64 get_index(Indices v);
};
} // namespace sm::cuda
