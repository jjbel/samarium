/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "samarium/core/types.hpp" // for f64
#include "samarium/math/Vec2.hpp"  // for Vec2

namespace sm
{
struct Mouse
{
    Vec2 pos{};
    Vec2 old_pos{};
    f64 scroll_amount{};
    bool left{};
    bool middle{};
    bool right{};
};
} // namespace sm
