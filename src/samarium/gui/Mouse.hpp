/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "samarium/core/types.hpp"   // for f64
#include "samarium/math/Vector2.hpp" // for Vector2

namespace sm
{
struct Mouse
{
    Vector2 pos{};
    Vector2 old_pos{};
    f64 scroll_amount{};
    bool left{};
    bool middle{};
    bool right{};
};
} // namespace sm
