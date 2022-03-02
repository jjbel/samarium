/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "Vector2.hpp"

namespace sm
{
struct Circle
{
    Vector2 centre{};
    f64 radius{};
};

struct LineSegment
{
    Vector2 p1{};
    Vector2 p2{};

    [[nodiscard]] constexpr auto vector() const { return p2 - p1; }

    [[nodiscard]] constexpr auto length() const { return vector().length(); }

    [[nodiscard]] constexpr auto length_sq() const
    {
        return vector().length_sq();
    }

    [[nodiscard]] constexpr auto slope() const { return vector().slope(); }

    constexpr auto translate(Vector2 amount)
    {
        this->p1 += amount;
        this->p2 += amount;
    }
};
} // namespace sm
