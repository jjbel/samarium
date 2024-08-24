/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "samarium/core/types.hpp" // for f64

#include "Vector2.hpp" // for Vector2

namespace sm
{
struct Circle
{
    Vector2 centre{};
    f64 radius{};

    [[nodiscard]] constexpr auto at_angle(f64 angle) const noexcept
    {
        return centre + Vector2::from_polar({.length = radius, .angle = angle});
    }

    /* Assuming point is on Circle, move it counter-clockwise */
    [[nodiscard]] constexpr auto move_along(Vector2 point, f64 distance) const noexcept
    {
        return centre + (point - centre).rotated(distance / this->radius);
    }
};

struct LineSegment
{
    Vector2 p1{};
    Vector2 p2{};

    struct StandardForm
    {
        f64 a{}; // coefficient of x
        f64 b{}; // coefficient of y
        f64 c{}; // constant term
    };

    [[nodiscard]] constexpr auto vector() const noexcept { return p2 - p1; }

    [[nodiscard]] constexpr auto length() const noexcept { return vector().length(); }

    [[nodiscard]] constexpr auto length_sq() const noexcept { return vector().length_sq(); }

    [[nodiscard]] constexpr auto slope() const noexcept { return vector().slope(); }

    [[nodiscard]] constexpr auto standard_form() const noexcept
    {
        const auto a = p1.y - p2.y;
        const auto b = p2.x - p1.x;
        const auto c = -a * p1.x - b * p1.y;
        return StandardForm{.a = a, .b = b, .c = c};
    }

    constexpr auto translate(Vector2 amount) noexcept
    {
        this->p1 += amount;
        this->p2 += amount;
    }

    constexpr auto for_each(auto&& function)
    {
        function(p1);
        function(p2);
    }
};

} // namespace sm
