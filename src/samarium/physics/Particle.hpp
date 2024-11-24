/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "samarium/core/types.hpp"  // for f64
#include "samarium/math/Vec2.hpp"   // for Vec2, operator*, Vec2_t
#include "samarium/math/shapes.hpp" // for Circle

namespace sm
{
template <typename Float = f64> struct Particle
{
    Vec2_t<Float> pos{};
    Vec2_t<Float> vel{};
    Vec2_t<Float> acc{};
    Float radius{1};
    Float mass{1};

    [[nodiscard]] constexpr auto as_circle() const noexcept { return Circle{pos, radius}; }

    constexpr auto apply_force(Vec2_t<Float> force) noexcept { acc += force / mass; }

    constexpr auto update(Float time_delta = 1.0 / 64) noexcept
    {
        // https://youtu.be/yGhfUcPjXuE&t=609
        // update pos using the average value of velocity
        const auto half_dv = acc * time_delta;
        vel += half_dv;
        pos += vel * time_delta;
        vel += half_dv;
        acc = Vec2_t<Float>{}; // reset acceleration
    }

    [[nodiscard]] constexpr bool operator==(const Particle&) const = default;
};
} // namespace sm
