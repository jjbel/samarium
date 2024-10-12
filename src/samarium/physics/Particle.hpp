/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "samarium/core/types.hpp"   // for f64
#include "samarium/math/Vector2.hpp" // for Vector2, operator*, Vector2_t
#include "samarium/math/shapes.hpp"  // for Circle

namespace sm
{
template <typename Float = f64> struct Particle
{
    Vector2_t<Float> pos{};
    Vector2_t<Float> vel{};
    Vector2_t<Float> acc{};
    Float radius{1};
    Float mass{1};

    [[nodiscard]] constexpr auto as_circle() const noexcept { return Circle{pos, radius}; }

    constexpr auto apply_force(Vector2_t<Float> force) noexcept { acc += force / mass; }

    constexpr auto update(Float time_delta = 1.0 / 64) noexcept
    {
        // https://youtu.be/yGhfUcPjXuE&t=609
        // update pos using the average value of velocity
        const auto half_dv = acc * time_delta;
        vel += half_dv;
        pos += vel * time_delta;
        vel += half_dv;
        acc = Vector2_t<Float>{}; // reset acceleration
    }

    [[nodiscard]] constexpr bool operator==(const Particle&) const = default;
};
} // namespace sm
