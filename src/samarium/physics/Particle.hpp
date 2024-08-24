/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
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
        vel += acc * time_delta;
        pos += vel * time_delta;
        acc = Vector2_t<Float>{}; // reset acceleration
    }

    [[nodiscard]] constexpr auto operator==(const Particle&) const -> bool = default;
};
} // namespace sm
