/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "samarium/graphics/Color.hpp"
#include "samarium/math/shapes.hpp"

namespace sm
{
struct Particle
{
    Vector2 pos{};
    Vector2 vel{};
    Vector2 acc{};
    double_t radius{1};
    double_t mass{1};
    Color color{};

    constexpr auto as_circle() const noexcept { return Circle{pos, radius}; }

    constexpr auto apply_force(Vector2 force) noexcept { acc += force / mass; }

    constexpr auto update(double_t time_delta = 1.0 / 64) noexcept
    {
        vel += acc * time_delta;
        pos += vel * time_delta;
        acc *= 0.0;
    }
};

constexpr auto update(auto& object, double_t time_delta = 1.0 / 60)
{
    object.update(time_delta);
}
} // namespace sm
