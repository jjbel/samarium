/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "../graphics/Color.hpp"
#include "../math/shapes.hpp"

namespace sm
{
struct Particle
{
    Vector2 pos{};
    Vector2 vel{};
    Vector2 acc{};
    f64 radius{1};
    f64 mass{1};
    Color color{};

    [[nodiscard]] constexpr auto as_circle() const noexcept { return Circle{pos, radius}; }

    constexpr auto apply_force(Vector2 force) noexcept { acc += force / mass; }

    constexpr auto update(f64 time_delta = 1.0 / 64) noexcept
    {
        vel += acc * time_delta;
        pos += vel * time_delta;
        acc = Vector2{}; // reset acceleration
    }
};

// constexpr auto update(auto& object, f64 time_delta = 1.0 / 60.0) { object.update(time_delta); }
} // namespace sm
