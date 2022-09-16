/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <compare>

#include "samarium/core/types.hpp"   // for f64
#include "samarium/math/Vector2.hpp" // for Vector2
#include "samarium/math/shapes.hpp"  // for Circle

namespace sm
{
struct Particle
{
    Vector2 pos{};
    Vector2 vel{};
    Vector2 acc{};
    f64 radius{1};
    f64 mass{1};

    [[nodiscard]] auto as_circle() const noexcept -> Circle;

    void apply_force(Vector2 force) noexcept;

    void update(f64 time_delta = 1.0 / 64) noexcept;

    auto operator<=>(const Particle&) const = default;
};
} // namespace sm


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_PARTICLE_IMPL)
#include "samarium/math/shapes.hpp" // for Circle

namespace sm
{
[[nodiscard]] auto Particle::as_circle() const noexcept -> Circle { return Circle{pos, radius}; }

void Particle::apply_force(Vector2 force) noexcept { acc += force / mass; }

void Particle::update(f64 time_delta) noexcept
{
    vel += acc * time_delta;
    pos += vel * time_delta;
    acc = Vector2{}; // reset acceleration
}
}; // namespace sm
#endif
