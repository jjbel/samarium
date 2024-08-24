/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "samarium/core/types.hpp"       // for f64
#include "samarium/math/Vector2.hpp"     // for Vector2_t, operator-, Vector2
#include "samarium/math/vector_math.hpp" // for distance

#include "Particle.hpp" // for Particle

namespace sm
{
template <typename Float = f64> struct Spring
{
    Particle<Float>& p1;
    Particle<Float>& p2;
    const f64 rest_length;
    const f64 stiffness;
    const f64 damping;

    Spring(Particle<Float>& particle1,
           Particle<Float>& particle2,
           f64 stiffness_ = 100.0,
           f64 damping_   = 10.0) noexcept
        : p1{particle1}, p2{particle2}, rest_length{math::distance(particle1.pos, particle2.pos)},
          stiffness{stiffness_}, damping{damping_}
    {
    }

    [[nodiscard]] auto length() const noexcept { return math::distance(p1.pos, p2.pos); }

    void update() noexcept
    {
        const auto vec    = p2.pos - p1.pos;
        const auto spring = (vec.length() - rest_length) * stiffness;
        auto damp         = Vector2::dot(vec.normalized(), p2.vel - p1.vel) * damping;

        const auto force = vec.with_length(spring + damp);

        p1.apply_force(force);
        p2.apply_force(-force);
    }
};
} // namespace sm
