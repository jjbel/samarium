/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "samarium/core/types.hpp"       // for f64
#include "samarium/math/Vec2.hpp"        // for Vec2_t, operator-, Vec2
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

    // TODO implement this as a filter on spring vector, not here
    // TODO doesn't work, explodes
    // TODO add a max_force to prevent explosions
    const f64 break_stretch;
    bool active = true;

    Spring(Particle<Float>& particle1,
           Particle<Float>& particle2,
           f64 stiffness_    = 100.0,
           f64 damping_      = 10.0,
           f64 break_stretch = 200000.0) noexcept
        : p1{particle1}, p2{particle2}, rest_length{math::distance(particle1.pos, particle2.pos)},
          stiffness{stiffness_}, damping{damping_}, break_stretch{break_stretch}
    {
    }

    [[nodiscard]] auto length() const noexcept { return math::distance(p1.pos, p2.pos); }

    void update() noexcept
    {
        // if (!active) { return; }
        const auto vec = p2.pos - p1.pos;
        const auto dx  = vec.length() - rest_length;
        // print(std::abs(dx) / rest_length);
        if (std::abs(dx) / rest_length > break_stretch)
        {
            active = false;
            // return;
        }
        const auto spring = dx * stiffness;
        auto damp         = Vec2::dot(vec.normalized(), p2.vel - p1.vel) * damping;

        const auto force = vec.with_length(spring + damp);

        p1.apply_force(force);
        p2.apply_force(-force);
    }
};
} // namespace sm
