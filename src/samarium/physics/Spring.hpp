/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
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
struct Spring
{
    Particle& p1;
    Particle& p2;
    const f64 rest_length;
    const f64 stiffness;
    const f64 damping;

    Spring(Particle& particle1,
           Particle& particle2,
           f64 stiffness_ = 100.0,
           f64 damping_   = 10.0) noexcept
        : p1{particle1}, p2{particle2}, rest_length{math::distance(particle1.pos, particle2.pos)},
          stiffness{stiffness_}, damping{damping_}
    {
    }

    [[nodiscard]] auto length() const noexcept -> f64;

    void update() noexcept;
};
} // namespace sm

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_SPRING_IMPL)

#include "samarium/math/Vector2.hpp"     // for Vector2
#include "samarium/math/vector_math.hpp" // for distance
#include "samarium/physics/Particle.hpp" // for Particle

namespace sm
{
[[nodiscard]] auto Spring::length() const noexcept -> f64 { return math::distance(p1.pos, p2.pos); }

void Spring::update() noexcept
{
    const auto vec    = p2.pos - p1.pos;
    const auto spring = (vec.length() - rest_length) * stiffness;
    auto damp         = Vector2::dot(vec.normalized(), p2.vel - p1.vel) * damping;

    const auto force = vec.with_length(spring + damp);

    p1.apply_force(force);
    p2.apply_force(-force);
}
} // namespace sm
#endif
