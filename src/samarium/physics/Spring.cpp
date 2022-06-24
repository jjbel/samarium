/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/math/Vector2.hpp"     // for Vector2
#include "samarium/math/vector_math.hpp" // for distance
#include "samarium/physics/Particle.hpp" // for Particle

#include "Spring.hpp"

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
