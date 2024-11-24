/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <optional> // for optional, nullopt

#include "samarium/core/types.hpp"       // for f64
#include "samarium/math/Vec2.hpp"        // for Vec2_t, operator+, opera...
#include "samarium/math/shapes.hpp"      // for LineSegment
#include "samarium/math/vector_math.hpp" // for distance, clamped_intersection

#include "Particle.hpp" // for Particle

namespace sm::phys
{
template <typename Float = f64>
[[nodiscard]] auto did_collide(const Particle<Float>& p1, const Particle<Float>& p2) -> bool
{
    return math::distance(p1.pos, p2.pos) < p1.radius + p2.radius;
}

namespace detail
{
[[nodiscard]] constexpr auto
solve_collision(f64& vel_left, f64& vel_right, f64 mass_left, f64 mass_right, f64 damping)
{
    const auto delta = damping * (vel_left - vel_right);
    vel_left  = (mass_left * vel_left + mass_right * vel_right - delta) / (mass_left + mass_right);
    vel_right = vel_left + delta;
}
} // namespace detail

template <typename Float = f64>
[[maybe_unused]] auto collide(Particle<Float>& p1, Particle<Float>& p2, f64 damping = 1.0) -> bool
{
    // https://jjbel.github.io/samarium/physics/two-particle-collision
    if (!did_collide(p1, p2)) { return false; }

    const auto angle_of_impact = Vec2::angle_between(p1.pos, p2.pos);

    // ensure p1 is on the left
    const auto swap_left_right =
        p1.pos.rotated(-angle_of_impact).x > p2.pos.rotated(-angle_of_impact).x;

    p1.vel.rotate(-angle_of_impact);
    p2.vel.rotate(-angle_of_impact);

    if (swap_left_right) { std::swap(p1, p2); }

    const auto did_collide = p2.vel.x <= p1.vel.x;
    if (did_collide) { detail::solve_collision(p1.vel.x, p2.vel.x, p1.mass, p2.mass, damping); }

    // cleanup:
    if (swap_left_right) { std::swap(p1, p2); }
    p1.vel.rotate(angle_of_impact);
    p2.vel.rotate(angle_of_impact);

    return did_collide;
}

template <typename Float = f64>
void collide(
    Particle<Float>& current, const LineSegment& l, f64 dt, f64 damping = 1.0, f64 friction = 1.0)
{
    const auto old_pos       = current.pos - current.vel * dt;
    const auto vec           = l.vector();
    const auto proj          = math::project(old_pos, l);
    const auto normal_vector = current.pos - proj;

    const auto radius_shift =
        (proj - old_pos)
            .with_length(current.radius); // keep track of the point on the circumference of
                                          // prev closest to l, which will cross l first

    const auto possible_collision =
        math::clamped_intersection({old_pos + radius_shift, current.pos + radius_shift}, l);

    if (!possible_collision) { return; }

    const auto point = possible_collision.value();

    auto leftover_vel = current.pos + radius_shift - point;
    leftover_vel.reflect(vec);
    current.vel.reflect(vec);
    current.pos = point + leftover_vel - radius_shift + normal_vector.with_length(0.05);

    current.vel.rotate(-vec.angle());
    current.vel.x *= friction;
    current.vel.y *= damping;
    current.vel.rotate(vec.angle());
}
} // namespace sm::phys
