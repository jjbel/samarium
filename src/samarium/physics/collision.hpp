/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <optional> // for optional, nullopt

#include "samarium/core/types.hpp"       // for f64
#include "samarium/math/Vector2.hpp"     // for Vector2_t, operator+, opera...
#include "samarium/math/shapes.hpp"      // for LineSegment
#include "samarium/math/vector_math.hpp" // for distance, clamped_intersection

#include "Particle.hpp" // for Particle

#include "samarium/util/print.hpp"

namespace sm::phys
{
[[nodiscard]] auto did_collide(const Particle& p1, const Particle& p2) -> bool;

[[maybe_unused]] auto collide(Particle& p1, Particle& p2, f64 damping = 1.0) -> bool;

void collide(f64 distance_threshold, Particle& p1, Particle& p2, f64 damping = 1.0);

void collide(
    Particle& current, const LineSegment& l, f64 dt, f64 damping = 1.0, f64 friction = 1.0);
} // namespace sm::phys


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_COLLISION_IMPL)

#include "samarium/math/Vector2.hpp"     // for Vector2
#include "samarium/math/shapes.hpp"      // for LineSegment
#include "samarium/math/vector_math.hpp" // for distance, clamped_intersection

namespace sm::phys
{
[[nodiscard]] auto did_collide(const Particle& p1, const Particle& p2) -> bool
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

[[maybe_unused]] auto collide(Particle& p1, Particle& p2, f64 damping) -> bool
{
    // https://strangequark1041.github.io/samarium/physics/two-particle-collision
    if (!did_collide(p1, p2)) { return false; }

    const auto angle_of_impact = Vector2::angle_between(p1.pos, p2.pos);

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

void collide(Particle& current, const LineSegment& l, f64 dt, f64 damping, f64 friction)
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

#endif // defined
