/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <optional> // for optional

#include "samarium/core/types.hpp"   // for f64
#include "samarium/math/Vector2.hpp" // for Vector2

#include "Particle.hpp"

namespace sm::phys
{
[[nodiscard]] auto did_collide(const Particle& p1, const Particle& p2) -> std::optional<Vector2>;

[[maybe_unused]] bool collide(Particle& p1, Particle& p2, f64 damping = 1.0);

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
[[nodiscard]] auto did_collide(const Particle& p1, const Particle& p2) -> std::optional<Vector2>
{
    if (math::distance(p1.pos, p2.pos) <= p1.radius + p2.radius)
    {
        return std::optional((p1.pos + p2.pos) / 2.0);
    }
    else { return std::nullopt; }
}

[[maybe_unused]] bool collide(Particle& p1, Particle& p2, f64 damping)
{
    /*
        https://courses.lumenlearning.com/boundless-physics/chapter/collisions/#:~:text=particles%20are%20involved%20in%20an-,elastic%20collision,-%2C%20the%20velocity%20of%20the%20first

        https://www.khanacademy.org/science/physics/linear-momentum/elastic-and-inelastic-collisions/a/what-are-elastic-and-inelastic-collisions
    */

    // if (&p1 == &p2) { return; } // prevent self-intersection

    if (const auto point = did_collide(p1, p2))
    {
        // position changes
        const auto shift  = (p1.radius + (math::distance(p1.pos, p2.pos) - p2.radius)) / 2;
        const auto centre = p1.pos + (p2.pos - p1.pos).with_length(shift);
        p1.pos            = centre + (p1.pos - centre).with_length(p1.radius);
        p2.pos            = centre + (p2.pos - centre).with_length(p2.radius);

        // velocity changes
        const auto line      = p1.pos - p2.pos;
        const auto length_sq = line.length_sq();
        const auto factor    = 2.0 / (p1.mass + p2.mass);
        const auto dot       = Vector2::dot(p1.vel - p2.vel, line);

        const auto vel1 = line * (p2.mass * factor * dot / length_sq);
        const auto vel2 = line * (-p1.mass * factor * dot / length_sq);
        // const auto dv              = p2.vel - p1.vel;
        // const auto factor_mass     = 1.0 / (p1.mass + p2.mass);
        // const auto factor_momentum = p1.mass * p1.vel + p2.mass * p2.vel;
        // const auto factor_damping  = damping * dv;
        // const auto vel1            = factor_mass * (factor_momentum + p2.mass * factor_damping);
        // const auto vel2            = factor_mass * (factor_momentum + p1.mass *
        // (-factor_damping));
        // const auto vel1 =
        //     (p1.mass * p1.vel + p2.mass * p2.vel + p2.mass * damping * (p2.vel - p1.vel)) /
        //     (p1.mass + p2.mass);
        // const auto vel2 =
        //     (p1.mass * p1.vel + p2.mass * p2.vel + p1.mass * damping * (p1.vel - p2.vel)) /
        //     (p1.mass + p2.mass);

        p1.vel -= vel1 * damping;
        p2.vel -= vel2 * damping;
        return true;
    }
    return false;
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
