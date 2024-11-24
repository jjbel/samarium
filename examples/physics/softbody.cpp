/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

#include "range/v3/view/concat.hpp"

using namespace sm;
using namespace sm::literals;

// !!!!! EDIT THIS !!!!!
struct Params
{
    f64 time_scale                 = 1.4;
    Vec2 gravity                   = -30.0_y;
    f64 coefficient_of_friction    = 0.95;
    f64 coefficient_of_restitution = 0.95; // bounciness
    f64 spring_stiffness           = 150.0;
    f64 spring_damping             = 55.0;
    f64 particle_mass              = 0.6;
    f64 particle_radius            = 1.6;
    Vec2 particle_velocity{10, 20};
    Dimensions particle_count_xy{4, 4};
    Vec2 softbody_area{25, 25};
};

auto main() -> i32
{
    const auto params = Params{};

    const auto get_dual_from_indices = [&](auto indices)
    {
        const auto x = interp::map_range<f64>(
            static_cast<f64>(indices.x), Extents<u64>{0UL, params.particle_count_xy.x}.cast<f64>(),
            Extents<f64>{-params.softbody_area.x / 2.0, params.softbody_area.x / 2.0});

        const auto y = interp::map_range<f64>(
            static_cast<f64>(indices.y), Extents<u64>{0UL, params.particle_count_xy.y}.cast<f64>(),
            Extents<f64>{-params.softbody_area.y / 2.0, params.softbody_area.y / 2.0});

        auto pos = Vec2{x, y};
        pos.rotate(1);

        const auto particle = Particle{
            pos, params.particle_velocity, {}, params.particle_radius, params.particle_mass};

        auto dual = Dual<Particle>();
        dual.prev = particle;
        dual.now  = particle;

        return dual;
    };


    auto particles =
        Grid2<Dual<Particle>>::generate(params.particle_count_xy, get_dual_from_indices);


    auto springs = [&]
    {
        std::vector<Spring> temp;
        temp.reserve(params.particle_count_xy.x * params.particle_count_xy.y * 4UL);

        for (auto i : loop::end(params.particle_count_xy.y))
        {
            for (auto j : loop::end(params.particle_count_xy.x))
            {
                if (j != 0) { temp.emplace_back(particles[{j, i}].now, particles[{j - 1, i}].now); }
                if (i != 0) { temp.emplace_back(particles[{j, i}].now, particles[{j, i - 1}].now); }
                if (i != 0 && j != 0)
                {
                    temp.emplace_back(particles[{j, i}].now, particles[{j - 1, i - 1}].now,
                                      params.spring_stiffness, params.spring_damping);
                }
                if (i != 0 && j != params.particle_count_xy.x - 1)
                {
                    temp.emplace_back(particles[{j, i}].now, particles[{j + 1, i - 1}].now,
                                      params.spring_stiffness, params.spring_damping);
                }
            }
        }

        return temp;
    }();

    auto app = App{{.dims{1600, 800}}};
    app.transform.scale *= 1.4;
    const auto viewport_box = app.viewport_box();

    const auto walls     = std::to_array({LineSegment{{-19, 20}, {19, 14}}});
    const auto colliders = ranges::views::concat(viewport_box, walls);

    auto watch = Stopwatch{};

    const auto update = [&](auto delta)
    {
        delta *= params.time_scale;

        for (auto&& spring : springs) { spring.update(); }

        for (auto&& particle : particles)
        {
            const auto mouse_pos = app.transform.apply_inverse(app.mouse.current_pos);

            particle->apply_force(particle->mass * params.gravity);

            if (math::within_distance(mouse_pos, particle->pos, particle->radius) && app.mouse.left)
            {
                particle->pos += app.mouse.vel() / app.transform.scale;
                particle->vel = Vec2{};
                particle->acc = Vec2{};
            }

            particle->update(delta);

            for (auto&& other_particle : particles)
            {
                phys::collide(particle.now, other_particle.now);
            }

            for (auto&& wall : colliders)
            {
                phys::collide(particle.now, wall, delta, params.coefficient_of_restitution,
                              params.coefficient_of_friction);
            }
        }
    };

    const auto draw = [&]
    {
        draw::background("#16161c"_c);

        for (const auto& ls : colliders) { app.draw_line_segment(ls, colors::white, 0.1); }

        for (const auto& spring : springs)
        {
            app.draw_line_segment(LineSegment{spring.p1.pos, spring.p2.pos},
                                  colors::white.with_multiplied_alpha(0.5), 0.1);
        }

        for (auto& particle : particles)
        {
            app.draw(particle.now, {.fill_color = colors::red});
            particle.prev = particle.now;
        }

        print("Framerate:", std::round(1.0 / watch.seconds()));
        watch.reset();
    };

    app.run(update, draw, 32);
}
