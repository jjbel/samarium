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
    f64 coefficient_of_restitution = 1.0; // bounciness
    f64 spring_stiffness           = 150.0;
    f64 spring_damping             = 55.0;
    f64 particle_mass              = 0.6;
    f64 particle_radius            = 1.6;
    Vec2 initial_vel{8, -15};
    Dimensions particle_count_xy{5, 5};
    Vec2 softbody_area{20, 20};
};

// TODO use instanced rendering
template <typename T> struct Dual
{
    T prev{};
    T now{};

    T& operator->() { return now; }
};

auto main() -> i32
{
    const auto params = Params{};

    const auto get_dual_from_indices = [&](auto indices)
    {
        const auto x = interp::map_range<f64>(
            static_cast<f64>(indices.x),
            Extents<u64>{0UL, params.particle_count_xy.x}.template as<f64>(),
            Extents<f64>{-params.softbody_area.x / 2.0, params.softbody_area.x / 2.0});

        const auto y = interp::map_range<f64>(
            static_cast<f64>(indices.y),
            Extents<u64>{0UL, params.particle_count_xy.y}.template as<f64>(),
            Extents<f64>{-params.softbody_area.y / 2.0, params.softbody_area.y / 2.0});

        auto pos = Vec2{x, y};
        pos.rotate(1);

        const auto particle =
            Particle{pos, params.initial_vel, {}, params.particle_radius, params.particle_mass};

        auto dual = Dual<Particle<f64>>();
        dual.prev = particle;
        dual.now  = particle;

        return dual;
    };


    auto particles =
        Grid2<Dual<Particle<f64>>>::generate(params.particle_count_xy, get_dual_from_indices);


    auto springs = [&]
    {
        std::vector<Spring<f64>> temp;
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

    auto window = Window{{.dims{1800, 900}}};
    window.camera.scale /= 70;
    window.display(); // to fix world_box?
    const auto viewport_box = window.world_box().line_segments();
    const auto walls        = std::to_array({LineSegment{{-19, -18}, {19, -25}}});
    const auto colliders    = ranges::views::concat(viewport_box, walls);

    auto watch = Stopwatch{};

    const auto update = [&](f64 delta)
    {
        delta *= params.time_scale;

        // TODO why auto&& was here
        for (auto& spring : springs) { spring.update(); }

        for (auto& particle : particles)
        {
            particle.now.apply_force(particle.now.mass * params.gravity);

            const auto mouse_pos     = window.mouse_pos();
            const auto mouse_pos_old = window.mouse_old_pos();

            if (window.mouse.left &&
                math::within_distance(mouse_pos, particle.now.pos,
                                      3 * particle.now.radius)) // or same for old pos
            {
                particle.now.vel = Vec2{};
                particle.now.acc = Vec2{};

                // TODO gives wrong with (too much) and without (too small) window.camera.scale.
                particle.now.pos += /* window.camera.scale * */ (mouse_pos - mouse_pos_old);
                // particle.now.acc += /* window.camera.scale * */ 0.01 * (mouse_pos -
                // mouse_pos_old);
            }

            particle.now.update(delta);

            // for (auto&& other_particle : particles)
            // {
            //     phys::collide(particle.now, other_particle.now);
            // }

            // TODO particles sometimes disappear, maybe coz NaN
            for (const auto& wall : colliders)
            {
                phys::collide(particle.now, wall, delta, params.coefficient_of_restitution,
                              params.coefficient_of_friction);
            }
        }
    };

    const auto draw = [&]
    {
        // drawing mouse later so do bg last
        draw::background("#16161c"_c);

        for (const auto& ls : colliders) { draw::line_segment(window, ls, colors::white, 0.45); }

        for (const auto& spring : springs)
        {
            if (spring.active)
            {
                draw::line_segment(window, LineSegment{spring.p1.pos, spring.p2.pos},
                                   colors::white.with_multiplied_alpha(0.5), 0.25);
            }
        }

        for (auto& particle : particles)
        {
            draw::circle(window, {particle.now.pos, particle.now.radius}, colors::red);
            particle.prev = particle.now;
        }

        if (window.mouse.left)
        {
            draw::circle(window, {window.mouse_pos(), 1.0}, Color{132, 30, 252});
        }

        // print("Framerate:", std::round(1.0 / watch.seconds()));
        watch.reset();
        window.pan([&] { return window.mouse.middle; });
        window.zoom_to_cursor();

        // std::this_thread::sleep_for(std::chrono::milliseconds(16));
    };

    // TODO many substeps needed else blows up
    run(window, update, draw, 32ULL);
}
