/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/gl/draw.hpp"
#include "samarium/math/Vector2.hpp"
#include "samarium/math/shapes.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    const auto particle_count    = u64{10000};
    const auto hash_grid_spacing = 2;

    auto window = Window{{1800, 900}};

    auto random   = RandomGenerator{1024, RandomMode::NonDeterministic};
    auto viewport = window.viewport();

    auto ps = ParticleSystem<Particle<f64>, 32>::generate(
        particle_count,
        [&](u64 /* index */)
        {
            return Particle<f64>{.pos    = random.vector(viewport) * 0.95,
                                 .vel    = random.polar_vector({0.0, 2.0}),
                                 .radius = .03,
                                 .mass   = 1.0};
        });

    // auto ps         = ParticleSystem{2};
    // ps.particles[0] = Particle{.pos = {-8, 0}, .vel = {8, 0}, .radius = 1.4, .mass = 1.0};
    // ps.particles[1] = Particle{.pos = {8, 0}, .vel = {0, 0}, .radius = 1.8, .mass = 1.0};
    auto watch  = Stopwatch{};
    auto update = [&]
    {
        const auto dt = 1.0 / 60.0;

        ps.for_each(
            [&](Particle<f64>& particle)
            {
                // particle.acc = Vector2{0, -1};
                for (const auto& wall : viewport.line_segments())
                {
                    phys::collide(particle, wall, dt, 0.97);
                }
            });

        watch.reset();
        const auto [count1, count2] = ps.self_collision(1.0);
        const auto time             = watch.seconds();
        fmt::print("Update: {:3.2}ms, {:4}/{:6} collisions\n", time * 1000.0, count1, count2);

        ps.update(dt);
    };

    auto draw = [&]
    {
        // zoom_pan(window);
        // print("mouse pos:", window.mouse.pos - window.mouse.old_pos);
        draw::background("#020407"_c);
        draw::grid_lines(window, {.spacing   = hash_grid_spacing,
                                  .color     = "#eeeeee"_c.with_multiplied_alpha(0.1),
                                  .thickness = 0.04});

        draw::bounding_box(window, viewport, "#eeeeee"_c, 0.2);

        const auto draw_bonds = [&]
        {
            for (auto i : loop::end(ps.particles.size()))
            {
                for (auto j : ps.hash_grid.neighbors(ps.particles[i].pos))
                {
                    if (i < j)
                    {
                        draw::line_segment(window,
                                           LineSegment{ps.particles[i].pos, ps.particles[j].pos},
                                           "#5640ff"_c.with_multiplied_alpha(0.3), 0.2);
                    }
                }
            }
        };
        // draw_bonds();

        watch.reset();
        for (const auto& i : ps)
        {
            // TODO gl error for polygon
            // draw::circle(window, i.as_circle(),
            //              {.fill_color   = "#5640ff"_c.with_multiplied_alpha(0.2),
            //               .border_color = {255, 30, 30, 255},
            //               .border_width = 1});
            draw::circle(window, i.as_circle(),
                         {.fill_color = "#ff0842"_c.with_multiplied_alpha(0.8)});
        }

        const auto time = watch.seconds();
        fmt::print("Draw: {:3.2}ms, {:8}/s\n", time * 1000.0,
                   static_cast<u32>(static_cast<f64>(ps.particles.size()) / time));
    };

    run(window, update, draw);
}
