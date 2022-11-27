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
    const auto particle_count    = u64{500};
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
                                 .radius = .1,
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
                particle.acc = Vector2{0, -1};
                for (const auto& wall : viewport.line_segments())
                {
                    phys::collide(particle, wall, dt, 0.97);
                }
            });

        watch.reset();
        const auto [count1, count2] = ps.self_collision(1.0);
        const auto time             = watch.seconds();
        fmt::print("{:4}/{:6} collisions in {:3.2}ms\n", count1, count2, time * 1000.0);

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

        for (auto i : range(ps.particles.size()))
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

        for (const auto& i : ps)
        {
            // TODO gl error for polygon
            // draw::circle(window, {i.pos, i.radius},
            //              {.fill_color   = "#5640ff"_c.with_multiplied_alpha(0.2),
            //               .border_color = {255, 30, 30, 255},
            //               .border_width = 1});
            draw::circle(window, {i.pos, i.radius},
                         {.fill_color = "#ff0842"_c.with_multiplied_alpha(1.0)});
        }
    };

    run(window, update, draw);
}
