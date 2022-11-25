/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    const auto particle_count    = u64{50000};
    const auto hash_grid_spacing = 0.2;

    auto window = Window{{1800, 900}};

    auto random   = RandomGenerator{1024, RandomMode::NonDeterministic};
    auto viewport = window.viewport();
    auto ps       = ParticleSystem::generate(particle_count,
                                             [&](u64 /* index */)
                                             {
                                           return Particle{.pos = random.vector(viewport) * 0.95,
                                                                 .vel = random.polar_vector({0.0, 2.0}),
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
            [&](Particle& particle)
            {
                for (const auto& wall : viewport.line_segments())
                {
                    phys::collide(particle, wall, dt, 1.0);
                }
            });

        watch.reset();
        const auto [count1, count2] = ps.self_collision(1.0, hash_grid_spacing);
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

        for (const auto& i : viewport.line_segments())
        {
            draw::line_segment(window, i, "#eeeeee"_c, 0.2);
        }

        for (const auto& i : ps)
        {
            // TODO gl error for polygon
            // draw::circle(window, {i.pos, i.radius},
            //              {.fill_color   = "#5640ff"_c.with_multiplied_alpha(0.2),
            //               .border_color = {255, 30, 30, 255},
            //               .border_width = 1});
            draw::circle(window, {i.pos, i.radius},
                         {.fill_color = "#5640ff"_c.with_multiplied_alpha(0.4)});
        }
    };

    run(window, update, draw);
}
