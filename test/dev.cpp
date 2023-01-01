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
    auto window = Window{{{1800, 900}}};

    auto rand  = RandomGenerator{};

    print(gpu::ParticleSystem::src);

    auto ps    = gpu::ParticleSystem(20);
    auto watch = Stopwatch{};

    for (auto& i : ps.particles)
    {
        i.pos    = rand.vector(window.viewport()).cast<f32>();
        i.vel    = rand.polar_vector({0, 4}).cast<f32>();
        i.radius = 0.4F;
    }

    auto frame = u64();

    const auto update = [&]
    {
        // if (frame == 60)
        {
            watch.reset();
            ps.update();
            watch.print();
        }
        frame++;
    };

    const auto draw = [&]
    {
        draw::background("#131417"_c);
        draw::grid_lines(window, {.spacing = 1, .color{255, 255, 255, 90}, .thickness = 0.028F});
        for (const auto& particle : ps.particles)
        {
            draw::circle(window, {particle.pos.cast<f64>(), particle.radius},
                         {.fill_color = "#ff0000"_c});
        }
    };

    run(window, update, draw);
}

// old texture stuff:
// auto texture = gl::Texture{gl::ImageFormat::R32F};
// imageStore( data, pos, vec4( in_val, 0.0, 0.0, 0.0 ) );
// buffer.bind_level(0, 0, gl::Access::ReadWrite);
