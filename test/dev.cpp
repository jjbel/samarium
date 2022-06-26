/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

int main()
{
    auto rand = RandomGenerator{};

    auto particles = ParticleSystem::generate(
        500,
        [&](u64 /* index */)
        {
            const auto mass = rand.range<f64>({0.5, 4.0});
            return Particle{.pos    = rand.vector(BoundingBox<f64>::square(70)),
                            .vel    = rand.polar_vector({0, 20}),
                            .radius = 0.7 * mass,
                            .mass   = mass};
        });

    auto app                = App{{.dims = dims720}};
    const auto viewport_box = app.viewport_box();
    auto clock              = Stopwatch{};

    auto trail = Trail{2000000};

    const auto update = [&](f64 dt)
    {
        particles.self_collision();

        particles.for_each(
            [viewport_box, dt](Particle& particle)
            {
                for (const auto& wall : viewport_box) { phys::collide(particle, wall, dt); }
            });

        particles.update(app.thread_pool, dt);
    };

    const auto draw = [&]
    {
        trail.push_back(particles[0].pos);

        app.fill("#0d1117"_c);
        app.draw(trail, "#2887ed"_c);

        for (const auto& p : particles)
        {
            app.draw(p, {.fill_color   = colors::transparent,
                         .border_color = "#fc2403"_c,
                         .border_width = 0.08});
        }
        print("fps:", std::round(1.0 / clock.seconds()));
        clock.reset();
    };

    app.run(update, draw, 16);
}
