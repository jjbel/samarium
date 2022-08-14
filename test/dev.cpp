/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/samarium.hpp"

#include "range/v3/view/concat.hpp"

using namespace sm;
using namespace sm::literals;

constexpr auto count           = 2000UL;
constexpr auto initial_speed   = 40.0;
constexpr auto class_size      = 4.0;
constexpr auto max_graph_speed = initial_speed * 3.0;
constexpr auto graph_width     = 90.0;
constexpr auto graph_height    = 60.0;
constexpr auto graph_centre    = Vector2{-47.5, 0.0};

int main()
{
    auto app = App{{.dims{1920, 1080}}};

    auto box = app.transformed_bounding_box();
    box.set_width(box.width() / 2.0);
    print(box);
    box.set_centre({app.transformed_dims().x / 4.0, 0.0});

    auto rand      = RandomGenerator{};
    auto particles = ParticleSystem::generate(
        count,
        [&](u64 /* index */)
        {
            return Particle{.pos = rand.vector(box),
                            // .vel = rand.polar_vector({0, initial_speed});
                            .vel    = rand.polar_vector({initial_speed, initial_speed + 0.001}),
                            .radius = 0.3,
                            .mass   = 0.1};
        });

    auto watch       = Stopwatch{};
    auto speeds      = std::vector<f64>(count);
    auto energy      = 0.0;
    const auto walls = box.line_segments();

    const auto update = [&](f64 dt)
    {
        particles.for_each(
            [&](Particle& particle)
            {
                for (const auto& wall : walls) { phys::collide(particle, wall, dt, 1.0); }
            });

        particles.self_collision();
        particles.update(dt);

        energy = 0.0;
        for (const auto& [speed, particle] : ranges::views::zip(speeds, particles))
        {
            speed = particle.vel.length();
            energy += 0.5 * particle.mass * speed * speed;
        }
    };

    const auto draw = [&]
    {
        // app.zoom_pan();

        auto frequencies = std::vector<f64>(static_cast<u64>(max_graph_speed / class_size + 1.0));

        for (auto speed : speeds)
        {
            speed = std::min(speed, max_graph_speed);
            frequencies.at(static_cast<u64>(speed / class_size)) += 1.0;
        }

        app.fill("#131417"_c);
        app.draw(App::GridLines{});
        for (const auto& particle : particles)
        {
            if (box.contains(particle.pos)) { app.draw(particle, {.fill_color = "#fc0330"_c}); }
        }

        auto points              = std::vector<Vector2>(frequencies.size());
        const auto max_frequency = ranges::max(frequencies);

        for (auto [i, frequency] : ranges::views::enumerate(frequencies))
        {
            const auto x = interp::map_range<u64, f64>(i, {0UL, frequencies.size()},
                                                       {-graph_width / 2.0, graph_width / 2.0});
            const auto y = interp::map_range<f64>(frequency, {0.0, max_frequency},
                                                  {-graph_height / 2.0, graph_height / 2.0});
            points[i]    = Vector2{x, y} + graph_centre;
        }
        app.draw_polyline(points, "#6179ff"_c, 0.2);

        fmt::print("\nfps: {}, energy: {:.5}, temperature: {:.4}", std::round(watch.current_fps()),
                   energy, energy / particles.size());
    };

    app.run(update, draw, 1);
}
