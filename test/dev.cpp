/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/gl/draw.hpp"
#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

#include "range/v3/algorithm/minmax_element.hpp"
#include "range/v3/view/concat.hpp"

using namespace sm;
using namespace sm::literals;

constexpr auto count           = 2000UL;
constexpr auto initial_speed   = 20.0;
constexpr auto class_size      = 4.0;
constexpr auto max_graph_speed = 60.0;
constexpr auto graph_width     = 80.0;
constexpr auto graph_height    = 40.0;
constexpr auto graph_centre    = Vector2{-37.5, 0.0};

int main()
{
    auto window = Window{{1600, 800}};
    auto rand   = RandomGenerator{count * 2, RandomMode::Stable, 73};
    auto particles =
        ParticleSystem<Particle<f64>, 64>{count, Particle<f64>{.radius = 0.15, .mass = 0.1}, 0.6};
    auto watch       = Stopwatch{};
    auto speeds      = std::vector<f64>(count);
    auto energy      = 0.0;
    auto frequencies = std::vector<f64>(static_cast<u64>(max_graph_speed / class_size + 1.0));

    auto box = window.viewport();
    box.set_width(box.width() / 2.0);
    print(box);
    // box.set_centre({app.transformed_dims().x / 4.0, 0.0});
    const auto walls = box.line_segments();

    for (auto& i : particles)
    {
        i.pos = rand.vector(box);
        // i.vel = rand.polar_vector({0, initial_speed});
        i.vel = rand.polar_vector({initial_speed, initial_speed + 0.001});
    }

    const auto update = [&]()
    {
        const auto dt = 1.0 / 600.0;
        particles.for_each(
            [&](Particle<f64>& particle)
            {
                for (const auto& wall : walls) { phys::collide(particle, wall, dt, 1.0); }
            });

        // particles.self_collision();
        // particles.update(dt);

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

        frequencies = std::vector<f64>(static_cast<u64>(max_graph_speed / class_size + 1.0));

        for (auto speed : speeds)
        {
            speed = std::min(speed, max_graph_speed);
            frequencies.at(static_cast<u64>(speed / class_size)) += 1.0;
        }

        draw::background("#131417"_c);
        draw::grid_lines(window);
        for (const auto& i : particles)
        {
            draw::circle(window, i.as_circle(), {.fill_color = "#fc0330"_c});
        }

        auto points              = std::vector<Vector2>(frequencies.size());
        const auto max_frequency = ranges::max(frequencies);
        for (auto [i, frequency] : ranges::views::enumerate(frequencies))
        {
            const auto x = interp::map_range<f64>(i, {0.0, static_cast<f64>(frequencies.size())},
                                                  {-graph_width / 2.0, graph_width / 2.0});
            const auto y = interp::map_range<f64>(frequency, {0.0, max_frequency},
                                                  {-graph_height / 2.0, graph_height / 2.0});
            points[i]    = Vector2{x, y} + graph_centre;
        }
        // draw::polyline(window, points, "#6179ff"_c, 0.2);


        // fmt::print("\nfps: {}, energy: {:.5}, speeds: ", std::round(watch.current_fps()),
        // energy);

        // for (auto i : frequencies) { fmt::print("{:.3}, ", i); }

        // for (auto i : speeds) { fmt::print("{:.3}, ", i); }
    };

    run(window, update, draw, 1);
}
