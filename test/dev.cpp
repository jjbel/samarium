/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

#include "range/v3/algorithm/max_element.hpp"
#include "range/v3/view/take.hpp"

using namespace sm;
using namespace sm::literals;

static constexpr auto initial_speed = 10.0;
static constexpr auto class_size    = 0.3;
static constexpr auto max_speed     = initial_speed * 3.0;
static constexpr auto plot_scale    = 0.9;

int main()
{
    auto rand = RandomGenerator{};

    auto app            = App{{.dims{1920, 1080}}};
    const auto viewport = app.transformed_bounding_box();
    auto half_viewport  = BoundingBox<f64>{.min = viewport.min, .max = {0.0, viewport.max.y}};

    auto ps = ParticleSystem::generate(
        2000,
        [&](u64 /* index */)
        {
            auto particle =
                Particle{.pos    = rand.vector(half_viewport),
                         .vel    = rand.polar_vector({0.0, max_speed}),
                         .radius = 0.2,
                         .mass   = 1.0};
            return particle;
        });

    const auto walls = viewport.line_segments();

    const auto update = [&](f64 dt)
    {
        ps.self_collision();
        for (auto& particle : ps)
        {
            for (const auto& wall : walls) { phys::collide(particle, wall, dt); }
        }
        ps.update(dt);
    };

    auto watch      = Stopwatch{};
    const auto draw = [&]
    {
        app.fill("#0e1117"_c);
        app.draw(App::GridLines{});

        for (const auto& particle : ps)
        {
            if (half_viewport.contains(particle.pos))
            {
                app.draw(particle, {.fill_color = "#ff150d"_c});
            }
        }

        const auto speed_data =
            ps | ranges::views::transform(
                     [](const auto& particle)
                     {
                         auto length = particle.vel.length();
                         length      = interp::clamp(length, {0.0, max_speed});
                         return static_cast<u64>(std::round(length / class_size));
                     });

        auto frequencies = std::vector<u16>(static_cast<u64>(max_speed / class_size));

        for (auto speed : speed_data) { frequencies.at(std::min(speed, frequencies.size() - 1))++; }
        const auto data_size     = frequencies.size();
        const auto max_frequency = ranges::max(frequencies);
        auto points              = std::vector<Vector2>(data_size);
        for (auto i : range(data_size))
        {
            const auto x = interp::map_range<u64, f64>(i, {0UL, data_size},
                                                       {0.0, viewport.max.x * plot_scale});
            const auto y = interp::map_range<f64, f64>(
                static_cast<f64>(frequencies[i]), {0.0, static_cast<f64>(max_frequency)},
                {viewport.min.y * plot_scale, viewport.max.y * plot_scale});

            points[i] = Vector2{x, y};
        }
        app.draw_polyline(points, "#0352fc"_c, 0.12);

        print(std::round(watch.current_fps()));
    };

    app.run(update, draw, 1);
}
