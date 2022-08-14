/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

#include "range/v3/algorithm/max_element.hpp"
#include "range/v3/numeric/accumulate.hpp"
#include "range/v3/view/drop.hpp"
#include "range/v3/view/take.hpp"

using namespace sm;
using namespace sm::literals;

static constexpr auto initial_speed = 10.0;
static constexpr auto class_size    = 0.3;
static constexpr auto max_speed     = initial_speed * 3.0;
static constexpr auto plot_scale    = 0.9;

template <typename T> inline auto moving_average(const std::vector<T>& data, u64 window_size)
{
    // if (data.size() <= window_size)
    // {
    //     return std::vector<f64>{{ranges::accumulate(data, 0) / static_cast<T>(data.size())}};
    // }

    auto result = std::vector<f64>(data.size() - window_size + 1);
    for (auto i : range(result.size()))
    {
        auto sub_range = data | ranges::views::drop(i) | ranges::views::take(window_size);
        result[i]      = ranges::accumulate(sub_range, 0) / static_cast<T>(window_size);
    }
    return result;
}

int main()
{
    auto rand = RandomGenerator{};

    auto app            = App{{.dims{1920, 1080}}};
    const auto viewport = app.transformed_bounding_box();
    auto half_viewport  = BoundingBox<f64>{.min = viewport.min, .max = {0.0, viewport.max.y}};

    auto ps = ParticleSystem::generate(
        500,
        [&](u64 /* index */)
        {
            auto particle =
                Particle{.pos    = rand.vector(half_viewport),
                         .vel    = rand.polar_vector({initial_speed, initial_speed + 0.00001}),
                         .radius = .2,
                         .mass   = 10};
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
        const auto smoothed_frequencies = moving_average(frequencies, 6);

        const auto data_size     = smoothed_frequencies.size();
        const auto max_frequency = ranges::max(smoothed_frequencies);

        auto points = std::vector<Vector2>(data_size);
        for (auto i : range(data_size))
        {
            const auto x = interp::map_range<u64, f64>(i, {0UL, data_size},
                                                       {0.0, viewport.max.x * plot_scale});
            const auto y = interp::map_range<f64, f64>(
                smoothed_frequencies[i], {0.0, max_frequency},
                {viewport.min.y * plot_scale, viewport.max.y * plot_scale});

            points[i] = Vector2{x, y};
        }
        app.draw_polyline(points, "#0352fc"_c, 0.12);

        print(std::round(watch.current_fps()));
    };

    app.run(update, draw, 2);
}
