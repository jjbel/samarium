/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

// TODO this still produces uneven results
// think about pushing each disk slightly, maybe on avg away from its neighbours

auto main() -> i32
{
    auto window = Window{{.dims = dims720}};

    // made error: mapped to window dims, which stretched it's aspect ratio, and made some discs
    // intersect
    // TODO some still intersecting

    // TODO with {40,30},10,2 some go out of window

    // default: window dims, but cud be anything
    // const auto region  = window.dims.cast<f64>();
    const auto region  = Vec2{1200, 700};
    const auto samples = 40UL;
    const auto radius  = 20.0;
    auto rand          = RandomGenerator{4096, RandomMode::Stable, /*seed*/ 41};
    auto points        = poisson_disc::uniform(rand, radius * 2.0, {{}, region}, samples);

    // TODO VVIMP work in pixel space
    // TODO this gives correct mouse pos, but drawing circles is in 1st quadrant only
    // see TODO below
    window.view = Transform{{-1.0, -1.0}, {2.0 / window.dims.x, 2.0 / window.dims.y}};

    // ideally use nC2 not n^2
    auto distances = std::vector<f64>();
    distances.reserve(points.size() * points.size());
    auto count1 = 0;
    auto count2 = 0;
    for (auto a : points)
    {
        for (auto b : points)
        {
            if (a != b)
            {
                const auto distance = math::distance(a, b);
                distances.push_back(distance);
                if (distance > 0 && distance < radius) { count1++; }
                if (distance > 0 && distance < 2 * radius) { count2++; }
            }
        }
    }
    std::sort(distances.begin(), distances.end());
    print("distances:", distances[0], distances[1], distances[distances.size() - 1]);
    print("total:", points.size() * points.size(), "\nd < radius:", count1,
          "\nd < 2 radius:", count2);

    run(window,
        [&]
        {
            draw::background("#15151f"_c);
            // TODO auto or const auto&
            for (auto i : loop::end(points.size()))
            {
                const auto& point = points[i];
                const auto c =
                    static_cast<u8>(static_cast<f64>(i) / static_cast<f64>(points.size()) * 255.0);
                const auto color = Color{c, c, c};
                // const auto color = "#ff0042"_c;

                // TODO -region/2
                draw::circle(window, Circle{.centre = point - region / 2.0, .radius = radius},
                             color);
            }
        });
}
