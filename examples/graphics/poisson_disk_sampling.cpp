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

auto main() -> i32
{
    auto window = Window{{.dims = dims720}};

    const auto region  = Vector2{40, 30};
    const auto samples = 10UL;
    const auto radius  = 1.0;
    auto rand          = RandomGenerator{};
    auto points        = rand.poisson_disc_points(radius, {region}, samples);

    auto box = window.viewport();

    auto mapper =
        interp::make_mapper<Vector2>({{}, region}, {box.min.cast<f64>(), box.max.cast<f64>()});

    run(window,
        [&]
        {
            app.fill("#15151f"_c);
            for (auto point : points)
            {
                draw::circle(window, Circle{.centre = mapper(point), .radius = radius},
                             {.fill_color = "#ff1438"_c});
            }
        });
}
