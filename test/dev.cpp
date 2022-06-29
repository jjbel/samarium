/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

#include "range/v3/view/enumerate.hpp"

using namespace sm;
using namespace sm::literals;

int main()
{
    auto app = App{{.dims{1600, 800}}};
    const auto count = 200;
    const auto size  = Vector2{70.0, 30.0};

    auto plot = std::vector<Vector2>(count);

    auto watch = Stopwatch{};

    auto rand  = RandomGenerator{};
    auto noise = util::PerlinNoise{};

    for (auto [i, value] : ranges::views::enumerate(plot))
    {
        value.x = interp::map_range<u64, f64>(i, {0, count}, {-size.x, size.x});
        value.y = interp::map_range<f64, f64>(noise(value.x / 4, 0.0), {0.0, 1.0}, {0.0, size.y});
    }

    const auto draw = [&]
    {
        app.fill("#16161c"_c);
        app.draw_polyline(plot, "#4542ff"_c, 0.1);
    };
    app.run(draw);
}
