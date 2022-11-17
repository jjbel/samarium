/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/lite.hpp"

#include "samarium/graphics/colors.hpp"
#include "samarium/util/Stopwatch.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto window = Window{{1800, 900}};
    window.view.scale /= 10.0;
    auto watch       = Stopwatch{};
    const auto scale = window.aspect_vector_min().length() * window.view.scale.length();
    print(scale, window.aspect_vector_max(), window.view.scale);

    const auto circle = [&](Vector2 pos)
    { draw::circle(window, {.centre = pos, .radius = .15}, {.fill_color = colors::red}); };

    while (window.is_open())
    {
        watch.reset();
        draw::background("#101113"_c);

        circle({});
        draw::grid_dots(window);

        const auto vec =
            window.view.apply_inverse(interp::map_range<Vector2>(
                window.mouse.pos, {{0, 0}, window.dims.cast<f64>()}, {{-1, -1}, {1, 1}})) *
            Vector2{1, -1};


        draw::line(window, {vec, {2.0, 1.0}}, colors::goldenrod, 0.1);
        window.display();
    }
}
