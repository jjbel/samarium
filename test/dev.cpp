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
    auto watch = Stopwatch{};

    const auto circle = [&](Vector2 pos)
    { draw::circle(window, {.centre = pos, .radius = .5}, {.fill_color = colors::red}); };

    auto counter = 0;
    auto total   = 0.0;

    while (window.is_open())
    {
        draw::background("#101113"_c);

        circle({});

        watch.reset();
        draw::grid_dots(window);
        total += watch.seconds();
        counter++;


        const auto vec =
            window.view.apply_inverse(interp::map_range<Vector2>(
                window.mouse.pos, {{0, 0}, window.dims.cast<f64>()}, {{-1, -1}, {1, 1}})) *
            Vector2{1, -1};


        draw::line(window, {vec, {2.0, 1.0}}, colors::goldenrod, 0.1);
        window.display();

        if (counter > 400) { break; }
    }

    fmt::print("Average: {:3.3f}ms\n", total / counter * 1000.0);
}
