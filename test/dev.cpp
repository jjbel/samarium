/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

struct MovablePoint
{
    Vector2 pos{};
    Color color{255, 255, 255, 255};
    f64 major_radius = 0.8;
    f64 minor_radius = 0.4;

    void draw(Window& window) const
    {
        draw::circle(window, {pos, major_radius}, {.fill_color = color.with_multiplied_alpha(0.4)});
        draw::circle(window, {pos, minor_radius}, {.fill_color = color});
    }
};

auto main() -> i32
{
    auto window       = Window{{{400, 400}}};
    window.view.scale = Vector2::combine(1.0 / 5.0);
    auto point        = MovablePoint{{1.0, 1.0}, "#4e22ff"_c};

    const auto update = [&]
    {
        const auto pos = window.mouse.pos;
        if (math::distance(pos, point.pos) <= point.major_radius) { print("Hello"); }
    };

    const auto draw = [&]
    {
        draw::background("#131417"_c);
        draw::grid_lines(window, {.spacing = 1, .color{255, 255, 255, 60}, .thickness = 0.03});
        draw::circle(window, {{0.0, 0.0}, 0.4}, {.fill_color = "#ff0e4e"_c});

        point.draw(window);
    };

    run(window, update, draw);
}
