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
    bool is_hovered  = false;
    bool was_hovered = false;

    void draw(Window& window) const
    {
        draw::circle(window, {pos, major_radius}, {.fill_color = color.with_multiplied_alpha(0.4)});
        draw::circle(window, {pos, minor_radius}, {.fill_color = color});
    }

    void update(const Mouse& mouse)
    {
        was_hovered = is_hovered;
        is_hovered  = math::distance(mouse.pos, pos) <= major_radius;
    }

    [[nodiscard]] auto entered() const { return is_hovered && !was_hovered; }

    [[nodiscard]] auto exited() const { return !is_hovered && was_hovered; }
};

auto main() -> i32
{
    auto window       = Window{{{1800, 900}}};
    window.view.scale = Vector2::combine(1.0 / 10.0);
    auto point        = MovablePoint{{1.0, 1.0}, "#4e22ff"_c};
    auto r            = 0.4;
    //     auto handler      = anim::Handler{};
    auto action       = anim::Action{};
    const auto update = [&]
    {
        point.update(window.mouse);
        //        if (point.entered()) { r = 2; }
        //        else if (point.exited()) { r = 6; }
        //        else { r = ; }
    };

    const auto draw = [&]
    {
        draw::background("#131417"_c);
        draw::grid_lines(window, {.spacing = 1, .color{255, 255, 255, 90}, .thickness = 0.028F});
        draw::grid_lines(window,
                         {.spacing = 1.0 / 3.0, .color{255, 255, 255, 40}, .thickness = 0.019F});
        draw::circle(window, {{0.0, 0.0}, r}, {.fill_color = "#ff0e4e"_c});

        point.draw(window);
    };

    run(window, update, draw);
}
