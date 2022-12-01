/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "range/v3/view/linear_distribute.hpp"

#include "samarium/graphics/Color.hpp" // for Color
#include "samarium/gui/Window.hpp"     // for Window
#include "samarium/math/Vector2.hpp"   // for Vector2f

namespace sm::draw
{
struct GridLines
{
    f64 spacing   = 1.0;
    Color color   = Color{200, 200, 200, 50};
    f32 thickness = 0.02F;
};
void grid_lines(Window& window, const GridLines& config = {});

struct GridDots
{
    f64 spacing     = 1.0;
    Color color     = Color{200, 200, 200, 50};
    f32 thickness   = 0.03F;
    u32 point_count = 4;
};
void grid_dots(Window& window, const GridDots& config = {});
} // namespace sm::draw

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_DRAW_IMPL)

#include "samarium/core/inline.hpp"
#include "samarium/gl/draw/grid.hpp"
#include "samarium/gl/draw/poly.hpp"
#include "samarium/gl/draw/shapes.hpp"

namespace sm::draw
{

SM_INLINE void grid_lines(Window& window, const GridLines& config)
{
    auto [x_max, y_max] = window.view.apply_inverse(Vector2{1.0, 1.0});
    auto [x_min, y_min] = window.view.apply_inverse(Vector2{-1.0, -1.0});
    x_min               = math::floor_to_nearest(x_min, config.spacing);
    y_min               = math::floor_to_nearest(y_min, config.spacing);
    x_max               = math::ceil_to_nearest(x_max, config.spacing);
    y_max               = math::ceil_to_nearest(y_max, config.spacing);

    for (auto i : ranges::views::linear_distribute(
             x_min, x_max, static_cast<i64>((x_max - x_min) / config.spacing)))
    {
        draw::line(window, {{i, 0.0}, {i, 1.0}}, config.color, config.thickness);
    }

    for (auto i : ranges::views::linear_distribute(
             y_min, y_max, static_cast<i64>((y_max - y_min) / config.spacing)))
    {
        draw::line(window, {{0.0, i}, {1.0, i}}, config.color, config.thickness);
    }
};

SM_INLINE void grid_dots(Window& window, const GridDots& config)
{
    auto [x_max, y_max] = window.view.apply_inverse(Vector2{1.0, 1.0});
    auto [x_min, y_min] = window.view.apply_inverse(Vector2{-1.0, -1.0});
    x_min               = math::floor_to_nearest(x_min, config.spacing);
    y_min               = math::floor_to_nearest(y_min, config.spacing);
    x_max               = math::ceil_to_nearest(x_max, config.spacing);
    y_max               = math::ceil_to_nearest(y_max, config.spacing);

    for (auto i : ranges::views::linear_distribute(
             x_min, x_max, static_cast<i64>((x_max - x_min) / config.spacing)))
    {
        for (auto j : ranges::views::linear_distribute(
                 y_min, y_max, static_cast<i64>((y_max - y_min) / config.spacing)))
        {
            draw::regular_polygon(window, {{i, j}, config.thickness}, config.point_count,
                                  {.fill_color = config.color});
        }
    }
};
} // namespace sm::draw
#endif
