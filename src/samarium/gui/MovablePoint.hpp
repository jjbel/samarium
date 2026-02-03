/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "samarium/gl/draw/shapes.hpp"

#include "Mouse.hpp"
#include "Window.hpp"

namespace sm
{
struct MovablePoint
{
    Vec2 pos{};
    Color color{255, 255, 255, 255};
    f64 major_radius = 0.8;
    f64 minor_radius = 0.4;
    bool is_hovered  = false;
    bool was_hovered = false;

    void draw(Window& window) const
    {
        draw::circle(window, {pos, major_radius}, color.with_multiplied_alpha(0.4));
        draw::circle(window, {pos, minor_radius}, color);
    }

    void update(const Mouse& mouse)
    {
        was_hovered = is_hovered;
        is_hovered  = math::distance(mouse.pos, pos) <= major_radius;
    }

    [[nodiscard]] auto entered() const { return is_hovered && !was_hovered; }

    [[nodiscard]] auto exited() const { return !is_hovered && was_hovered; }
};
} // namespace sm
