/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "samarium/gl/draw/poly.hpp"
#include "samarium/graphics/Color.hpp" // for Color, ShapeColor
#include "samarium/gui/Window.hpp"     // for Window
#include "samarium/math/shapes.hpp"    // for LineSegment, Circle


namespace sm::draw
{
// TODO swap transform and point_count order for uniformity
inline void
circle(Window& window, Circle circle, Color color, const glm::mat4& transform, u32 point_count = 16)
{
    regular_polygon(window, circle, point_count, color, transform);
}

inline void circle(Window& window, Circle circle_, Color color, u32 point_count = 16)
{
    circle(window, circle_, color, window.world2gl(), point_count);
}

// inline void circle(Window& window,
//                    Circle circle,
//                    ShapeColor color,
//                    const glm::mat4& transform,
//                    u32 point_count = 16)
// {
//     regular_polygon(window, circle, point_count, color, transform);
// }

// inline void circle(Window& window, Circle circle_, ShapeColor color, u32 point_count = 16)
// {
//     circle(window, circle_, color, window.world2gl(), point_count);
// }

inline void line_segment(
    Window& window, const LineSegment& line, Color color, f32 thickness, const glm::mat4& transform)
{
    const auto thickness_vector =
        Vec2::from_polar(
            {.length = thickness / 2., .angle = line.vector().angle() + math::pi / 2.0})
            .cast<f32>();

    const auto first = line.p1.cast<f32>();
    const auto last  = line.p2.cast<f32>();
    auto points      = std::to_array<Vec2f>({first - thickness_vector, first + thickness_vector,
                                             last + thickness_vector, last - thickness_vector});
    polygon(window, points, color, transform);
}

inline void
line_segment(Window& window, const LineSegment& line, Color color, f32 thickness = 0.02F)
{
    line_segment(window, line, color, thickness, window.world2gl());
}

inline void line(
    Window& window, const LineSegment& line, Color color, f32 thickness, const glm::mat4& transform)
{
    const auto midpoint = (line.p1 + line.p2) / 2.0;
    // const auto scale         = window.aspect_vector_max().length() * window.view.scale.length();
    // TODO calculate proper scale. see grid
    const auto scale         = 1000.0;
    const auto extended_line = LineSegment{(line.p1 - midpoint) * scale + midpoint,
                                           (line.p2 - midpoint) * scale + midpoint};
    line_segment(window, extended_line, color, thickness, transform);
}
inline void line(Window& window, const LineSegment& line_, Color color, f32 thickness = 0.02F)
{
    line(window, line_, color, thickness, window.world2gl());
}

inline void bounding_box(
    Window& window, const Box2<f64>& box, Color color, f32 thickness, const glm::mat4& transform)
{
    for (const auto& line : box.line_segments())
    {
        line_segment(window, line, color, thickness, transform);
    }
}
inline void bounding_box(Window& window, const Box2<f64>& box, Color color, f32 thickness = 0.02F)
{
    bounding_box(window, box, color, thickness, window.world2gl());
}

// quick and dirty
inline void polyline_segments(Window& window,
                              std::span<const Vec2f> points,
                              f32 thickness,
                              Color color,
                              const glm::mat4& transform)
{
    // TODO assumes pts >= 2
    for (auto i : loop::end(points.size() - 1UL))
    {
        draw::line_segment(window, {points[i].cast<f64>(), points[i + 1UL].cast<f64>()}, color,
                           thickness, transform);
    }
}

inline void
polyline_segments(Window& window, std::span<const Vec2f> points, f32 thickness, Color color)
{
    polyline_segments(window, points, thickness, color, window.world2gl());
}

inline void
polygon_segments(Window& window, std::span<const Vec2f> points, f32 thickness, Color color)
{
    polyline_segments(window, points, thickness, color);
    draw::line_segment(window, {points[points.size() - 1UL].cast<f64>(), points[0].cast<f64>()},
                       color, thickness);
}
} // namespace sm::draw
