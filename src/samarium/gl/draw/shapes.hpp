/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "samarium/graphics/Color.hpp" // for Color, ShapeColor
#include "samarium/gui/Window.hpp"     // for Window
#include "samarium/math/shapes.hpp"    // for LineSegment, Circle

namespace sm::draw
{
void circle(
    Window& window, Circle circle, Color color, const glm::mat4& transform, u32 point_count = 16);
void circle(Window& window, Circle circle_, Color color, u32 point_count = 16);

void circle(Window& window,
            Circle circle,
            ShapeColor color,
            const glm::mat4& transform,
            u32 point_count = 16);
void circle(Window& window, Circle circle_, ShapeColor color, u32 point_count = 16);

void line_segment(Window& window,
                  const LineSegment& line,
                  Color color,
                  f32 thickness,
                  const glm::mat4& transform);
void line_segment(Window& window, const LineSegment& line, Color color, f32 thickness = 0.02F);

void line(Window& window,
          const LineSegment& line,
          Color color,
          f32 thickness,
          const glm::mat4& transform);
void line(Window& window, const LineSegment& line_, Color color, f32 thickness = 0.02F);

void bounding_box(Window& window,
                  const BoundingBox<f64>& box,
                  Color color,
                  f32 thickness,
                  const glm::mat4& transform);
void bounding_box(Window& window, const BoundingBox<f64>& box, Color color, f32 thickness = 0.02F);
} // namespace sm::draw

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_DRAW_IMPL)

#include "samarium/core/inline.hpp"
#include "samarium/gl/draw/poly.hpp"
#include "samarium/gl/draw/shapes.hpp"

namespace sm::draw
{
    // TODO dwap transform and point_count order for uniformity
SM_INLINE void
circle(Window& window, Circle circle, Color color, const glm::mat4& transform, u32 point_count)
{
    regular_polygon(window, circle, point_count, color, transform);
}

SM_INLINE void circle(Window& window, Circle circle_, Color color, u32 point_count)
{
    circle(window, circle_, color, window.world2gl(), point_count);
}

SM_INLINE void
circle(Window& window, Circle circle, ShapeColor color, const glm::mat4& transform, u32 point_count)
{
    regular_polygon(window, circle, point_count, color, transform);
}

SM_INLINE void circle(Window& window, Circle circle_, ShapeColor color, u32 point_count)
{
    circle(window, circle_, color, window.world2gl(), point_count);
}


SM_INLINE void line_segment(
    Window& window, const LineSegment& line, Color color, f32 thickness, const glm::mat4& transform)
{
    const auto thickness_vector =
        Vector2::from_polar(
            {.length = thickness / 2., .angle = line.vector().angle() + math::pi / 2.0})
            .cast<f32>();

    const auto first = line.p1.cast<f32>();
    const auto last  = line.p2.cast<f32>();
    auto points      = std::to_array<Vector2f>({first - thickness_vector, first + thickness_vector,
                                                last + thickness_vector, last - thickness_vector});
    polygon(window, points, {.fill_color = color}, transform);
}

SM_INLINE void line_segment(Window& window, const LineSegment& line, Color color, f32 thickness)
{
    line_segment(window, line, color, thickness, window.world2gl());
}

SM_INLINE void bounding_box(Window& window,
                            const BoundingBox<f64>& box,
                            Color color,
                            f32 thickness,
                            const glm::mat4& transform)
{
    for (const auto& line : box.line_segments())
    {
        line_segment(window, line, color, thickness, transform);
    }
}

SM_INLINE void bounding_box(Window& window, const BoundingBox<f64>& box, Color color, f32 thickness)
{
    bounding_box(window, box, color, thickness, window.world2gl());
}

SM_INLINE void line(
    Window& window, const LineSegment& line, Color color, f32 thickness, const glm::mat4& transform)
{
    const auto midpoint = (line.p1 + line.p2) / 2.0;
    // const auto scale         = window.aspect_vector_max().length() * window.view.scale.length();
    // TODO calculate proper scale
    const auto scale         = 1000.0;
    const auto extended_line = LineSegment{(line.p1 - midpoint) * scale + midpoint,
                                           (line.p2 - midpoint) * scale + midpoint};
    line_segment(window, extended_line, color, thickness, transform);
}

SM_INLINE void line(Window& window, const LineSegment& line_, Color color, f32 thickness)
{
    line(window, line_, color, thickness, window.world2gl());
}
} // namespace sm::draw
#endif
