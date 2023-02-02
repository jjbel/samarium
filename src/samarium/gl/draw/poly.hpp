/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "samarium/core/types.hpp"       // for u32, f32
#include "samarium/gl/Context.hpp"       // for Context
#include "samarium/gl/draw/vertices.hpp" // for Buffer
#include "samarium/graphics/Color.hpp"   // for Color
#include "samarium/gui/Window.hpp"       // for Window
#include "samarium/math/Vector2.hpp"     // for Vector2f
#include "samarium/math/vector_math.hpp" // for regular_polygon_points

namespace sm::draw
{
void polyline(Window& window,
              std::span<const Vector2f> points,
              Color color,
              f32 thickness,
              const glm::mat4& transform);
void polyline(Window& window, std::span<const Vector2f> points, Color color, f32 thickness);

void polygon(Window& window,
             std::span<const Vector2f> points,
             ShapeColor color,
             const glm::mat4& transform);
void polygon(Window& window, std::span<const Vector2f> points, ShapeColor color);

void regular_polygon(Window& window,
                     Circle border_circle,
                     u32 point_count,
                     ShapeColor color,
                     const glm::mat4& transform);
void regular_polygon(Window& window, Circle border_circle, u32 point_count, ShapeColor color);
} // namespace sm::draw

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_DRAW_IMPL)

#include "samarium/core/inline.hpp"
#include "samarium/gl/draw/poly.hpp"
#include "samarium/math/vector_math.hpp"

namespace sm::draw
{
SM_INLINE void polyline_impl(Window& window,
                             std::span<const Vector2f> points,
                             Color color,
                             f32 thickness,
                             const glm::mat4& transform)
{
    const auto& shader = window.context.shaders.at("polyline");
    shader.bind();
    shader.set("thickness", thickness);
    shader.set("screen_dims", window.dims.cast<f64>());

    shader.set("view", transform);
    shader.set("color", color);

    const auto& buffer = window.context.shader_storage_buffers.at("default");
    buffer.set_data(points);
    buffer.bind();
    window.context.vertex_arrays.at("empty").bind();
    glDrawArrays(GL_TRIANGLES, 0, 6 * (static_cast<i32>(points.size()) - 3));
}

SM_INLINE void polyline(Window& window,
                        std::span<const Vector2f> points,
                        Color color,
                        f32 thickness,
                        const glm::mat4& transform)
{
    auto new_points = std::vector<Vector2f>{points.begin(), points.end()};
    new_points.insert(new_points.begin(), 2.0F * new_points[0] - new_points[1]);
    const auto last = points.back();
    new_points.push_back(2.0F * last - new_points[new_points.size() - 2]);
    polyline_impl(window, {new_points}, color, thickness, transform);
}

SM_INLINE void
polyline(Window& window, std::span<const Vector2f> points, Color color, f32 thickness)
{
    polyline(window, points, color, thickness, window.view);
}

SM_INLINE void polygon(Window& window,
                       std::span<const Vector2f> points,
                       ShapeColor color,
                       const glm::mat4& transform)
{
    if (color.fill_color.a != 0)
    {
        const auto& shader = window.context.shaders.at("Pos");
        window.context.set_active(shader);
        shader.set("view", transform);
        shader.set("color", color.fill_color);

        const auto& buffer = window.context.vertex_buffers.at("default");

        auto& vao = window.context.vertex_arrays.at("Pos");
        window.context.set_active(vao);
        buffer.set_data(points);
        vao.bind(buffer, sizeof(Vector2_t<f32>));

        glDrawArrays(GL_TRIANGLE_FAN, 0, static_cast<i32>(points.size()));
    }

    if (color.border_color.a != 0 && color.border_width != 0.0)
    {
        auto new_points = std::vector<Vector2f>{points.begin(), points.end()};
        new_points.insert(new_points.begin(), new_points.back());
        new_points.push_back(new_points[1]);
        new_points.push_back(new_points[2]);
        polyline_impl(window, {new_points}, color.border_color,
                      static_cast<f32>(color.border_width), transform);
    }
}

SM_INLINE void polygon(Window& window, std::span<const Vector2f> points, ShapeColor color)
{
    polygon(window, points, color, window.view);
}

SM_INLINE void regular_polygon(Window& window,
                               Circle border_circle,
                               u32 point_count,
                               ShapeColor color,
                               const glm::mat4& transform)
{
    const auto points = math::regular_polygon_points<f32>(point_count, border_circle);
    polygon(window, {points.begin(), point_count}, color, transform);
}

SM_INLINE void
regular_polygon(Window& window, Circle border_circle, u32 point_count, ShapeColor color)
{
    regular_polygon(window, border_circle, point_count, color, window.view);
}
} // namespace sm::draw
#endif
