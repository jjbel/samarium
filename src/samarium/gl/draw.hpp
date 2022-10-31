/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <array> // for array,to_array
#include <cmath>
#include <numbers>
#include <span>   // for span
#include <vector> // for vector, allocator

#include "glad/glad.h"                     // for glDrawArrays, GL_COLOR_BUFFER...
#include "glm/mat4x4.hpp"                  // for mat4
#include "range/v3/algorithm/for_each.hpp" // for for_each
#include "range/v3/algorithm/minmax.hpp"   // for minmax
#include "range/v3/to_container.hpp"       // for to

#include "samarium/core/types.hpp"        // for f32, f64, u64, i32
#include "samarium/gl/Context.hpp"        // for Context
#include "samarium/gl/Shader.hpp"         // for Shader
#include "samarium/graphics/Color.hpp"    // for Color, ShapeColor
#include "samarium/graphics/Gradient.hpp" // for Gradient
#include "samarium/gui/Window.hpp"        // for Window
#include "samarium/math/BoundingBox.hpp"  // for BoundingBox
#include "samarium/math/Extents.hpp"      // for range
#include "samarium/math/Transform.hpp"    // for Transform
#include "samarium/math/Vector2.hpp"      // for Vector2f, Vector2_t, operator*
#include "samarium/math/interp.hpp"       // for lerp, lerp_inverse
#include "samarium/math/math.hpp"         // for two_pi
#include "samarium/math/shapes.hpp"       // for Circle, LineSegment
#include "samarium/util/Map.hpp"          // for Map
#include "samarium/util/format.hpp"       // for format

#include "gl.hpp" // for Buffer, VertexArray

namespace sm::draw
{
void vertices(gl::Context& context,
              std::span<gl::Vertex<gl::Layout::Pos>> verts,
              Color color,
              gl::Primitive primitive,
              const glm::mat4& transform);
void vertices(Window& window,
              std::span<gl::Vertex<gl::Layout::Pos>> verts,
              Color color,
              gl::Primitive primitive);

void vertices(gl::Context& context,
              std::span<Vector2f> verts,
              Color color,
              gl::Primitive primitive,
              const glm::mat4& transform);
void vertices(Window& window, std::span<Vector2f> verts, Color color, gl::Primitive primitive);

void vertices(gl::Context& context,
              std::span<gl::Vertex<gl::Layout::PosColor>> verts,
              gl::Primitive primitive,
              const glm::mat4& transform);
void vertices(Window& window,
              std::span<gl::Vertex<gl::Layout::PosColor>> verts,
              gl::Primitive primitive);

void circle(Window& window, Circle circle, ShapeColor color, const glm::mat4& transform);
void circle(Window& window, Circle circle_, ShapeColor color);

void line_segment(
    Window& window, LineSegment line, Color color, f32 thickness, const glm::mat4& transform);
void line_segment(Window& window, LineSegment line, Color color, f32 thickness = 0.01F);

void background(Color color);

/**
 * @brief               Fill the background with a gradient
 *
 * @tparam size
 * @param  window
 * @param  gradient
 * @param  angle
 */
template <u64 size> void background(Window& window, const Gradient<size>& gradient, f32 angle = 0.0)
{
    /*
    The technique is to rotate a box to just cover the diagonal of the screen,
    and then fill the box with a gradient by putting vertices on its edges

    This is done in screenspace coordinates which distort with the screen aspect ratio, so adjust
    the angle
    */

    // use Vector2f to preserve obtuse angles (atan2)
    angle = (Vector2f::from_polar({.length = 1.0, .angle = angle}) *
             window.aspect_vector_max().cast<f32>())
                .angle();

    const auto transformed_angle = angle + std::numbers::pi_v<f32> / 4;
    const auto max =
        std::numbers::sqrt2_v<f32> *
        ranges::max(math::abs(std::sin(transformed_angle)), math::abs(std::cos(transformed_angle)));

    auto verts = std::array<gl::Vertex<gl::Layout::PosColor>, size * 2>();

    for (auto i : range(size))
    {
        const auto factor =
            interp::lerp_inverse<f32>(static_cast<f64>(i), {0.0F, static_cast<f32>(size - 1)});

        verts[2 * i].pos =
            Vector2f{static_cast<f32>(interp::clamped_lerp<f32>(factor, {-max, max})), max}.rotated(
                angle);

        verts[2 * i + 1].pos =
            Vector2f{static_cast<f32>(interp::clamped_lerp<f32>(factor, {-max, max})), -max}
                .rotated(angle);

        verts[2 * i].color     = gradient.colors[i];
        verts[2 * i + 1].color = gradient.colors[i];
    }

    vertices(window, verts, gl::Primitive::TriangleStrip, glm::mat4{1.0F});
}

void polyline(Window& window,
              std::span<Vector2f> points,
              Color color,
              f32 thickness,
              const glm::mat4& transform);
void polyline(Window& window, std::span<Vector2f> points, Color color, f32 thickness);

void polygon(Window& window,
             std::span<Vector2f> points,
             ShapeColor color,
             const glm::mat4& transform);
void polygon(Window& window, std::span<Vector2f> points, ShapeColor color);

void regular_polygon(Window& window,
                     Vector2_t<f32> pos,
                     f32 radius,
                     u64 point_count,
                     ShapeColor color,
                     const glm::mat4& transform);
void regular_polygon(
    Window& window, Vector2_t<f32> pos, f32 radius, u64 point_count, ShapeColor color);
} // namespace sm::draw


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_DRAW_IMPL)

#include "samarium/core/inline.hpp"

#include "draw.hpp"
#include "gl.hpp"

namespace sm::draw
{
SM_INLINE void vertices(gl::Context& context,
                        std::span<gl::Vertex<gl::Layout::Pos>> verts,
                        Color color,
                        gl::Primitive primitive,
                        const glm::mat4& transform)
{
    const auto& shader = context.shaders.at("Pos");
    context.set_active(shader);
    shader.set("view", transform);
    shader.set("color", color);

    auto& vao = context.vertex_arrays.at("Pos");
    context.set_active(vao);

    const auto& buffer = context.vertex_buffers.at("default");
    buffer.set_data(verts);
    vao.bind(buffer, sizeof(verts[0]));

    glDrawArrays(static_cast<i32>(primitive), 0, static_cast<i32>(verts.size()));
}

SM_INLINE void vertices(Window& window,
                        std::span<gl::Vertex<gl::Layout::Pos>> verts,
                        Color color,
                        gl::Primitive primitive)
{
    vertices(window.context, verts, color, primitive, window.view);
}

SM_INLINE void vertices(gl::Context& context,
                        std::span<Vector2f> verts,
                        Color color,
                        gl::Primitive primitive,
                        const glm::mat4& transform)
{
    const auto& shader = context.shaders.at("Pos");
    context.set_active(shader);
    shader.set("view", transform);
    shader.set("color", color);

    auto& vao = context.vertex_arrays.at("Pos");
    context.set_active(vao);

    const auto& buffer = context.vertex_buffers.at("default");
    buffer.set_data(verts);
    vao.bind(buffer, sizeof(verts[0]));

    glDrawArrays(static_cast<i32>(primitive), 0, static_cast<i32>(verts.size()));
}

SM_INLINE void
vertices(Window& window, std::span<Vector2f> verts, Color color, gl::Primitive primitive)
{
    vertices(window.context, verts, color, primitive, window.view);
}

SM_INLINE void vertices(gl::Context& context,
                        std::span<gl::Vertex<gl::Layout::PosColor>> verts,
                        gl::Primitive primitive,
                        const glm::mat4& transform)
{
    const auto& shader = context.shaders.at("PosColor");
    context.set_active(shader);
    shader.set("view", transform);

    auto& vao = context.vertex_arrays.at("PosColor");
    context.set_active(vao);

    const auto& buffer = context.vertex_buffers.at("default");
    buffer.set_data(verts);
    vao.bind(buffer, sizeof(verts[0]));

    glDrawArrays(static_cast<i32>(primitive), 0, static_cast<i32>(verts.size()));
}

SM_INLINE void
vertices(Window& window, std::span<gl::Vertex<gl::Layout::PosColor>> verts, gl::Primitive primitive)
{
    vertices(window.context, verts, primitive, window.view);
}

SM_INLINE void background(Color color)
{
    glClearColor(static_cast<f32>(color.r) / 255.0F, static_cast<f32>(color.g) / 255.0F,
                 static_cast<f32>(color.b) / 255.0F, static_cast<f32>(color.a) / 255.0F);
    glClear(GL_COLOR_BUFFER_BIT);
}

SM_INLINE void polyline_impl(Window& window,
                             std::span<Vector2f> points,
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

    auto& buffer = window.context.shader_storage_buffers.at("default");
    buffer.set_data(points);
    buffer.bind();
    window.context.vertex_arrays.at("empty").bind();
    glDrawArrays(GL_TRIANGLES, 0, 6 * (static_cast<i32>(points.size()) - 3));
}

SM_INLINE void polyline(Window& window,
                        std::span<Vector2f> points,
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

SM_INLINE void polyline(Window& window, std::span<Vector2f> points, Color color, f32 thickness)
{
    polyline(window, points, color, thickness, window.view);
}

SM_INLINE void
polygon(Window& window, std::span<Vector2f> points, ShapeColor color, const glm::mat4& transform)
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

SM_INLINE void polygon(Window& window, std::span<Vector2f> points, ShapeColor color)
{
    polygon(window, points, color, window.view);
}

SM_INLINE void regular_polygon(Window& window,
                               Circle border_circle,
                               u64 point_count,
                               ShapeColor color,
                               const glm::mat4& transform)
{
    auto points = std::vector<Vector2_t<f32>>{};
    points.reserve(point_count);

    for (auto i = 0.0F; i < static_cast<f32>(math::two_pi);
         i += static_cast<f32>(math::two_pi) / static_cast<f32>(point_count))
    {
        points.push_back(Vector2_t<f32>::from_polar({static_cast<f32>(border_circle.radius), i}) +
                         border_circle.centre.cast<f32>());
    }
    polygon(window, points, color, transform);
}

SM_INLINE void
regular_polygon(Window& window, Circle border_circle, u64 point_count, ShapeColor color)
{
    regular_polygon(window, border_circle, point_count, color, window.view);
}

SM_INLINE void circle(Window& window, Circle circle, ShapeColor color, const glm::mat4& transform)
{
    regular_polygon(window, circle, 64, color, transform);
}

SM_INLINE void circle(Window& window, Circle circle_, ShapeColor color)
{
    circle(window, circle_, color, window.view);
}

SM_INLINE void line_segment(
    Window& window, LineSegment line, Color color, f32 thickness, const glm::mat4& transform)
{
    const auto thickness_vector =
        Vector2::from_polar(
            {.length = thickness / 2., .angle = line.vector().angle() + math::pi / 2.0})
            .cast<f32>();

    const auto first = line.p1.cast<f32>();
    const auto last  = line.p2.cast<f32>();
    auto points      = std::to_array<Vector2f>({first - thickness_vector, first + thickness_vector,
                                                last + thickness_vector, last - thickness_vector});
    polygon(window, points, {.fill_color = color});
}

SM_INLINE void line_segment(Window& window, LineSegment line, Color color, f32 thickness)
{
    line_segment(window, line, color, thickness, window.view);
}
} // namespace sm::draw

#endif
