/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
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
// TODO add simple and fast polyline versions, which just draw rectangle or whatev
// TODO cud add a bias from -1 to 1 to offset it inward or outward

// return points for a triangle strip
auto make_polyline(std::span<const Vector2f> in_pts, f32 thickness) -> std::vector<Vector2f>;

// void polyline(Window& window,
//               std::span<const Vector2f> points,
//               Color color,
//               f32 thickness,
//               const glm::mat4& transform);
void polyline(Window& window, std::span<const Vector2f> points, f32 thickness, Color color);

void polygon(Window& window,
             std::span<const Vector2f> points,
             Color color,
             const glm::mat4& transform);
void polygon(Window& window, std::span<const Vector2f> points, Color color);

void regular_polygon(
    Window& window, Circle border_circle, u32 point_count, Color color, const glm::mat4& transform);
void regular_polygon(Window& window, Circle border_circle, u32 point_count, Color color);

// SHAPECOLOR stuf todo -----------------------------------------------------
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
#include "samarium/math/loop.hpp" // for start_end
#include "samarium/math/vector_math.hpp"


namespace sm::draw
{
constexpr auto sign_of(f32 x)
{
    // assumes x != 0
    return static_cast<f32>(x > 0) * 2.0F - 1.0F;
}

// which side of v1 does v0 lie on?
constexpr auto which_side_of(Vector2f v0, Vector2f v1)
{
    return sign_of(v1.y * v0.x - v0.y * v1.x);
}

SM_INLINE
auto make_polyline(std::span<const Vector2f> in_pts, f32 thickness) -> std::vector<Vector2f>
{
    const auto count = in_pts.size();

    if (count == 1) { return {}; }

    auto out_pts = std::vector<Vector2f>(count * 2);

    auto edge  = (in_pts[1] - in_pts[0]).normalized();
    auto disp  = edge.rotated(static_cast<f32>(math::pi / 2.0)) * thickness;
    out_pts[0] = in_pts[0] + disp;
    out_pts[1] = in_pts[0] - disp;

    for (auto i : loop::start_end(u64(1), count - u64(1)))
    {
        // TODO overoptimized?
        // okay to split it up into many variables?
        const auto new_edge  = (in_pts[i + 1] - in_pts[i]).normalized();
        edge                 = -edge; // reversing later calcns simpler
        const auto cos       = Vector2f::dot(edge, new_edge);
        const auto sin_theta = std::sqrt(1.0F - cos * cos);
        auto new_disp        = thickness * (new_edge + edge) / sin_theta;
        // TODO link the onenote diag here

        // TODO explanation
        // GL_TRIANGLE_STRIP need the vertices to e specified in zig zag order
        new_disp *= which_side_of(disp, edge) * which_side_of(new_disp, edge);

        out_pts[2 * i]     = in_pts[i] + new_disp;
        out_pts[2 * i + 1] = in_pts[i] - new_disp;

        edge = new_edge;
        disp = new_disp;
    }

    disp                         = edge.rotated(static_cast<f32>(math::pi / 2.0)) * thickness;
    out_pts[2 * (count - 1)]     = in_pts[count - 1] + disp;
    out_pts[2 * (count - 1) + 1] = in_pts[count - 1] - disp;

    return out_pts;
}

SM_INLINE void
polyline(Window& window, std::span<const Vector2f> in_pts, f32 thickness, Color color)
{
    const auto out_pts = make_polyline(in_pts, thickness);

    const auto& shader = window.context.shaders.at("Pos");
    window.context.set_active(shader);

    shader.set("view", window.world2gl());
    shader.set("color", color);

    const auto& buffer = window.context.vertex_buffers.at("default");

    auto& vao = window.context.vertex_arrays.at("Pos");
    window.context.set_active(vao);
    buffer.set_data(out_pts);
    vao.bind(buffer, sizeof(Vector2_t<f32>));

    glDrawArrays(GL_TRIANGLE_STRIP, 0, static_cast<i32>(out_pts.size()));
}

// SM_INLINE void polyline_impl(Window& window,
//                              std::span<const Vector2f> points,
//                              Color color,
//                              f32 thickness,
//                              const glm::mat4& transform)
// {
//     const auto& shader = window.context.shaders.at("polyline");
//     shader.bind();
//     shader.set("thickness", thickness);
//     shader.set("screen_dims", window.dims.cast<f64>());

//     shader.set("view", transform);
//     shader.set("color", color);

//     const auto& buffer = window.context.shader_storage_buffers.at("default");
//     buffer.set_data(points);
//     buffer.bind();
//     window.context.vertex_arrays.at("empty").bind();
//     glDrawArrays(GL_TRIANGLES, 0, 6 * (static_cast<i32>(points.size()) - 3));
// }

// SM_INLINE void polyline(Window& window,
//                         std::span<const Vector2f> points,
//                         Color color,
//                         f32 thickness,
//                         const glm::mat4& transform)
// {
//     auto new_points = std::vector<Vector2f>{points.begin(), points.end()};
//     new_points.insert(new_points.begin(), 2.0F * new_points[0] - new_points[1]);
//     const auto last = points.back();
//     new_points.push_back(2.0F * last - new_points[new_points.size() - 2]);
//     polyline_impl(window, {new_points}, color, thickness, transform);
// }

// SM_INLINE void
// polyline(Window& window, std::span<const Vector2f> points, Color color, f32 thickness)
// {
//     polyline(window, points, color, thickness, window.world2gl());
// }

SM_INLINE void polygon(Window& window,
                       std::span<const Vector2f> points,
                       Color fill_color,
                       const glm::mat4& transform)
{
    // TODO assumes clockwise/anticlockwise?

    const auto& shader = window.context.shaders.at("Pos");
    window.context.set_active(shader);

    shader.set("view", transform);
    shader.set("color", fill_color);

    const auto& buffer = window.context.vertex_buffers.at("default");

    auto& vao = window.context.vertex_arrays.at("Pos");
    window.context.set_active(vao);
    buffer.set_data(points);
    vao.bind(buffer, sizeof(Vector2_t<f32>));

    // TODO this requires convex...we should use triangle strip...
    glDrawArrays(GL_TRIANGLE_FAN, 0, static_cast<i32>(points.size()));
}

SM_INLINE void polygon(Window& window, std::span<const Vector2f> points, Color color)
{
    polygon(window, points, color, window.world2gl());
}

SM_INLINE void regular_polygon(
    Window& window, Circle border_circle, u32 point_count, Color color, const glm::mat4& transform)
{
    // TODO cache this. its allocated millions of times
    const auto points = math::regular_polygon_points<f32>(point_count, border_circle);
    polygon(window, {points.begin(), point_count}, color, transform);
}

SM_INLINE void regular_polygon(Window& window, Circle border_circle, u32 point_count, Color color)
{
    regular_polygon(window, border_circle, point_count, color, window.world2gl());
}

// TODO SHAPECOLOR STUFF -------------------------------------------------------------------------

SM_INLINE void polygon(Window& window,
                       std::span<const Vector2f> points,
                       ShapeColor color,
                       const glm::mat4& transform)
{
    if (color.fill_color.a != 0)
    {
        // TODO
    }

    if (color.border_color.a != 0 && color.border_width != 0.0)
    {
        auto new_points = std::vector<Vector2f>{points.begin(), points.end()};
        new_points.insert(new_points.begin(), new_points.back());
        new_points.push_back(new_points[1]);
        new_points.push_back(new_points[2]);
        // polyline_impl(window, {new_points}, color.border_color,
        //   static_cast<f32>(color.border_width), transform);
    }
}

SM_INLINE void polygon(Window& window, std::span<const Vector2f> points, ShapeColor color)
{
    polygon(window, points, color, window.world2gl());
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
    regular_polygon(window, border_circle, point_count, color, window.world2gl());
}
} // namespace sm::draw
#endif
