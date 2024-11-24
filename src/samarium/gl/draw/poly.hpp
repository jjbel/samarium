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
#include "samarium/math/Vec2.hpp"        // for Vec2f
#include "samarium/math/loop.hpp"        // for start_end
#include "samarium/math/vector_math.hpp" // for regular_polygon_points


namespace sm
{
inline auto points_to_f32(std::span<const Vec2> in_pts) -> std::vector<Vec2f>
{
    auto out_pts = std::vector<Vec2f>(in_pts.size());
    for (auto i : loop::end(in_pts.size())) { out_pts[i] = in_pts[i].template cast<f32>(); }
    return out_pts;
}
} // namespace sm

namespace sm::draw
{
namespace detail
{
constexpr auto sign_of(f32 x)
{
    // assumes x != 0
    return static_cast<f32>(x > 0) * 2.0F - 1.0F;
}

// which side of v1 does v0 lie on?
constexpr auto which_side_of(Vec2f v0, Vec2f v1)
{
    return detail::sign_of(v1.y * v0.x - v0.y * v1.x);
}
} // namespace detail

// TODO cud add a bias from -1 to 1 to offset it inward or outward
inline auto make_polyline(std::span<const Vec2f> in_pts, f32 thickness) -> std::vector<Vec2f>
{
    // TODO perhaps almost parallel edges are a problem

    const auto count = in_pts.size();

    if (count <= 2) { return {}; }

    auto out_pts = std::vector<Vec2f>(count * 2);

    auto edge  = (in_pts[1] - in_pts[0]).normalized();
    auto disp  = edge.rotated(static_cast<f32>(math::pi / 2.0)) * thickness;
    out_pts[0] = in_pts[0] + disp;
    out_pts[1] = in_pts[0] - disp;

    for (auto i : loop::start_end(u64(1), count - u64(1)))
    {
        // TODO overoptimized?
        // okay to split it up into many variables?
        const auto new_edge = (in_pts[i + 1] - in_pts[i]).normalized();
        edge                = -edge; // reversing later calcns simpler
        const auto cos      = Vec2f::dot(edge, new_edge);
        // TODO link the onenote diag here
        // GL_TRIANGLE_STRIP need the vertices to e specified in zig zag order
        auto new_disp          = Vec2f{};
        constexpr auto epsilon = 1e-4F;

        // if antiparallel
        // TODO cud also handle parallel
        if (cos + 1 < epsilon) { new_disp = disp; }
        else
        {
            const auto sin_theta = std::sqrt(1.0F - cos * cos);
            new_disp             = thickness * (new_edge + edge) / /* std::abs */ (sin_theta);
            new_disp *= detail::which_side_of(disp, edge) * detail::which_side_of(new_disp, edge);
        }

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

inline void polyline(Window& window,
                     std::span<const Vec2f> in_pts,
                     f32 thickness,
                     Color color,
                     const glm::mat4& transform)
{
    const auto out_pts = make_polyline(in_pts, thickness);

    const auto& shader = window.context.shaders.at("Pos");
    window.context.set_active(shader);

    shader.set("view", transform);
    shader.set("color", color);

    const auto& buffer = window.context.vertex_buffers.at("default");

    auto& vao = window.context.vertex_arrays.at("Pos");
    window.context.set_active(vao);
    buffer.set_data(out_pts);
    vao.bind(buffer, sizeof(Vec2f));

    glDrawArrays(GL_TRIANGLE_STRIP, 0, static_cast<i32>(out_pts.size()));
}

inline void polyline(Window& window, std::span<const Vec2f> in_pts, f32 thickness, Color color)
{
    polyline(window, in_pts, thickness, color, window.world2gl());
}

inline void polyline(Window& window, std::span<const Vec2> in_pts, f32 thickness, Color color)
{
    polyline(window, points_to_f32(in_pts), thickness, color, window.world2gl());
}

inline void
polygon(Window& window, std::span<const Vec2f> points, Color fill_color, const glm::mat4& transform)
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
    vao.bind(buffer, sizeof(Vec2f));

    // TODO this requires convex...we should use triangle strip...
    glDrawArrays(GL_TRIANGLE_FAN, 0, static_cast<i32>(points.size()));
}

inline void polygon(Window& window, std::span<const Vec2f> points, Color color)
{
    polygon(window, points, color, window.world2gl());
}

inline void regular_polygon(
    Window& window, Circle border_circle, u32 point_count, Color color, const glm::mat4& transform)
{
    // TODO cache this. its allocated millions of times
    const auto points = math::regular_polygon_points<f32>(point_count, border_circle);
    polygon(window, {points.begin(), point_count}, color, transform);
}

inline void regular_polygon(Window& window, Circle border_circle, u32 point_count, Color color)
{
    regular_polygon(window, border_circle, point_count, color, window.world2gl());
}

// SHAPECOLOR stuf todo -----------------------------------------------------
// inline void polygon(Window& window,
//                     std::span<const Vec2f> points,
//                     ShapeColor color,
//                     const glm::mat4& transform)
// {
// if (color.fill_color.a != 0)
// {
// TODO
// }

// if (color.border_color.a != 0 && color.border_width != 0.0)
// {
//     auto new_points = std::vector<Vec2f>{points.begin(), points.end()};
//     new_points.insert(new_points.begin(), new_points.back());
//     new_points.push_back(new_points[1]);
//     new_points.push_back(new_points[2]);
// polyline_impl(window, {new_points}, color.border_color,
//   static_cast<f32>(color.border_width), transform);
// }
// }

// inline void polygon(Window& window, std::span<const Vec2f> points, ShapeColor color)
// {
//     polygon(window, points, color, window.world2gl());
// }

// inline void regular_polygon(Window& window,
//                             Circle border_circle,
//                             u32 point_count,
//                             ShapeColor color,
//                             const glm::mat4& transform)
// {
//     const auto points = math::regular_polygon_points<f32>(point_count, border_circle);
//     polygon(window, {points.begin(), point_count}, color, transform);
// }

// inline void regular_polygon(Window& window, Circle border_circle, u32 point_count, ShapeColor
// color)
// {
//     regular_polygon(window, border_circle, point_count, color, window.world2gl());
// }
} // namespace sm::draw

namespace sm::draw
{
// SM_INLINE void polyline_impl(Window& window,
//                              std::span<const Vec2f> points,
//                              Color color,
//                              f32 thickness,
//                              const glm::mat4& transform)
// {
//     const auto& shader = window.context.shaders.at("polyline");
//     shader.bind();
//     shader.set("thickness", thickness);
//     shader.set("screen_dims", window.dims.template cast<f64>());

//     shader.set("view", transform);
//     shader.set("color", color);

//     const auto& buffer = window.context.shader_storage_buffers.at("default");
//     buffer.set_data(points);
//     buffer.bind();
//     window.context.vertex_arrays.at("empty").bind();
//     glDrawArrays(GL_TRIANGLES, 0, 6 * (static_cast<i32>(points.size()) - 3));
// }

// SM_INLINE void polyline(Window& window,
//                         std::span<const Vec2f> points,
//                         Color color,
//                         f32 thickness,
//                         const glm::mat4& transform)
// {
//     auto new_points = std::vector<Vec2f>{points.begin(), points.end()};
//     new_points.insert(new_points.begin(), 2.0F * new_points[0] - new_points[1]);
//     const auto last = points.back();
//     new_points.push_back(2.0F * last - new_points[new_points.size() - 2]);
//     polyline_impl(window, {new_points}, color, thickness, transform);
// }

// SM_INLINE void
// polyline(Window& window, std::span<const Vec2f> points, Color color, f32 thickness)
// {
//     polyline(window, points, color, thickness, window.world2gl());
// }
} // namespace sm::draw
