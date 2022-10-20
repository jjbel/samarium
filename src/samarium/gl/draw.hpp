#pragma once

#include <array> // for array,to_array
#include <numbers>
#include <span>   // for span
#include <vector> // for vector, allocator

#include "glad/glad.h"                     // for glDrawArrays, GL_COLOR_BUFFER...
#include "glm/mat4x4.hpp"                  // for mat4
#include "range/v3/algorithm/for_each.hpp" // for for_each
#include "range/v3/algorithm/minmax.hpp"   //for minmax

#include "samarium/core/types.hpp"        // for f32, f64, u64, i32
#include "samarium/gl/Context.hpp"        // for Context
#include "samarium/gl/Shader.hpp"         // for Shader
#include "samarium/graphics/Color.hpp"    // for Color, ShapeColor
#include "samarium/graphics/Gradient.hpp" // for Gradient
#include "samarium/gui/Window.hpp"        // for Window
#include "samarium/math/Extents.hpp"      // for range
#include "samarium/math/Vector2.hpp"      // for Vector2f, Vector2_t, operator*
#include "samarium/math/interp.hpp"       // for lerp, lerp_inverse
#include "samarium/math/math.hpp"         // for two_pi
#include "samarium/math/shapes.hpp"       // for Circle
#include "samarium/util/Map.hpp"          // for Map
#include "samarium/util/format.hpp"       // for format

#include "gl.hpp" // for Buffer, VertexArray

namespace sm::draw
{
void circle(Window& window, Circle circle, ShapeColor color, u64 point_count = 64);

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
    using Vertex = gl::Vertex<gl::Layout::PosColor>;

    auto vertices    = std::array<Vertex, 2 * size>{};
    const auto ratio = static_cast<f32>(window.aspect_ratio());

    for (auto i : range(size))
    {
        const auto factor =
            interp::lerp_inverse<f32>(static_cast<f64>(i), {0.0F, static_cast<f32>(size - 1)});
        vertices[2 * i].pos =
            Vector2f{static_cast<f32>(interp::clamped_lerp<f32>(factor, {-1.0F, 1.0F})), 1.0}
                .rotated(angle) *
            Vector2f{1.0, ratio};

        vertices[2 * i + 1].pos =
            Vector2f{static_cast<f32>(interp::clamped_lerp<f32>(factor, {-1.0F, 1.0F})), -1.0}
                .rotated(angle) *
            Vector2f{1.0, ratio};

        vertices[2 * i].color     = gradient.colors[i];
        vertices[2 * i + 1].color = gradient.colors[i];
    }

    const auto& shader = window.context.shaders.at("PosColor");
    window.context.set_active(shader);
    shader.set("view", glm::mat4{1.0});

    auto& vao = window.context.vertex_arrays.at("PosColor");
    window.context.set_active(vao);

    const auto& buffer = window.context.vertex_buffers.at("default");
    buffer.set_data(vertices);
    vao.bind(buffer, sizeof(Vertex));

    glDrawArrays(GL_TRIANGLE_STRIP, 0, static_cast<i32>(vertices.size()));
}

void polyline(Window& window, std::span<Vector2f> points, Color color, f32 thickness);

void polygon(Window& window, std::span<Vector2f> points, ShapeColor color);

void regular_polygon(
    Window& window, Vector2_t<f32> pos, f32 radius, u64 point_count, ShapeColor color);
} // namespace sm::draw


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_DRAW_IMPL)

#include "samarium/core/inline.hpp"

#include "draw.hpp"
#include "gl.hpp"

namespace sm::draw
{
SM_INLINE void background(Color color)
{
    glClearColor(static_cast<f32>(color.r) / 255.0f, static_cast<f32>(color.g) / 255.0f,
                 static_cast<f32>(color.b) / 255.0f, static_cast<f32>(color.a) / 255.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

SM_INLINE void polyline_impl(Window& window, std::span<Vector2f> points, Color color, f32 thickness)
{
    const auto& shader = window.context.shaders.at("polyline");
    shader.bind();
    shader.set("thickness", thickness);
    shader.set("screen_dims", window.dims.cast<f64>());

    shader.set("view", window.view.as_matrix());
    shader.set("color", color);

    auto& buffer = window.context.shader_storage_buffers.at("default");
    buffer.set_data(points);
    buffer.bind();
    window.context.vertex_arrays.at("empty").bind();
    glDrawArrays(GL_TRIANGLES, 0, 6 * (static_cast<i32>(points.size()) - 3));
}

SM_INLINE void polyline(Window& window, std::span<Vector2f> points, Color color, f32 thickness)
{
    auto new_points = std::vector<Vector2f>{points.begin(), points.end()};
    new_points.insert(new_points.begin(), 2.0f * new_points[0] - new_points[1]);
    const auto last = points.back();
    new_points.push_back(2.0f * last - new_points[new_points.size() - 2]);
    polyline_impl(window, {new_points}, color, thickness);
}

SM_INLINE void polygon(Window& window, std::span<Vector2f> points, ShapeColor color)
{
    if (color.fill_color.a != 0)
    {
        auto& shader = window.context.shaders.at("Pos");
        window.context.set_active(shader);
        shader.set("view", window.view.as_matrix());
        shader.set("color", color.fill_color);

        const auto& buffer = window.context.vertex_buffers.at("default");

        auto& vao = window.context.vertex_arrays.at("Pos");
        window.context.set_active(vao);
        buffer.set_data(points);
        vao.bind();
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
                      static_cast<f32>(color.border_width));
    }
}

SM_INLINE void
regular_polygon(Window& window, Vector2_t<f32> pos, f32 radius, u64 point_count, ShapeColor color)
{
    auto points = std::vector<Vector2_t<f32>>{};
    points.reserve(point_count);
    for (auto i = 0.0f; i < static_cast<f32>(math::two_pi);
         i += static_cast<f32>(math::two_pi) / static_cast<f32>(point_count))
    {
        points.push_back(Vector2_t<f32>::from_polar({radius, i}) + pos);
    }
    polygon(window, points, color);
}

SM_INLINE void circle(Window& window, Circle circle, ShapeColor color, u64 point_count)
{
    regular_polygon(window, circle.centre.cast<f32>(), static_cast<f32>(circle.radius), point_count,
                    color);
}
} // namespace sm::draw

#endif
