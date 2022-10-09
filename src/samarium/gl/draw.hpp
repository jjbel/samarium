#pragma once

#include <vector>

#include "Window.hpp"
#include "gl.hpp"

#include "samarium/graphics/Color.hpp"
#include "samarium/math/Vector2.hpp"
#include "samarium/math/shapes.hpp"
#include "samarium/math/vector_math.hpp"

namespace sm::draw
{
void fill(Color color);

void polyline(Window& window, std::span<Vector2f> points, Color color, f32 thickness);

void polygon(Window& window, std::span<Vector2f> points, ShapeColor color);

void regular_polygon(
    Window& window, Vector2_t<f32> pos, f32 radius, u64 point_count, ShapeColor color);

void circle(Window& window, Circle circle, ShapeColor color, u64 point_count = 64);
} // namespace sm::draw


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_DRAW_IMPL)

#include "draw.hpp"
#include "gl.hpp"

namespace sm::draw
{
void fill(Color color)
{
    glClearColor(static_cast<f32>(color.r) / 255.0f, static_cast<f32>(color.g) / 255.0f,
                 static_cast<f32>(color.b) / 255.0f, static_cast<f32>(color.a) / 255.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void polyline_impl(Window& window, std::span<Vector2f> points, Color color, f32 thickness)
{
    const auto& shader = window.context.shaders.at("polyline");
    shader.use();
    shader.set("thickness", thickness);
    shader.set("screen_dims", window.dims().as<f64>());

    shader.set("view", window.view);
    shader.set("color", color);

    auto& buffer = window.context.shader_storage_buffers.at("default");
    buffer.set_data(points);
    buffer.bind();
    window.context.vertex_arrays.at("empty").bind();
    glDrawArrays(GL_TRIANGLES, 0, 6 * (static_cast<i32>(points.size()) - 3));
}

void polyline(Window& window, std::span<Vector2f> points, Color color, f32 thickness)
{
    auto new_points = std::vector<Vector2f>{points.begin(), points.end()};
    new_points.insert(new_points.begin(), 2.0f * new_points[0] - new_points[1]);
    const auto last = points.back();
    new_points.push_back(2.0f * last - new_points[new_points.size() - 2]);
    polyline_impl(window, {new_points}, color, thickness);
}

void polygon(Window& window, std::span<Vector2f> points, ShapeColor color)
{
    if (color.fill_color.a != 0)
    {
        auto& shader = window.context.shaders.at("Pos");
        shader.use();
        shader.set("view", window.view);
        shader.set("color", color.fill_color);
        auto& buffer = window.context.vertex_buffers.at("default");
        auto& vao    = window.context.vertex_arrays.at("Pos");
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

void regular_polygon(
    Window& window, Vector2_t<f32> pos, f32 radius, u64 point_count, ShapeColor color)
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

void circle(Window& window, Circle circle, ShapeColor color, u64 point_count)
{
    regular_polygon(window, circle.centre.as<f32>(), static_cast<f32>(circle.radius), point_count,
                    color);
}
} // namespace sm::draw

#endif
