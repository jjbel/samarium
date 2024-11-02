/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <span> // for span

#include "samarium/gl/Context.hpp"     // for Context
#include "samarium/gl/Vertex.hpp"      // for Vertex
#include "samarium/gl/gl.hpp"          // for Primitive
#include "samarium/graphics/Color.hpp" // for Color
#include "samarium/gui/Window.hpp"     // for Window
#include "samarium/math/Vector2.hpp"   // for Vector2f

namespace sm::draw
{
// TODO too many overloads?
void vertices(gl::Context& context,
              std::span<const gl::Vertex<gl::Layout::Pos>> verts,
              Color color,
              gl::Primitive primitive,
              const glm::mat4& transform);
void vertices(Window& window,
              std::span<const gl::Vertex<gl::Layout::Pos>> verts,
              Color color,
              gl::Primitive primitive);

void vertices(gl::Context& context,
              std::span<const Vector2f> verts,
              Color color,
              gl::Primitive primitive,
              const glm::mat4& transform);
void vertices(Window& window,
              std::span<const Vector2f> verts,
              Color color,
              gl::Primitive primitive);

void vertices(gl::Context& context,
              std::span<const gl::Vertex<gl::Layout::PosColor>> verts,
              gl::Primitive primitive,
              const glm::mat4& transform);
void vertices(Window& window,
              std::span<const gl::Vertex<gl::Layout::PosColor>> verts,
              gl::Primitive primitive);
} // namespace sm::draw

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_DRAW_IMPL)

#include "samarium/core/inline.hpp"
#include "samarium/gl/draw/vertices.hpp"

namespace sm::draw
{
SM_INLINE void vertices(gl::Context& context,
                        std::span<const gl::Vertex<gl::Layout::Pos>> verts,
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
                        std::span<const gl::Vertex<gl::Layout::Pos>> verts,
                        Color color,
                        gl::Primitive primitive)
{
    vertices(window.context, verts, color, primitive, window.view);
}

SM_INLINE void vertices(gl::Context& context,
                        std::span<const Vector2f> verts,
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
vertices(Window& window, std::span<const Vector2f> verts, Color color, gl::Primitive primitive)
{
    vertices(window.context, verts, color, primitive, window.view);
}

SM_INLINE void vertices(gl::Context& context,
                        std::span<const gl::Vertex<gl::Layout::PosColor>> verts,
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

SM_INLINE void vertices(Window& window,
                        std::span<const gl::Vertex<gl::Layout::PosColor>> verts,
                        gl::Primitive primitive)
{
    vertices(window.context, verts, primitive, window.view);
}
} // namespace sm::draw

#endif
