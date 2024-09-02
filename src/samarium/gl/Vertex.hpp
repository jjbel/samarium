/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <vector> // for vector

#include "glad/glad.h"

#include "samarium/math/Vector2.hpp" // for Vector2_t

#include "Buffer.hpp"
#include "gl.hpp"

namespace sm::gl
{
struct VertexAttribute
{
    i32 size{};
    GLenum type{};
    u32 offset{};
    GLboolean normalized{GL_FALSE};
};

struct VertexArray
{
    u32 handle{};

    VertexArray();

    explicit VertexArray(const std::vector<VertexAttribute>& attributes);

    VertexArray(const VertexArray&) = delete;

    auto operator=(const VertexArray&) -> VertexArray& = delete;

    VertexArray(VertexArray&& other) noexcept : handle{other.handle} { other.handle = 0; }

    auto operator=(VertexArray&& other) noexcept -> VertexArray&
    {
        if (this != &other)
        {
            glDeleteVertexArrays(1, &handle);
            handle       = other.handle;
            other.handle = 0;
        }
        return *this;
    }

    void bind() const;

    void bind(const VertexBuffer& buffer, i32 stride);

    void bind(const ElementBuffer& buffer);

    void make_attribute(u32 index, const VertexAttribute& attribute);

    ~VertexArray();
};

enum class Layout
{
    Pos,        ///< Each vertex only has a position, all vertices have the same color
    PosColor,   ///< Each vertex has a position and a color
    PosTex,     ///< Each vertex has a position and a texture coordinate
    PosColorTex ///< Each vertex has a position, color and a texture coordinate
};

template <Layout mode> struct Vertex
{
    Vector2_t<f32> pos{};
};

template <> struct Vertex<Layout::PosColor>
{
    Vector2_t<f32> pos{};
    Color color{};
};

template <> struct Vertex<Layout::PosTex>
{
    Vector2_t<f32> pos{};
    Vector2_t<f32> tex_coord{};
};

template <> struct Vertex<Layout::PosColorTex>
{
    Vector2_t<f32> pos{};
    Color color{};
    Vector2_t<f32> tex_coord{};
};
} // namespace sm::gl

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_VERTEX_IMPL)

#include "glad/glad.h" // for glCreateVertexArrays, glBindVer...

#include "samarium/core/inline.hpp" // for SM_INLINE
#include "samarium/math/loop.hpp"   // for end

#include "Vertex.hpp"
namespace sm::gl
{
SM_INLINE VertexArray::VertexArray() { glCreateVertexArrays(1, &handle); }

SM_INLINE VertexArray::VertexArray(const std::vector<VertexAttribute>& attributes)
{
    glCreateVertexArrays(1, &handle);
    for (auto i : loop::end(attributes.size()))
    {
        make_attribute(static_cast<u32>(i), attributes[i]);
    }
}

SM_INLINE void VertexArray::bind() const
{
    glBindVertexArray(handle); // make active, creating if necessary
}

SM_INLINE void VertexArray::make_attribute(u32 index, const VertexAttribute& attribute)
{
    glEnableVertexArrayAttrib(handle, index);
    glVertexArrayAttribBinding(handle, index, 0);
    glVertexArrayAttribFormat(handle, index, attribute.size, attribute.type, attribute.normalized,
                              attribute.offset);
}

SM_INLINE void VertexArray::bind(const VertexBuffer& buffer, i32 stride)
{
    glVertexArrayVertexBuffer(handle, 0, buffer.handle, 0, stride);
}

SM_INLINE void VertexArray::bind(const ElementBuffer& buffer)
{
    glVertexArrayElementBuffer(handle, buffer.handle);
}

SM_INLINE VertexArray::~VertexArray() { glDeleteVertexArrays(1, &handle); }
} // namespace sm::gl
#endif
