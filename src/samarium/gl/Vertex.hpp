/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <stdexcept> // for logic_error
#include <vector>    // for vector

#include "glad/glad.h"

#include "samarium/core/types.hpp"     // for u32
#include "samarium/math/Vector2.hpp"   // for Vector2f
#include "samarium/util/byte_size.hpp" // for byte_size

#include "gl.hpp"

namespace sm::gl
{
template <BufferType type> struct Buffer
{
    u32 handle;

    Buffer()
    {
        glCreateBuffers(1, &handle); // generate a name
    }

    explicit Buffer(const ranges::range auto& array, Usage usage = Usage::StaticDraw)
    {
        glCreateBuffers(1, &handle); // generate a name

        glNamedBufferData(handle, util::range_byte_size(array), std::data(array),
                          static_cast<GLenum>(usage)); // copy data
    }

    Buffer(const Buffer&) = delete;

    auto operator=(const Buffer&) -> Buffer& = delete;

    Buffer(Buffer&& other) noexcept : handle{other.handle} { other.handle = 0; }

    auto operator=(Buffer&& other) noexcept -> Buffer&
    {
        if (this != &other)
        {
            glDeleteBuffers(1, &handle);
            handle       = other.handle;
            other.handle = 0;
        }
        return *this;
    }

    void bind(u32 index = 0) const
    {
        if constexpr (type == BufferType::ShaderStorage)
        {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, handle);
        }
        else
        {
            throw std::logic_error{"Binding points can only be used for ShaderStorage objects"};
        }
    }

    void set_data(const ranges::range auto& array, Usage usage = Usage::StaticDraw) const
    {
        glNamedBufferData(handle, static_cast<i64>(util::range_byte_size(array)), std::data(array),
                          static_cast<GLenum>(usage)); // copy data
    }

    void set_sub_data(const ranges::range auto& array, i64 offset = 0) const
    {
        glNamedBufferSubData(handle, offset, util::range_byte_size(array), std::data(array));
    }

    ~Buffer() { glDeleteBuffers(1, &handle); }
};

using VertexBuffer  = Buffer<BufferType::Vertex>;
using ElementBuffer = Buffer<BufferType::Element>;
using ShaderStorageBuffer = Buffer<BufferType::ShaderStorage>;

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

#include "samarium/core/inline.hpp"

#include "Vertex.hpp"
namespace sm::gl
{
SM_INLINE VertexArray::VertexArray() { glCreateVertexArrays(1, &handle); }

SM_INLINE VertexArray::VertexArray(const std::vector<VertexAttribute>& attributes)
{
    glCreateVertexArrays(1, &handle);
    for (auto i : range(attributes.size())) { make_attribute(static_cast<u32>(i), attributes[i]); }
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
