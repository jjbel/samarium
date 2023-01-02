/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <span>   // for span
#include <vector> // for vector

#include "glad/glad.h"
#include "range/v3/range_concepts.hpp" // for range

#include "samarium/core/types.hpp"     // for u32
#include "samarium/math/Vector2.hpp"   // for Vector2f
#include "samarium/util/Error.hpp"     // for Error
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
        else { throw Error{"Binding points can only be used for ShaderStorage objects"}; }
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

    template <typename T> auto read(u64 count) const -> std::vector<T>
    {
        auto data = std::vector<T>(count);
        glGetNamedBufferSubData(handle, 0, count * sizeof(T), data.data());
        return data;
    }

    template <typename T> auto read_to(std::span<T> buffer) const
    {
        glGetNamedBufferSubData(handle, 0, buffer.size() * sizeof(T), buffer.data());
    }

    ~Buffer() { glDeleteBuffers(1, &handle); }
};

using VertexBuffer        = Buffer<BufferType::Vertex>;
using ElementBuffer       = Buffer<BufferType::Element>;
using ShaderStorageBuffer = Buffer<BufferType::ShaderStorage>;
} // namespace sm::gl
