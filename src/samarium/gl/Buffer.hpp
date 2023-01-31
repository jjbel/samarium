/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <span> // for span
#include <tl/expected.hpp>
#include <vector> // for vector

#include "glad/glad.h"
#include "range/v3/algorithm/fill.hpp" // for fill
#include "range/v3/range_concepts.hpp" // for range

#include "samarium/core/types.hpp"     // for u32
#include "samarium/math/Vector2.hpp"   // for Vector2f
#include "samarium/util/Error.hpp"     // for Error
#include "samarium/util/Result.hpp"    // for Result
#include "samarium/util/byte_size.hpp" // for byte_size

#include "samarium/util/print.hpp" // for print

#include "gl.hpp"

namespace sm::gl
{
// https://www.khronos.org/opengl/wiki/Buffer_Object
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

// https://www.khronos.org/opengl/wiki/Buffer_Object#Persistent_mapping
template <typename T> struct MappedBuffer
{
    static constexpr GLbitfield mapping_flags =
        GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    static constexpr GLbitfield storage_flags = GL_DYNAMIC_STORAGE_BIT | mapping_flags;

    u32 handle;
    std::span<T> data;

  private:
    MappedBuffer() = default;

  public:
    static auto make(i32 size) -> Result<MappedBuffer<T>>
    {
        auto buffer = MappedBuffer<T>{};
        glCreateBuffers(1, &buffer.handle);
        if (!buffer.resize(size))
        {
            return make_unexpected(fmt::format(
                "MappedBuffer: glMapNamedBufferRange failed to create buffer of size {}", size));
        }
        return {std::move(buffer)};
    }

    static auto make(i32 size, const T& initial_value) -> Result<MappedBuffer<T>>
    {
        auto buffer = MappedBuffer<T>{};
        glCreateBuffers(1, &buffer.handle);
        if (!buffer.resize(size))
        {
            return make_unexpected(fmt::format(
                "MappedBuffer: glMapNamedBufferRange failed to create buffer of size {}", size));
        }
        buffer.fill(initial_value);
        return {std::move(buffer)};
    }

    auto resize(i32 new_size)
    {
        glNamedBufferStorage(handle, new_size * sizeof(T), nullptr, storage_flags);
        void* pointer = glMapNamedBufferRange(handle, 0, new_size * sizeof(T), mapping_flags);
        if (pointer == nullptr) { return false; }
        data = std::span<T>(reinterpret_cast<T*>(pointer), static_cast<u64>(new_size));
        return true;
    }

    auto fill(const T& value) { ranges::fill(data, value); }

    auto bind(u32 index = 0) const { glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, handle); }

    auto read() { glGetNamedBufferSubData(handle, 0, data.size() * sizeof(T), data.data()); }

    MappedBuffer(const MappedBuffer&)                    = delete;
    auto operator=(const MappedBuffer&) -> MappedBuffer& = delete;

    MappedBuffer(MappedBuffer&& other) noexcept : handle{other.handle}, data{other.data}
    {
        other.handle = 0;
        other.data   = {};
    }

    auto operator=(MappedBuffer&& other) noexcept -> MappedBuffer&
    {
        if (this != &other)
        {
            glDeleteBuffers(1, &handle);
            handle       = other.handle;
            data         = other.data;
            other.handle = 0;
            other.data   = {};
        }
        return *this;
    }

    ~MappedBuffer()
    {
        // TODO: segfaults
        // glUnmapNamedBuffer(handle);

        glDeleteBuffers(1, &handle);
    }
};
} // namespace sm::gl
