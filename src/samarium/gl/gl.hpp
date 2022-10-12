#pragma once

#include <stdexcept> // for logic_error
#include <vector>    // for vector

#include "fmt/core.h"  // for print
#include "glad/glad.h" // for GLenum, GL_SHADER_STORAGE_BUFFER

#include "samarium/core/types.hpp"     // for f32, i64, u32, i32
#include "samarium/graphics/Color.hpp" // for Color
#include "samarium/math/Vector2.hpp"   // for Vector2_t

namespace sm::gl
{
inline constexpr auto version_major = 4;
inline constexpr auto version_minor = 6;

namespace detail
{
constexpr auto element_size(const auto& array) { return sizeof(array[0]); }
constexpr auto array_byte_size(const auto& array) { return array.size() * element_size(array); }
} // namespace detail

enum class BufferType
{
    Vertex        = GL_ARRAY_BUFFER,
    Element       = GL_ELEMENT_ARRAY_BUFFER,
    Uniform       = GL_UNIFORM_BUFFER,
    ShaderStorage = GL_SHADER_STORAGE_BUFFER
};

enum class Usage
{
    StaticDraw  = GL_STATIC_DRAW,
    StaticRead  = GL_STATIC_READ,
    StaticCopy  = GL_STATIC_COPY,
    StreamDraw  = GL_STREAM_DRAW,
    StreamRead  = GL_STREAM_READ,
    StreamCopy  = GL_STREAM_COPY,
    DynamicDraw = GL_DYNAMIC_DRAW,
    DynamicRead = GL_DYNAMIC_READ,
    DynamicCopy = GL_DYNAMIC_COPY
};


template <BufferType type> struct Buffer
{
    u32 handle;

    Buffer()
    {
        glCreateBuffers(1, &handle); // generate a name
    }

    explicit Buffer(const std::ranges::range auto& array, Usage usage = Usage::StaticDraw)
    {
        glCreateBuffers(1, &handle); // generate a name

        glNamedBufferData(handle, detail::array_byte_size(array), array.data(),
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

    void bind() const
    {
        if constexpr (type == BufferType::ShaderStorage)
        {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, handle);
        }
        else
        {
            glBindBuffer(static_cast<GLenum>(type), handle); // make active, creating if necessary
        }
    }

    void bind(u32 index) const
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

    void set_data(const std::ranges::range auto& array, Usage usage = Usage::StaticDraw) const
    {
        glNamedBufferData(handle, static_cast<i64>(detail::array_byte_size(array)), array.data(),
                          static_cast<GLenum>(usage)); // copy data
    }

    void set_raw_data(const void* data, i64 byte_size, Usage usage = Usage::StaticDraw) const
    {
        glNamedBufferData(handle, byte_size, data,
                          static_cast<GLenum>(usage)); // copy data
    }

    void set_sub_data(const std::ranges::range auto& array, i64 offset = 0) const
    {
        glNamedBufferSubData(handle, offset, detail::array_byte_size(array), array.data());
    }

    void set_sub_raw_data(const void* data, i64 byte_size, i64 offset = 0) const
    {
        glNamedBufferSubData(handle, offset, byte_size, data);
    }

    ~Buffer() { glDeleteBuffers(1, &handle); }
};

using VertexBuffer  = Buffer<BufferType::Vertex>;
using ElementBuffer = Buffer<BufferType::Element>;

struct VertexAttribute
{
    i32 size;
    GLenum type;
    u32 offset;
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

    void bind();

    void bind(const VertexBuffer& buffer, i32 stride);

    void bind(const ElementBuffer& buffer);

    void make_attribute(u32 index, const VertexAttribute& attribute);

    ~VertexArray();
};

enum class Format
{
    Pos,
    PosColor,
    PosTex,
    PosColorTex
};

template <Format mode> struct Vertex
{
    Vector2_t<f32> pos{};
};

template <> struct Vertex<Format::PosColor>
{
    Vector2_t<f32> pos{};
    Color color{};
};

template <> struct Vertex<Format::PosTex>
{
    Vector2_t<f32> pos{};
    Vector2_t<f32> tex_coord{};
};

template <> struct Vertex<Format::PosColorTex>
{
    Vector2_t<f32> pos{};
    Color color{};
    Vector2_t<f32> tex_coord{};
};

inline auto get_current(GLenum object)
{
    auto handle = 0;
    glGetIntegerv(object, &handle);
    return handle;
}

inline void message_callback(GLenum source,
                             GLenum type,
                             GLuint id,
                             GLenum severity,
                             GLsizei /* length */,
                             GLchar const* message,
                             void const* /* user_param */)
{
    const auto src_str = [source]
    {
        switch (source)
        {
        case GL_DEBUG_SOURCE_API: return "api";
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM: return "window system";
        case GL_DEBUG_SOURCE_SHADER_COMPILER: return "shader compiler";
        case GL_DEBUG_SOURCE_THIRD_PARTY: return "third party";
        case GL_DEBUG_SOURCE_APPLICATION: return "application";
        case GL_DEBUG_SOURCE_OTHER: return "other";
        default: return "";
        }
    }();

    const auto type_str = [type]
    {
        switch (type)
        {
        case GL_DEBUG_TYPE_ERROR: return "ERROR";
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: return "DEPRECATED_BEHAVIOR";
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: return "UNDEFINED_BEHAVIOR";
        case GL_DEBUG_TYPE_PORTABILITY: return "PORTABILITY";
        case GL_DEBUG_TYPE_PERFORMANCE: return "PERFORMANCE";
        case GL_DEBUG_TYPE_MARKER: return "MARKER";
        case GL_DEBUG_TYPE_OTHER: return "OTHER";
        default: return "";
        }
    }();

    const auto severity_str = [severity]
    {
        switch (severity)
        {
        case GL_DEBUG_SEVERITY_NOTIFICATION: return "info";
        case GL_DEBUG_SEVERITY_LOW: return "low";
        case GL_DEBUG_SEVERITY_MEDIUM: return "medium";
        case GL_DEBUG_SEVERITY_HIGH: return "high";
        default: return "";
        }
    }();

    fmt::print("gl: {}, {}, {}, id:{} : {}\n", src_str, type_str, severity_str, id, message);
}

inline auto enable_debug_output()
{
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback(message_callback, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr,
                          GL_FALSE);
}
} // namespace sm::gl
