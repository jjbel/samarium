/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <array>     // for to_array
#include <stdexcept> // for logic_error
#include <vector>    // for vector

#include "fmt/core.h"                  // for print
#include "glad/glad.h"                 // for GLenum, GL_SHADER_STORAGE_BUFFER
#include "range/v3/range/concepts.hpp" // for range

#include "samarium/core/types.hpp"     // for f32, i64, u32, i32
#include "samarium/graphics/Color.hpp" // for Color
#include "samarium/math/Vector2.hpp"   // for Vector2_t

namespace sm::gl
{
inline constexpr auto version_major = 4;
inline constexpr auto version_minor = 6;

static constexpr auto unit_square =
    std::to_array<Vector2f>({{-1.0F, -1.0F}, {-1.0F, 1.0F}, {1.0F, 1.0F}, {1.0F, -1.0F}});

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

enum class Access
{
    Read      = GL_READ_ONLY,
    Write     = GL_WRITE_ONLY,
    ReadWrite = GL_READ_WRITE
};

enum class Primitive
{
    Points        = GL_POINTS,
    Lines         = GL_LINES,
    LineLoop      = GL_LINE_LOOP,
    LineStrip     = GL_LINE_STRIP,
    Triangles     = GL_TRIANGLES,
    TriangleStrip = GL_TRIANGLE_STRIP,
    TriangleFan   = GL_TRIANGLE_FAN
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
                             [[maybe_unused]] GLsizei length,
                             GLchar const* message,
                             [[maybe_unused]] void const* user_param)
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
