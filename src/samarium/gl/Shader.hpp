/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <filesystem>  // for path
#include <string>      // for string
#include <string_view> // for string_view
#include <type_traits> // for decay<>::type
#include <utility>     // for move

#include "fmt/core.h"                  // for format
#include "glad/glad.h"                 // for glDeleteProgram, glDeleteShader
#include "glm/ext/matrix_float4x4.hpp" // for mat4
#include "tl/expected.hpp"             // for make_unexpected

#include "samarium/core/types.hpp"     // for u32, u64, i32, f32
#include "samarium/graphics/Color.hpp" // for Color
#include "samarium/math/Transform.hpp" // for Transform
#include "samarium/math/Vector2.hpp"   // for Vector2
#include "samarium/util/Result.hpp"    // for Result
#include "samarium/util/file.hpp"      // for read


#include "Texture.hpp"

namespace sm::gl
{
struct VertexShader
{
    u32 handle{};

    VertexShader(const VertexShader&) = delete;

    auto operator=(const VertexShader&) -> VertexShader& = delete;

    [[nodiscard]] static auto from_file(const std::filesystem::path& path)
    {
        return expect(make(*file::read(path)));
    }

    [[nodiscard]] static auto make(std::string source) -> Result<VertexShader>
    {
        auto program_handle = glCreateShader(GL_VERTEX_SHADER);
        source              = "#version 460 core\n" + source;
        const auto src      = source.c_str();
        glShaderSource(program_handle, 1, &src, nullptr);
        glCompileShader(program_handle);

        auto success = 0;
        glGetShaderiv(program_handle, GL_COMPILE_STATUS, &success);

        if (success)
        {
            auto shader   = VertexShader{};
            shader.handle = program_handle;
            return {std::move(shader)};
        }

        auto log_size = 0;
        char info_log[1024];
        glGetShaderInfoLog(program_handle, 1024, &log_size, info_log);

        return make_unexpected(fmt::format("Vertex shader compilation error:\n{}",
                                           std::string_view{info_log, static_cast<u64>(log_size)}));
    }

    VertexShader(VertexShader&& other) noexcept : handle{other.handle} { other.handle = 0; }

    auto operator=(VertexShader&& other) noexcept -> VertexShader&
    {
        if (this != &other)
        {
            glDeleteShader(handle);
            handle       = other.handle;
            other.handle = 0;
        }
        return *this;
    }

    ~VertexShader() { glDeleteShader(handle); }

  private:
    VertexShader() = default;
};

struct FragmentShader
{
    u32 handle{};

    explicit FragmentShader(const std::string& source);

    FragmentShader(const FragmentShader&) = delete;

    auto operator=(const FragmentShader&) -> FragmentShader& = delete;

    static auto from_file(const std::filesystem::path& path)
    {
        return FragmentShader(*file::read(path));
    }

    static inline auto make(std::string source) -> Result<FragmentShader>
    {
        auto program_handle = glCreateShader(GL_FRAGMENT_SHADER);
        source              = "#version 460 core\n" + source;
        const auto src      = source.c_str();
        glShaderSource(program_handle, 1, &src, nullptr);
        glCompileShader(program_handle);

        auto success = 0;
        glGetShaderiv(program_handle, GL_COMPILE_STATUS, &success);

        if (success)
        {
            auto shader   = FragmentShader{};
            shader.handle = program_handle;
            return Result<FragmentShader>{std::move(shader)};
        }

        auto log_size = 0;
        char info_log[1024];
        glGetShaderInfoLog(program_handle, 1024, &log_size, info_log);

        return make_unexpected(fmt::format("Vertex shader compilation error:\n{}",
                                           std::string_view{info_log, static_cast<u64>(log_size)}));
    }

    FragmentShader(FragmentShader&& other) noexcept : handle{other.handle} { other.handle = 0; }

    auto operator=(FragmentShader&& other) noexcept -> FragmentShader&
    {
        if (this != &other)
        {
            glDeleteShader(handle);
            handle       = other.handle;
            other.handle = 0;
        }
        return *this;
    }

    ~FragmentShader() { glDeleteShader(handle); }

  private:
    FragmentShader() = default;
};

struct Shader
{
    u32 handle{glCreateProgram()};

    Shader(const VertexShader& vertex, const FragmentShader& fragment)
    {
        glAttachShader(handle, vertex.handle);
        glAttachShader(handle, fragment.handle);
        glLinkProgram(handle);
    }

    Shader(const Shader&) = delete;

    auto operator=(const Shader&) -> Shader& = delete;

    Shader(Shader&& other) noexcept : handle{other.handle} { other.handle = 0; }

    auto operator=(Shader&& other) noexcept -> Shader&
    {
        if (this != &other)
        {
            glDeleteProgram(handle);
            handle       = other.handle;
            other.handle = 0;
        }
        return *this;
    }

    [[nodiscard]] auto get_uniform_location(const std::string& name) const -> i32;

    void set(const std::string& name, bool value) const;
    void set(const std::string& name, i32 value) const;
    void set(const std::string& name, f32 value) const;
    void set(const std::string& name, Color value) const;
    void set(const std::string& name, Vector2 value) const;
    void set(const std::string& name, const glm::mat4& value) const;

    // TODO implicit cast to glm mat4 but now explicit overload
    // void set(const std::string& name, const Transform& value) const;

    void bind() const { glUseProgram(handle); }

    ~Shader() { glDeleteProgram(handle); }

  private:
    Shader() = default;
};

struct ComputeShader
{
    u32 handle;

    ComputeShader(const ComputeShader&)                    = delete;
    auto operator=(const ComputeShader&) -> ComputeShader& = delete;
    ComputeShader(ComputeShader&& other) noexcept : handle{other.handle} { other.handle = 0; }
    auto operator=(ComputeShader&& other) noexcept -> ComputeShader&
    {
        if (this != &other)
        {
            glDeleteProgram(handle);
            handle       = other.handle;
            other.handle = 0;
        }
        return *this;
    }

    static inline auto make(std::string source,
                            i32 local_size_x = 1,
                            i32 local_size_y = 1,
                            i32 local_size_z = 1) -> Result<ComputeShader>
    {
        auto program_handle = glCreateShader(GL_COMPUTE_SHADER);
        source = fmt::format("#version 460 core\nlayout(local_size_x = {}, local_size_y = {}, "
                             "local_size_z = {}) in;\n",
                             local_size_x, local_size_y, local_size_z) +
                 source;
        const auto src_pointer = source.c_str();
        glShaderSource(program_handle, 1, &src_pointer, nullptr);
        glCompileShader(program_handle);

        auto success = 0;
        glGetShaderiv(program_handle, GL_COMPILE_STATUS, &success);

        if (success != 0)
        {
            auto shader   = ComputeShader{};
            shader.handle = glCreateProgram();
            glAttachShader(shader.handle, program_handle);
            glLinkProgram(shader.handle);
            return {std::move(shader)};
        }

        auto log_size = 0;
        auto log_str  = std::string(1024, ' ');
        glGetShaderInfoLog(program_handle, 1024, &log_size, log_str.data());

        return make_unexpected(
            fmt::format("ComputeShader compilation error:\n{}",
                        std::string_view{log_str.data(), static_cast<u64>(log_size)}));
    }

    explicit ComputeShader(const std::string& source,
                           i32 local_size_x = 1,
                           i32 local_size_y = 1,
                           i32 local_size_z = 1)
    {
        auto result = make(source);
        if (result) { *this = std::move(*result); }
        else { throw Error{result.error()}; }
        // TODO use local_sizes?
        (void)local_size_x;
        (void)local_size_y;
        (void)local_size_z;
    }

    void bind();
    void run(u32 x, u32 y = 1U, u32 z = 1U);

    [[nodiscard]] auto get_uniform_location(const std::string& name) const -> i32;

    void set(const std::string& name, bool value) const;
    void set(const std::string& name, i32 value) const;
    void set(const std::string& name, f32 value) const;
    void set(const std::string& name, Color value) const;
    void set(const std::string& name, Vector2 value) const;
    void set(const std::string& name, const glm::mat4& value) const;

    void set(const std::string& name, const Texture& texture) const;

    ~ComputeShader() { glDeleteProgram(handle); }

  private:
    ComputeShader() = default;
};
} // namespace sm::gl


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_SHADER_IMPL)

#include "glad/glad.h"          // for GL_COMPILE_STATUS, glCompileS...
#include "glm/gtc/type_ptr.hpp" // forvalue_ptr

#include "Shader.hpp"

namespace sm::gl
{
auto Shader::get_uniform_location(const std::string& name) const -> i32
{
    return glGetUniformLocation(handle, name.c_str());
}

void Shader::set(const std::string& name, bool value) const
{
    glUniform1i(get_uniform_location(name), static_cast<i32>(value));
}

void Shader::set(const std::string& name, i32 value) const
{
    glUniform1i(get_uniform_location(name), value);
}

void Shader::set(const std::string& name, f32 value) const
{
    glUniform1f(get_uniform_location(name), value);
}

void Shader::set(const std::string& name, Color value) const
{
    glUniform4f(get_uniform_location(name), static_cast<f32>(value.r) / 255.0F,
                static_cast<f32>(value.g) / 255.0F, static_cast<f32>(value.b) / 255.0F,
                static_cast<f32>(value.a) / 255.0F);
}

void Shader::set(const std::string& name, Vector2 value) const
{
    glUniform2f(get_uniform_location(name), static_cast<f32>(value.x), static_cast<f32>(value.y));
}

void Shader::set(const std::string& name, const glm::mat4& value) const
{
    glUniformMatrix4fv(get_uniform_location(name), 1, GL_FALSE, glm::value_ptr(value));
}

// void Shader::set(const std::string& name, const Transform& value) const
// {
//     glUniform4f(get_uniform_location(name), static_cast<f32>(value.pos.x),
//                 static_cast<f32>(value.pos.y), static_cast<f32>(value.scale.x),
//                 static_cast<f32>(value.scale.y));
// }

void ComputeShader::bind() { glUseProgram(handle); }

void ComputeShader::run(u32 x, u32 y, u32 z)
{
    glDispatchCompute(x, y, z);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

auto ComputeShader::get_uniform_location(const std::string& name) const -> i32
{
    return glGetUniformLocation(handle, name.c_str());
}

void ComputeShader::set(const std::string& name, bool value) const
{
    glUniform1i(get_uniform_location(name), static_cast<i32>(value));
}

void ComputeShader::set(const std::string& name, i32 value) const
{
    glUniform1i(get_uniform_location(name), value);
}

void ComputeShader::set(const std::string& name, f32 value) const
{
    glUniform1f(get_uniform_location(name), value);
}

void ComputeShader::set(const std::string& name, Color value) const
{
    glUniform4f(get_uniform_location(name), static_cast<f32>(value.r) / 255.0F,
                static_cast<f32>(value.g) / 255.0F, static_cast<f32>(value.b) / 255.0F,
                static_cast<f32>(value.a) / 255.0F);
}

void ComputeShader::set(const std::string& name, Vector2 value) const
{
    glUniform2f(get_uniform_location(name), static_cast<f32>(value.x), static_cast<f32>(value.y));
}

void ComputeShader::set(const std::string& name, const glm::mat4& value) const
{
    glUniformMatrix4fv(get_uniform_location(name), 1, GL_FALSE, glm::value_ptr(value));
}

void ComputeShader::set(const std::string& name, const Texture& texture) const
{
    glUniform1i(get_uniform_location(name), static_cast<i32>(texture.handle));
}
} // namespace sm::gl
#endif
