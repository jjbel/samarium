#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "glad/glad.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"

#include "samarium/util/Expected.hpp"
#include "samarium/util/file.hpp"
#include "samarium/util/print.hpp"

namespace sm::gl
{
struct VertexShader
{
    u32 handle{};

    explicit VertexShader(const std::string& source);

    VertexShader(const VertexShader&) = delete;

    auto operator=(const VertexShader&) -> VertexShader& = delete;

    static auto from_file(const std::filesystem::path& path)
    {
        return VertexShader(expect(file::read(path)));
    }

    static inline auto make(const std::string& source) -> Expected<VertexShader, std::string>
    {
        auto program_handle = glCreateShader(GL_VERTEX_SHADER);
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

        return tl::make_unexpected(
            fmt::format("Vertex shader compilation error:\n{}",
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
        return FragmentShader(expect(file::read(path)));
    }

    static inline auto make(const std::string& source) -> Expected<FragmentShader, std::string>
    {
        auto program_handle = glCreateShader(GL_FRAGMENT_SHADER);
        const auto src      = source.c_str();
        glShaderSource(program_handle, 1, &src, nullptr);
        glCompileShader(program_handle);

        auto success = 0;
        glGetShaderiv(program_handle, GL_COMPILE_STATUS, &success);

        if (success)
        {
            auto shader   = FragmentShader{};
            shader.handle = program_handle;
            return Expected<FragmentShader, std::string>{std::move(shader)};
        }

        auto log_size = 0;
        char info_log[1024];
        glGetShaderInfoLog(program_handle, 1024, &log_size, info_log);

        return tl::make_unexpected(
            fmt::format("Vertex shader compilation error:\n{}",
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
    u32 handle;

    Shader(const VertexShader& vertex, const FragmentShader& fragment) : handle(glCreateProgram())
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

    void use() const { glUseProgram(handle); }

    ~Shader() { glDeleteProgram(handle); }

  private:
    Shader() = default;
};

struct ComputeShader
{
    u32 handle;

    explicit ComputeShader(const std::string& source);

    ComputeShader(const ComputeShader&) = delete;

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

    static inline auto make(const std::string& source) -> Expected<ComputeShader, std::string>
    {
        auto program_handle = glCreateShader(GL_COMPUTE_SHADER);
        const auto src      = source.c_str();
        glShaderSource(program_handle, 1, &src, nullptr);
        glCompileShader(program_handle);

        auto success = 0;
        glGetShaderiv(program_handle, GL_COMPILE_STATUS, &success);

        if (success)
        {
            auto shader   = ComputeShader{};
            shader.handle = glCreateProgram();
            glAttachShader(shader.handle, program_handle);
            glLinkProgram(shader.handle);
            return {std::move(shader)};
        }

        auto log_size = 0;
        char info_log[1024];
        glGetShaderInfoLog(program_handle, 1024, &log_size, info_log);

        return tl::make_unexpected(
            fmt::format("Compute shader compilation error:\n{}",
                        std::string_view{info_log, static_cast<u64>(log_size)}));
    }

    void use(u32 x, u32 y, u32 z = 1U) const;

    ~ComputeShader() { glDeleteProgram(handle); }

  private:
    ComputeShader() = default;
};
} // namespace sm::gl
