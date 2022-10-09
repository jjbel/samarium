#include "Shader.hpp"

namespace sm::gl
{
VertexShader::VertexShader(const std::string& source) : handle(glCreateShader(GL_VERTEX_SHADER))
{
    const auto str = source.c_str();
    glShaderSource(handle, 1, &str, nullptr);
    glCompileShader(handle);

    auto success = 0;
    glGetShaderiv(handle, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        auto log_size = 0;
        char info_log[1024];
        glGetShaderInfoLog(handle, 1024, &log_size, info_log);
        sm::error("vertex shader compilation failed: ", info_log);
    }
}

FragmentShader::FragmentShader(const std::string& source) : handle(glCreateShader(GL_FRAGMENT_SHADER))
{
    const auto str = source.c_str();
    glShaderSource(handle, 1, &str, nullptr);
    glCompileShader(handle);

    auto success = 0;
    glGetShaderiv(handle, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        auto log_size = 0;
        char info_log[1024];
        glGetShaderInfoLog(handle, 1024, &log_size, info_log);
        error("fragment shader compilation failed: ", info_log);
    }
}

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
    glUniform4f(get_uniform_location(name), static_cast<f32>(value.r) / 255.0f,
                static_cast<f32>(value.g) / 255.0f, static_cast<f32>(value.b) / 255.0f,
                static_cast<f32>(value.a) / 255.0f);
}

void Shader::set(const std::string& name, Vector2 value) const
{
    glUniform2f(get_uniform_location(name), static_cast<f32>(value.x), static_cast<f32>(value.y));
}

void Shader::set(const std::string& name, const glm::mat4& value) const
{
    glUniformMatrix4fv(get_uniform_location(name), 1, GL_FALSE, glm::value_ptr(value));
}

ComputeShader::ComputeShader(const std::string& source)
{
    auto shader_handle = glCreateShader(GL_COMPUTE_SHADER);
    const auto src     = source.c_str();
    glShaderSource(shader_handle, 1, &src, nullptr);
    glCompileShader(shader_handle);

    auto success = 0;
    glGetShaderiv(shader_handle, GL_COMPILE_STATUS, &success);

    if (!success)
    {
        auto log_size = 0;
        char info_log[1024];
        glGetShaderInfoLog(shader_handle, 1024, &log_size, info_log);
        error("compute shader compilation failed:\n",
              std::string_view{info_log, static_cast<u64>(log_size)});
    }

    handle = glCreateProgram();
    glAttachShader(handle, shader_handle);
    glLinkProgram(handle);
}

void ComputeShader::use(u32 x, u32 y, u32 z) const
{
    glUseProgram(handle);
    glDispatchCompute(x, y, z);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}
} // namespace sm::gl
