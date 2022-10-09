#pragma once

#include <memory>    // for unique_ptr
#include <stdexcept> // for string
#include <string>    // for runtime_error

#include "Context.hpp"
#include "gl.hpp"

#include "GLFW/glfw3.h"
#include "samarium/math/Vector2.hpp"
#include "samarium/util/Expected.hpp"
#include "samarium/util/print.hpp"

namespace sm
{
struct Window
{
    struct Deleter
    {
        auto operator()(GLFWwindow* ptr) const
        {
            glfwDestroyWindow(ptr);
            glfwTerminate();
        }
    };

    using Handle = std::unique_ptr<GLFWwindow, Deleter>;

    Handle handle{};
    gl::Context context{};
    glm::mat4 view{1.0f};

    static inline bool resized;

    static void
    framebuffer_size_callback([[maybe_unused]] GLFWwindow* window, i32 new_width, i32 new_height)
    {
        glViewport(0, 0, new_width, new_height);
        Window::resized = true;
    }

    explicit Window(Dimensions dims, const std::string& title = "Samarium Window")
    {
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, gl::version_major);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, gl::version_minor);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 4);

        handle = Handle(glfwCreateWindow(static_cast<i32>(dims.x), static_cast<i32>(dims.y),
                                         title.c_str(), nullptr, nullptr));

        if (!handle) { throw std::runtime_error("Error: failed to create window"); }

        glfwMakeContextCurrent(handle.get());
        glfwSetFramebufferSizeCallback(handle.get(), framebuffer_size_callback);

        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            throw std::runtime_error("Error: failed to initialize GLAD");
        }

        gl::enable_debug_output();

        glEnable(GL_MULTISAMPLE);

        glEnable(GL_BLEND); // enable blending function
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        context.init();
    }

    Window(const Window&) = delete;

    [[nodiscard]] auto is_open() -> bool;

    void close();

    void display();

    [[nodiscard]] auto dims() const -> Dimensions;

    [[nodiscard]] auto aspect_ratio() const -> f64;

    [[nodiscard]] auto get_image() const -> Image;
};
} // namespace sm


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_WINDOW_IMPL)

#include "Window.hpp"

namespace sm
{
auto Window::is_open() -> bool { return !glfwWindowShouldClose(handle.get()); }

void Window::close() { glfwSetWindowShouldClose(handle.get(), true); }

void Window::display()
{
    resized = false;
    glfwSwapBuffers(handle.get());
    glfwPollEvents();

    const auto aspect = static_cast<f32>(aspect_ratio());
    view              = glm::ortho(-aspect, aspect, -1.0f, 1.0f, -10.0f, 10.0f);
}

auto Window::dims() const -> Dimensions
{
    auto width  = 0;
    auto height = 0;

    glfwGetWindowSize(handle.get(), &width, &height);
    return {static_cast<u64>(width), static_cast<u64>(height)};
}

auto Window::aspect_ratio() const -> f64
{
    const auto current_dims = dims().as<f64>();
    return current_dims.x / current_dims.y;
}

auto Window::get_image() const -> Image
{
    const auto current_dims = this->dims();
    auto image              = Image{current_dims};
    glReadPixels(0, 0, static_cast<i32>(current_dims.x), static_cast<i32>(current_dims.y), GL_RGBA,
                 GL_UNSIGNED_BYTE, static_cast<void*>(&image.front()));
    return image;
}
} // namespace sm

#endif
