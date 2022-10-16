/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <memory>    // for allocator, unique_ptr
#include <stdexcept> // for runtime_error
#include <string>    // for string

#include "glad/glad.h" // for gladLoadGLLoader, glEnable

#include "GLFW/glfw3.h"                // for glfwWindowHint, glfwGetMous...
#include "glm/ext/matrix_float4x4.hpp" // for mat4

#include "samarium/core/types.hpp"     // for f64, i32, u64, f32
#include "samarium/gl/Context.hpp"     // for Context
#include "samarium/gl/gl.hpp"          // for enable_debug_output, versio...
#include "samarium/graphics/Image.hpp" // for Image
#include "samarium/math/Transform.hpp" // for Transform
#include "samarium/math/Vector2.hpp"   // for Dimensions, Vector2_t, Vector2
#include "samarium/util/Grid.hpp"      // for Grid

#include "Mouse.hpp"    // for Mouse
#include "keyboard.hpp" // for keyboard

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
    Dimensions dims{};
    Transform view{};
    Mouse mouse{};
    keyboard::Keymap keymap{};

    static inline bool resized;

    static void
    framebuffer_size_callback([[maybe_unused]] GLFWwindow* window, i32 new_width, i32 new_height)
    {
        glViewport(0, 0, new_width, new_height);
        Window::resized = true;
    }

    explicit Window(Dimensions dims_, const std::string& title = "Samarium Window");

    Window(const Window&)            = delete;
    Window& operator=(const Window&) = delete;

    Window(Window&&) noexcept            = default;
    Window& operator=(Window&&) noexcept = default;

    ~Window() = default;

    [[nodiscard]] auto is_open() const -> bool;

    void close();

    void get_inputs();

    void display();

    [[nodiscard]] auto is_key_pressed(Key key) const -> bool;

    [[nodiscard]] auto aspect_ratio() const -> f64;

    [[nodiscard]] auto get_image() const -> Image;
};
} // namespace sm


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_WINDOW_IMPL)

#include "glm/gtc/matrix_transform.hpp" // for ortho

#include "samarium/core/inline.hpp"

#include "Window.hpp"

namespace sm
{
SM_INLINE Window::Window(Dimensions dims_, const std::string& title) : dims{dims_}
{
    if (glfwInit() == 0) { throw std::runtime_error("Error: failed to initialize glfw"); }

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

    keymap.push_back(keyboard::OnKeyPress{*handle, {Key::Escape}, [this] { this->close(); }});
}

SM_INLINE auto Window::is_open() const -> bool { return glfwWindowShouldClose(handle.get()) == 0; }

SM_INLINE void Window::close() { glfwSetWindowShouldClose(handle.get(), true); }

SM_INLINE void Window::get_inputs()
{
    glfwPollEvents();
    mouse.left    = glfwGetMouseButton(handle.get(), GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    mouse.right   = glfwGetMouseButton(handle.get(), GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    mouse.old_pos = mouse.pos;

    auto xpos = 0.0;
    auto ypos = 0.0;
    glfwGetCursorPos(handle.get(), &xpos, &ypos);
    mouse.pos = {xpos, ypos};

    keymap.run();
}

SM_INLINE void Window::display()
{
    resized = false;
    glfwSwapBuffers(handle.get());
    get_inputs();

    auto width  = 0;
    auto height = 0;

    glfwGetWindowSize(handle.get(), &width, &height);
    dims = {static_cast<u64>(width), static_cast<u64>(height)};
    // view.scale.y = view.scale.x * aspect_ratio();
}

SM_INLINE auto Window::is_key_pressed(Key key) const -> bool
{
    return glfwGetKey(handle.get(), static_cast<i32>(key)) == GLFW_PRESS;
}

SM_INLINE auto Window::aspect_ratio() const -> f64
{
    const auto current_dims = dims.as<f64>();
    return current_dims.x / current_dims.y;
}

SM_INLINE auto Window::get_image() const -> Image
{
    auto image = Image{dims};
    glReadPixels(0, 0, static_cast<i32>(dims.x), static_cast<i32>(dims.y), GL_RGBA,
                 GL_UNSIGNED_BYTE, static_cast<void*>(&image.front()));
    return image;
}
} // namespace sm

#endif
