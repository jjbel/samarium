/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <memory>    // for allocator, unique_ptr
#include <stdexcept> // for runtime_error
#include <string>    // for string

#include "glad/glad.h" // for gladLoadGLLoader, glEnable

#include "GLFW/glfw3.h"                            // for glfwWindowHint, glfwGetMous...
#include "glm/ext/matrix_float4x4.hpp"             // for mat4
#include "samarium/util/call_thunk/call_thunk.hpp" // for thunk

#include "samarium/core/types.hpp"       // for f64, i32, u64, f32
#include "samarium/gl/Context.hpp"       // for Context
#include "samarium/gl/Texture.hpp"       // for Texture
#include "samarium/gl/gl.hpp"            // for enable_debug_output, versio...
#include "samarium/math/BoundingBox.hpp" // for BoundingBox
#include "samarium/math/Transform.hpp"   // for Transform
#include "samarium/math/Vector2.hpp"     // for Dimensions, Vector2_t, Vector2
#include "samarium/math/math.hpp"        // for min, max
#include "samarium/util/Grid.hpp"        // for Image

#include "Mouse.hpp"    // for Mouse
#include "keyboard.hpp" // for keyboard

namespace sm
{
enum class Space
{
    World,
    Screen
};

struct WindowConfig
{
    Dimensions dims   = dims720;
    std::string title = "Samarium Window";
    bool resizable    = true;
};

struct ScrollCallback
{
    struct Holder
    {
        double scroll{};
        void function([[maybe_unused]] GLFWwindow* window,
                      [[maybe_unused]] double xoffset,
                      double yoffset)
        {
            scroll = yoffset;
        }
    };

    Holder holder{};
    call_thunk::thunk<GLFWscrollfun> thunk{holder, &Holder::function};
};

struct ResizeCallback
{
    struct Holder
    {
        Dimensions dims{};
        bool resized{};
        void function([[maybe_unused]] GLFWwindow* window, i32 new_width, i32 new_height)
        {
            glViewport(0, 0, new_width, new_height);
            dims    = Dimensions::make(new_width, new_height);
            resized = true;
        }
    };

    Holder holder{};
    call_thunk::thunk<GLFWframebuffersizefun> thunk{holder, &Holder::function};
};

/**
 * @brief               An RAII wrapper around GLFWwindow
 *
 */
struct Window
{
    struct Deleter
    {
        auto operator()(GLFWwindow* ptr) const { glfwDestroyWindow(ptr); }
    };

    using Handle = std::unique_ptr<GLFWwindow, Deleter>;

    /**
     * @brief               An RAII helper which creates and cleans up the GLFWwindow
     *
     */
    struct Init
    {
        Init(const WindowConfig& config, Handle& handle)
        {
            if (glfwInit() == 0) { throw Error{"failed to initialize glfw"}; }

            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, gl::version_major);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, gl::version_minor);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

            handle = Handle(glfwCreateWindow(static_cast<i32>(config.dims.x),
                                             static_cast<i32>(config.dims.y), config.title.c_str(),
                                             nullptr, nullptr));

            if (!handle) { throw Error{"failed to create window"}; }

            glfwMakeContextCurrent(handle.get());
            if (gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)) == 0)
            {
                throw Error{"failed to initialize GLAD"};
            }

            glViewport(0, 0, static_cast<i32>(config.dims.x), static_cast<i32>(config.dims.y));
        }

        Init(const Init&)                    = delete;
        auto operator=(const Init&) -> Init& = delete;

        Init(Init&&) noexcept                    = default;
        auto operator=(Init&&) noexcept -> Init& = default;

        ~Init() { glfwTerminate(); }
    };

    ScrollCallback scroll_callback{};
    ResizeCallback resize_callback{};

    Handle handle{};
    Dimensions dims{};

    Transform view{.scale = Vector2::combine(1.0 / 20.0)};
    // applying view goes from GL clip space (ie -1 to 1) to graph space
    // TODO but actually seems to be smthg else
    // https://learnopengl.com/Getting-started/Coordinate-Systems
    // TODO make a function to return a pixel_space transform

    Mouse mouse{};
    keyboard::Keymap keymap{};

    [[no_unique_address]] Init init;
    gl::Context context;

    explicit Window(const WindowConfig& config = {});

    Window(const Window&)                    = delete;
    auto operator=(const Window&) -> Window& = delete;

    Window(Window&&) noexcept                    = delete;
    auto operator=(Window&&) noexcept -> Window& = delete;

    ~Window() = default;

    [[nodiscard]] auto is_open() const -> bool;

    void close();

    void get_inputs();

    void display();

    [[nodiscard]] auto is_key_pressed(Key key) const -> bool;

    /**
     * @brief               The aspect ratio of the window given by width / height
     *
     * @return f64
     */
    [[nodiscard]] auto aspect_ratio() const -> f64;

    /**
     * @brief               A Vector2 given by [width, height]  / max(width, height)
     *
     * @return Vector2
     */
    [[nodiscard]] auto aspect_vector_min() const -> Vector2;

    /**
     * @brief               A Vector2 given by [width, height]  / min(width, height)
     *
     * @return Vector2
     */
    [[nodiscard]] auto aspect_vector_max() const -> Vector2;

    /**
     * @brief               The bounds of the visible region of the window
     *
     * @tparam space
     * @return BoundingBox<f64>
     */
    template <Space space = Space::World> [[nodiscard]] auto viewport() const -> BoundingBox<f64>
    {
        const auto ratio = aspect_ratio();
        const auto box   = BoundingBox<f64>{{-ratio, -1.0}, {ratio, 1.0}};

        if constexpr (space == Space::World)
        {
            auto transform = view;
            transform.scale *= 2.0;
            // TODO without dividing by 2 stuff is off
            return transform.apply_inverse(box);
        }
    }
    // TODO implement it for Space::Screen

    // ##################################
    // Coordinate Systems
    // gl: x: [-1, 1], y: [-r, r] where r = 1/aspect_ratio. +ve y is upward. origin at center of
    // screen
    //
    // px: pixel space: x: [0, dims.x), y: [0, dims.y). +ve y is downard. origin at top left corner of screen
    [[nodiscard]] auto view_px2gl() const -> Transform
    {
        const auto dimsf  = dims.cast<f64>();
        const auto factor = dimsf.y / dims.x;
        const auto scale  = Vector2{2.0, -2.0 * factor} / dimsf; // 2 map [0,1] to [-1,1].
        return Transform{{-1.0, factor}, scale};                 // todo why +factor not -factor
    }

    [[nodiscard]] auto view_gl2px() const -> Transform
    {
        return view_px2gl().inverse();
    }

    

    /**
     * @brief               Get the pixels currently rendered
     *
     * @return Image
     */
    [[nodiscard]] auto get_image() const -> Image;

    /**
     * @brief               Write current pixels to target.
     * Assumes window size is >= image size
     * @return void
     */
    [[nodiscard]] auto get_image(Image& target) const -> void;
};
} // namespace sm


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_WINDOW_IMPL)

#include "glm/gtc/matrix_transform.hpp" // for ortho

#include "Window.hpp"
#include "samarium/core/inline.hpp" // for SM_INLINE

namespace sm
{
SM_INLINE Window::Window(const WindowConfig& config)
    : dims{config.dims}, init{config, handle}, context{config.dims}
{
    gl::enable_debug_output();

    glEnable(GL_BLEND); // enable blending function
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    keymap.push_back(keyboard::OnKeyPress{
        *handle, {Key::Escape}, [this] { this->close(); }}); // by default, exit on escape

    resize_callback.holder.dims = dims;
    glfwSetFramebufferSizeCallback(handle.get(), resize_callback.thunk);
    glfwSetScrollCallback(handle.get(), scroll_callback.thunk);
}

SM_INLINE auto Window::is_open() const -> bool { return glfwWindowShouldClose(handle.get()) == 0; }

SM_INLINE void Window::close() { glfwSetWindowShouldClose(handle.get(), true); }

SM_INLINE void Window::get_inputs()
{
    scroll_callback.holder.scroll = 0.0;

    glfwPollEvents();

    mouse.left    = glfwGetMouseButton(handle.get(), GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    mouse.right   = glfwGetMouseButton(handle.get(), GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    mouse.old_pos = mouse.pos;

    auto xpos = 0.0;
    auto ypos = 0.0;
    glfwGetCursorPos(handle.get(), &xpos, &ypos);

    // since the view can change based on mouse movement
    // keep mouse pos in pixel coords, which are stable
    // TODO what abt when moving window
    // mouse.pos = view_pixel_to_gl().apply(Vector2{xpos, ypos});
    mouse.pos = Vector2{xpos, ypos};

    mouse.scroll_amount = scroll_callback.holder.scroll;

    keymap.run();
}

// TODO should call display once at the start or fix view.scale.y
SM_INLINE void Window::display()
{
    context.draw_frame();

    resize_callback.holder.resized = false;
    glfwSwapBuffers(handle.get());
    dims         = resize_callback.holder.dims;
    view.scale.y = view.scale.x * aspect_ratio(); // shd be inside below if?

    if (resize_callback.holder.resized)
    {
        context.frame_texture =
            gl::Texture{gl::ImageFormat::RGBA8, dims, gl::Texture::Wrap::ClampEdge,
                        gl::Texture::Filter::Nearest, gl::Texture::Filter::Nearest};
        context.framebuffer.bind_texture(context.frame_texture);
    }

    get_inputs();
}

SM_INLINE auto Window::is_key_pressed(Key key) const -> bool
{
    return glfwGetKey(handle.get(), static_cast<i32>(key)) == GLFW_PRESS;
}

SM_INLINE auto Window::aspect_ratio() const -> f64
{
    const auto current_dims = dims.cast<f64>();
    return current_dims.x / current_dims.y;
}

SM_INLINE auto Window::aspect_vector_min() const -> Vector2
{
    return dims.cast<f64>() / static_cast<f64>(math::max(dims.x, dims.y));
}

SM_INLINE auto Window::aspect_vector_max() const -> Vector2
{
    return dims.cast<f64>() / static_cast<f64>(math::min(dims.x, dims.y));
}

SM_INLINE auto Window::get_image(Image& target) const -> void
{
    glReadPixels(0, 0, static_cast<i32>(target.dims.x), static_cast<i32>(target.dims.y), GL_RGBA,
                 GL_UNSIGNED_BYTE, static_cast<void*>(&target.front()));
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
