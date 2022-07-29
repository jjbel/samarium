/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <array>  // for array
#include <span>   // for span
#include <string> // for allocator, string

#include "SFML/Graphics/RenderWindow.hpp"  // for RenderWindow
#include "SFML/Graphics/Texture.hpp"       // for Texture
#include "SFML/System/String.hpp"          // for String
#include "SFML/Window/ContextSettings.hpp" // for ContextSettings
#include "SFML/Window/VideoMode.hpp"       // for VideoMode
#include "SFML/Window/WindowStyle.hpp"     // for Close, Titlebar

#include "samarium/core/types.hpp"       // for f64, u32, u64
#include "samarium/graphics/Color.hpp"   // for Color, ShapeColor
#include "samarium/graphics/Image.hpp"   // for Image, dimsFHD
#include "samarium/gui/Keyboard.hpp"     // for Keyboard, Keyboard::Key
#include "samarium/gui/Mouse.hpp"        // for Mouse
#include "samarium/math/BoundingBox.hpp" // for BoundingBox
#include "samarium/math/Transform.hpp"   // for Transform
#include "samarium/math/Vector2.hpp"     // for Vector2, Dimensions
#include "samarium/util/Stopwatch.hpp"   // for Stopwatch
#include "samarium/util/ThreadPool.hpp"  // for ThreadPool

namespace sm
{
class Trail;
struct Particle;
struct LineSegment;
template <class F> class FunctionRef;
struct Circle;

enum class VertexMode
{
    Points,
    Lines,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
    Quads
};

/**
 * @brief               App encapsulates an event loop, rendering and user input
 */
class App
{
  public:
    sf::RenderWindow sf_render_window;
    sf::Texture texture;
    u64 target_framerate;

    struct Settings
    {
        Dimensions dims{sm::dimsFHD};
        std::string name{"Samarium Window"};
        u32 framerate{64};
    };

    Stopwatch clock{};
    Image image;
    ThreadPool thread_pool{};
    Transform transform;
    u64 frame_counter{};
    Keyboard::Keymap keymap{};
    Mouse mouse{sf_render_window};

    explicit App(const Settings& settings)
        : sf_render_window(
              sf::VideoMode(static_cast<u32>(settings.dims.x), static_cast<u32>(settings.dims.y)),
              settings.name,
              sf::Style::Titlebar | sf::Style::Close,
              sf::ContextSettings{0, 0, /* antialiasing factor */ 8}),
          target_framerate{settings.framerate}, image{settings.dims},
          transform{.pos   = image.dims.as<f64>() / 2.,
                    .scale = Vector2::combine(10) * Vector2{1.0, -1.0}}
    {
        texture.create(static_cast<u32>(settings.dims.x), static_cast<u32>(settings.dims.y));

        sf_render_window.setFramerateLimit(settings.framerate);

        keymap.push_back(Keyboard::OnKeyDown({Keyboard::Key::Escape}, // exit by default with Escape
                                             [&sf_render_window = this->sf_render_window]
                                             { sf_render_window.close(); }));
    }

    /// Load the current GPU pixels to the image to draw on manually
    void load_pixels();

    /// Store the current pixels of the image in the GPU
    void store_pixels();

    /// Display the current contents of the screen, update `clock` and
    void display();

    /// Is the window currently open
    auto is_open() const -> bool;

    /// Load the current `keymap` and `mouse` events
    void get_input();

    /// Dimensions of the viewport in screenspace coordinates
    auto dims() const -> Dimensions;

    /// The `Dimensions` of the viewport in worldspace coordinates
    auto transformed_dims() const -> Vector2;

    /// The `BoundingBox` formed by the viewport in screenspace coordinates
    auto bounding_box() const -> BoundingBox<u64>;

    /// The `BoundingBox` formed by the viewport in worldspace coordinates
    auto transformed_bounding_box() const -> BoundingBox<f64>;

    /// The 4 `LineSegment`'s forming the viewportm, in worldspace coordinates
    auto viewport_box() const -> std::array<LineSegment, 4>;

    /// Get a copy of the pixels currently on the GPU
    auto get_image() -> Image;

    /**
     * @brief               Fill the entire screen with `color`
     *
     * @param  color
     */
    void fill(Color color);

    void draw(Circle circle, ShapeColor color, u64 vertex_count = 64);

    void draw(BoundingBox<f64> box, ShapeColor color);

    void draw(const Particle& particle, ShapeColor color);

    void draw_line_segment(const LineSegment& ls,
                           Color color   = Color{255, 255, 255},
                           f64 thickness = 0.1);


    void draw_world_space(FunctionRef<Color(Vector2)> callable);
    void draw_world_space(FunctionRef<Color(Vector2)> callable,
                          const BoundingBox<f64>& bounding_box);

    void draw_screen_space(FunctionRef<Color(Indices)> callable);
    void draw_screen_space(FunctionRef<Color(Indices)> callable,
                           const BoundingBox<u64>& bounding_box);

    void draw_polyline(std::span<const Vector2> vertices,
                       Color color   = Color{255, 255, 255},
                       f64 thickness = 1.0);

    struct GridLines
    {
        f64 scale          = 5.0;
        Color line_color   = {240, 240, 240, 50};
        f64 line_thickness = 0.02;
        Color axis_color   = {240, 240, 240, 80};
        f64 axis_thickness = 0.1;
        u64 levels         = 2UL;
    };

    struct GridDots
    {
        f64 scale     = 5.0;
        Color color   = {240, 240, 240, 50};
        f64 thickness = 0.1;
    };

    void draw(const GridLines& settings);
    void draw(const GridDots& settings);

    /**
     * @brief               Draw a polygon from points
     *
     * @param  vertices     Input points
     * @param  color
     * @param  thickness
     */
    void draw_polygon(std::span<const Vector2> vertices,
                      Color color   = Color{255, 255, 255},
                      f64 thickness = 1.0);

    /**
     * @brief               Draw a Trail
     *
     * @param  trail        Trail object
     * @param  color        Trail Color
     * @param  thickness    Width of trail
     * @param  fade_factor  Lerp color from alpha = 1.0 to fade_factor
     */
    void draw(const Trail& trail,
              Color color     = Color{255, 255, 255},
              f64 thickness   = 0.1,
              f64 fade_factor = 0.0);

    void draw_vertices(std::span<const Vector2> vertices, VertexMode mode = VertexMode::LineStrip);

    /**
     * @brief               Start the event loop and call func every iteration
     *
     * @param  func         Draw/Update function
     */
    void run(FunctionRef<void()> func);

    /**
     * @brief               Start the event loop, call `update` substeps times, `draw` once
     *
     * @param  update       Called `substeps` times
     * @param  draw
     * @param  substeps
     */
    void run(FunctionRef<void(f64)> update, FunctionRef<void()> draw, u64 substeps = 1UL);

    /**
     * @brief               Start the event loop, call `update` substeps times, `draw` once
     *
     * @param  handle_input Called first
     * @param  update       Called `substeps` times
     * @param  draw
     * @param  substeps
     */
    void run(FunctionRef<void()> handle_input,
             FunctionRef<void(f64)> update,
             FunctionRef<void()> draw,
             u64 substeps = 1UL);

    /**
     * @brief               Zoom and pan by adjusting `transform`, by default on left-click
     *
     * @param  zoom_factor  Factor to zoom in by
     * @param  pan_factor   Factor to scale panning by (normally 1)
     */
    void zoom_pan(f64 zoom_factor = 0.1, f64 pan_factor = 1.0);

    /**
     * @brief               Zoom and pan by adjusting `transform`, given conditions on when to zoom
     * and pan
     *
     * @param  zoom_condition
     * @param  pan_condition
     * @param  zoom_factor
     * @param  pan_factor
     */
    void zoom_pan(FunctionRef<bool()> zoom_condition,
                  FunctionRef<bool()> pan_condition,
                  f64 zoom_factor = 0.1,
                  f64 pan_factor  = 1.0);
};
} // namespace sm
