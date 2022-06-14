/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <algorithm>
#include <span>
#include <tuple>

#include "SFML/Graphics.hpp"

#include "../core/ThreadPool.hpp"
#include "../graphics/Image.hpp"
#include "../graphics/Trail.hpp"
#include "../gui/Keyboard.hpp"
#include "../gui/Mouse.hpp"
#include "../math/BoundingBox.hpp"
#include "../math/Extents.hpp"
#include "../math/Transform.hpp"
#include "../math/vector_math.hpp"
#include "../physics/Particle.hpp"
#include "../util/FunctionRef.hpp"
#include "../util/Stopwatch.hpp"

namespace sm
{
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
    sf::RenderWindow sf_render_window;
    sf::Texture texture;
    Stopwatch watch{};
    u64 target_framerate;

  public:
    struct Settings
    {
        Dimensions dims{sm::dimsFHD};
        std::string name{"Samarium Window"};
        uint32_t framerate{64};
    };

    Image image;
    ThreadPool thread_pool{};
    Transform transform;
    u64 frame_counter{};
    Keymap keymap;
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

        keymap.push_back({Keyboard::Key::LControl, Keyboard::Key::Q}, // exit by default with Ctrl+Q
                         [&sf_render_window = this->sf_render_window]
                         { sf_render_window.close(); });
    }

    /**
     * @brief               Load the current GPU pixels to the image to draw on manually
     */
    void load_pixels();

    /**
     * @brief               Store the current pixels of the image in the GPU
     */
    void store_pixels();

    /**
     * @brief               Display the current contents of the screen, update `watch` and `frame_counter`
     */
    void display();

    /**
     * @brief               Is the window currently open
     * 
     * @return true         
     * @return false        
     */
    auto is_open() const -> bool;

    /**
     * @brief               Load the current `keymap` and `mouse` events
     */
    void get_input();

    /**
     * @brief               Dimensions of the viewport in screenspace coordinates
     */
    auto dims() const -> Dimensions;

    /**
     * @brief               The `Dimensions` of the viewport with `transform` applied
     */
    auto transformed_dims() const -> Vector2;

    /**
     * @brief               The `BoundingBox` formed by the viewport
     * 
     * @return BoundingBox<size_t> 
     */
    auto bounding_box() const -> BoundingBox<size_t>;

    /**
     * @brief               The 4 `LineSegment`'s forming the viewportm, in worldspace coordinates
     */
    auto viewport_box() const -> std::array<LineSegment, 4>;

    /**
     * @brief               Get a copy of the pixels currently on the GPU
     */
    auto get_image() -> Image;

    /**
     * @brief               Fill the entire screen with `color`
     * 
     * @param  color
     */
    void fill(Color color);

    void draw(Circle circle, Color color);

    void draw(const Particle& particle);

    void draw_line_segment(const LineSegment& ls,
                           Color color   = Color{255, 255, 255},
                           f64 thickness = 0.1);


    void draw_world_space(FunctionRef<Color(Vector2)> callable);
    void draw_world_space(FunctionRef<Color(Vector2)> callable,
                          const BoundingBox<f64>& bounding_box);

    void draw_polyline(std::span<const Vector2> vertices,
                       Color color   = Color{255, 255, 255},
                       f64 thickness = 1.0);

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
     * @param  trail        
     * @param  color        
     * @param  thickness    
     */
    void draw(Trail trail, Color color = Color{255, 255, 255}, f64 thickness = 1.0);

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
};
} // namespace sm
