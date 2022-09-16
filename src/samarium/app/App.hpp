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
#include "samarium/util/FunctionRef.hpp" // for ThreadPool
#include "samarium/util/Stopwatch.hpp"   // for Stopwatch
#include "samarium/util/ThreadPool.hpp"  // for ThreadPool

namespace sm
{
class Trail;
struct Particle;
struct LineSegment;
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

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_APP_IMPL)

#include <tuple> // for ignore

#include "SFML/Graphics/CircleShape.hpp"   // for CircleShape
#include "SFML/Graphics/Color.hpp"         // for Color
#include "SFML/Graphics/Image.hpp"         // for Image
#include "SFML/Graphics/PrimitiveType.hpp" // for PrimitiveType, Quads
#include "SFML/Graphics/RenderWindow.hpp"  // for RenderWindow
#include "SFML/Graphics/Sprite.hpp"        // for Sprite
#include "SFML/Graphics/Texture.hpp"       // for Texture
#include "SFML/Graphics/Vertex.hpp"        // for Vertex
#include "SFML/Graphics/VertexArray.hpp"   // for VertexArray
#include "SFML/System/Vector2.hpp"         // for Vector2f
#include "SFML/Window/Event.hpp"           // for Event, Event::(anonymous)
#include "SFML/Window/Mouse.hpp"           // for Mouse, Mouse::VerticalWheel
#include "range/v3/algorithm/copy.hpp"     // for copy_fn, copy

#include "samarium/graphics/Trail.hpp"   // for Trail
#include "samarium/gui/sfml.hpp"         // for sfml
#include "samarium/math/Extents.hpp"     // for Extents, Extents<>::Iterator
#include "samarium/math/math.hpp"        // for pi
#include "samarium/math/shapes.hpp"      // for LineSegment, Circle
#include "samarium/math/vector_math.hpp" // for area
#include "samarium/physics/Particle.hpp" // for Particle
#include "samarium/util/FunctionRef.hpp" // for FunctionRef

namespace sm
{
void App::load_pixels()
{
    this->texture.update(this->sf_render_window);
    const auto sf_image = texture.copyToImage();
    const auto ptr      = sf_image.getPixelsPtr();

    ranges::copy(ptr, ptr + image.size() * 4UL, reinterpret_cast<u8*>(&image[0]));
}

void App::store_pixels()
{
    this->texture.update(reinterpret_cast<const u8*>(&image[0]));
    sf::Sprite sprite(texture);

    this->sf_render_window.draw(sprite);
}

void App::display()
{
    sf_render_window.display();
    frame_counter++;
    clock.reset();
}

void App::fill(Color color) { sf_render_window.clear(sfml(color)); }

auto App::is_open() const -> bool { return sf_render_window.isOpen(); }

void App::get_input()
{
    this->mouse.scroll_amount = 0.0;

    sf::Event event;
    while (sf_render_window.pollEvent(event))
    {
        if (event.type == sf::Event::Closed) { sf_render_window.close(); }
        else if (event.type == sf::Event::MouseWheelScrolled &&
                 event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel)
        {
            this->mouse.scroll_amount = static_cast<f64>(event.mouseWheelScroll.delta);
        }
    }

    this->keymap.run();
    this->mouse.update(this->sf_render_window);
}

auto App::dims() const -> Dimensions { return image.dims; }

auto App::transformed_dims() const -> Vector2 { return this->dims().as<f64>() / transform.scale; }

auto App::bounding_box() const -> BoundingBox<u64> { return this->image.bounding_box(); }

auto App::transformed_bounding_box() const -> BoundingBox<f64>
{
    const auto box = this->image.bounding_box();
    return this->transform.apply_inverse(BoundingBox<f64>{
        box.min.as<f64>(),
        box.max.as<f64>() +
            Vector2{1.0,
                    1.0}}); // add 1 to compensate for inclusive-exclusive from Image::bounding_box
}

auto App::viewport_box() const -> std::array<LineSegment, 4>
{
    const auto f64_dims = this->image.dims.as<f64>();

    return std::array{
        this->transform.apply_inverse(LineSegment{{}, {0.0, f64_dims.y}}),
        this->transform.apply_inverse(LineSegment{{}, {f64_dims.x, 0.0}}),
        this->transform.apply_inverse(LineSegment{{f64_dims.x, 0.0}, {f64_dims.x, f64_dims.y}}),
        this->transform.apply_inverse(LineSegment{{0.0, f64_dims.y}, {f64_dims.x, f64_dims.y}})};
}

auto App::get_image() -> Image
{
    this->load_pixels();
    return this->image;
}

void App::draw(Circle circle, ShapeColor color, u64 vertex_count)
{
    const auto border_width = static_cast<f32>(color.border_width * transform.scale.x);
    const auto radius       = static_cast<f32>(circle.radius * transform.scale.x);
    const auto pos          = transform.apply(circle.centre).as<f32>();

    auto shape = sf::CircleShape{radius, vertex_count};
    shape.setFillColor(sfml(color.fill_color));
    shape.setOutlineColor(sfml(color.border_color));
    shape.setOutlineThickness(border_width);
    shape.setOrigin(radius, radius);
    shape.setPosition(pos.x, pos.y);

    sf_render_window.draw(shape);
}

void App::draw(BoundingBox<f64> box, ShapeColor color)
{
    const auto border_width = static_cast<f32>(color.border_width * transform.scale.x);
    const auto pos          = transform.apply(box.centre()).as<f32>();
    const auto displacement = sfml((box.displacement() * transform.scale.x).as<f32>().abs());

    auto shape = sf::RectangleShape{displacement};
    shape.setFillColor(sfml(color.fill_color));
    shape.setOutlineColor(sfml(color.border_color));
    shape.setOutlineThickness(border_width);
    shape.setPosition(pos.x - displacement.x / 2.0f, pos.y - displacement.y / 2.0f);

    sf_render_window.draw(shape);
}

void App::draw(const Particle& particle, ShapeColor color)
{
    this->draw(particle.as_circle(), color);
}

void App::draw_line_segment(const LineSegment& ls, Color color, f64 thickness)
{
    const auto sfml_color = sfml(color);
    const auto thickness_vector =
        Vector2::from_polar({.length = thickness, .angle = ls.vector().angle() + math::pi / 2.0});

    auto vertices = std::array<sf::Vertex, 4>{};

    vertices[0].position = sfml(transform.apply(ls.p1 - thickness_vector).as<f32>());
    vertices[1].position = sfml(transform.apply(ls.p1 + thickness_vector).as<f32>());
    vertices[2].position = sfml(transform.apply(ls.p2 + thickness_vector).as<f32>());
    vertices[3].position = sfml(transform.apply(ls.p2 - thickness_vector).as<f32>());

    vertices[0].color = sfml_color;
    vertices[1].color = sfml_color;
    vertices[2].color = sfml_color;
    vertices[3].color = sfml_color;

    sf_render_window.draw(vertices.data(), vertices.size(), sf::Quads);
}

void App::draw_world_space(FunctionRef<Color(Vector2)> callable)
{
    const auto bounding_box    = image.bounding_box();
    const auto transformed_box = transform.apply_inverse(BoundingBox<u64>{
        .min = bounding_box.min,
        .max = bounding_box.max + Indices{1, 1}}.template as<f64>());

    this->draw_world_space(callable, transformed_box);
}

void App::draw_world_space(FunctionRef<Color(Vector2)> callable,
                           const BoundingBox<f64>& bounding_box)
{
    load_pixels();

    const auto box = this->transform.apply(bounding_box)
                         .clamped_to(image.bounding_box().template as<f64>())
                         .template as<u64>();

    if (math::area(box) == 0UL) { return; }

    const auto x_range = box.x_range(); // inclusive range
    const auto y_range = box.y_range();

    const auto job = [&](auto min, auto max)
    {
        for (auto y : range(min, max))
        {
            for (auto x : x_range)
            {
                const auto coords = Indices{x, y};
                const auto col    = callable(transform.apply_inverse(coords.template as<f64>()));

                image[coords].add_alpha_over(col);
            }
        }
    };

    thread_pool.parallelize_loop(y_range.min, y_range.max + 1, job, thread_pool.get_thread_count())
        .wait();

    store_pixels();
}

void App::draw_screen_space(FunctionRef<Color(Indices)> callable)
{
    const auto bounding_box = image.bounding_box();
    this->draw_screen_space(callable, bounding_box);
}

void App::draw_screen_space(FunctionRef<Color(Indices)> callable,
                            const BoundingBox<u64>& bounding_box)
{
    load_pixels();

    const auto box = bounding_box.clamped_to(image.bounding_box());

    if (math::area(box) == 0UL) { return; }

    const auto x_range = box.x_range();
    const auto y_range = box.y_range();

    const auto job = [&](auto min, auto max)
    {
        for (auto y : range(min, max))
        {
            for (auto x : x_range)
            {
                const auto coords = Indices{x, y};
                const auto col    = callable(coords);

                image[coords].add_alpha_over(col);
            }
        }
    };
    thread_pool.parallelize_loop(y_range.min, y_range.max + 1, job, thread_pool.get_thread_count())
        .wait();
    store_pixels();
}

void App::draw_polyline(std::span<const Vector2> vertices, Color color, f64 thickness)
{
    if (vertices.size() < 2) { return; }

    for (auto i : range(vertices.size() - 1UL))
    {
        draw_line_segment(LineSegment{vertices[i], vertices[i + 1UL]}, color, thickness);
    }
}

void App::draw_polygon(std::span<const Vector2> vertices, Color color, f64 thickness)
{
    if (vertices.size() < 3) { return; }

    for (auto i : range(vertices.size() - 1UL))
    {
        draw_line_segment(LineSegment{vertices[i], vertices[i + 1UL]}, color, thickness);
    }
    draw_line_segment(LineSegment{vertices.back(), vertices.front()}, color, thickness);
}

void App::draw_vertices(std::span<const Vector2> vertices, VertexMode mode)
{
    if (vertices.empty()) { return; }
    auto converted = sf::VertexArray(static_cast<sf::PrimitiveType>(mode), vertices.size());
    for (auto i : range(vertices.size())) { converted[i].position = sfml(vertices[i].as<f32>()); }
    sf_render_window.draw(converted);
}

void App::draw(const Trail& trail, Color color, f64 thickness, f64 fade_factor)
{
    if (trail.size() < 2) { return; }
    const auto vertices = trail.span();
    for (auto i : range(vertices.size() - 1UL))
    {
        const auto factor = 1.0 - static_cast<f64>(i) / static_cast<f64>(vertices.size() - 1UL);
        const auto faded_color = color.with_multiplied_alpha(-fade_factor * factor + 1.0);
        draw_line_segment(LineSegment{vertices[i], vertices[i + 1UL]}, faded_color, thickness);
    }
}

void App::draw(const App::GridLines& settings)
{
    const auto box            = this->transformed_bounding_box();
    const auto axis_thickness = settings.axis_thickness /*  / this->transform.scale.y */;
    const auto line_thickness = settings.line_thickness /*  / this->transform.scale.y */;

    this->draw_line_segment({{0.0, box.min.y}, {0.0, box.max.y}}, settings.axis_color,
                            axis_thickness);

    this->draw_line_segment({{box.min.x, 0.0}, {box.max.x, 0.0}}, settings.axis_color,
                            axis_thickness);

    if (settings.levels <= 1) { return; }

    for (auto i = settings.scale; i <= box.max.x; i += settings.scale)
    {
        this->draw_line_segment({{i, box.min.y}, {i, box.max.y}}, settings.line_color,
                                line_thickness);
    }
    for (auto i = -settings.scale; i >= box.min.x; i -= settings.scale)
    {
        this->draw_line_segment({{i, box.min.y}, {i, box.max.y}}, settings.line_color,
                                line_thickness);
    }

    for (auto i = settings.scale; i <= box.max.y; i += settings.scale)
    {
        this->draw_line_segment({{box.min.x, i}, {box.max.x, i}}, settings.line_color,
                                line_thickness);
    }
    for (auto i = -settings.scale; i >= box.min.y; i -= settings.scale)
    {
        this->draw_line_segment({{box.min.x, i}, {box.max.x, i}}, settings.line_color,
                                line_thickness);
    }
}

void App::draw(const App::GridDots& settings)
{
    const auto box    = this->transformed_bounding_box();
    const auto radius = settings.thickness /*  / this->transform.scale.x */;

    const auto from_y = math::ceil_to_nearest(box.min.y, settings.scale);
    const auto to_y   = math::floor_to_nearest(box.max.y, settings.scale);

    const auto from_x = math::ceil_to_nearest(box.min.x, settings.scale);
    const auto to_x   = math::floor_to_nearest(box.max.x, settings.scale);

    for (auto y = from_y; y <= to_y; y += settings.scale)
    {
        for (auto x = from_x; x <= to_x; x += settings.scale)
        {
            this->draw(Circle{{x, y}, radius}, {.fill_color = settings.color}, 4);
        }
    }
}

void App::run(FunctionRef<void()> func)
{
    while (this->is_open())
    {
        this->get_input();
        func();
        this->display();
    }
}

void App::run(FunctionRef<void(f64)> update, FunctionRef<void()> draw, u64 substeps)
{
    while (this->is_open())
    {
        this->get_input();

        for (auto i : range(substeps))
        {
            std::ignore = i;
            update(1.0 / static_cast<f64>(substeps) / static_cast<f64>(this->target_framerate));
            // update(clock.seconds() / static_cast<f64>(substeps));
        }
        draw();

        this->display();
    }
}

void App::run(FunctionRef<void()> handle_input,
              FunctionRef<void(f64)> update,
              FunctionRef<void()> draw,
              u64 substeps)
{
    while (this->is_open())
    {
        this->get_input();
        handle_input();
        for (auto i : range(substeps))
        {
            std::ignore = i;
            update(1.0 / static_cast<f64>(substeps) / static_cast<f64>(this->target_framerate));
        }
        draw();

        this->display();
    }
}

void App::zoom_pan(f64 zoom_factor, f64 pan_factor)
{
    if (mouse.left) { transform.pos += pan_factor * (mouse.current_pos - mouse.old_pos); }

    const auto scale = 1.0 + zoom_factor * mouse.scroll_amount;
    transform.scale *= Vector2::combine(scale);
    transform.pos = mouse.current_pos + scale * pan_factor * (transform.pos - mouse.current_pos);
}

void App::zoom_pan(FunctionRef<bool()> zoom_condition,
                   FunctionRef<bool()> pan_condition,
                   f64 zoom_factor,
                   f64 pan_factor)
{
    if (pan_condition()) { transform.pos += pan_factor * (mouse.current_pos - mouse.old_pos); }
    if (zoom_condition())
    {
        const auto scale = 1.0 + zoom_factor * mouse.scroll_amount;
        transform.scale *= Vector2::combine(scale);
        transform.pos =
            mouse.current_pos + scale * pan_factor * (transform.pos - mouse.current_pos);
    }
}
} // namespace sm
#endif
