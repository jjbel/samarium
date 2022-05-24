/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <algorithm>
#include <tuple>

#include "SFML/Graphics.hpp"

#include "core/ThreadPool.hpp"
#include "graphics/Color.hpp"
#include "graphics/Image.hpp"
#include "gui/Keyboard.hpp"
#include "gui/Mouse.hpp"
#include "math/BoundingBox.hpp"
#include "math/Extents.hpp"
#include "math/Transform.hpp"
#include "math/vector_math.hpp"
#include "physics/Particle.hpp"
#include "util/FunctionRef.hpp"
#include "util/Stopwatch.hpp"

namespace sm
{

enum class CoordinateSpace
{
    Screen,
    World
};

namespace concepts
{

template <typename T, CoordinateSpace cs = CoordinateSpace::World> struct is_drawable
{
    static constexpr auto value =
        ((cs == CoordinateSpace::Screen) && std::is_invocable_r_v<Color, T, Indices>) ||
        ((cs == CoordinateSpace::World) && std::is_invocable_r_v<Color, T, Vector2>);
};

template <typename T, CoordinateSpace cs = CoordinateSpace::World>
concept DrawableLambda = is_drawable<T, cs>::value;
} // namespace concepts

class App
{
    sf::RenderWindow sf_render_window;
    sf::Texture texture;
    Stopwatch watch{};
    u64 target_framerate;
    Image image;

  public:
    struct Settings
    {
        Dimensions dims{sm::dimsFHD};
        std::string name{"Samarium Window"};
        uint32_t framerate{64};
    };

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

    void load_pixels();

    void store_pixels();

    void display();

    auto is_open() const -> bool;

    void get_input();

    auto dims() const -> Dimensions;

    auto transformed_dims() const -> Vector2;

    auto bounding_box() const -> BoundingBox<size_t>;

    auto viewport_box() const -> std::array<LineSegment, 4>;

    auto get_image() const -> Image;

    void fill(Color color);

    void draw(Circle circle, Color color);

    void draw(const Particle& particle);

    void draw_line_segment(const LineSegment& ls,
                           Color color   = Color{255, 255, 255},
                           f64 thickness = 0.1);

    template <typename T>
    requires concepts::DrawableLambda<T, CoordinateSpace::World>
    void draw(T&& fn) { this->draw<CoordinateSpace::World>(std::forward<T>(fn)); }

    template <CoordinateSpace cs, typename T>
    requires concepts::DrawableLambda<T, cs>
    void draw(T&& fn)
    {
        const auto bounding_box    = image.bounding_box();
        const auto transformed_box = transform.apply_inverse(BoundingBox<u64>{
            .min = bounding_box.min,
            .max = bounding_box.max + Indices{1, 1}}.template as<f64>());

        this->draw<cs>(std::forward<T>(fn), transformed_box);
    }

    template <CoordinateSpace cs, typename T>
    requires concepts::DrawableLambda<T, cs>
    void draw(T&& fn, const BoundingBox<f64>& bounding_box)
    {
        load_pixels();

        const auto box = this->transform.apply(bounding_box)
                             .clamped_to(image.bounding_box().template as<f64>())
                             .template as<u64>();

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
                    const auto col    = invoke_fn<cs>(fn, coords);

                    image[coords].add_alpha_over(col);
                }
            }
        };

        thread_pool.parallelize_loop(y_range.min, y_range.max + 1, job,
                                     thread_pool.get_thread_count());

        store_pixels();
    }


    void run(FunctionRef<void(f64)> update, FunctionRef<void()> draw, u64 substeps = 1UL);

    void run(FunctionRef<void()> func);

  private:
    template <CoordinateSpace cs>
    inline auto invoke_fn(const auto& fn, Indices coords) const
        requires(cs == CoordinateSpace::Screen)
    {
        return fn(coords);
    }

    template <CoordinateSpace cs>
    inline auto invoke_fn(const auto& fn, Indices coords) const
        requires(cs == CoordinateSpace::World)
    {
        return fn(transform.apply_inverse(coords.template as<f64>()));
    }
};
} // namespace sm
