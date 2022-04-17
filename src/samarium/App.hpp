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
#include "math/geometry.hpp"
#include "physics/Particle.hpp"
#include "util/Stopwatch.hpp"

namespace sm
{

namespace concepts
{
// takes a const Vector2& and returns a Color
template <typename T>
concept DrawableLambda = std::is_invocable_r_v<Color, T, const Vector2&>;
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
        uint32_t framerate{60};
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
                    .scale = Vector2::combine(16) * Vector2{1.0, -1.0}}
    {
        texture.create(static_cast<u32>(settings.dims.x), static_cast<u32>(settings.dims.y));

        sf_render_window.setFramerateLimit(settings.framerate);

        keymap.push_back({Keyboard::Key::LControl, Keyboard::Key::Q}, // exit by default with Ctrl+Q
                         [&sf_render_window = this->sf_render_window]
                         { sf_render_window.close(); });
    }

    void sync_window_to_image();

    void sync_image_to_window();

    void display();

    auto is_open() const -> bool;

    void get_input();

    auto bounding_box() const -> BoundingBox<size_t>;

    auto viewport_box() const -> std::array<LineSegment, 4>;

    void fill(Color color);

    void draw(Circle circle, Color color);

    void draw(const Particle& particle);

    void draw_line_segment(const LineSegment& ls,
                           Color color   = Color{255, 255, 255},
                           f64 thickness = 0.1);

    template <concepts::DrawableLambda T> void draw(T&& fn)
    {
        const auto bounding_box = image.bounding_box();
        this->draw(std::forward<T>(fn), transform.apply_inverse(BoundingBox<u64>{
                                            .min = bounding_box.min,
                                            .max = bounding_box.max +
                                                   Indices{1, 1}}.template as<f64>()));
    }

    void draw(concepts::DrawableLambda auto&& fn, const BoundingBox<f64>& bounding_box)
    {
        sync_window_to_image();

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
                    const auto coords_transformed =
                        transform.apply_inverse(coords.template as<f64>());

                    const auto col = fn(coords_transformed);

                    image[coords].add_alpha_over(col);
                }
            }
        };

        thread_pool.parallelize_loop(y_range.min, y_range.max + 1, job,
                                     thread_pool.get_thread_count());

        sync_image_to_window();
    }

    void run(std::invocable<f64> auto&& update, std::invocable auto&& draw, u64 substeps = 1UL)
    {
        while (this->is_open())
        {
            this->get_input();

            for (auto i : range(substeps))
            {
                std::ignore = i;
                update(1.0 / static_cast<f64>(substeps) / static_cast<f64>(this->target_framerate));
            }
            draw();

            this->display();
        }
    }

    void run(std::invocable auto&& func)
    {
        while (this->is_open())
        {
            this->get_input();
            func();
            this->display();
        }
    }
};
} // namespace sm
