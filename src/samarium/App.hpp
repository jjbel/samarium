/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <algorithm>
#include <tuple>

#include "./graphics/Renderer.hpp"
#include "./gui/Window.hpp"

namespace sm
{
class App
{
    sf::RenderWindow sf_render_window;
    sf::Texture texture;
    sf::Sprite sprite{texture};
    Image image;
    Stopwatch watch{};
    u64 target_framerate;

    void sync_texture_to_image();

    void sync_image_to_texture();

  public:
    using Settings = Window::Settings;

    Transform transform;
    u64 frame_counter{};
    Keymap keymap;
    Mouse mouse{sf_render_window};

    explicit App(const Settings& settings)
        : sf_render_window(
              sf::VideoMode(static_cast<u32>(settings.dims.x), static_cast<u32>(settings.dims.y)),
              settings.name,
              sf::Style::Titlebar | sf::Style::Close,
              sf::ContextSettings{0, 0, 8}),
          image{settings.dims}, target_framerate{settings.framerate},
          transform{.pos   = image.dims.as<f64>() / 2.,
                    .scale = Vector2::combine(16) * Vector2{1.0, -1.0}}
    {
        texture.create(static_cast<u32>(settings.dims.x), static_cast<u32>(settings.dims.y));

        sf_render_window.setFramerateLimit(settings.framerate);

        keymap.push_back({Keyboard::Key::LControl, Keyboard::Key::Q}, // exit by default with Ctrl+Q
                         [&sf_render_window = this->sf_render_window]
                         { sf_render_window.close(); });
    }

    void display();

    auto is_open() const -> bool;

    void get_input();

    auto viewport_box() const -> std::array<LineSegment, 4>;

    void fill(Color color);

    void draw(Circle circle, Color color);

    void draw(const Particle& particle);

    void draw_line_segment(const LineSegment& ls,
                           Color color   = Color{255, 255, 255},
                           f64 thickness = 0.1);

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
};
} // namespace sm
