/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "../graphics/Renderer.hpp" // for Renderer
#include "../util/Stopwatch.hpp"    // for Stopwatch

#include "Keyboard.hpp"
#include "Mouse.hpp"

namespace sm
{
class Window
{
    protected:
    // --------MEMBER VARS--------
    sf::Image im;
    sf::Texture sftexture;
    sf::Sprite sf_buffer_sprite;
    sf::RenderWindow window;
    sm::util::Stopwatch watch{};
    size_t target_framerate{};

  public:
    size_t frame_counter{};
    Keymap keymap;
    Mouse mouse{window};

    // --------MEMBER TYPES--------
    struct Settings
    {
        Dimensions dims{sm::dimsFHD};
        std::string name{"Samarium Window"};
        uint32_t framerate{64};
    };

    struct Manager
    {
        Window& window;
        const Renderer& renderer;

        Manager(Window& win, const Renderer& rn) : window(win), renderer(rn) { window.get_input(); }

        Manager(const Manager&) = delete;

        ~Manager() // NOSONAR
        {
            window.draw_image(renderer.image);
            window.display();
        }
    };

    // --------MEMBER FUNCTIONS--------

    explicit Window(const Settings& settings)
        : window(
              sf::VideoMode(static_cast<u32>(settings.dims.x), static_cast<u32>(settings.dims.y)),
              settings.name,
              sf::Style::Titlebar | sf::Style::Close),
          target_framerate{settings.framerate}
    {
        window.setFramerateLimit(settings.framerate);
        keymap.push_back({Keyboard::Key::LControl, Keyboard::Key::Q}, // exit by default with Ctrl+Q
                         [&window = this->window] { window.close(); });
    }

    auto is_open() const -> bool;

    void get_input();

    void draw_image(const Image& image);

    void display();

    void run(const Renderer& rn, std::invocable auto&& call_every_frame)
    {
        while (this->is_open())
        {
            const auto wm = Manager(*this, rn);
            call_every_frame();
        }
    }

    template <typename UpdateFunction, typename DrawFunction>
    void run(const Renderer& rn,
             UpdateFunction&& update,
             DrawFunction&& draw,
             size_t substeps    = 1,
             size_t frame_limit = 1000)
    {
        while (this->is_open() && this->frame_counter < frame_limit)
        {
            const auto wm = Manager(*this, rn);
            for (size_t i = 0; i < substeps; i++)
            {
                update(1.0 / static_cast<f64>(target_framerate * substeps));
            }

            draw();
        }
    }

    auto current_framerate() const -> f64;

    auto time_delta() const -> f64;

    auto sf_window() const -> const sf::Window&;
};
} // namespace sm
