/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "../graphics/Renderer.hpp"
#include "../util/util.hpp"

#include "Keymap.hpp"
#include "Mouse.hpp"

namespace sm
{
class Window
{
    // --------MEMBER VARS--------
    sf::Image im;
    sf::Texture sftexture;
    sf::Sprite sfbufferSprite;
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
            window.draw(renderer.image);
            window.display();
        }
    };

    // --------MEMBER FUNCTIONS--------

    Window(const Settings& settings)
        : window(
              sf::VideoMode(static_cast<u32>(settings.dims.x), static_cast<u32>(settings.dims.y)),
              settings.name,
              sf::Style::Titlebar | sf::Style::Close),
          target_framerate{settings.framerate}
    {
        window.setFramerateLimit(settings.framerate);
        keymap.push_back({sf::Keyboard::LControl, sf::Keyboard::Q}, // exit by default with Ctrl+Q
                         [&window = this->window] { window.close(); });
    }

    bool is_open() const;

    void get_input();

    void draw(const Image& image);

    void display();

    void run(Renderer& rn, std::invocable auto&& call_every_frame)
    {
        while (this->is_open())
        {
            const auto wm = Manager(*this, rn);
            call_every_frame();
        }
    }

    template <typename UpdateFunction, typename DrawFunction>
    void run(Renderer& rn,
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

    f64 current_framerate() const;

    f64 time_delta() const;

    const sf::Window& sf_window() const;
};
} // namespace sm
