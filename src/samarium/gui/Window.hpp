/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <chrono>
#include <thread>

#include "SFML/Graphics.hpp"

#include "../graphics/Renderer.hpp"
#include "../util/util.hpp"

#include "Keymap.hpp"

namespace sm
{
class Window
{
    sf::Image im;
    sf::Texture sftexture;
    sf::Sprite sfbufferSprite;
    sf::RenderWindow window;
    sm::util::Stopwatch watch{};
    size_t target_framerate{};

  public:
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

    size_t frame_counter{};
    Keymap keymap;

    Window(Settings settings)
        : window(sf::VideoMode(static_cast<uint32_t>(settings.dims.x),
                               static_cast<uint32_t>(settings.dims.y)),
                 settings.name,
                 sf::Style::Titlebar | sf::Style::Close),
          target_framerate{settings.framerate}
    {
        window.setFramerateLimit(settings.framerate);
        keymap.push_back({sf::Keyboard::LControl, sf::Keyboard::Q},
                         [&window = this->window] { window.close(); });
    }

    bool is_open() const;

    void get_input();

    void draw(const Image& image);

    void display();

    template <std::invocable T> void run(Renderer& rn, T&& call_every_frame)
    {
        while (this->is_open())
        {
            const auto wm = Manager(*this, rn);
            call_every_frame();
        }
    }

    template <typename T, typename U>
    void run(Renderer& rn, T&& update, U&& draw, size_t substeps = 1, size_t frame_limit = 1000)
    {
        while (this->is_open() and this->frame_counter < frame_limit)
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
};
} // namespace sm
