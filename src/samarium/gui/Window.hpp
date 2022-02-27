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

#include "samarium/graphics/Renderer.hpp"
#include "samarium/util/util.hpp"

namespace sm
{
class Window
{
    sf::Image im;
    sf::Texture sftexture;
    sf::Sprite sfbufferSprite;
    sf::RenderWindow window;
    sm::util::Stopwatch watch{};

  public:
    struct Manager
    {
        Window& window;
        Renderer& renderer;

        Manager(Window& win, Renderer& rn, Color color)
            : window(win), renderer(rn)
        {
            window.get_input();
            renderer.fill(color);
        }

        Manager(const Manager&) = delete;

        ~Manager() // NOSONAR
        {
            renderer.render();

            window.draw(renderer.image);
            window.display();
        }
    };

    size_t frame_counter{};

    Window(Dimensions dims         = sm::dimsFHD,
           const std::string& name = "Samarium Window",
           uint32_t framerate      = 64)
        : window(sf::VideoMode(static_cast<uint32_t>(dims.x),
                               static_cast<uint32_t>(dims.y)),
                 name,
                 sf::Style::Titlebar | sf::Style::Close)
    {
        window.setFramerateLimit(framerate);
    }

    bool is_open() const;

    void get_input();

    void draw(const Image& image);

    void display();

    template <typename T>
    requires std::invocable<T>
    void run(Renderer& rn, Color color, T call_every_frame)
    {
        while (this->is_open())
        {
            const auto wm = Manager(*this, rn, color);
            call_every_frame();
        }
    }

    double current_framerate() const;
};
} // namespace sm
