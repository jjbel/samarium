/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "Window.hpp"

namespace sm
{
bool Window::is_open() const { return window.isOpen(); }

void Window::get_input()
{
    sf::Event event;
    while (window.pollEvent(event))
    {
        if (event.type == sf::Event::Closed) window.close();
    }

    this->keymap.run();
}

void Window::draw(const Image& image)
{
    im.create(static_cast<uint32_t>(image.dims.x), static_cast<uint32_t>(image.dims.y),
              reinterpret_cast<const sf::Uint8*>(&image[0]));
    sftexture.loadFromImage(im);
    sfbufferSprite.setTexture(sftexture, true);
}

void Window::display()
{
    window.draw(sfbufferSprite);
    window.display();
    ++frame_counter;
    watch.reset();
}

f64 Window::current_framerate() const { return this->watch.time().count() * 1000; }
} // namespace sm
