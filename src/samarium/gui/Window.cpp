/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "Window.hpp"

namespace sm
{
auto Window::is_open() const -> bool { return window.isOpen(); }

void Window::get_input()
{
    this->mouse.scroll_amount = 0.0;

    sf::Event event;
    while (window.pollEvent(event))
    {
        if (event.type == sf::Event::Closed) { window.close(); }
        else if (event.type == sf::Event::MouseWheelScrolled &&
                 event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel)
        {
            this->mouse.scroll_amount = static_cast<f64>(event.mouseWheelScroll.delta);
        }
    }

    this->keymap.run();
    this->mouse.update(this->window);
}

void Window::draw_image(const Image& image)
{
    im.create(static_cast<u32>(image.dims.x), static_cast<u32>(image.dims.y),
              reinterpret_cast<const u8*>(image.begin()));
    sftexture.loadFromImage(im);
    sf_buffer_sprite.setTexture(sftexture, true);
}

void Window::display()
{
    window.draw(sf_buffer_sprite);
    window.display();
    frame_counter++;
    watch.reset();
}

auto Window::current_framerate() const -> f64 { return this->watch.time().count() * 1000; }

auto Window::sf_window() const -> const sf::Window& { return this->window; }

auto Window::time_delta() const -> f64 { return 1.0 / static_cast<f64>(this->target_framerate); }

} // namespace sm
