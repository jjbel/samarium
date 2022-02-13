/*
 *                                  MIT License
 *
 *                               Copyright (c) 2022
 *
 *       Project homepage: <https://github.com/strangeQuark1041/samarium/>
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the Software), to deal
 *  in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *     copies of the Software, and to permit persons to whom the Software is
 *            furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 *                copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *                                   SOFTWARE.
 *
 *  For more information, please refer to <https://opensource.org/licenses/MIT/>
 */

#pragma once

#include <chrono>
#include <thread>

#include "SFML/Graphics.hpp"

namespace sm
{
class Window
{
    sf::Image im;
    sf::Texture sftexture;
    sf::Sprite sfbufferSprite;
    sf::RenderWindow window;

  public:
    size_t frame_counter;

    Window(Dimensions dims         = sm::dimsFHD,
           const std::string& name = "Samarium Window",
           uint32_t framerate      = 65536)
        : window(
              sf::VideoMode(static_cast<uint32_t>(dims.x), static_cast<uint32_t>(dims.y)),
              name,
              sf::Style::Titlebar | sf::Style::Close),
          frame_counter{ 0 }
    {
        window.setFramerateLimit(framerate);
    }

    auto is_open() const { return window.isOpen(); }

    auto get_input()
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed) window.close();
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::LControl) &&
            sf::Keyboard::isKeyPressed(sf::Keyboard::Q))
            window.close();
    }

    auto draw(const Image& image)
    {
        im.create(static_cast<uint32_t>(image.dims.x),
                  static_cast<uint32_t>(image.dims.y),
                  reinterpret_cast<const sf::Uint8*>(&image[0]));
        sftexture.loadFromImage(im);
        sfbufferSprite.setTexture(sftexture, true);
    }

    auto display()
    {
        window.draw(sfbufferSprite);
        window.display();
        ++frame_counter;
        // std::this_thread::sleep_for(std::chrono::milliseconds{ 16 });
    }

    operator bool() const { return window.isOpen(); }
};
} // namespace sm
