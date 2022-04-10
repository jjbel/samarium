/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "./graphics/Renderer.hpp"
#include "./gui/Window.hpp"

namespace sm
{
struct App : public Renderer, public Window
{
    App(const Settings& settings) : Renderer(Image{settings.dims}), Window(settings) {}

    void display()
    {
        im.create(static_cast<u32>(image.dims.x), static_cast<u32>(image.dims.y),
                  reinterpret_cast<const u8*>(image.begin()));
        sftexture.loadFromImage(im);
        sf_buffer_sprite.setTexture(sftexture, true);

        window.draw(sf_buffer_sprite);
        window.display();
        frame_counter++;
        watch.reset();
    }
};
} // namespace sm
