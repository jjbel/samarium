/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "./graphics/Renderer.hpp"
#include "./gui/Window.hpp"
#include <tuple>

namespace sm
{
struct App : public Renderer, public Window
{
    explicit App(const Settings& settings) : Renderer(Image{settings.dims}), Window(settings) {}

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

    void run(std::invocable auto&& update)
    {
        while (this->is_open())
        {
            this->get_input();
            update();
            this->display();
        }
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
};
} // namespace sm
