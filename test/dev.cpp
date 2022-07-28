/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

int main()
{
    const auto speed = 2;

    auto app = App{{.dims = {1800, 900}}};
    app.transform.scale *= 6;

    auto player     = Particle{};
    const auto disp = Vector2{3, 3};

    // app.keymap.push_back({Keyboard::Key::Space}, [] { print("Jump"); });
    auto jumper =
        Keyboard::EventListener<Keyboard::Event::Down, Keyboard::Key::Space>{[] { print("Jump"); }};

    const auto update = [&](f64 dt)
    {
        jumper();
        app.zoom_pan();
        app.transform.pos.x -= speed * dt * app.transform.scale.x;
        player.pos.x += speed * dt;
    };

    const auto draw = [&]
    {
        app.fill("#141724"_c);
        app.draw(App::GridLines{.scale = 1.0, .axis_thickness = 0.016, .line_thickness = 0.01});
        // app.draw(App::GridDots{.scale = 1.0, .thickness = 0.03});
        app.draw(BoundingBox<f64>::from_centre_width_height(player.pos, disp.x, disp.y),
                 {.fill_color = "#ff6c17"_c, .border_width = 0.1});
        // app.draw(BoundingBox<f64>{{-4, -4}, {4, 5}},
        //          {.border_color = "#fc035a"_c, .border_width = 0.1});
    };

    app.run(update, draw);
}
