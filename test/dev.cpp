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
    const auto speed = 4.0;

    auto app = App{{.dims = {1800, 900}}};
    app.transform.scale *= 6;
    app.transform.pos = app.dims().as<f64>() * Vector2{0.1, 0.85};

    auto player     = Particle{.pos = {0.0, 2.0}};
    const auto disp = Vector2{1.5, 1.5};
    auto trail      = Trail{200};
    auto jumper     = Keyboard::OnKeyDown{{Keyboard::Key::Space},
                                      [&]
                                      {
                                          player.vel.y += 12;
                                          print("Jump");
                                      }};

    const auto update = [&](f64 dt)
    {
        jumper();
        // app.zoom_pan();
        app.transform.pos.x -= speed * dt * app.transform.scale.x;

        player.acc = Vector2{0.0, -30};

        player.update(dt);
        player.pos.x += speed * dt;

        if (player.pos.y <= 0.0)
        {
            player.pos.y = 0;
            player.vel.y = 0;
        }
        trail.push_back(player.pos + disp / 2.0);
    };

    const auto draw = [&]
    {
        app.fill("#151726"_c);
        // app.draw(App::GridLines{.scale = 1.0, .line_thickness = 0.01, .axis_thickness = 0.016});
        app.draw(App::GridDots{.scale = 1.0, .thickness = 0.03});
        app.draw(trail, "#17ff70"_c, 0.4, 1.0);
        app.draw(BoundingBox<f64>{player.pos, player.pos + disp},
                 {.fill_color = "#ff6c17"_c, .border_width = 0.1});
        // app.draw(BoundingBox<f64>{{-4, -4}, {4, 5}},
        //          {.border_color = "#fc035a"_c, .border_width = 0.1});
    };

    app.run(update, draw);
}
