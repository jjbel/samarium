/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

#include "range/v3/view/enumerate.hpp"

using namespace sm;
using namespace sm::literals;

int main()
{
    auto app         = App{{.dims{1280, 720}}};
    const auto count = 200;
    const auto size  = Vector2{70.0, 30.0};

    auto plot = std::vector<Vector2>(count);

    auto watch = Stopwatch{};

    auto rand = RandomGenerator{};
    // auto noise = util::noise{};

    const auto draw = [&]
    {
        if (app.mouse.left) { app.transform.pos += app.mouse.current_pos - app.mouse.old_pos; }

        const auto scale = 1.0 + 0.1 * app.mouse.scroll_amount;
        app.transform.scale *= Vector2::combine(scale);
        const auto mouse_pos = app.mouse.current_pos;
        app.transform.pos    = mouse_pos + scale * (app.transform.pos - mouse_pos);

        app.fill("#16161c"_c);
        // app.draw_polyline(plot, "#4542ff"_c, 0.1);
        app.draw_world_space(
            [&](Vector2 pos)
            {
                auto noise =
                    noise::perlin_2d(pos.as<f64>(), {.scale = 1.0, .detail = 8, .roughness = 0.9}) *
                    255.0;
                return Color::from_grayscale(static_cast<u8>(noise));
            });

        print(watch.seconds() * 1000.0);
        // print(std::round(1.0 / watch.seconds()));
        watch.reset();
    };
    app.run(draw);
    // for (auto i : range(100, 200)) { print(static_cast<f64>(ValueNoise_2D(0, i, 7, 0.5, 0) *
    // 255)); }
}
