/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#include "samarium/graphics/colors.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto app   = App{{.dims{1800, 900}}};
    auto watch = Stopwatch{};

    auto radius  = 3.0;
    auto dynamic = SecondOrderDynamics<Vec2>{{}, 4.5, 0.5};
    auto trail   = Trail{20000};

    const auto update = [&](f64 dt)
    {
        const auto mouse_pos = app.transform.apply_inverse(app.mouse.current_pos);

        dynamic.update(dt, mouse_pos);
        print(mouse_pos, dynamic.value);
        trail.push_back(dynamic.value);
    };

    const auto draw = [&]
    {
        draw::background("#0D0D13"_c);
        app.draw(trail, "#BBD4FF"_c, 0.3, 1.0);
        app.draw(Circle{dynamic.value, radius}, {.fill_color = colors::red});
    };

    app.run(update, draw, 2);
}
