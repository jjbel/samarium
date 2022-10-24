/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/gradients.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

int main()
{
    auto window = Window{{720, 720}};
    auto watch  = Stopwatch{};

    auto angle = 0.0;
    while (window.is_open())
    {
        draw::background(window, "#1c151b"_c);
        draw::circle(window, {{-0.8, -0.8}, 0.1}, {.fill_color = "#fa2844"_c});
        draw::circle(window, {{-0.8, 0.8}, 0.1}, {.fill_color = "#fa2844"_c});

        draw::background(window, gradients::heat, angle);
        const auto throughput = static_cast<i32>(watch.current_fps());
        if (window.is_key_pressed(Key::Space)) angle += 0.5_degrees;

        draw::circle(window, {{0.8, 0.8}, 0.1}, {.fill_color = "#fa2844"_c});
        draw::circle(window, {{0.8, -0.8}, 0.1}, {.fill_color = "#fa2844"_c});

        window.display();
    }
}
