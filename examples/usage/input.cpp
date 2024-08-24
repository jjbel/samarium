/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto window = Window{{1024, 1024}};
    auto watch  = Stopwatch{};

    auto action = keyboard::OnKeyPress(*window.handle, {Key::W}, [] { print("Key"); });

    while (window.is_open())
    {
        draw::background("#1c151b"_c);
        draw::circle(window, {{0.2, 0.3}, 0.4}, {.fill_color = "#fa2844"_c});
        watch.reset();

        // TODO: no way of detecting resized?
        // if (Window::resized) { print("Resized"); }
        if (window.mouse.left) { print("Left"); }
        if (window.mouse.right) { print("Right"); }
        if (window.is_key_pressed(Key::Space)) { print("Space"); }
        action();
        window.display();
    }
}
    