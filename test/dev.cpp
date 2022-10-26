/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/gl/draw.hpp"
#include "samarium/graphics/gradients.hpp"
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

int main()
{
    auto window = Window{{720, 720}};

    while (window.is_open())
    {
        draw::background(Color{.a = 0});
        draw::circle(window, {{-0.8, -0.8}, 0.1}, {.fill_color = Color{0, 0, 255, 255}});
        draw::circle(window, {{-0.8, 0.8}, 0.1}, {.fill_color = Color{0, 0, 255, 255}});
        draw::background(window, gradients::heat);

        draw::circle(window, {{0.8, 0.8}, 0.1}, {.fill_color = Color{0, 0, 255, 255}});
        draw::circle(window, {{0.8, -0.8}, 0.1}, {.fill_color = Color{0, 0, 255, 25}});

        window.display();
    }
    print(sizeof(glm::mat4), sizeof(f32), sizeof(Transform));
}
