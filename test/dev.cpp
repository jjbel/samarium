/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/gl/draw.hpp"
#include "samarium/gl/gl.hpp"
#include "samarium/graphics/Gradient.hpp"
#include "samarium/graphics/Image.hpp"
#include "samarium/graphics/gradients.hpp"
#include "samarium/math/Vector2.hpp"
#include "samarium/samarium.hpp"
#include <GLFW/glfw3.h>

using namespace sm;
using namespace sm::literals;

int main()
{
    auto window = Window{{720, 720}};
    auto watch  = Stopwatch{};

    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    auto angle = 0.0;
    while (window.is_open())
    {
        draw::background("#1c151b"_c);
        draw::circle(window, {{-0.8, -0.8}, 0.1}, {.fill_color = "#fa2844"_c});
        draw::circle(window, {{-0.8, 0.8}, 0.1}, {.fill_color = "#fa2844"_c});

        watch.reset();
        draw::background(window, Gradient<2>{"#000000"_c, "#FFFFFF"_c}, angle);
        const auto throughput = static_cast<i32>(watch.current_fps());
        print(fmt::format(std::locale("en_US.UTF-8"), "{:L}", throughput));
        angle += 2.0_degrees;

        draw::circle(window, {{0.8, 0.8}, 0.1}, {.fill_color = "#fa2844"_c});
        draw::circle(window, {{0.8, -0.8}, 0.1}, {.fill_color = "#fa2844"_c});

        // print(window.view, window.viewport());
        window.display();
    }
}
