/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

int main()
{
    if (glfwInit() == 0) { throw std::runtime_error("Error: failed to initialize glfw"); }

    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);
    auto window = Window{{720, 720}};
    glfwSetWindowAttrib(window.handle.get(), GLFW_DECORATED, GLFW_FALSE);

    while (window.is_open())
    {
        draw::background(Color{.a = 0});
        draw::circle(window, {{-0.8, -0.8}, 0.1}, {.fill_color = Color{0, 0, 255, 255}});
        draw::circle(window, {{-0.8, 0.8}, 0.1}, {.fill_color = Color{0, 0, 255, 255}});

        draw::circle(window, {{0.8, 0.8}, 0.1}, {.fill_color = Color{0, 0, 255, 255}});
        draw::circle(window, {{0.8, -0.8}, 0.1}, {.fill_color = Color{0, 0, 255, 25}});

        window.display();
    }
}
