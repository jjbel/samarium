/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <memory> // for unique_ptr

#include "GLFW/glfw3.h"

namespace sm
{
struct WindowDeleter
{
    auto operator()(GLFWwindow* ptr) const
    {
        glfwDestroyWindow(ptr);
        glfwTerminate();
    }
};

using WindowHandle = std::unique_ptr<GLFWwindow, WindowDeleter>;
} // namespace sm
