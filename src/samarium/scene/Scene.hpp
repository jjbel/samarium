/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "samarium/gui/Window.hpp"

namespace sm
{
struct Scene
{
    struct Settings
    {
        double time_delta{1};
        Vector2 gravity{};
        double friction_coefficient{};
        double restitution_coefficient{};
        double spring_constant{1};
    };

    Settings settings;
    Window window;
    Renderer rn;
};
} // namespace sm
