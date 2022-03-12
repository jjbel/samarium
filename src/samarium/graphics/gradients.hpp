/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "Gradient.hpp"

namespace sm::gradients
{
constexpr inline auto blue       = Gradient<2>(Color{101, 199, 247}, Color{0, 82, 212});
constexpr inline auto purple     = Gradient<2>{Color{142, 45, 226}, Color{74, 0, 224}};
constexpr inline auto blue_green = Gradient<2>{Color{0, 242, 96}, Color{5, 117, 230}};
constexpr inline auto horizon    = Gradient<2>{Color{18, 194, 233}, Color{246, 79, 89}};
constexpr inline auto heat =
    Gradient<3>{Color{27, 9, 128}, Color{107, 21, 21}, Color{255, 126, 40}};
} // namespace sm::gradients
