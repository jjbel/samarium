/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "samarium/util/Grid.hpp"

#include "Color.hpp"

namespace sm
{
using Image       = Grid<Color>;
using ScalarField = Grid<f64>;
using VectorField = Grid<Vector2>;

constexpr inline auto dims4K  = Dimensions{3840UL, 2160UL};
constexpr inline auto dimsFHD = Dimensions{1920UL, 1080UL};
constexpr inline auto dims720 = Dimensions{1280UL, 720UL};
constexpr inline auto dims480 = Dimensions{640UL, 480UL};
constexpr inline auto dimsP2  = Dimensions{2048UL, 1024UL};
} // namespace sm
