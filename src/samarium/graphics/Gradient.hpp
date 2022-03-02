/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <array>

#include "samarium/math/interp.hpp"

#include "Color.hpp"

namespace sm
{
template <size_t size> class Gradient;

template <> class Gradient<2>
{
    Color from{};
    Color to{};

  public:
    constexpr Gradient(Color from_, Color to_) : from{from_}, to{to_} {}
    constexpr auto operator()(f64 factor) const
    {
        return interp::lerp_rgb(factor, from, to);
    }
};

template <> class Gradient<3>
{
    Color from{};
    Color mid{};
    Color to{};

  public:
    constexpr Gradient(Color from_, Color mid_, Color to_)
        : from{from_}, mid{mid_}, to{to_}
    {
    }
    constexpr auto operator()(f64 factor) const
    {
        factor = Extents<f64>{0.0, 1.0}.clamp(factor);
        if (factor < 0.5) { return interp::lerp_rgb(2.0 * factor, from, mid); }
        else
            return interp::lerp_rgb(2.0 * (factor - 0.5), mid, to);
    }
};
} // namespace sm
