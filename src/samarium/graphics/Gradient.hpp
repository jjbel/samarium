/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <array>

#include "samarium/math/interp.hpp"
#include "samarium/util/print.hpp"

#include "Color.hpp"

namespace sm
{
template <u64 size> struct Gradient
{
    std::array<Color, size> colors;

    explicit constexpr Gradient(auto&&... colors_) : colors{colors_...} {}

    [[nodiscard]] auto operator()(f64 factor) const
    {
        const auto mapped = factor * (size - 1UL) + 0.01;
        // TODO the +0.1 prevents the map range from dividing by 0

        const auto lower = static_cast<u64>(mapped);            // static_cast rounds down
        const auto upper = static_cast<u64>(std::ceil(mapped)); // round up
        const auto mapped_factor =
            interp::map_range<f64>(mapped, {std::floor(mapped), std::ceil(mapped)}, {0.0, 1.0});

        return interp::lerp_rgb(mapped_factor, colors[lower], colors[upper]);
    }
};

template <typename... Args> Gradient(Args... args) -> Gradient<sizeof...(Args)>;
} // namespace sm
