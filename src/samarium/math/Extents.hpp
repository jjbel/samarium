/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "../core/concepts.hpp"

namespace sm
{
template <concepts::Arithmetic T> class Extents
{
  public:
    T min{};
    T max{};

    [[nodiscard]] static constexpr auto find_min_max(T a, T b)
    {
        return (a < b) ? Extents{a, b} : Extents{b, a};
    }

    [[nodiscard]] constexpr auto size() const { return max - min; }

    [[nodiscard]] constexpr auto contains(T value) const { return min <= value and value <= max; }

    [[nodiscard]] constexpr auto clamp(T value) const
    {
        if (value < min) return min;
        else if (value > max)
            return max;
        else
            return value;
    }

    [[nodiscard]] constexpr auto lerp(f64 factor) const
    {
        return min * (1.0 - factor) + max * factor;
    }

    [[nodiscard]] constexpr auto clamped_lerp(f64 factor) const
    {
        return min * (1.0 - this->clamp(factor)) + max * factor;
    }

    [[nodiscard]] constexpr f64 lerp_inverse(T value) const { return (value - min) / this->size(); }

    template <typename Function>
    constexpr auto
    for_each(Function&& fn) const requires concepts::Integral<T> && std::invocable<Function, T>
    {
        for (auto i = min; i <= max; i++) { fn(i); }
    }
};
} // namespace sm
