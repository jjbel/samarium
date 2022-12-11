/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <compare>
#include <iterator>

#include "samarium/core/concepts.hpp"
#include "samarium/math/math.hpp"

namespace sm
{
template <typename T> struct Extents
{
    T min{};
    T max{};

    [[nodiscard]] static constexpr auto find_min_max(T a, T b) noexcept
    {
        return (a < b) ? Extents{a, b} : Extents{b, a};
    }

    [[nodiscard]] constexpr auto size() const noexcept { return max - min; }

    [[nodiscard]] constexpr auto contains(T value) const noexcept
    {
        return min <= value && value <= max;
    }

    [[nodiscard]] constexpr auto overlaps(Extents<T> other) const noexcept
    {
        return min <= other.max && other.min <= max;
    }

    [[nodiscard]] constexpr auto union_with(Extents<T> other) const noexcept
    {
        return Extents<T>{math::min(min, other.min), math::max(max, other.max)};
    }

    [[nodiscard]] constexpr auto clamp(T value) const noexcept
    {
        if (value < min) { return min; }
        else if (value > max) { return max; }
        else { return value; }
    }

    [[nodiscard]] constexpr auto lerp(f64 factor) const noexcept
        requires concepts::Number<T>
    {
        return static_cast<f64>(min) * (1.0 - factor) +
               static_cast<f64>(max) * factor; // prevent conversion warnings
    }

    [[nodiscard]] constexpr auto lerp(f64 factor) const noexcept
        requires(!concepts::Number<T>)
    {
        return min * (1.0 - factor) + max * factor;
    }

    [[nodiscard]] constexpr auto clamped_lerp(f64 factor) const noexcept
    {
        return min * (1.0 - this->clamp(factor)) + max * factor;
    }

    [[nodiscard]] constexpr auto lerp_inverse(T value) const noexcept -> f64
    {
        return (value - min) / this->size();
    }

    template <typename Function>
    constexpr auto for_each(Function&& fn) const
        requires concepts::Integral<T>
    {
        for (auto i = min; i < max; i++) { fn(i); }
    }

    template <typename U> [[nodiscard]] constexpr auto as() const noexcept
    {
        return Extents<U>{static_cast<U>(min), static_cast<U>(max)};
    }
};
} // namespace sm
