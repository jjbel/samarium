// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Jai Bellare
// Project homepage: https://github.com/strangeQuark1041/samarium
#pragma once

#include <cmath>
#include <numbers>

#include "samarium/core/concepts.hpp"

namespace sm::math
{
constexpr inline auto EPSILON = 1.e-4;

template <concepts::FloatingPoint T>
[[nodiscard]] constexpr auto almost_equal(T a, T b) noexcept
{
    return std::abs(a - b) <= EPSILON;
}

template <typename T>
[[nodiscard]] constexpr auto min(T value0, T value1) noexcept
{
    if (value0 < value1) return value0;
    else
        return value1;
}

template <typename T>
[[nodiscard]] constexpr auto max(T value0, T value1) noexcept
{
    if (value0 > value1) return value0;
    else
        return value1;
}

template <u32 n> [[nodiscard]] constexpr auto power(auto x)
{
    if constexpr (n == 0) return 1;

    return x * power<n - 1>(x);
}
} // namespace sm::math

namespace sm::literals
{
consteval auto operator"" _degrees(long double angle)
{
    return angle / 180.0 * std::numbers::pi;
}

consteval auto operator"" _radians(long double angle)
{
    return angle * 180.0 / std::numbers::pi;
}
} // namespace sm::literals
