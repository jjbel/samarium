// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Jai Bellare
// Project homepage: https://github.com/strangeQuark1041/samarium
#pragma once

#include <cmath>
#include <numbers>

#include "../core/concepts.hpp"

namespace sm::math
{
constexpr inline auto EPSILON = 1.e-4;

template <concepts::FloatingPoint T> [[nodiscard]] constexpr auto almost_equal(T a, T b) noexcept
{
    return std::abs(a - b) <= EPSILON;
}

template <typename T> [[nodiscard]] constexpr auto min(T value0, T value1) noexcept
{
    if (value0 < value1) return value0;
    else
        return value1;
}

template <typename T> [[nodiscard]] constexpr auto max(T value0, T value1) noexcept
{
    if (value0 > value1) return value0;
    else
        return value1;
}

template <u32 n> [[nodiscard]] constexpr auto power(auto x) noexcept
{
    if constexpr (n == 0) return 1;

    return x * power<n - 1>(x);
}

[[nodiscard]] constexpr auto to_degrees(auto angle) noexcept
{
    return angle * 180.0 / std::numbers::pi;
}

[[nodiscard]] constexpr auto to_radians(auto angle) noexcept
{
    return angle / 180.0 * std::numbers::pi;
}

template <concepts::Number T> [[nodiscard]] constexpr auto abs(T x) noexcept
{
    return x >= T{} ? x : -x;
}

template <typename T> [[nodiscard]] constexpr int sign(T x, std::false_type is_signed)
{
    return T(0) < x;
}

template <typename T> [[nodiscard]] constexpr int sign(T x, std::true_type is_signed)
{
    return (T(0) < x) - (x < T(0));
}

template <typename T> [[nodiscard]] constexpr int sign(T x)
{
    return sign(x, std::is_signed<T>());
}

} // namespace sm::math

namespace sm::literals
{
consteval auto operator"" _degrees(f80 angle) noexcept { return math::to_radians(angle); }

consteval auto operator"" _radians(f80 angle) noexcept { return angle; }
} // namespace sm::literals
