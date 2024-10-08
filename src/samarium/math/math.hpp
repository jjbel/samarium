/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <cmath>
#include <numbers>

#include "samarium/core/concepts.hpp"

namespace sm::math
{
constexpr inline auto epsilon       = 1.e-4;
constexpr inline auto pi            = std::numbers::pi;
constexpr inline auto two_thirds_pi = 2.0 * pi / 3.0;
constexpr inline auto two_pi        = 2.0 * pi;
constexpr inline auto e             = std::numbers::e;
constexpr inline auto sqrt2         = std::numbers::sqrt2;

/**
 * @brief               Check if 2 floating point values are equal
 *
 * @tparam T            Floating point type
 * @param  a            First value
 * @param  b            Second value
 * @param  threshold    threshold within which to compare a and b, by default sm::math::epsilon
 * @return true         if a and b are within threshold else false
 */
template <concepts::FloatingPoint T>
[[nodiscard]] constexpr auto almost_equal(T a, T b, f64 threshold = epsilon) noexcept
{
    return std::abs(a - b) <= threshold;
}

/**
 * @brief               Minimum of 2 values
 *
 * @tparam T
 * @param  value0       First value
 * @param  value1       Second value
 * @return T
 */
template <typename T> [[nodiscard]] constexpr auto min(T value0, T value1) noexcept
{
    if (value0 < value1) { return value0; }
    else { return value1; }
}

/**
 * @brief               Maximum of 2 values
 *
 * @tparam T
 * @param  value0       First value
 * @param  value1       Second value
 * @return T
 */
template <typename T> [[nodiscard]] constexpr auto max(T value0, T value1) noexcept
{
    if (value0 > value1) { return value0; }
    else { return value1; }
}

/**
 * @brief               Raise a number to a fixed power
 *
 * @tparam n            Exponent
 * @param  x            Base
 * @return auto
 */
template <u32 n> [[nodiscard]] constexpr auto power(auto x) noexcept
{
    if constexpr (n == 0) { return 1; }
    else // for whatever reason else is needed after return
    {
        return x * power<n - 1>(x);
    }
}

/**
 * @brief               Convert from radians to degrees
 *
 * @param  angle        Angle in radians
 * @return auto
 */
[[nodiscard]] constexpr auto to_degrees(concepts::FloatingPoint auto angle) noexcept
{
    return angle * 180.0 / math::pi;
}

/**
 * @brief               Convert from degrees to radians
 *
 * @param  angle        Angle in degrees
 * @return auto
 */
[[nodiscard]] constexpr auto to_radians(concepts::FloatingPoint auto angle) noexcept
{
    return angle / 180.0 * math::pi;
}

/**
 * @brief               Absolute value of a value
 *
 * @tparam T            A numeric type
 * @param  x            Value
 * @return T
 */
template <concepts::Number T> [[nodiscard]] constexpr auto abs(T x) noexcept
{
    return x >= T{} ? x : -x;
}

/**
 * @brief               Sign of an unsigned type
 *
 * @tparam T            Unsigned Type
 * @param  x            Value
 * @return 1            if x is greater than 0 else 0
 */
template <typename T>
[[nodiscard]] constexpr auto sign(T x, std::false_type /* is_signed */) noexcept -> i32
{
    return T{} < x;
}

/**
 * @brief               Sign of a signed type
 *
 * @tparam T            Signed type
 * @param x             Value
 * @return              -1 if x < 0, 1 if x > 0, 0 is x == 0
 */
template <typename T>
[[nodiscard]] constexpr auto sign(T x, std::true_type /* is_signed */) noexcept -> i32
{
    return (T{} < x) - (x < T{});
}

/**
 * @brief               Sign of a value
 *
 * @tparam T            Numeric type
 * @param x             Value
 * @return              -1 if x < 0, 1 if x > 0, 0 is x == 0
 */
template <typename T> [[nodiscard]] constexpr auto sign(T x) noexcept -> i32
{
    return sign(x, std::is_signed<T>{});
}

/**
 * @brief               Modulus of 2 floating-point values
 * (see https://stackoverflow.com/a/67098028)
 *
 * @tparam T
 * @param  x            First value
 * @param  y            Second value
 */
template <concepts::FloatingPoint T> constexpr auto mod(T x, T y) noexcept
{
    return x - std::trunc(x / y) * y;
}

/**
 * @brief               Wrap x to range [0, max)
 * (see https://stackoverflow.com/a/29871193/17100530)
 *
 * @tparam T
 * @param  x            Value
 * @param  max          Upper limit of range
 */
template <concepts::FloatingPoint T> constexpr auto wrap_max(T x, T max)
{
    /* integer math: `(max + x % max) % max` */
    return std::fmod(max + std::fmod(x, max), max);
}

/**
 * @brief               Wrap x to range [min, max)
 * (see https://stackoverflow.com/a/29871193/17100530)
 *
 * @tparam T
 * @param  x            Value
 * @param  min          Lower limit of range
 * @param  max          Upper limit of range
 */
template <concepts::FloatingPoint T> constexpr auto wrap_min_max(T x, T min, T max)
{
    return min + wrap_max(x - min, max - min);
}

/**
 * @brief               Round value to nearest multiple of target
 *
 * @tparam T
 * @param  value
 * @param  target
 */
template <typename T> constexpr auto round_to_nearest(T value, T target)
{
    return target * std::round(value / target);
}

/**
 * @brief               Ceil value to nearest multiple of target
 *
 * @tparam T
 * @param  value
 * @param  target
 */
template <typename T> constexpr auto ceil_to_nearest(T value, T target)
{
    return target * std::ceil(value / target);
}

/**
 * @brief               Floor value to nearest multiple of target
 *
 * @tparam T
 * @param  value
 * @param  target
 */
template <typename T> constexpr auto floor_to_nearest(T value, T target)
{
    return target * std::floor(value / target);
}
} // namespace sm::math

namespace sm::literals
{
consteval auto operator"" _degrees(long double angle) noexcept
{
    return math::to_radians(static_cast<f64>(angle));
}

consteval auto operator"" _radians(long double angle) noexcept { return static_cast<f64>(angle); }
} // namespace sm::literals
