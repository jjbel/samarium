/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <concepts>
// #include <iterator>

// #include "range/v3/range/concepts.hpp"

#include "types.hpp"

namespace sm::concepts
{

constexpr inline auto reason(const char* const /* reason_string */) { return true; };

template <typename T>
concept Integral =
    reason("NOTE: T should be of integral type, eg int or size_t") && std::is_integral_v<T>;

template <typename T>
concept FloatingPoint = reason("NOTE: T should be of floating point type, eg float or double") &&
                        std::is_floating_point_v<T>;

template <typename T>
concept Number = Integral<T> || FloatingPoint<T>;

template <typename T>
concept Arithmetic = requires(T a, T b) {
    { a + b } -> std::same_as<T>;
    { a - b } -> std::same_as<T>;
};

template <typename T, typename... U>
concept AnyOf = (std::same_as<T, U> || ...);

// removing coz cuda complains about range-v3
// /**
//  * @brief               Is T a range of V's
//  *
//  * @tparam T
//  * @tparam V
//  */
// template <class T, class V>
// concept Iterable = std::ranges::range<T> && std::same_as<V, std::ranges::range_value_t<T>>;
} // namespace sm::concepts
