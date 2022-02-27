/* 
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <concepts>

#include "types.hpp"

namespace sm::concepts
{

constexpr bool reason(const char* const) { return true; };

template <typename T>
concept Integral =
    reason("NOTE: T should be of integral type, eg int or size_t") &&
    std::is_integral_v<T>;

template <typename T>
concept FloatingPoint =
    reason("NOTE: T should be of floating point type, eg float or double") &&
    std::is_floating_point_v<T>;

template <typename T>
concept Number = Integral<T> || FloatingPoint<T>;

template <typename T>
concept Arithmetic = requires(T a, T b)
{
    {
        a + b
        } -> std::same_as<T>;
    {
        a - b
        } -> std::same_as<T>;
};

template <typename T, typename... U>
concept AnyOf = (std::same_as<T, U> || ...);

template <class ContainerType>
concept Container = requires(ContainerType a, const ContainerType b)
{
    requires std::regular<ContainerType>;
    requires std::swappable<ContainerType>;
    requires std::destructible<typename ContainerType::value_type>;
    requires std::same_as<typename ContainerType::reference,
                          typename ContainerType::value_type&>;
    requires std::same_as<typename ContainerType::const_reference,
                          const typename ContainerType::value_type&>;
    requires std::forward_iterator<typename ContainerType::iterator>;
    requires std::forward_iterator<typename ContainerType::const_iterator>;
    requires std::signed_integral<typename ContainerType::difference_type>;
    requires std::same_as<typename ContainerType::difference_type,
                          typename std::iterator_traits<
                              typename ContainerType::iterator>::difference_type>;
    requires std::same_as<
        typename ContainerType::difference_type,
        typename std::iterator_traits<
            typename ContainerType::const_iterator>::difference_type>;
    {
        a.begin()
        } -> std::same_as<typename ContainerType::iterator>;
    {
        a.end()
        } -> std::same_as<typename ContainerType::iterator>;
    {
        b.begin()
        } -> std::same_as<typename ContainerType::const_iterator>;
    {
        b.end()
        } -> std::same_as<typename ContainerType::const_iterator>;
    {
        a.cbegin()
        } -> std::same_as<typename ContainerType::const_iterator>;
    {
        a.cend()
        } -> std::same_as<typename ContainerType::const_iterator>;
    {
        a.size()
        } -> std::same_as<typename ContainerType::size_type>;
    {
        a.max_size()
        } -> std::same_as<typename ContainerType::size_type>;
    {
        a.empty()
        } -> std::same_as<bool>;
};
} // namespace sm::concepts