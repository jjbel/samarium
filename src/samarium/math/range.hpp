/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <compare>
#include <type_traits> // for make_signed_t

#include "samarium/core/concepts.hpp" // for Integral

namespace sm::range
{
enum class Interval
{
    Open,
    Closed
};

template <concepts::Integral T> struct IntegerIterator
{
    using iterator_category = std::contiguous_iterator_tag;
    using difference_type   = T;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T;

    T index{};

    constexpr auto operator<=>(const IntegerIterator&) const noexcept = default;

    constexpr auto operator*() const noexcept { return index; }

    // Prefix increment
    constexpr auto operator++() noexcept -> IntegerIterator&
    {
        index++;
        return *this;
    }

    // Postfix increment
    constexpr auto operator++(int) noexcept -> IntegerIterator
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }
};

template <concepts::Integral T, Interval interval> struct IntegerSentinel
{
    T index{};

    [[nodiscard]] constexpr auto operator==(IntegerIterator<T> other) const noexcept -> bool
    {
        if constexpr (interval == Interval::Closed) { return other.index > index; }
        else { return other.index >= index; }
    }
};

template <concepts::FloatingPoint T, Interval interval = Interval::Open> struct FloatIterator
{
    using iterator_category = std::contiguous_iterator_tag;
    using difference_type   = T;
    using value_type        = T;
    using pointer           = T*;
    using reference         = T;

    T start{};
    T end{};
    u64 index_end{};
    u64 index{};

    constexpr auto operator<=>(const FloatIterator& other) const noexcept
    {
        return index <=> other.index;
    }

    constexpr auto operator*() const noexcept
    {
        return start + (end - start) * static_cast<T>(index) / static_cast<T>(index_end);
    }

    // Prefix increment
    constexpr auto operator++() noexcept -> FloatIterator&
    {
        index++;
        return *this;
    }

    // Postfix increment
    constexpr auto operator++(int) noexcept -> FloatIterator
    {
        auto tmp = *this;
        ++(*this);
        return tmp;
    }
};

template <concepts::Integral T = u64, Interval interval = Interval::Open> struct Integral
{
    T min{};
    T max{};

    [[nodiscard]] constexpr auto begin() const noexcept { return IntegerIterator<T>{min}; }

    [[nodiscard]] constexpr auto end() const noexcept { return IntegerSentinel<T, interval>{max}; }
};

template <concepts::FloatingPoint T = f64, Interval interval = Interval::Open> struct FloatRange
{
    T min{};
    T max{};
    u64 count{};

    [[nodiscard]] constexpr auto begin() const noexcept
    {
        if constexpr (interval == Interval::Closed) { return FloatIterator<T>{min, max, count, 0}; }
        else { return FloatIterator<T>{min, max, count - 1, 0}; }
    }

    [[nodiscard]] constexpr auto end() const noexcept
    {
        if constexpr (interval == Interval::Closed)
        {
            return FloatIterator<T>{min, max, count, count};
        }
        else { return FloatIterator<T>{min, max, count + 1, count + 1}; }
    }
};

template <concepts::Integral T, Interval interval = Interval::Open>
[[nodiscard]] constexpr auto end(T end) noexcept
{
    return Integral<T, interval>{.max{end}};
}

template <concepts::Integral T, Interval interval = Interval::Open>
[[nodiscard]] constexpr auto start_end(T start, T end) noexcept
{
    return Integral<T, interval>{start, end};
}
} // namespace sm::range
