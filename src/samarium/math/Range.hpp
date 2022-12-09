/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "samarium/core/concepts.hpp" // for Integral

namespace sm
{
enum class Interval
{
    Closed,
    Open
};

template <concepts::Integral T> struct IntegerIterator
{
    using iterator_category = std::contiguous_iterator_tag;
    using difference_type   = T;
    using value_type        = T;
    using pointer           = T*; // or also value_type*
    using reference         = T;  // or also value_type&

    T index;

    constexpr auto operator<=>(const IntegerIterator&) const noexcept = default;

    constexpr auto operator*() const noexcept -> reference { return index; }

    constexpr auto operator->() noexcept -> pointer { return &index; }

    // Prefix increment
    constexpr auto operator++() noexcept -> IntegerIterator&
    {
        index++;
        return *this;
    }

    // Postfix increment
    constexpr auto operator++(int) noexcept -> IntegerIterator
    {
        IntegerIterator tmp = *this;
        ++(*this);
        return tmp;
    }
};

// template <concepts::FloatingPoint T = f64, Interval interval = Interval::Open> struct Range
//{
//     f64 start{};
//     f64 end{};
//     u64 count{};
//
//     [[nodiscard]] constexpr auto size() const noexcept { return count; }
//
//     [[nodiscard]] constexpr auto begin() const noexcept { return start; }
//     [[nodiscard]] constexpr auto begin() const noexcept { return e; }
// };
} // namespace sm
