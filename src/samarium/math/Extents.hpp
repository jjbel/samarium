/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <compare>
#include <iterator>

#include "../core/concepts.hpp"

namespace sm
{
template <concepts::Arithmetic T> class Extents
{
  public:
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

    [[nodiscard]] constexpr auto clamp(T value) const noexcept
    {
        if (value < min) return min;
        else if (value > max)
            return max;
        else
            return value;
    }

    [[nodiscard]] constexpr auto lerp(f64 factor) const noexcept
    {
        return min * (1.0 - factor) + max * factor;
    }

    [[nodiscard]] constexpr auto clamped_lerp(f64 factor) const noexcept
    {
        return min * (1.0 - this->clamp(factor)) + max * factor;
    }

    [[nodiscard]] constexpr f64 lerp_inverse(T value) const noexcept
    {
        return (value - min) / this->size();
    }

    template <typename Function>
    constexpr auto
    for_each(Function&& fn) const requires concepts::Integral<T> && std::invocable<Function, T>
    {
        for (auto i = min; i <= max; i++) { fn(i); }
    }

    struct Iterator
    {
        using iterator_category = std::contiguous_iterator_tag;
        using difference_type   = T;
        using value_type        = T;
        using pointer           = T*; // or also value_type*
        using reference         = T;  // or also value_type&

        T index;
        constexpr auto operator<=>(const Iterator&) const noexcept = default;

        constexpr reference operator*() const noexcept { return index; }

        constexpr pointer operator->() noexcept { return &index; }

        // Prefix increment
        constexpr Iterator& operator++() noexcept
        {
            index++;
            return *this;
        }

        // Postfix increment
        constexpr Iterator operator++(int) noexcept
        {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }
    };

    [[nodiscard]] constexpr auto begin() const noexcept requires concepts::Integral<T>
    {
        return Iterator{min};
    }

    [[nodiscard]] constexpr auto end() const noexcept requires concepts::Integral<T>
    {
        return Iterator{max};
    }

    [[nodiscard]] constexpr auto operator[](u64 index) const { return min + index; }
};


[[nodiscard]] constexpr auto range(u64 max) { return Extents<u64>{0UL, max}; }

[[nodiscard]] constexpr auto range(u64 min, u64 max) { return Extents<u64>{min, max}; }
} // namespace sm
