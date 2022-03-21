/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <compare>

#include "../core/types.hpp"

namespace sm
{
struct Range
{
    u64 min;
    u64 max;

    struct Iterator
    {
        using iterator_category = std::contiguous_iterator_tag;
        using difference_type   = u64;
        using value_type        = u64;
        using pointer           = u64*; // or also value_type*
        using reference         = u64;  // or also value_type&

        u64 index;
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

    constexpr Range(u64 max_) noexcept : min{}, max{max_} {}

    constexpr Range(u64 min_, u64 max_) noexcept : min{min_}, max{max_} {}

    [[nodiscard]] constexpr auto begin() const noexcept { return Iterator{min}; }

    [[nodiscard]] constexpr auto end() const noexcept { return Iterator{max}; }
};
} // namespace sm
