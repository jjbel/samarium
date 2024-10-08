/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <span> // for span

#include "samarium/core/concepts.hpp" // for FloatingPoint

namespace sm::math
{
template <concepts::FloatingPoint T = f64>
[[nodiscard]] constexpr auto sum(std::span<T> data) noexcept
{
    // TODO use std alogrithms
    // TODO use threading/std::execution

    auto sum = T{};
    for (const auto& i : data) { sum += i; }
    return sum;
}

template <concepts::FloatingPoint T = f64>
[[nodiscard]] constexpr auto mean(std::span<T> data) noexcept
{
    return sum(data) / static_cast<T>(data.size());
}
} // namespace sm::math
