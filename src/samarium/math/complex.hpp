/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <complex>

#include "Vec2.hpp"

namespace sm
{
template <typename T> [[nodiscard]] constexpr auto to_complex(Vec2_t<T> vec) noexcept
{
    return std::complex{vec.x, vec.y};
}

template <typename T> [[nodiscard]] constexpr auto from_complex(std::complex<T> complex) noexcept
{
    return Vec2_t<T>{.x = complex.real(), .y = complex.imag()};
}

namespace math
{
static const auto two_pi_i = 2.0 * pi * std::complex<f64>{0.0, 1.0};
} // namespace math
} // namespace sm
