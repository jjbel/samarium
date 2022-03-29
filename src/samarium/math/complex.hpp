/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <complex>

#include "Vector2.hpp"

namespace sm
{
template <typename T> [[nodiscard]] constexpr auto to_complex(Vector2_t<T> vec) noexcept
{
    return std::complex{vec.x, vec.y};
}

template <typename T> [[nodiscard]] constexpr auto from_complex(std::complex<T> complex) noexcept
{
    return Vector2_t<T>{.x = complex.real(), .y = complex.imag()};
}
} // namespace sm
