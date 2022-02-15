/*
 *                                  MIT License
 *
 *                               Copyright (c) 2022
 *
 *       Project homepage: <https://github.com/strangeQuark1041/samarium/>
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the Software), to deal
 *  in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *     copies of the Software, and to permit persons to whom the Software is
 *            furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 *                copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *                                   SOFTWARE.
 *
 *  For more information, please refer to <https://opensource.org/licenses/MIT/>
 */

#pragma once

#include <cmath>

#include "core/concepts.hpp"

namespace sm::math
{
constexpr inline auto PI      = 3.14159265358979323846;
constexpr inline auto EPSILON = 1.e-4;

template <concepts::floating_point T>
[[nodiscard]] constexpr auto equals(T a, T b) noexcept
{
    return std::abs(a - b) <= EPSILON;
}

template <typename T> [[nodiscard]] constexpr inline auto min(T value0, T value1) noexcept
{
    if (value0 < value1) return value0;
    else
        return value1;
}

template <typename T> [[nodiscard]] constexpr inline auto max(T value0, T value1) noexcept
{
    if (value0 > value1) return value0;
    else
        return value1;
}

template <u32 n> [[nodiscard]] constexpr inline auto power(auto x)
{
    if constexpr (n == 0) return 1;
    // if constexpr (n == 1) return x;

    return x * power<n-1>(x);
}
} // namespace sm::math
