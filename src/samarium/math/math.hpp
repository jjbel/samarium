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
#include <numbers>

#include "samarium/core/concepts.hpp"

namespace sm::math
{
constexpr inline auto EPSILON = 1.e-4;

template <concepts::FloatingPoint T>
[[nodiscard]] constexpr auto almost_equal(T a, T b) noexcept
{
    return std::abs(a - b) <= EPSILON;
}

template <typename T>
[[nodiscard]] constexpr auto min(T value0, T value1) noexcept
{
    if (value0 < value1) return value0;
    else
        return value1;
}

template <typename T>
[[nodiscard]] constexpr auto max(T value0, T value1) noexcept
{
    if (value0 > value1) return value0;
    else
        return value1;
}

template <u32 n> [[nodiscard]] constexpr auto power(auto x)
{
    if constexpr (n == 0) return 1;

    return x * power<n - 1>(x);
}
} // namespace sm::math

namespace sm::literals
{
consteval auto operator"" _degrees(long double angle)
{
    return angle / 180.0 * std::numbers::pi;
}

consteval auto operator"" _radians(long double angle)
{
    return angle * 180.0 / std::numbers::pi;
}
} // namespace sm::literals
