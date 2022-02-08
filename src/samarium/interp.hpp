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

#include "Extents.hpp"

namespace sm::interp
{
template <typename T>
[[nodiscard]] constexpr inline auto in_range(T value, Extents<T> range)
{
    return range.contains(value);
}

template <typename T>
[[nodiscard]] constexpr inline auto clamp(T value, Extents<T> range) noexcept
{
    return range.clamp(value);
}

template <typename T, typename U>
[[nodiscard]] constexpr inline auto lerp(T value, Extents<U> range)
{
    return range.lerp(value);
}

// https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another

template <typename T, typename U, bool shouldClamp = false>
[[nodiscard]] constexpr inline auto
map_range(T value, T from_min, T from_max, U to_min, U to_max)
{
    if constexpr (shouldClamp) value = clamp(value, from_min, from_max);
    const auto fromRange = from_max - from_min;
    const auto toRange   = to_max - to_min;
    return from_min + (value - from_min) * toRange / fromRange;
}
} // namespace sm::interp
