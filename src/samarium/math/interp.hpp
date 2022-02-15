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
[[nodiscard]] constexpr auto smooth_step()
{
    return [](auto x) { return x * x * (3. - 2. * x); };
}

[[nodiscard]] constexpr auto smoother_step()
{
    return [](auto x) { return x * x * x * (x * (x * 6. - 15.) + 10.); };
}

template <typename T> [[nodiscard]] constexpr auto in_range(T value, Extents<T> range)
{
    return range.contains(value);
}

template <typename T>
[[nodiscard]] constexpr auto clamp(T value, Extents<T> range) noexcept
{
    return range.clamp(value);
}

template <typename T> [[nodiscard]] constexpr auto lerp(double factor, Extents<T> range)
{
    return range.lerp(factor);
}

template <typename T>
[[nodiscard]] constexpr auto lerp_inverse(double value, Extents<T> range)
{
    return range.lerp_inverse(value);
}

// https://stackoverflow.com/questions/1969240/mapping-a-range-of-values-to-another

template <typename T, typename Output_t = T>
[[nodiscard]] constexpr auto map_range(T value, Extents<T> from, Extents<T> to)
{
    return static_cast<Output_t>(from.min + (value - from.min) * to.size() / from.size());
}

template <typename T, typename Output_t = T>
[[nodiscard]] constexpr auto map_range_clamp(T value, Extents<T> from, Extents<T> to)
{
    return static_cast<Output_t>(from.min + (from.clamp(value) - from.min) * to.size() /
                                                from.size());
}

template <typename T, typename Output_t = T>
[[nodiscard]] constexpr auto make_mapper(Extents<T> from, Extents<T> to)
{
    return [from_min = from.min, from_max = from.max, from_range = from.size(),
            to_range = to.size()](T value) {
        return static_cast<Output_t>(from_min +
                                     (value - from_min) * to_range / from_range);
    };
}

template <typename T, typename Output_t = T>
[[nodiscard]] constexpr auto make_clamped_mapper(Extents<T> from, Extents<T> to)
{
    return [from, from_min = from.min, from_max = from.max,
            from_range = from.max - from.min, to_range = to.max - to.min](T value)
    {
        return static_cast<Output_t>(from_min + (from.clamp(value) - from_min) *
                                                    to_range / from_range);
    };
}
} // namespace sm::interp
