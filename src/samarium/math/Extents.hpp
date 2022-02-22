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

#include <tuple>

#include "samarium/graphics/Color.hpp"

#include "math.hpp"

namespace sm
{
template <concepts::arithmetic T> class Extents
{
  public:
    T min{};
    T max{};

    [[nodiscard]] static constexpr auto find_min_max(T a, T b)
    {
        return (a < b) ? Extents{a, b} : Extents{b, a};
    }

    [[nodiscard]] constexpr auto size() const { return max - min; }

    [[nodiscard]] constexpr auto contains(T value) const { return min <= value and value <= max; }

    [[nodiscard]] constexpr auto clamp(T value) const
    {
        return value < min ? min : value > max ? max : value;
    }

    [[nodiscard]] constexpr auto lerp(double_t factor) const
    {
        return min * (1. - factor) + max * factor;
    }

    [[nodiscard]] constexpr auto clamped_lerp(double_t factor) const
    {
        return min * (1. - this->clamp(factor)) + max * factor;
    }

    [[nodiscard]] constexpr double_t lerp_inverse(T value) const
    {
        return (value - min) / this->size();
    }
};
} // namespace sm
