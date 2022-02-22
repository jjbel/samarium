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

#include <array>

#include "samarium/math/interp.hpp"

#include "Color.hpp"

namespace sm
{
template <size_t size> class Gradient
{
    static_assert(size > 1, "gradient must have at least 2 colors");

    // std::array<Color, size> colors;
};

template <> class Gradient<2>
{
    Color from{};
    Color to{};

  public:
    constexpr Gradient(Color from_, Color to_) : from{from_}, to{to_} {}
    constexpr auto operator()(double factor) const
    {
        return interp::lerp_rgb(factor, from, to);
    }
};

template <> class Gradient<3>
{
    Color from{};
    Color mid{};
    Color to{};

  public:
    constexpr Gradient(Color from_, Color mid_,Color to_) : from{from_}, to{to_} {}
    constexpr auto operator()(double factor) const
    {
        factor = Extents<double_t>{0.0, 1.0}.clamp(factor);
        if (factor < 0.5) { return interp::lerp_rgb(2.0 * factor, from, mid); }
        else
            return interp::lerp_rgb(2.0 * (factor - 0.5), mid, to);
    }
};
} // namespace sm
