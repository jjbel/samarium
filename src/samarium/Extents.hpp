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

#include "math.hpp"

namespace sm
{
template <concepts::arithmetic T> class Extents
{
  public:
    constexpr Extents(T a, T b)
    {
        std::tie(m_min, m_max) = (a < b) ? std::pair{ a, b } : std::pair{ b, a };
    }
    
    [[nodiscard]] constexpr auto min() const { return m_min; }
    [[nodiscard]] constexpr auto max() const { return m_max; }

    [[nodiscard]] constexpr auto contains(T value) const
    {
        return m_min <= value and value <= m_max;
    }

    [[nodiscard]] constexpr auto clamp(T value) const
    {
        return value < m_min ? m_min : value > m_max ? m_max : value;
    }

    template <concepts::floating_point U>
    [[nodiscard]] constexpr auto lerp(U factor) const
    {
        return m_min * (static_cast<U>(1.) - factor) + m_max * factor;
    }

  private:
    T m_min;
    T m_max;
};
} // namespace sm
