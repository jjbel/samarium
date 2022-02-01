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
#include "Vector2.hpp"
#include "math.hpp"

namespace sm
{
template <sm::util::number T = double> class Rect
{
  public:
    // tag dispatching modes:
    static class default_init_t
    {
    } default_init;
    static class from_min_max_t
    {
    } from_min_max;
    static class centre_width_height_t
    {
    } centre_width_height;

    // members:
    const Vector2_t<T> min;
    const Vector2_t<T> max;

    // constructors:
    constexpr Rect() noexcept : min{}, max{} {}

    constexpr Rect(Vector2_t<T> vec1, Vector2_t<T> vec2) noexcept
        : min(std::min(vec1.x, vec2.x), std::min(vec1.y, vec2.y)),
          max(std::max(vec1.x, vec2.x), std::max(vec1.y, vec2.y))
    {
    }

    constexpr Rect(from_min_max_t, Vector2_t<T> min_, Vector2_t<T> max_) noexcept
        : min{ min_ }, max{ max_ }
    {
    }

    constexpr Rect(centre_width_height_t, Vector2_t<T> centre, T width, T height) noexcept
        : min{ centre - Vector2_t<T>{ -std::abs(width), -std::abs(height) } }, max{
              centre + Vector2_t<T>{ std::abs(width), std::abs(height) }
          }
    {
    }
};
} // namespace sm
