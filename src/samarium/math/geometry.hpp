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

#include "interp.hpp"
#include "shapes.hpp"

namespace sm::math
{
[[nodiscard]] constexpr auto distance(Vector2 p1, Vector2 p2)
{
    return (p1 - p2).length();
}

[[nodiscard]] constexpr auto project(Vector2 point, LineSegment ls)
{
    // https://stackoverflow.com/a/1501725/17100530
    const auto vec = ls.vector();
    const auto t   = Vector2::dot(point - ls.p1, vec) / ls.length_sq();
    return interp::lerp(t, Extents<Vector2>{ls.p1, ls.p2});
}

[[nodiscard]] constexpr auto project_clamped(Vector2 point, LineSegment ls)
{
    // https://stackoverflow.com/a/1501725/17100530
    const auto vec = ls.vector();
    const auto t =
        interp::clamp(Vector2::dot(point - ls.p1, vec) / ls.length_sq(), {0., 1.});
    return interp::lerp(t, Extents<Vector2>{ls.p1, ls.p2});
}

[[nodiscard]] constexpr auto distance(Vector2 point, LineSegment ls)
{
    const auto l2 = ls.length_sq();
    if (almost_equal(l2, 0.)) return distance(point, ls.p1); // p1 == p2 case
    return distance(point, project(point, ls));
}

[[nodiscard]] constexpr auto clamped_distance(Vector2 point, LineSegment ls)
{
    const auto l2 = ls.length_sq();
    if (almost_equal(l2, 0.)) return distance(point, ls.p1); // p1 == p2 case
    return distance(point, project_clamped(point, ls));
}
} // namespace sm::math
