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

#include <optional>

#include "interp.hpp"
#include "shapes.hpp"

#include "samarium/util/print.hpp"

namespace sm::math
{
[[nodiscard]] constexpr auto distance(Vector2 p1, Vector2 p2)
{
    return (p1 - p2).length();
}

[[nodiscard]] constexpr auto lerp_along(Vector2 point, const LineSegment& ls)
{
    return Vector2::dot(point - ls.p1, ls.vector()) / ls.length_sq();
}

[[nodiscard]] constexpr auto clamped_lerp_along(Vector2 point, const LineSegment& ls)
{
    return interp::clamp(Vector2::dot(point - ls.p1, ls.vector()) /
                             ls.length_sq(),
                         Extents<double>{0.0, 1.0});
}

[[nodiscard]] constexpr auto project(Vector2 point, const LineSegment& ls)
{
    // https://stackoverflow.com/a/1501725/17100530
    const auto vec = ls.vector();
    const auto t   = Vector2::dot(point - ls.p1, vec) / ls.length_sq();
    return interp::lerp(t, Extents<Vector2>{ls.p1, ls.p2});
}

[[nodiscard]] constexpr auto project_clamped(Vector2 point, const LineSegment& ls)
{
    // https://stackoverflow.com/a/1501725/17100530
    const auto vec = ls.vector();
    const auto t   = interp::clamp(
          Vector2::dot(point - ls.p1, vec) / ls.length_sq(), {0., 1.});
    return interp::lerp(t, Extents<Vector2>{ls.p1, ls.p2});
}

[[nodiscard]] constexpr auto distance(Vector2 point, const LineSegment& ls)
{
    const auto l2 = ls.length_sq();
    if (almost_equal(l2, 0.)) return distance(point, ls.p1); // p1 == p2 case
    return distance(point, project(point, ls));
}

[[nodiscard]] constexpr auto clamped_distance(Vector2 point, const LineSegment& ls)
{
    const auto l2 = ls.length_sq();
    if (almost_equal(l2, 0.)) return distance(point, ls.p1); // p1 == p2 case
    return distance(point, project_clamped(point, ls));
}

[[nodiscard]] constexpr auto lies_in_segment(Vector2 point, const LineSegment& l)
{
    return interp::in_range(
        Vector2::dot(point - l.p1, l.vector()) / l.length_sq(), {0., 1.});
}

[[nodiscard]] constexpr std::optional<Vector2> intersection(const LineSegment& l1,
                                                            const LineSegment& l2)
{
    const auto denom1 = l1.p2.x - l1.p1.x;
    const auto denom2 = l2.p2.x - l2.p1.x;

    const auto denom1_is_0 = almost_equal(denom1, 0.0);
    const auto denom2_is_0 = almost_equal(denom2, 0.0);

    if (denom1_is_0 && denom2_is_0) return std::nullopt;

    if (denom1_is_0)
        return std::optional{
            Vector2{l1.p1.x, l2.slope() * (l1.p1.x - l2.p1.x) + l2.p1.y}};
    else if (denom2_is_0)
        return std::optional{
            Vector2{l2.p1.x, l1.slope() * (l2.p1.x - l1.p1.x) + l1.p1.y}};
    else
    {
        const auto m1 = l1.slope();
        const auto m2 = l2.slope();

        const auto x =
            (m2 * l2.p1.x - m1 * l1.p1.x + l1.p1.y - l2.p1.y) / (m2 - m1);
        return std::optional{Vector2{x, m1 * (x - l1.p1.x) + l1.p1.y}};
    }
}

[[nodiscard]] constexpr std::optional<Vector2>
clamped_intersection(const LineSegment& l1, const LineSegment& l2)
{
    const auto point = intersection(l1, l2);
    if (!point) return std::nullopt;
    if (lies_in_segment(*point, l1) && lies_in_segment(*point, l2)) return point;
    else
        return std::nullopt;
}
} // namespace sm::math
