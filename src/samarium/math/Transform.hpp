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

#include "fmt/format.h"

#include "Rect.hpp"
#include "shapes.hpp"

namespace sm
{
class Transform
{
  public:
    Vector2 pos{};
    Vector2 scale{};

    constexpr auto apply(Vector2 vec) const { return vec * scale + pos; }

    constexpr auto apply(Rect<double_t> rect) const
    {
        return Rect<double_t>{apply(rect.min), apply(rect.max)};
    }

    constexpr auto apply_inverse(Vector2 vec) const { return (vec - pos) / scale; }

    constexpr auto apply_inverse(Rect<double_t> rect) const
    {
        return Rect<double_t>::find_min_max( // -ve sign may invalidate min, max so recalculate it
            apply_inverse(rect.min), apply_inverse(rect.max));
    }

    constexpr auto apply_inverse(LineSegment l) const
    {
        return LineSegment( // -ve sign may invalidate min, max so recalculate it
            apply_inverse(l.p1), apply_inverse(l.p2));
    }
};
} // namespace sm

template <> class fmt::formatter<sm::Transform>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) const { return ctx.begin(); }

    template <typename FormatContext> auto format(const sm::Transform& p, FormatContext& ctx)
    {
        return format_to(ctx.out(), "Transform[pos: {}, scale: {}]", p.pos, p.scale);
    }
};
