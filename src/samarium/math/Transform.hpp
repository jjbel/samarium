/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

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

    constexpr auto apply(const Rect<f64>& rect) const
    {
        return Rect<f64>{apply(rect.min), apply(rect.max)}.validated();
    }

    constexpr auto apply_inverse(Vector2 vec) const
    {
        return (vec - pos) / scale;
    }

    constexpr auto apply_inverse(const Rect<f64>& rect) const
    {
        return Rect<f64>::find_min_max(
            this->apply_inverse(rect.min),
            this->apply_inverse(
                rect.max)); // -ve sign may invalidate min, max, so recalculate it
    }

    constexpr auto apply_inverse(const LineSegment& l) const
    {
        return LineSegment( // -ve sign may invalidate min, max so recalculate it
            apply_inverse(l.p1), apply_inverse(l.p2));
    }
};
} // namespace sm

template <> class fmt::formatter<sm::Transform>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) const
    {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const sm::Transform& p, FormatContext& ctx)
    {
        return format_to(ctx.out(), "Transform[pos: {}, scale: {}]", p.pos,
                         p.scale);
    }
};
