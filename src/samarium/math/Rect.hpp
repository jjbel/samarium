/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "Vector2.hpp"

namespace sm
{
template <concepts::Number T = f64> struct Rect
{
    Vector2_t<T> min;
    Vector2_t<T> max;

    template <concepts::Number U> [[nodiscard]] constexpr auto as() const
    {
        // very weird: https://stackoverflow.com/a/3505738/17100530
        return Rect<U>{min.template as<U>(), max.template as<U>()};
    }

    [[nodiscard]] static constexpr auto find_min_max(Vector2_t<T> p1,
                                                     Vector2_t<T> p2)
    {
        return Rect<T>{{math::min(p1.x, p2.x), math::min(p1.y, p2.y)},
                       {math::max(p1.x, p2.x), math::max(p1.y, p2.y)}};
    }

    [[nodiscard]] static constexpr auto
    from_centre_width_height(Vector2_t<T> centre, T width, T height)
    {
        const auto vec = Vector2_t{.x = width, .y = height};
        return Rect{centre - vec, centre + vec};
    }

    [[nodiscard]] constexpr auto contains(const Vector2_t<T>& vec) const noexcept
    {
        return vec.x >= min.x && vec.x <= max.x && vec.y >= min.y &&
               vec.y <= max.y;
    }

    [[nodiscard]] constexpr auto clamped_to(Rect<T> bounds) const
    {
        return Rect<T>{{math::max(this->min.x, bounds.min.x),
                        math::max(this->min.y, bounds.min.y)},
                       {math::min(this->min.x, bounds.min.x),
                        math::min(this->min.y, bounds.min.y)}};
    }

    [[nodiscard]] constexpr friend bool
    operator==(const Rect<T>& lhs, const Rect<T>& rhs) noexcept = default;
};
} // namespace sm


template <sm::concepts::Number T> class fmt::formatter<sm::Rect<T>>
{
  public:
    constexpr auto parse(const format_parse_context& ctx) { return ctx.begin(); }

    auto format(const sm::Rect<T>& p, auto& ctx)
    {
        return format_to(ctx.out(),
        R"(
Rect(min = {},
     max = {}))", p.min, p.max);
    }
};
