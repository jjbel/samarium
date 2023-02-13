/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <array> // for array

#include "range/v3/algorithm/minmax.hpp" // for minmax

#include "Extents.hpp" // for Extents
#include "Vector2.hpp" // for Vector2
#include "shapes.hpp"  // for LineSegment

namespace sm
{
template <concepts::Number T = f64> struct BoundingBox
{
    using VecType = Vector2_t<T>;
    VecType min;
    VecType max;

    /**
     * @brief               Make a BoundingBox which fits a range of points
     *
     * @param  points       points to fit around
     */
    [[nodiscard]] static constexpr auto fit(const auto& points)
    {
        auto [x_min, x_max] = ranges::minmax(points, {}, &VecType::x);
        auto [y_min, y_max] = ranges::minmax(points, {}, &VecType::y);
        return BoundingBox{{x_min.x, y_min.y}, {x_max.x, y_max.y}};
    }

    template <concepts::Number U> [[nodiscard]] constexpr auto as() const
    {
        return BoundingBox<U>{min.template cast<U>(), max.template cast<U>()};
    }

    [[nodiscard]] static constexpr auto square(T width) noexcept
        requires std::is_signed_v<T>
    {
        width = std::abs(width / 2); // recenter, make +ve
        return BoundingBox{.min{-width, -width}, .max{width, width}};
    }

    [[nodiscard]] static constexpr auto find_min_max(VecType p1, VecType p2)
    {
        return BoundingBox<T>{{math::min(p1.x, p2.x), math::min(p1.y, p2.y)},
                              {math::max(p1.x, p2.x), math::max(p1.y, p2.y)}};
    }

    constexpr auto validate() { *this = find_min_max(this->min, this->max); }

    [[nodiscard]] constexpr auto validated() const
    {
        auto bounding_box = *this;
        bounding_box.validate();
        return bounding_box;
    }

    [[nodiscard]] static constexpr auto
    from_centre_width_height(VecType centre, T width, T height) noexcept
    {
        const auto vec = VecType{.x = width / static_cast<T>(2), .y = height / static_cast<T>(2)};
        return BoundingBox{centre - vec, centre + vec};
    }

    [[nodiscard]] constexpr auto contains(VecType vec) const noexcept
    {
        return vec.x >= min.x && vec.x <= max.x && vec.y >= min.y && vec.y <= max.y;
    }

    [[nodiscard]] constexpr auto diagonal() const noexcept { return max - min; }

    [[nodiscard]] constexpr auto clamp(VecType vec) const
    {
        return VecType{Extents<T>{min.x, max.x}.clamp(vec.x),
                       Extents<T>{min.y, max.y}.clamp(vec.y)};
    }

    [[nodiscard]] constexpr auto clamped_to(BoundingBox<T> bounds) const
    {
        using ext        = Extents<T>;
        const auto ext_x = ext{bounds.min.x, bounds.max.x};
        const auto ext_y = ext{bounds.min.y, bounds.max.y};
        return BoundingBox<T>{{ext_x.clamp(min.x), ext_y.clamp(min.y)},
                              {ext_x.clamp(max.x), ext_y.clamp(max.y)}};
    }

    [[nodiscard]] constexpr auto width() const { return max.x - min.x; }
    [[nodiscard]] constexpr auto height() const { return max.y - min.y; }

    [[nodiscard]] constexpr auto x_range() const { return Extents<T>{min.x, max.x}; }
    [[nodiscard]] constexpr auto y_range() const { return Extents<T>{min.y, max.y}; }

    [[nodiscard]] constexpr auto operator==(const BoundingBox<T>&) const -> bool = default;

    [[nodiscard]] constexpr auto centre() const noexcept { return (min + max) / static_cast<T>(2); }

    constexpr auto set_centre(VecType new_centre) noexcept
    {
        const auto shift = new_centre - centre();
        min += shift;
        max += shift;
    }

    constexpr auto set_width(f64 new_width) noexcept
    {
        const auto half_width     = math::abs(new_width) / static_cast<T>(2);
        const auto current_centre = centre();
        min.x                     = current_centre.x - half_width;
        max.x                     = current_centre.x + half_width;
    }

    constexpr auto set_height(f64 new_height) noexcept
    {
        const auto half_height    = math::abs(new_height) / static_cast<T>(2);
        const auto current_centre = centre();
        min.y                     = current_centre.y - half_height;
        max.y                     = current_centre.y + half_height;
    }

    [[nodiscard]] constexpr auto line_segments() const noexcept
        requires concepts::FloatingPoint<T>
    {
        const auto top_right   = Vector2{max.x, min.y};
        const auto bottom_left = Vector2{min.x, max.y};
        return std::array<LineSegment, 4>{
            {{min, top_right}, {top_right, max}, {max, bottom_left}, {bottom_left, min}}};
    }

    [[nodiscard]] constexpr auto points() const noexcept
    {
        return std::to_array<VecType>({min, {min.x, max.y}, max, {max.x, min.y}});
    }
};
} // namespace sm
