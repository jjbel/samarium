/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <algorithm> // for minmax
#include <array>     // for array

#include "Extents.hpp" // for Extents
#include "Vec2.hpp"    // for Vec2
#include "loop.hpp"    // for start_end
#include "shapes.hpp"  // for LineSegment


namespace sm
{
enum class PlacementX
{
    Left   = 0,
    Middle = 1,
    Right  = 2
};

enum class PlacementY
{
    Bottom = 0,
    Middle = 1,
    Top    = 2
};

struct Placement
{
    PlacementX x{};
    PlacementY y{};
};

// use Box2{{}, max}, not Box2{max} for min={0, 0}
template <concepts::Number T = f64> struct Box2
{
    using VecType = Vec2_t<T>;
    VecType min;
    VecType max;

    /**
     * @brief               Make a Box2 which fits a range of points
     *
     * @param  points       points to fit around
     */
    [[nodiscard]] static constexpr auto fit_points(const auto& points)
    {
        const auto [x_min, x_max] = std::ranges::minmax(points, {}, &VecType::x);
        const auto [y_min, y_max] = std::ranges::minmax(points, {}, &VecType::y);
        return Box2{{x_min.x, y_min.y}, {x_max.x, y_max.y}};
    }

    // TODO rename this union or smthg
    // return a box which tightly encloses a and b
    [[nodiscard]] static constexpr auto fit_boxes(Box2<T> a, Box2<T> b)
    {
        auto box  = Box2<T>{};
        box.min.x = a.min.x < b.min.x ? a.min.x : b.min.x;
        box.min.y = a.min.y < b.min.y ? a.min.y : b.min.y;
        box.max.x = a.max.x > b.max.x ? a.max.x : b.max.x;
        box.max.y = a.max.y > b.max.y ? a.max.y : b.max.y;
        return box;
    }

    [[nodiscard]] static constexpr auto fit_boxes(const auto& boxes)
    {
        auto box = boxes[0];
        for (auto i : loop::start_end(1, boxes.size())) { box = fit_boxes(box, boxes[i]); }
        return box;
    }

    // TODO rename cast
    template <concepts::Number U> [[nodiscard]] constexpr auto cast() const
    {
        return Box2<U>{min.template cast<U>(), max.template cast<U>()};
    }

    [[nodiscard]] static constexpr auto square(T width) noexcept
        requires std::is_signed_v<T>
    {
        width = std::abs(width / 2); // recenter, make +ve
        return Box2{.min{-width, -width}, .max{width, width}};
    }

    [[nodiscard]] static constexpr auto find_min_max(VecType p1, VecType p2)
    {
        return Box2<T>{{math::min(p1.x, p2.x), math::min(p1.y, p2.y)},
                       {math::max(p1.x, p2.x), math::max(p1.y, p2.y)}};
    }

    // TODO remove validation
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
        return Box2{centre - vec, centre + vec};
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

    [[nodiscard]] constexpr auto clamped_to(Box2<T> bounds) const
    {
        using ext        = Extents<T>;
        const auto ext_x = ext{bounds.min.x, bounds.max.x};
        const auto ext_y = ext{bounds.min.y, bounds.max.y};
        return Box2<T>{{ext_x.clamp(min.x), ext_y.clamp(min.y)},
                       {ext_x.clamp(max.x), ext_y.clamp(max.y)}};
    }

    [[nodiscard]] constexpr auto width() const { return max.x - min.x; }
    [[nodiscard]] constexpr auto height() const { return max.y - min.y; }

    [[nodiscard]] constexpr auto x_range() const { return Extents<T>{min.x, max.x}; }
    [[nodiscard]] constexpr auto y_range() const { return Extents<T>{min.y, max.y}; }

    [[nodiscard]] constexpr bool operator==(const Box2<T>&) const = default;

    [[nodiscard]] constexpr auto centre() const noexcept { return (min + max) / static_cast<T>(2); }

    constexpr auto set_centre(VecType new_centre) noexcept
    {
        const auto shift = new_centre - centre();
        min += shift;
        max += shift;
    }

    constexpr auto set_width(T new_width) noexcept
    {
        const auto half_width     = math::abs(new_width) / static_cast<T>(2);
        const auto current_centre = centre();
        min.x                     = current_centre.x - half_width;
        max.x                     = current_centre.x + half_width;
    }

    constexpr auto set_height(T new_height) noexcept
    {
        const auto half_height    = math::abs(new_height) / static_cast<T>(2);
        const auto current_centre = centre();
        min.y                     = current_centre.y - half_height;
        max.y                     = current_centre.y + half_height;
    }

    [[nodiscard]] constexpr auto scaled(T scale) const noexcept
    {
        return from_centre_width_height(centre(), width() * scale, height() * scale);
    }

    [[nodiscard]] constexpr auto scaled_x(T scale) const noexcept
    {
        return from_centre_width_height(centre(), width() * scale, height());
    }

    [[nodiscard]] constexpr auto scaled_y(T scale) const noexcept
    {
        return from_centre_width_height(centre(), width(), height() * scale);
    }

    [[nodiscard]] constexpr auto line_segments() const noexcept
        requires concepts::FloatingPoint<T>
    {
        const auto top_right   = Vec2{max.x, min.y};
        const auto bottom_left = Vec2{min.x, max.y};
        return std::array<LineSegment, 4>{
            {{min, top_right}, {top_right, max}, {max, bottom_left}, {bottom_left, min}}};
    }

    [[nodiscard]] constexpr auto points() const noexcept
    {
        return std::to_array<VecType>({min, {min.x, max.y}, max, {max.x, min.y}});
    }

    // map from {(0 0) bottom-left, (1, 1) bottom-right} to {min, max}
    // basically lerp
    [[nodiscard]] constexpr auto map_position(VecType pos) { return min + (max - min) * pos; }

    [[nodiscard]] constexpr auto get_placement(Placement p)
    {
        return map_position({static_cast<f64>(p.x) / 2.0, static_cast<f64>(p.y) / 2.0});
    }
};
} // namespace sm
