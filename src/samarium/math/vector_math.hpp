/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <optional>

#include "samarium/core/concepts.hpp"     // for FloatingPoint
#include "samarium/util/SmallVector.hpp"  // for SmallVector
#include "samarium/util/StaticVector.hpp" // for StaticVector

#include "BoundingBox.hpp"
#include "Vector2.hpp" // for Vector2_t
#include "interp.hpp"
#include "loop.hpp" // for start_end
#include "shapes.hpp"

namespace sm::math
{
[[nodiscard]] constexpr auto distance(Vector2 p1, Vector2 p2) noexcept
{
    return (p1 - p2).length();
}

[[nodiscard]] constexpr auto distance_sq(Vector2 p1, Vector2 p2) noexcept
{
    return (p1 - p2).length_sq();
}

[[nodiscard]] constexpr auto within_distance(Vector2 p1, Vector2 p2, f64 distance) noexcept
{
    return distance_sq(p1, p2) <= distance * distance;
}

[[nodiscard]] constexpr auto lerp_along(Vector2 point, const LineSegment& ls) noexcept
{
    return Vector2::dot(point - ls.p1, ls.vector()) / ls.length_sq();
}

[[nodiscard]] constexpr auto clamped_lerp_along(Vector2 point, const LineSegment& ls) noexcept
{
    return interp::clamp(Vector2::dot(point - ls.p1, ls.vector()) / ls.length_sq(),
                         Extents<f64>{0.0, 1.0});
}

[[nodiscard]] constexpr auto project(Vector2 point, const LineSegment& ls) noexcept
{
    // https://stackoverflow.com/a/1501725/17100530
    const auto vec = ls.vector();
    const auto t   = Vector2::dot(point - ls.p1, vec) / ls.length_sq();
    return interp::lerp(t, Extents<Vector2>{ls.p1, ls.p2});
}

[[nodiscard]] constexpr auto project_clamped(Vector2 point, const LineSegment& ls) noexcept
{
    // https://stackoverflow.com/a/1501725/17100530
    const auto vec = ls.vector();
    const auto t   = interp::clamp(Vector2::dot(point - ls.p1, vec) / ls.length_sq(), {0., 1.});
    return interp::lerp(t, Extents<Vector2>{ls.p1, ls.p2});
}

[[nodiscard]] constexpr auto distance(Vector2 point, const LineSegment& ls) noexcept
{
    if (almost_equal(ls.length_sq(), 0.0)) return distance(point, ls.p1); // p1 == p2 case

    return distance(point, project(point, ls));
}

[[nodiscard]] constexpr auto clamped_distance(Vector2 point, const LineSegment& ls) noexcept
{
    if (const auto l2 = ls.length_sq(); almost_equal(l2, 0.))
    {
        return distance(point, ls.p1); // p1 == p2 case
    }
    return distance(point, project_clamped(point, ls));
}

[[nodiscard]] constexpr auto lies_in_segment(Vector2 point, const LineSegment& l) noexcept
{
    return interp::in_range(Vector2::dot(point - l.p1, l.vector()) / l.length_sq(), {0., 1.});
}

[[nodiscard]] constexpr auto intersection(const LineSegment& l1,
                                          const LineSegment& l2) noexcept -> std::optional<Vector2>
{
    const auto denom1 = l1.p2.x - l1.p1.x;
    const auto denom2 = l2.p2.x - l2.p1.x;

    const auto denom1_is_0 = almost_equal(denom1, 0.0);
    const auto denom2_is_0 = almost_equal(denom2, 0.0);

    if (denom1_is_0 && denom2_is_0) { return std::nullopt; }

    if (denom1_is_0) { return {Vector2{l1.p1.x, l2.slope() * (l1.p1.x - l2.p1.x) + l2.p1.y}}; }

    if (denom2_is_0) { return {Vector2{l2.p1.x, l1.slope() * (l2.p1.x - l1.p1.x) + l1.p1.y}}; }

    const auto m1 = l1.slope();
    const auto m2 = l2.slope();

    const auto x = (m2 * l2.p1.x - m1 * l1.p1.x + l1.p1.y - l2.p1.y) / (m2 - m1);
    return {Vector2{x, m1 * (x - l1.p1.x) + l1.p1.y}};
}

[[nodiscard]] inline auto
clamped_intersection(const LineSegment& l1,
                     const LineSegment& l2) noexcept -> std::optional<Vector2>
{
    const auto point = intersection(l1, l2);
    if (!point) { return std::nullopt; }
    if (lies_in_segment(*point, l1) && lies_in_segment(*point, l2)) { return point; }
    return std::nullopt;
}

[[nodiscard]] inline auto intersection(const LineSegment& line_segment, const BoundingBox<f64>& box)
{
    // TODO check if point is vertex of box
    auto points = StaticVector<Vector2, 4>{};
    for (const auto& line : box.line_segments())
    {
        if (const auto result = clamped_intersection(line_segment, line))
        {
            points.push_back(result.value());
        }
    }

    return points;
}

template <typename T> [[nodiscard]] constexpr auto area(BoundingBox<T> bounding_box) noexcept
{
    return (bounding_box.max.x - bounding_box.min.x) * (bounding_box.max.y - bounding_box.min.y);
}

[[nodiscard]] constexpr auto area(Circle circle)
{
    return math::pi * circle.radius * circle.radius;
}

template <typename T> [[nodiscard]] constexpr auto abs_area(BoundingBox<T> bounding_box) noexcept
{
    return math::abs(area(bounding_box));
}

[[nodiscard]] constexpr auto abs_area(Circle circle) noexcept { return math::abs(area(circle)); }

template <u32 point_count, concepts::FloatingPoint Float = f64>
[[nodiscard]] constexpr auto regular_polygon_points() noexcept
{
    auto points           = std::array<Vector2_t<Float>, point_count>{};
    points[0].x           = 1; // first vertex is always (1, 0)
    const auto base_angle = static_cast<Float>(math::two_pi) / static_cast<Float>(point_count);
    for (auto i : loop::start_end(u32{1}, point_count))
    {
        const auto angle = static_cast<Float>(i) * base_angle;
        points[i]        = Vector2_t<Float>::from_polar({1.0, angle});
    }
    return points;
}

template <u32 point_count, concepts::FloatingPoint Float = f64>
[[nodiscard]] constexpr auto regular_polygon_points(Circle circumcircle) noexcept
{
    const auto centre = circumcircle.centre.cast<Float>();
    auto points       = regular_polygon_points<point_count, Float>();
    for (auto& point : points) { point = point * circumcircle.radius + centre; }
    return points;
}

template <concepts::FloatingPoint Float = f64>
[[nodiscard]] constexpr auto regular_polygon_points(u32 point_count) noexcept
{
    auto points           = SmallVector<Vector2_t<Float>, 64>(point_count);
    points[0].x           = 1; // first vertex is always (1, 0)
    const auto base_angle = static_cast<Float>(math::two_pi) / static_cast<Float>(point_count);
    for (auto i : loop::start_end(u32{1}, point_count))
    {
        const auto angle = static_cast<Float>(i) * base_angle;
        points[i]        = Vector2_t<Float>::from_polar({1.0, angle});
    }
    return points;
}

template <concepts::FloatingPoint Float = f64>
[[nodiscard]] constexpr auto regular_polygon_points(u32 point_count, Circle circumcircle) noexcept
{
    const auto centre = circumcircle.centre.cast<Float>();
    const auto radius = static_cast<Float>(circumcircle.radius);
    auto points       = regular_polygon_points<Float>(point_count);
    for (auto& point : points) { point = point * radius + centre; }
    return points;
}
} // namespace sm::math
