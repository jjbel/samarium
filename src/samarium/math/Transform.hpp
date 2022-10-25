/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "glm/ext/matrix_float4x4.hpp" // for mat4
#include "glm/gtx/transform.hpp"

#include "BoundingBox.hpp"
#include "samarium/math/math.hpp"
#include "shapes.hpp"

namespace sm
{
class Transform
{
  public:
    Vector2 pos{};
    Vector2 scale{1.0, 1.0};
    f64 rotation{};

    [[nodiscard]] constexpr auto apply(Vector2 vec) const noexcept { return vec * scale + pos; }

    [[nodiscard]] constexpr auto apply(const BoundingBox<f64>& bounding_box) const noexcept
    {
        return BoundingBox<f64>{apply(bounding_box.min), apply(bounding_box.max)}.validated();
    }

    [[nodiscard]] constexpr auto apply_inverse(Vector2 vec) const noexcept
    {
        return (vec - pos) / scale;
    }

    [[nodiscard]] constexpr auto apply_inverse(const BoundingBox<f64>& bounding_box) const noexcept
    {
        return BoundingBox<f64>::find_min_max(
            this->apply_inverse(bounding_box.min),
            this->apply_inverse(
                bounding_box.max)); // -ve sign may invalidate min, max, so recalculate it
    }

    [[nodiscard]] constexpr auto apply_inverse(const LineSegment& l) const noexcept
    {
        return LineSegment{// -ve sign may invalidate min, max so recalculate it
                           apply_inverse(l.p1), apply_inverse(l.p2)};
    }

    [[nodiscard]] auto as_matrix() const noexcept
    {
        return glm::translate(glm::rotate(glm::scale(glm::vec3{static_cast<f32>(scale.x),
                                                               static_cast<f32>(scale.y), 1.0F}),
                                          static_cast<f32>(rotation), glm::vec3{0.0F, 0.0F, 1.0F}),
                              glm::vec3{static_cast<f32>(pos.x), static_cast<f32>(pos.y), 0.0F});
    }

    [[nodiscard]] operator glm::mat4() const noexcept { return as_matrix(); }
};
} // namespace sm
