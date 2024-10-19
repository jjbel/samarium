/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
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

    // add back rotation?
    // maybe useful for animating scaled rotations, or for vector math
    // but for drawing, most likely will be axis aligned, ie multiple of 90 degrees

    [[nodiscard]] static constexpr auto map_boxes_from_to(BoundingBox<f64> from,
                                                          BoundingBox<f64> to)
    {
        // equations:
        // min1 * scale + pos = min2
        // width1 * scale = width2
        const auto scale = to.diagonal() / from.diagonal();
        const auto pos   = to.min - scale * from.min;
        return Transform{pos, scale};
    }

    // TODO we shouldn't have 2 overloads doing the same thing. remove apply
    [[nodiscard]] constexpr auto apply(Vector2 vec) const noexcept { return vec * scale + pos; }
    [[nodiscard]] constexpr auto operator()(Vector2 vec) const noexcept
    {
        return vec * scale + pos;
    }

    [[nodiscard]] constexpr auto apply(Vector2f vec) const noexcept
    {
        return vec * scale.cast<f32>() + pos.cast<f32>();
    }
    [[nodiscard]] constexpr auto operator()(Vector2f vec) const noexcept
    {
        return vec * scale.cast<f32>() + pos.cast<f32>();
    }

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

    [[nodiscard]] constexpr auto inverse() const noexcept
    {
        const auto new_scale = 1.0 / scale;
        return Transform{-pos * new_scale, new_scale};
    }

    [[nodiscard]] auto as_matrix() const noexcept
    {
        // VVIMP
        // glm translate POST multiplies a translation matrix
        // http://www.c-jump.com/bcc/common/Talk3/Math/GLM/GLM.html#W01_0110_glmtranslate
        // bug I encountered: aspect ratio was getting multiplied twice
        const auto gl_scale = glm::vec3{static_cast<f32>(scale.x), static_cast<f32>(scale.y), 1.0F};
        const auto gl_pos   = glm::vec3{static_cast<f32>(pos.x), static_cast<f32>(pos.y), 0.0F};
        // return glm::translate(glm::scale(gl_scale), gl_pos);
        return glm::scale(glm::translate(glm::mat4(1.0F), gl_pos), gl_scale);
    }

    [[nodiscard]] operator glm::mat4() const noexcept { return as_matrix(); }

    [[nodiscard]] constexpr auto then(Transform next) const noexcept
    {
        return Transform{pos * next.scale + next.pos, scale * next.scale};
    }
};
} // namespace sm
