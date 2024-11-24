/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "samarium/gl/Context.hpp"        // for Context
#include "samarium/graphics/Color.hpp"    // for Color
#include "samarium/graphics/Gradient.hpp" // for Gradient
#include "samarium/gui/Window.hpp"        // for Window
#include "samarium/math/Vec2.hpp"         // for Vec2f

#include "vertices.hpp"

namespace sm::draw
{
void background(Color color);

/**
 * @brief               Fill the background with a gradient
 *
 * @param  window
 * @param  gradient
 * @param  angle
 */
void background(Window& window, const Gradient& gradient, f32 angle = 0.0);
} // namespace sm::draw

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_DRAW_IMPL)

#include "samarium/core/inline.hpp"
#include "samarium/gl/draw/background.hpp"

namespace sm::draw
{
SM_INLINE void background(Color color)
{
    glClearColor(static_cast<f32>(color.r) / 255.0F, static_cast<f32>(color.g) / 255.0F,
                 static_cast<f32>(color.b) / 255.0F, static_cast<f32>(color.a) / 255.0F);
    glClear(GL_COLOR_BUFFER_BIT);
}

void background(Window& window, const Gradient& gradient, f32 angle)
{
    /*
    The technique is to rotate a box to just cover the diagonal of the screen,
    and then fill the box with a gradient by putting vertices on its edges

    This is done in screenspace coordinates which distort with the screen aspect ratio, so adjust
    the angle
    */
    print("1");
    // use Vec2f to preserve obtuse angles (atan2)
    angle = (Vec2f::from_polar({.length = 1.0, .angle = angle}) *
             window.aspect_vector_max().cast<f32>())
                .angle();

    const auto transformed_angle = angle + std::numbers::pi_v<f32> / 4;
    const auto max =
        std::numbers::sqrt2_v<f32> *
        ranges::max(math::abs(std::sin(transformed_angle)), math::abs(std::cos(transformed_angle)));
    const auto size = gradient.colors.size();
    auto verts      = std::vector<gl::Vertex<gl::Layout::PosColor>>(size);
    print("2");

    for (auto i : loop::end(size))
    {
        const auto factor =
            interp::lerp_inverse<f32>(static_cast<f32>(i), {0.0F, static_cast<f32>(size - 1)});

        verts[2 * i].pos =
            Vec2f{static_cast<f32>(interp::clamped_lerp<f32>(factor, {-max, max})), max}.rotated(
                angle);

        verts[2 * i + 1].pos =
            Vec2f{static_cast<f32>(interp::clamped_lerp<f32>(factor, {-max, max})), -max}.rotated(
                angle);

        verts[2 * i].color     = gradient.colors[i];
        verts[2 * i + 1].color = gradient.colors[i];
    }
    print("3");

    vertices(window.context, verts, gl::Primitive::TriangleStrip, glm::mat4{1.0F});
    print("4");
}
} // namespace sm::draw
#endif
