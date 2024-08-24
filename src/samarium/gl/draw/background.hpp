/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "samarium/gl/Context.hpp"        // for Context
#include "samarium/graphics/Color.hpp"    // for Color
#include "samarium/graphics/Gradient.hpp" // for Gradient
#include "samarium/gui/Window.hpp"        // for Window
#include "samarium/math/Vector2.hpp"      // for Vector2f

namespace sm::draw
{
void background(Color color);

/**
 * @brief               Fill the background with a gradient
 *
 * @tparam size
 * @param  window
 * @param  gradient
 * @param  angle
 */
template <u64 size> void background(Window& window, const Gradient<size>& gradient, f32 angle = 0.0)
{
    /*
    The technique is to rotate a box to just cover the diagonal of the screen,
    and then fill the box with a gradient by putting vertices on its edges

    This is done in screenspace coordinates which distort with the screen aspect ratio, so adjust
    the angle
    */

    // use Vector2f to preserve obtuse angles (atan2)
    angle = (Vector2f::from_polar({.length = 1.0, .angle = angle}) *
             window.aspect_vector_max().cast<f32>())
                .angle();

    const auto transformed_angle = angle + std::numbers::pi_v<f32> / 4;
    const auto max =
        std::numbers::sqrt2_v<f32> *
        ranges::max(math::abs(std::sin(transformed_angle)), math::abs(std::cos(transformed_angle)));

    auto verts = std::array<gl::Vertex<gl::Layout::PosColor>, size * 2>();

    for (auto i : loop::end(size))
    {
        const auto factor =
            interp::lerp_inverse<f32>(static_cast<f64>(i), {0.0F, static_cast<f32>(size - 1)});

        verts[2 * i].pos =
            Vector2f{static_cast<f32>(interp::clamped_lerp<f32>(factor, {-max, max})), max}.rotated(
                angle);

        verts[2 * i + 1].pos =
            Vector2f{static_cast<f32>(interp::clamped_lerp<f32>(factor, {-max, max})), -max}
                .rotated(angle);

        verts[2 * i].color     = gradient.colors[i];
        verts[2 * i + 1].color = gradient.colors[i];
    }

    vertices(window, verts, gl::Primitive::TriangleStrip, glm::mat4{1.0F});
}
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
} // namespace sm::draw
#endif
