/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "samarium/graphics/Trail.hpp"   // for Trail
#include "samarium/gui/Window.hpp"       // for Window
#include "samarium/util/FunctionRef.hpp" // for FunctionRef

namespace sm::draw
{
void trail(Window& window, const Trail& trail, Color color, f32 thickness);
void trail(Window& window,
           const Trail& trail,
           FunctionRef<Color(f64)> dynamic_color,
           f32 thickness);
} // namespace sm::draw


#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_DRAW_IMPL)

#include "samarium/gl/draw/poly.hpp"
#include "samarium/gl/draw/trail.hpp"

namespace sm::draw
{
SM_INLINE void trail(Window& window, const Trail& trail, Color color, f32 thickness)
{
    if (trail.size() < 2) { return; }
    const auto points = trail.trail |
                        ranges::views::transform([](Vector2 vec) { return vec.cast<f32>(); }) |
                        ranges::to<std::vector>;
    // TODO ranges::to iwyu
    // TODO draw
    // polyline(window, points, thickness, color);
}

// SM_INLINE void
// trail(Window& window, const Trail& trail, FunctionRef<Color(f64)> dynamic_color, f32 thickness)
// {
// TODO
// }
} // namespace sm::draw

#endif
