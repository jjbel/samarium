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
    // TODO check < 2 in polyline or not?
    if (trail.size() < 2) { return; }
    polyline(window, trail.trail, thickness, color);
}

// SM_INLINE void
// trail(Window& window, const Trail& trail, FunctionRef<Color(f64)> dynamic_color, f32 thickness)
// {
// TODO
// }
} // namespace sm::draw

#endif
