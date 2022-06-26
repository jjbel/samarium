/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <optional> // for optional

#include "samarium/core/types.hpp"   // for f64
#include "samarium/math/Vector2.hpp" // for Vector2

namespace sm
{
struct LineSegment;
}
namespace sm
{
struct Particle;
}

namespace sm::phys
{
[[nodiscard]] auto did_collide(const Particle& p1, const Particle& p2) -> std::optional<Vector2>;

void collide(Particle& p1, Particle& p2, f64 damping = 1.0);

void collide(Particle& current, const LineSegment& l, f64 dt);
} // namespace sm::phys
