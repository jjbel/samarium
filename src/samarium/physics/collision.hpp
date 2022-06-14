/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "../math/Vector2.hpp"
#include "../math/vector_math.hpp"

#include "Particle.hpp"

namespace sm::phys
{
[[nodiscard]] auto did_collide(const Particle& p1, const Particle& p2) -> std::optional<Vector2>;

void collide(Particle& p1, Particle& p2);

void collide(Particle& current, const Particle& old, const LineSegment& l);
} // namespace sm::phys
