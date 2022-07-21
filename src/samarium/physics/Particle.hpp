/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <compare>

#include "samarium/core/types.hpp"   // for f64
#include "samarium/math/Vector2.hpp" // for Vector2
#include "samarium/math/shapes.hpp"  // for Circle

namespace sm
{
struct Particle
{
    Vector2 pos{};
    Vector2 vel{};
    Vector2 acc{};
    f64 radius{1};
    f64 mass{1};

    [[nodiscard]] Circle as_circle() const noexcept;

    void apply_force(Vector2 force) noexcept;

    void update(f64 time_delta = 1.0 / 64) noexcept;

    auto operator<=>(const Particle&) const = default;
};
} // namespace sm
