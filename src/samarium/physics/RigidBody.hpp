/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "samarium/core/types.hpp" // for f64
#include "samarium/math/Vec2.hpp"  // for Vec2, operator*, operator/

namespace sm
{
struct RigidBody
{
    Vec2 pos{};
    Vec2 vel{};
    Vec2 acc{};
    f64 mass{1};

    f64 a_pos{};
    f64 a_vel{};
    f64 a_acc{};
    f64 a_mass{};

    constexpr auto apply_force(Vec2 force) noexcept { acc += force / mass; }

    constexpr auto apply_torque(f64 torque) noexcept { a_acc += torque / a_mass; }

    constexpr auto apply_force(Vec2 force, Vec2 relative_pos) noexcept
    {
        acc += force / mass;
        // TODO use relative_pos
        (void)relative_pos;
    }

    constexpr auto update(f64 time_delta = 1.0 / 64) noexcept
    {
        vel += acc * time_delta;
        pos += vel * time_delta;
        acc = Vec2{}; // reset acceleration

        a_vel += a_acc * time_delta;
        a_pos += a_vel * time_delta;
        a_acc = 0.0; // reset acceleration
    }

    [[nodiscard]] constexpr auto operator==(const RigidBody&) const -> bool = default;
};
} // namespace sm
