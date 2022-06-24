/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "samarium/core/types.hpp"     // for f64
#include "samarium/math/Transform.hpp" // for Transform
#include "samarium/math/Vector2.hpp"   // for Vector2
namespace sf
{
class RenderWindow;
}

namespace sm
{
struct Mouse
{
    enum class Button
    {
        Left,
        Right
    };

    Vector2 current_pos;
    Vector2 old_pos;
    f64 scroll_amount{};
    bool left;
    bool middle;
    bool right;

    explicit Mouse(const sf::RenderWindow& window) { this->update(window); }

    void update(const sf::RenderWindow& window);

    [[nodiscard]] auto apply(Transform transform, Mouse::Button btn = Button::Left) const
        -> Transform;

    [[nodiscard]] auto vel() const -> Vector2;
};
} // namespace sm
