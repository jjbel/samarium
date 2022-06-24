/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "SFML/Window/Mouse.hpp"          // for Mouse
#include "SFML/Graphics/RenderWindow.hpp" // for RenderWindow

#include "samarium/gui/sfml.hpp"       // for sfml
#include "samarium/math/Transform.hpp" // for Transform

#include "Mouse.hpp"

namespace sm
{
Vector2 Mouse::vel() const { return this->current_pos - this->old_pos; }

void Mouse::update(const sf::RenderWindow& window)
{
    this->old_pos     = this->current_pos;
    this->current_pos = sfml(sf::Mouse::getPosition(window)).as<f64>();
    this->left        = sf::Mouse::isButtonPressed(sf::Mouse::Left);
    this->middle      = sf::Mouse::isButtonPressed(sf::Mouse::Middle);
    this->right       = sf::Mouse::isButtonPressed(sf::Mouse::Right);
}

Transform Mouse::apply(Transform transform, Mouse::Button btn) const
{
    if ((btn == Mouse::Button::Left && this->left) || (btn == Mouse::Button::Right && this->right))
        transform.pos += this->vel();

    return transform;
}
} // namespace sm
