/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "Keymap.hpp"

namespace sm
{
void Keymap::clear()
{
    this->map.clear();
    this->actions.clear();
}

void Keymap::run() const
{
    for (size_t i = 0; i < map.size(); i++)
    {
        for (auto key : map[i])
        {
            if (!sf::Keyboard::isKeyPressed(key)) return;
        }

        actions[i]();
    }
}
} // namespace sm
