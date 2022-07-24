/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "range/v3/algorithm/all_of.hpp"
#include "range/v3/view/enumerate.hpp" // for enumerate

#include "samarium/math/Extents.hpp" // for range
#include "samarium/util/print.hpp"   // for range

#include "Keyboard.hpp"

namespace sm
{
void Keymap::clear()
{
    this->map.clear();
    this->actions.clear();
}

void Keymap::run() const
{
    print(map.size(), actions.size());
    for (auto [i, key_combination] : ranges::views::enumerate(map))
    {
        if (ranges::all_of(key_combination, [](auto key) { return Keyboard::is_key_pressed(key); }))
        {
            actions[i]();
        }
    }
}
} // namespace sm
