/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "range/v3/algorithm/all_of.hpp"
#include "range/v3/view/enumerate.hpp" // for enumerate

#include "Keyboard.hpp"

#include "samarium/util/print.hpp"

namespace sm::Keyboard
{
void OnKeyPress::operator()() const
{
    if (ranges::all_of(key_set, is_key_pressed)) { action(); }
}

void OnKeyDown::operator()()
{
    const auto current = ranges::all_of(key_set, is_key_pressed);
    if (!previous && current) { action(); }
    previous = current;
}

void OnKeyUp::operator()()
{
    const auto current = ranges::all_of(key_set, is_key_pressed);
    if (!current && previous) { action(); }
    previous = current;
}

void Keymap::clear() { this->actions.clear(); }

void Keymap::run() const
{
    for (const auto& action : actions) { action(); }
}
} // namespace sm::Keyboard
