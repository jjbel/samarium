/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include <functional>
#include <vector>

#include "SFML/Window/Keyboard.hpp"

namespace sm
{
class Keymap
{
    using Keys_t   = std::vector<sf::Keyboard::Key>;
    using Action_t = std::function<void()>;

    std::vector<Keys_t> map;
    std::vector<Action_t> actions;

  public:
    Keymap() = default;

    explicit Keymap(std::vector<std::pair<Keys_t, Action_t>>&& input_keymap)
    {
        map.reserve(input_keymap.size());
        actions.reserve(input_keymap.size());

        for (const auto& [key, action] : input_keymap)
        {
            map.push_back(key);
            actions.push_back(action);
        }
    }

    void push_back(const Keys_t& keys, std::invocable auto&& fn)
    {
        map.push_back(keys);
        actions.emplace_back(fn);
    }

    void clear();

    void run() const;
};
} // namespace sm
