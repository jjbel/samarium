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

    Keymap(std::vector<std::pair<Keys_t, Action_t>>&& _)
    {
        map.reserve(_.size());
        actions.reserve(_.size());

        for (auto&& i : _)
        {
            map.push_back(i.first);
            actions.push_back(i.second);
        }
    }

    template <typename Fn> void push_back(Keys_t keys, Fn&& fn) requires std::invocable<Fn>
    {
        map.push_back(keys);
        actions.emplace_back(fn);
    }

    void clear();

    void run() const;
};
} // namespace sm
