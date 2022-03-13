/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include <vector>
#include <functional>

#include "SFML/Window/Keyboard.hpp"

namespace sm
{
class Keymap
{
    using Keys_t   = std::vector<sf::Keyboard::Key>;
    using Action_t = std::function<void()>;

    std::vector<Keys_t> map;
    std::vector<Action_t> functions;

  public:
    Keymap() = default;

    Keymap(std::vector<std::pair<Keys_t, Action_t>>&& _)
    {
        for (auto&& i : _)
        {
            map.push_back(i.first);
            functions.push_back(i.second);
        }
    }

    template <typename Fn> void push_back(Keys_t keys, Fn&& fn) requires std::invocable<Fn>
    {
        map.push_back(keys);
        functions.emplace_back(fn);
    }

    void clear();

    void run() const;
};
} // namespace sm
