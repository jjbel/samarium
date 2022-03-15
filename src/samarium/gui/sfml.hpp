/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "SFML/Graphics.hpp"
#include "SFML/Window.hpp"

namespace sm
{
template <typename T> [[nodiscard]] constexpr Vector2_t<T> sfml(sf::Vector2<T> vec) noexcept
{
    return {vec.x, vec.y};
}
} // namespace sm
