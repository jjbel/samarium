/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <span>
#include <vector>

#include "samarium/math/Vector2.hpp"

namespace sm
{
class Trail
{
    std::vector<Vector2> trail;

  public:
    const size_t max_length;

    explicit Trail(size_t length) : trail(), max_length{length} {}

    void push_back(Vector2 pos);

    std::span<Vector2> span() const;
};
} // namespace sm
