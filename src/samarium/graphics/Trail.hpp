/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <span>
#include <vector>

#include "../math/Vector2.hpp"

namespace sm
{
class Trail
{
    std::vector<Vector2> trail;

  public:
    const size_t max_length;

    explicit Trail(size_t length = 50) : max_length{length} { trail.reserve(length); }

    [[nodiscard]] auto size() const -> size_t;

    void push_back(Vector2 pos);

    [[nodiscard]] auto span() const -> std::span<const Vector2>;
};
} // namespace sm
