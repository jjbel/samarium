/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <span>   // for span
#include <vector> // for vector

#include "range/v3/algorithm/rotate.hpp" // for rotate, rotate_fn

#include "samarium/core/types.hpp"   // for u64
#include "samarium/math/Vector2.hpp" // for Vector2

namespace sm
{
class Trail
{
    std::vector<Vector2> trail;

  public:
    const u64 max_length;

    explicit Trail(u64 length = 50) : max_length{length} { trail.reserve(length); }

    auto begin() noexcept { return trail.begin(); }
    auto end() noexcept { return trail.end(); }

    auto begin() const noexcept { return trail.cbegin(); }
    auto end() const noexcept { return trail.cend(); }

    auto cbegin() const noexcept { return trail.cbegin(); }
    auto cend() const noexcept { return trail.cend(); }

    auto size() const noexcept { return trail.size(); }
    auto empty() const noexcept { return trail.empty(); }

    auto operator[](u64 index) noexcept { return trail[index]; }
    auto operator[](u64 index) const noexcept { return trail[index]; }

    void push_back(Vector2 pos);

    [[nodiscard]] auto span() const -> std::span<const Vector2>;
};
} // namespace sm

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_TRAIL_IMPL)

#include <utility> // for move

#include "range/v3/algorithm/rotate.hpp" // for rotate

namespace sm
{
void Trail::push_back(Vector2 pos)
{
    if (this->max_length > this->trail.size()) { this->trail.push_back(pos); }
    else
    {
        ranges::rotate(this->trail, this->trail.begin() + 1);
        this->trail.back() = pos;
    }
}

std::span<const Vector2> Trail::span() const { return std::span(this->trail); }
} // namespace sm
#endif
