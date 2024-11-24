/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <span>   // for span
#include <vector> // for vector

#include "range/v3/algorithm/rotate.hpp" // for rotate, rotate_fn

#include "samarium/core/types.hpp" // for u64
#include "samarium/math/Vec2.hpp"  // for Vec2

namespace sm
{
struct Trail
{
    std::vector<Vec2> trail;
    u64 max_length;

    explicit Trail(u64 length = 50) : max_length{length} { trail.reserve(length); }

    [[nodiscard]] auto begin() noexcept { return trail.begin(); }
    [[nodiscard]] auto end() noexcept { return trail.end(); }

    [[nodiscard]] auto begin() const noexcept { return trail.cbegin(); }
    [[nodiscard]] auto end() const noexcept { return trail.cend(); }

    [[nodiscard]] auto cbegin() const noexcept { return trail.cbegin(); }
    [[nodiscard]] auto cend() const noexcept { return trail.cend(); }

    [[nodiscard]] auto size() const noexcept { return trail.size(); }
    [[nodiscard]] auto empty() const noexcept { return trail.empty(); }

    auto operator[](u64 index) noexcept { return trail[index]; }
    auto operator[](u64 index) const noexcept { return trail[index]; }

    void push_back(Vec2 pos);

    [[nodiscard]] auto span() const -> std::span<const Vec2>;
};
} // namespace sm

#if defined(SAMARIUM_HEADER_ONLY) || defined(SAMARIUM_TRAIL_IMPL)

#include <utility> // for move

#include "range/v3/algorithm/rotate.hpp" // for rotate

namespace sm
{
void Trail::push_back(Vec2 pos)
{
    if (this->max_length > this->trail.size()) { this->trail.push_back(pos); }
    else
    {
        ranges::rotate(this->trail, this->trail.begin() + 1);
        this->trail.back() = pos;
    }
}

auto Trail::span() const -> std::span<const Vec2> { return {this->trail}; }
} // namespace sm
#endif
