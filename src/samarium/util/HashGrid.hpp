/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <array>

#include "range/v3/algorithm/copy.hpp"

#include "samarium/core/types.hpp"
#include "samarium/math/Extents.hpp"
#include "samarium/math/Vector2.hpp"
#include "samarium/math/math.hpp"

#include "SmallVector.hpp"
#include "unordered.hpp"

namespace sm
{
template <typename T, usize CellInlineCapacity = 127> struct HashGrid
{
    using Key       = Vector2_t<i32>;
    using Container = Map<Key, SmallVector<T, CellInlineCapacity>>;

    Container map{};
    const f64 spacing;

    explicit HashGrid(f64 spacing_ = 1.0) : spacing{spacing_} {}

    auto to_coords(Vector2 pos) const
    {
        return Key::make(math::floor_to_nearest(pos.x, spacing),
                         math::floor_to_nearest(pos.y, spacing));
    }

    auto insert(Vector2 pos, T value) { map[to_coords(pos)].push_back(value); }

    auto cell_containing(Vector2 pos) -> typename Container::iterator
    {
        return map.find(to_coords(pos));
    }

    auto cell_containing(Vector2 pos) const -> typename Container::const_iterator
    {
        return map.find(to_coords(pos));
    }

    auto neighbors(Vector2 pos) const
    {
        constexpr auto offsets = std::to_array<Vector2>(
            {{0, 0}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}});

        auto out = SmallVector<T, math::min(CellInlineCapacity * offsets.size(), u64(127))>();

        for (auto offset : offsets)
        {
            const auto iter = cell_containing(pos + offset * spacing);
            if (iter == map.end()) { continue; }
            for (const auto& i : iter->second) { out.push_back(i); }
        }

        return out;
    }
};
} // namespace sm
