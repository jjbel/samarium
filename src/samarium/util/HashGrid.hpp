/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
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
template <typename T, usize CellInlineCapacity = 127, typename Float = f64> struct HashGrid
{
    using Vector2Float = Vector2_t<Float>;

    using Key = Vector2_t<i32>;

    // using Value    = SmallVector<T, CellInlineCapacity>;
    // TODO small vector isn't worth it?
    using Value = std::vector<T>;

    using Container = Map<Key, Value>;

    Container map{};
    const Float spacing;

    explicit HashGrid(Float spacing_ = 1.0F) : spacing{spacing_} {}

    auto to_coords(Vector2Float pos) const
    {
        return Key::make(math::floor_to_nearest(pos.x, spacing),
                         math::floor_to_nearest(pos.y, spacing));
    }

    auto insert(Vector2Float pos, T value) { map[to_coords(pos)].push_back(value); }

    auto cell_containing(Vector2Float pos) -> typename Container::iterator
    {
        return map.find(to_coords(pos));
    }

    auto cell_containing(Vector2Float pos) const -> typename Container::const_iterator
    {
        return map.find(to_coords(pos));
    }

    auto neighbors(Vector2Float pos) const
    {
        constexpr auto offsets = std::to_array<Vector2Float>(
            {{0, 0}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}});

        auto out = std::vector<T>();
        // auto out = SmallVector<T, math::min(CellInlineCapacity * offsets.size(), u64(127))>();

        for (auto offset : offsets)
        {
            const auto iter = cell_containing(pos + offset * spacing);
            if (iter == map.end()) { continue; }
            for (const auto& i : iter->second) { out.push_back(i); }
        }

        return out;
    }

    auto print_occupancy()
    {
        // u64 min_occ = CellInlineCapacity;
        // u64 max_occ = 0;
        // u64 total   = 0;
        // for (const auto& [key, vec] : map)
        // {
        //     min_occ = std::min(min_occ, vec.size());
        //     max_occ = std::max(max_occ, vec.size());
        //     total += vec.size();
        // }
        // fmt::print("min_cc: {} max_occ: {} | total size: {} total inline capacity: {}\n",
        // min_occ,
        //            max_occ, total, CellInlineCapacity * map.size());
        // TODO not giving correct values
    }
};
} // namespace sm
