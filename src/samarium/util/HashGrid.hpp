/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
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

#include "StaticVector.hpp"
#include "unordered.hpp"

template <> struct ankerl::unordered_dense::hash<sm::Vector2_t<sm::i32>>
{
    using is_avalanching = void;

    [[nodiscard]] auto operator()(const sm::Vector2_t<sm::i32>& vec) const noexcept -> uint64_t
    {
        static_assert(std::has_unique_object_representations_v<sm::Vector2_t<sm::i32>>);
        return ankerl::unordered_dense::detail::wyhash::hash(&vec, sizeof(vec));
    }
};

namespace sm
{
template <typename T, usize CellInlineCapacity = 32> struct HashGrid
{
    using Key       = Vector2_t<i32>;
    using Container = Map<Key, StaticVector<T, CellInlineCapacity>>;

    Container map{};
    const f64 spacing;

    explicit HashGrid(f64 spacing_ = 1.0) : spacing{spacing_} {}

    auto to_coords(Vector2 pos) const
    {
        // return Key::make(math::floor_to_nearest(pos.x, spacing),
        //                  math::floor_to_nearest(pos.y, spacing));
        return (pos / spacing).cast<i32>();
    }

    auto insert(Vector2 pos, T value) { map[to_coords(pos)].push_back(value); }

    auto find(Vector2 pos) -> typename Container::iterator { return map.find(to_coords(pos)); }

    auto find(Vector2 pos) const -> typename Container::const_iterator
    {
        return map.find(to_coords(pos));
    }

    auto neighbors(Vector2 pos) const
    {
        constexpr auto offsets = std::to_array<Vector2>(
            {{0, 0}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}});

        auto out = StaticVector<T, CellInlineCapacity * offsets.size()>();

        for (auto offset : offsets)
        {
            const auto& iter = map.find(to_coords(pos + offset * spacing));
            if (iter == map.end()) { continue; }
            for (const auto& i : iter->second) { out.push_back(i); }
        }

        return out;
    }
};
} // namespace sm
