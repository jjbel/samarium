/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <array>

#include "samarium/core/types.hpp"
#include "samarium/math/Extents.hpp"
#include "samarium/math/Vector2.hpp"
#include "samarium/math/math.hpp"

#include "range/v3/algorithm/copy.hpp"

#include "Map.hpp"
#include "StaticVector.hpp"

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
template <typename T, usize Count = 32> struct HashGrid
{
    using Key       = Vector2_t<i32>;
    using Container = Map<Key, StaticVector<T, Count>>;

    Container map{};
    const f64 spacing;

    explicit HashGrid(f64 spacing_ = 1.0) : spacing{spacing_} {}

    auto to_coords(Vector2 pos) const
    {
        return Key::make(math::floor_to_nearest(pos.x, spacing),
                         math::floor_to_nearest(pos.y, spacing));
    }

    auto insert(Vector2 pos, T value) { map[to_coords(pos)].push_back(value); }

    typename Container::iterator find(Vector2 pos) { return map.find(to_coords(pos)); }

    typename Container::const_iterator find(Vector2 pos) const { return map.find(to_coords(pos)); }

    auto neighbors(Vector2 pos) const
    {
        constexpr auto offsets = std::to_array<Key>(
            {{0, 0}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}});

        const auto coords = to_coords(pos);
        auto out          = StaticVector<T, Count * 9>(Count * 9);
        auto current_size = usize(0);

        for (auto i : range(9))
        {
            const auto iter = map.find(coords + offsets[i]);
            if (iter == map.end()) { continue; }
            ranges::copy(iter->second, out.begin() + current_size + iter->second.size());
            current_size += iter->second.size();
        }
        out.resize(current_size);
        return out;
    }
};
} // namespace sm
