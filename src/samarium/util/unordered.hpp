/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "ankerl/unordered_dense.h"

#include "samarium/math/Vector2.hpp"

namespace sm
{
template <class Key,
          class T,
          class Hash                 = ankerl::unordered_dense::hash<Key>,
          class KeyEqual             = std::equal_to<Key>,
          class AllocatorOrContainer = std::allocator<std::pair<Key, T>>,
          class Bucket               = ankerl::unordered_dense::bucket_type::standard>
using Map = ankerl::unordered_dense::map<Key, T, Hash, KeyEqual, AllocatorOrContainer, Bucket>;

template <class Key,
          class Hash                 = ankerl::unordered_dense::hash<Key>,
          class KeyEqual             = std::equal_to<Key>,
          class AllocatorOrContainer = std::allocator<Key>,
          class Bucket               = ankerl::unordered_dense::bucket_type::standard>
using Set = ankerl::unordered_dense::set<Key, Hash, KeyEqual, AllocatorOrContainer, Bucket>;
} // namespace sm

template <typename T> struct ankerl::unordered_dense::hash<sm::Vector2_t<T>>
{
    using is_avalanching = void;

    [[nodiscard]] auto operator()(const sm::Vector2_t<T>& vec) const noexcept -> uint64_t
    {
        static_assert(std::has_unique_object_representations_v<sm::Vector2_t<T>>);
        return ankerl::unordered_dense::detail::wyhash::hash(&vec, sizeof(vec));
    }
};
