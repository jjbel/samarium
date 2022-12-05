/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "ankerl/unordered_dense.h"

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
