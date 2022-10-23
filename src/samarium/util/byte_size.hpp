/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "range/v3/range/concepts.hpp"
#include "range/v3/range/primitives.hpp"

namespace sm::util
{
constexpr auto element_byte_size(const auto& array) { return sizeof(array[0]); }

constexpr auto range_byte_size(const ranges::range auto& array)
{
    return ranges::size(array) * element_byte_size(array);
}
} // namespace sm::util
