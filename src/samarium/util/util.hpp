/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <stdexcept> // for invalid_argument

#include "range/v3/action/transform.hpp"

#include "../core/types.hpp" // for u8

namespace sm::util
{
[[nodiscard]] consteval u64 strlen(const char* str) // NOSONAR
{
    return *str ? 1UL + strlen(str + 1UL) : 0UL;
}

[[nodiscard]] consteval u8 hex_to_int_safe(char ch)
{
    if ('0' <= ch && ch <= '9') { return static_cast<u8>(ch - '0'); }
    if ('a' <= ch && ch <= 'f') { return static_cast<u8>(ch - 'a' + 10); }
    if ('A' <= ch && ch <= 'F') { return static_cast<u8>(ch - 'A' + 10); }
    throw std::invalid_argument("hex character must be 0-9, a-f, or A-F");
}

template <typename Range, typename To> [[nodiscard]] inline auto range_cast(const Range& range)
{
    return ranges::actions::transform(range, [](const auto& value) { return (To)value; });
}
} // namespace sm::util
