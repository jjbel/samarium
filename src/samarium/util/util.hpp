/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <stdexcept> // for invalid_argument

#include "range/v3/action/transform.hpp" // for transform
#include "range/v3/view/transform.hpp"   // for transform

#include "samarium/core/types.hpp" // for u8, u64

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

template <typename To, typename Range> [[nodiscard]] inline auto range_cast(const Range& range)
{
    return ranges::actions::transform(range, [](const auto& value) { return (To)value; });
}

template <typename To> [[nodiscard]] inline auto cast_view()
{
    return ranges::views::transform([](const auto& value) { return (To)value; });
}

template <typename T>
[[nodiscard]] inline auto project_view(T&& proj)
    requires std::is_member_object_pointer_v<T>
{
    return ranges::views::transform([proj](const auto& value) { return value.*proj; });
}
} // namespace sm::util
