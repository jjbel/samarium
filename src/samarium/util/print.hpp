/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include <string_view> // for string_view

#include "fmt/color.h"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "fmt/std.h"

#include "format.hpp"                       // for fmt::formatter
#include "samarium/util/SourceLocation.hpp" // for SourceLocation

namespace sm
{
template <typename... Args> auto print(Args&&... args)
{
    (fmt::print("{} ", std::forward<Args>(args)), ...);
    fmt::print("\n");
}

template <typename T> auto print_single(T&& arg) { print(std::forward<T>(arg)); }

inline auto log(std::string_view message, SourceLocation location = SourceLocation::current())
{
    fmt::print(fg(fmt::color::steel_blue) | fmt::emphasis::bold, "[{}:{}: {}]: {}",
               location.file_name(), location.line(), location.function_name(), message);
}
} // namespace sm
