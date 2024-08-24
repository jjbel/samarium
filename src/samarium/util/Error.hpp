/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <stdexcept>   // for runtime_error
#include <string>      // for string
#include <string_view> // for string_view

#include "SourceLocation.hpp"
#include "format.hpp"

namespace sm
{
namespace detail
{
[[nodiscard]] inline auto format_message(std::string_view message,
                                         const SourceLocation& source_location,
                                         std::string_view name)
{
    return fmt::format("samarium: {}\nat {}\n{}",
                       fmt::styled(name, fg(fmt::color::crimson) | fmt::emphasis::bold),
                       source_location, fmt::styled(message, fmt::emphasis::bold));
}
} // namespace detail

struct Error : public std::runtime_error
{
    explicit Error(std::string_view message              = "",
                   const SourceLocation& source_location = SourceLocation::current(),
                   std::string_view name                 = "Error")
        : std::runtime_error{detail::format_message(message, source_location, name)}
    {
    }
};
} // namespace sm
