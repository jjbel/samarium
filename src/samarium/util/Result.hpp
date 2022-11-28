/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <string> // for string

#include "tl/expected.hpp"

#include "Error.hpp"

namespace sm
{
template <typename T> using Result = tl::expected<T, std::string>;

struct BadResultAccess : Error
{
    explicit BadResultAccess(const std::string& message,
                             const SourceLocation& source_location = SourceLocation::current())
        : Error{message, source_location, "Bad Result Access"}
    {
    }
};

template <typename T>
[[nodiscard]] inline auto expect(Result<T>&& value,
                                 const SourceLocation& source_location = SourceLocation::current())
{
    if (value) { return std::move(value.value()); }
    throw BadResultAccess{value.error(), source_location};
}
} // namespace sm
