/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <stdexcept> // for runtime_error
#include <string>    // for string

#include "tl/expected.hpp"

namespace sm
{
template <typename T> using Result = tl::expected<T, std::string>;

struct BadResultAccess : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

template <typename T> [[nodiscard]] inline auto expect(Result<T>&& value)
{
    if (value) { return std::move(value.value()); }
    throw BadResultAccess{value.error()};
}

template <typename T> [[nodiscard]] inline auto operator*(Result<T>&& value)
{
    if (value) { return std::move(value.value()); }
    throw BadResultAccess{value.error()};
}
} // namespace sm
