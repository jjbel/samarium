/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <stdexcept>

#include "fmt/format.h"
#include "tl/expected.hpp"

namespace sm
{
template <typename T, typename E> using expected = tl::expected<T, E>;

template <typename T, typename E> [[nodiscard]] inline auto expect(expected<T, E>&& value)
{
    if (value) { return std::move(value.value()); }
    else { throw std::runtime_error{fmt::format("Bad expected access:\n{}", value.error())}; }
}
} // namespace sm
