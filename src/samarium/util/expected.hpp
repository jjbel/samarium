/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include <stdexcept>

#include "tl/expected.hpp"

namespace sm
{
template <typename T, typename E> using expected = tl::expected<T, E>;

template <typename T> [[nodiscard]] inline auto expect(const expected<T, std::string>& value)
{
    if (value) { return value.value(); }
    else { throw std::runtime_error{"Bad expected access: " + value.error()}; }
}
} // namespace sm
