/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "types.hpp"

namespace sm
{
struct Version
{
    u8 major{1};
    u8 minor{1};
    u8 patch{0};
};

[[maybe_unused]] static constexpr auto version = Version{};
} // namespace sm
