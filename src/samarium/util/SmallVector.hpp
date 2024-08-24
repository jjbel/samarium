/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "ankerl/svector.h"

namespace sm
{
template <typename T, size_t MinInlineCapacity>
using SmallVector = ankerl::svector<T, MinInlineCapacity>;
}
