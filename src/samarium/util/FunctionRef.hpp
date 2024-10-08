/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022-2024 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/jjbel/samarium
 */

#pragma once

#include "tl/function_ref.hpp"

namespace sm
{
template <typename F> using FunctionRef = tl::function_ref<F>;
}
