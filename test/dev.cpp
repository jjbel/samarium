/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#pragma once

#include "samarium/samarium.hpp"
#include "samarium/util/Result.hpp"
#include "samarium/util/file.hpp"

using namespace sm;

auto main() -> i32
{
    print("Hello");
    auto x = expect(file::read("/usr/nonexistent"));
}
