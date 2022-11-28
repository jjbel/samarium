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
#include <vector>

using namespace sm;

auto main() -> i32
{
    print("Hello");
    auto search_paths = std::vector<file::Path>{"/home/jb/sm_", "/home/jb/sm"};
    print("Found:", expect(file::find("file.hpp", {"/home/jb", "/home/jb/sm"})));
}
