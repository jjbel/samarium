/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/samarium.hpp"
#include "samarium/util/Stopwatch.hpp"
#include "samarium/util/file.hpp"
#include <stdexcept>

using namespace sm;
using namespace sm::literals;

auto main() -> i32
{
    auto watch = Stopwatch{};
    // auto window = Window{{1800, 900}};
    watch.print();
    print(*file::read("/home/test/dev.\pp"));
    // throw std::runtime_error("/home/jb/sm/test/dev.pp does not exist");
}
