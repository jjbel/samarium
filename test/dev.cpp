/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/graphics/Image.hpp"
#include "samarium/samarium.hpp"
#include "samarium/util/Stopwatch.hpp"
#include "samarium/util/file.hpp"
#include "samarium/util/fpng/fpng.hpp"
#include <chrono>
#include <thread>

using namespace sm;
using namespace sm::literals;

int main()
{
    auto body = RigidBody{};
    auto im   = expect(file::read_image("/home/jb/Pictures/alphatuari-f1.jpg"));

    auto watch = Stopwatch{};
    file::write(file::Png{}, im);
    watch.print();

    watch.reset();
    file::write(file::Targa{}, im);
    watch.print();

    watch.reset();
    file::write(file::Bmp{}, im);
    watch.print();
}
