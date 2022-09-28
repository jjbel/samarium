/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/math/Vector2.hpp"
#define SAMARIUM_HEADER_ONLY

// #include "samarium/samarium.hpp"

// using namespace sm;
// using namespace sm::literals;

int main()
{
    auto x = 42;
    x++;
    auto y = x;
    auto v = sm::Vector2{-12, 16};
    v.x -= -5;
    v = v.abs();
}
