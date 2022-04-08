/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "../src/samarium/graphics/gradients.hpp"
#include "../src/samarium/math/complex.hpp"
#include "../src/samarium/samarium.hpp"
#include <limits>

using sm::print;

int main()
{
    for (size_t i = 0; i < 20; i++) { print(sm::random::random()); }

}
