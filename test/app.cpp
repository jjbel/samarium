/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "../src/samarium/util/ostream.hpp"
#include <bitset>
#include <complex>

#include "../src/samarium/graphics/gradients.hpp"
#include "../src/samarium/math/complex.hpp"
#include "../src/samarium/samarium.hpp"

using namespace sm;

int main()
{
    for (int i = 0; i < 12; i++) print(random::random());
}
