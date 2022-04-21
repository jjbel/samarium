/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/samarium.hpp"
#include "samarium/util/random.hpp"

using namespace sm;

int main() {for(auto i : range(10)) print(random::choice({1, -2, -4, 5})); }
