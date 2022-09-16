/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */
#define SAMARIUM_HEADER_ONLY
#include "samarium/samarium.hpp"

using namespace sm;
using namespace sm::literals;

int main() { print(polynomial_from_roots<3>({-1.0, 1.0}).coeffs); }
