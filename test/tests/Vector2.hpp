/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/math/Vector2.hpp"

TEST(util_format, vector2)
{

    using namespace std::literals;
    EXPECT_EQ(fmt::format("{}", sm::Vector2{2, 3}), "Vec( 2.000,  3.000)"s);
    EXPECT_EQ(fmt::format("{}", sm::Vector2_t<size_t>{2, 3}), "Vec(  2,   3)"s);
}

TEST(math_Vector2, literals)
{
    using namespace sm::literals;
    const auto a = 1.0_x;
    const auto b = sm::Vector2{1.0, 0};
    EXPECT_EQ(a, b);
}
