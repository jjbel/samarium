/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/math/Vector2.hpp"

TEST_CASE("util.format", "Vector2")
{
    using namespace std::literals;
    CHECK(fmt::format("{}", sm::Vector2{2, 3}) == "Vec( 2.000,  3.000)"s);
    CHECK(fmt::format("{}", sm::Indices{2, 3}) == "Vec(  2,   3)"s);
}

TEST_CASE("literals", "math.Vector2")
{
    using namespace sm::literals;

    const auto a_x = 1.0_x;
    const auto b_x = sm::Vector2{1.0, 0};
    CHECK(a_x == b_x);

    const auto a_y = 1.0_y;
    const auto b_y = sm::Vector2{0, 1.0};
    CHECK(a_y == b_y);
}

TEST_CASE("math.Vector2", "Vector2")
{
    static_assert(std::is_same_v<sm::Vector2::value_type, sm::f64>);

    SECTION("x vector")
    {
        const auto a = sm::Vector2{1.0, 0.0};
        CHECK(sm::math::almost_equal(a.length(), 1.0));
        CHECK(sm::math::almost_equal(a.length_sq(), 1.0));
        CHECK(sm::math::almost_equal(a.angle(), 0.0));
        CHECK(sm::math::almost_equal(a.slope(), 0.0));
    }

    SECTION("xy vector")
    {
        const auto b = sm::Vector2{1.0, 1.0};
        CHECK(sm::math::almost_equal(b.length(), std::sqrt(2.0)));
        CHECK(sm::math::almost_equal(b.length_sq(), 2.0));
        CHECK(sm::math::almost_equal(b.angle(), sm::math::to_radians(45)));
        CHECK(sm::math::almost_equal(b.slope(), 1.0));
    }

    SECTION("y vector")
    {
        const auto c = sm::Vector2{0.0, 1.0};
        CHECK(sm::math::almost_equal(c.length(), 1.0));
        CHECK(sm::math::almost_equal(c.length_sq(), 1.0));
        CHECK(sm::math::almost_equal(c.angle(), sm::math::to_radians(90)));
    }

    SECTION("origin vector")
    {
        const auto d = sm::Vector2{0.0, 0.0};
        CHECK(sm::math::almost_equal(d.length(), 0.0));
        CHECK(sm::math::almost_equal(d.length_sq(), 0.0));
    }
}
