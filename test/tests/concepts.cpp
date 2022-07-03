/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2022 Jai Bellare
 * See <https://opensource.org/licenses/MIT/> or LICENSE.md
 * Project homepage: https://github.com/strangeQuark1041/samarium
 */

#include "samarium/core/concepts.hpp"

#include "catch2/catch_test_macros.hpp"

TEST_CASE("concepts")
{
    SECTION("Integral: Unsigned")
    {
        REQUIRE(sm::concepts::Integral<sm::u8>);
        REQUIRE(sm::concepts::Integral<sm::u16>);
        REQUIRE(sm::concepts::Integral<sm::u32>);
        REQUIRE(sm::concepts::Integral<sm::u64>);
    }

    SECTION("Integral: Signed")
    {
        REQUIRE(sm::concepts::Integral<sm::i8>);
        REQUIRE(sm::concepts::Integral<sm::i16>);
        REQUIRE(sm::concepts::Integral<sm::i32>);
        REQUIRE(sm::concepts::Integral<sm::i64>);
    }

    SECTION("FloatingPoint")
    {
        REQUIRE(sm::concepts::FloatingPoint<sm::f32>);
        REQUIRE(sm::concepts::FloatingPoint<sm::f64>);
    }

    SECTION("Number: Unsigned")
    {
        REQUIRE(sm::concepts::Number<sm::u8>);
        REQUIRE(sm::concepts::Number<sm::u16>);
        REQUIRE(sm::concepts::Number<sm::u32>);
        REQUIRE(sm::concepts::Number<sm::u64>);
    }

    SECTION("Number: Signed")
    {
        REQUIRE(sm::concepts::Number<sm::i8>);
        REQUIRE(sm::concepts::Number<sm::i16>);
        REQUIRE(sm::concepts::Number<sm::i32>);
        REQUIRE(sm::concepts::Number<sm::i64>);
    }

    SECTION("Number: Floating Point")
    {
        REQUIRE(sm::concepts::Number<sm::f32>);
        REQUIRE(sm::concepts::Number<sm::f64>);
    }
}
